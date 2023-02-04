"""Financial programming utilities.

v0.4.1 alpha: API MAY (IN FACT, CERTAINLY WILL) BREAK AT ANY TIME!

author: [@sparshsah](https://github.com/sparshsah)


# Notes
* Each `get_est_{whatever}_of_r()` function estimates its
    specified market param based on the given data sample.
* Each `get_exante_{whatever}_of_w()` function calculates its
    specified portfolio stat taking as ground truth the given market params.
"""

from __future__ import annotations

from typing import Tuple, Dict, Union, Optional
from collections import OrderedDict
import pandas as pd
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
import foggy_statslib.core as fsc
import foggy_statslib.stats.est.tsa as fsset

from foggy_statslib.core import FloatSeries, FloatDF, FloatSeriesOrDF, Floatlike, DEFAULT_DATETIME_FREQ
from foggy_statslib.stats.est.tsa import DEFAULT_AVG_KIND, DEFAULT_DE_AVG_KIND, \
    DEFAULT_SMOOTHING_HORIZON, DEFAULT_EST_WINDOW_KIND, DEFAULT_EST_HORIZON, DEFAULT_EVAL_WINDOW_KIND


# market facts that we can't control
# approx duration of a 10Y US treasury note in a "normal" rates climate
DEFAULT_BOND_DUR: float = 7
# calendar, business
DAYCOUNTS: pd.Series = fsc.get_series([
    ("d", 1), ("Bd", 1),
    ("W", 7), ("BW", 5),
    ("Cm", 30), ("Bm", 21),
    ("CQ", 91), ("BQ", 65),
    ("CY", 365), ("BY", 261)
])
# Observe info at `t`, trade on it the course of `t+1`, earn at `t+2`:
# At the end of each session (t), we review the data,
# then trade up or down over the next session (t+1) to hit target leverage,
# finally earning the corresponding return over the next-next session (t+2).
# Under this model, you get no execution during t+1, until the close when
# you get all the execution at once. This seems pretty unrealistic,
# but it's actually conservative: The alternative is to assume you trade fast
# and thereby start earning the return intraday during t+1.
IMPL_LAG: int = 2

# analytical or portfolio-construction choices that we do control
DEFAULT_R_KIND: str = "log"
DEFAULT_PLOT_CUM_R_KIND: str = DEFAULT_R_KIND
# nice standard number to target
DEFAULT_VOL: float = 0.10
CASH_NAME: str = "cash"

# smoothing, estimation, evaluation, etc
HORIZONS: pd.Series = fsc.get_series([
    ("micro", 3),
    ("mini", DAYCOUNTS["BW"]),
    ("short", DAYCOUNTS["Bm"]),
    ("sweet", 42),
    ("med", DAYCOUNTS["BQ"]),
    ("long", DAYCOUNTS["BY"]),
    ("super", 3 * DAYCOUNTS["BY"]),
    ("hyper", 5 * DAYCOUNTS["BY"]),
    ("ultra", 10 * DAYCOUNTS["BY"])
])

ROUND_DPS: pd.Series = fsc.get_series([
    ("alpha_t", 2),
    ("corr", 2),
    ("Sharpe", 2),
    ("t-stat", 2),
    ("ER", 4),
    ("Vol", 4),
    ("Frac valid timesteps", 3)
])


########################################################################################################################
## DAT SELLSIDE SHIT ###################################################################################################
########################################################################################################################

def __get_pv(fv: float, t: float = 0, discount_r: float = 0) -> float:
    return fv / (1 + discount_r)**t


def _get_pv(fv: FloatSeries, discount_r: float = 0) -> FloatSeries:
    return pd.Series(
        {
            t: __get_pv(_fv, t=t, discount_r=discount_r)
            for (t, _fv) in fv.items()
        }
    )


def _get_dur(fv: FloatSeries, discount_r: float = 0) -> float:
    """Not exactly right, but good enough for me.

    I come from macro-land, where duration is simply calculated as a recent
    realized beta to shifts in the global yield curve.
    Obviously, in private credit, you can't do the same thing
    for a pre-inception security whose price history doesn't even exist yet.
    """
    pv = _get_pv(fv=fv, discount_r=discount_r)
    dur = np.average(pv.index, weights=pv.values)
    return dur


########################################################################################################################
## RETURN MANIPULATIONS ################################################################################################
########################################################################################################################

def __get_r_from_mult(mult: float=1, kind: str=DEFAULT_R_KIND) -> float:
    if kind in ["geom", "arith"]:
        r = mult-1
    elif kind == "log":
        r = np.log(mult)
    else:
        raise ValueError(kind)
    return r


def _get_r_from_mult(mult: FloatSeries, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    r = mult.apply(__get_r_from_mult, kind=kind)
    r = r.rename(mult.name)
    return r


def __get_mult(r: float=0, kind: str=DEFAULT_R_KIND) -> float:
    if kind in ["geom", "arith"]:
        mult = 1+r
    elif kind == "log":
        mult = np.exp(r)
    else:
        raise ValueError(kind)
    return mult


def _get_mult(r: FloatSeries, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    mult = r.apply(__get_mult, kind=kind)
    mult = mult.rename(r.name)
    return mult


def __rekind_r(r: float=0, from_kind: str=DEFAULT_R_KIND, to_kind: str=DEFAULT_R_KIND) -> float:
    mult = __get_mult(r=r, kind=from_kind)
    r = __get_r_from_mult(mult=mult, kind=to_kind)
    return r


def _rekind_r(
        r: FloatSeries,
        from_kind: str=DEFAULT_R_KIND,
        to_kind: str=DEFAULT_R_KIND
    ) -> FloatSeries:
    r = r.apply(__rekind_r, from_kind=from_kind, to_kind=to_kind)
    return r


def rekind_r(
        r: FloatDF,
        from_kind: str=DEFAULT_R_KIND,
        to_kind: str=DEFAULT_R_KIND
    ) -> FloatDF:
    r = r.apply(_rekind_r, from_kind=from_kind, to_kind=to_kind)
    return r


def _get_r_from_px(
        px: FloatSeries,
        seed_value: Optional[float]=1.00,
        kind: str=DEFAULT_R_KIND
    ) -> FloatSeries:
    mult = px / px.shift()
    # if possible, fill in the NaN at the beginning
    seed_value = fsc.maybe(seed_value, np.nan)
    mult.iloc[0] = px.iloc[0] / seed_value
    r = _get_r_from_mult(mult=mult, kind=kind)
    r = r.rename(px.name)
    return r


def _get_r_from_yld(
        yld: FloatSeries,
        dur: Union[float, FloatSeries]=DEFAULT_BOND_DUR,
        kind: str=DEFAULT_R_KIND,
        annualizer: int=DAYCOUNTS["BY"]
    ) -> FloatSeries:
    """Approximation assuming log yields."""
    # TODO(sparshsah): prove and implement for every r_kind
    _ = kind
    single_day_carry_exposed_return = yld.shift() / annualizer
    # remember: duration is in years, so we must use annualized yields
    dur = FloatSeries(dur, index=yld.index)
    delta_yld = yld - yld.shift()
    duration_exposed_return = -dur.shift() * delta_yld
    # combine the two effects
    r = single_day_carry_exposed_return + duration_exposed_return
    r = r.rename(name=yld.name)
    return r


def _get_xr(r: FloatSeries, cash_r: FloatSeries, r_kind: str=DEFAULT_R_KIND) -> FloatSeries:
    """Excess-of-cash returns."""
    cash_r = cash_r.reindex(index=r.index)
    if r_kind == "geom":
        xr = (1+r) / (1+cash_r) - 1
    elif r_kind in ["log", "arith"]:
        # obviously works for arith r_kind
        # but, works for log too: ln(e^r / e^cash_r) = r - cash_r
        xr = r - cash_r
    else:
        raise ValueError(r_kind)
    xr = xr.rename(r.name)
    return xr


def get_xr(r: FloatDF, cash_r: FloatSeries) -> FloatSeries:
    """Excess-of-cash returns."""
    xr = r.apply(_get_xr, cash_r=cash_r)
    return xr


def __get_levered_xr(lev: float=1, xr: float=0, kind: str=DEFAULT_R_KIND) -> float:
    """Levered excess-of-cash return at a _single timestep_ for a _single asset_.

    The portfolio return on NAV you'd get if you invested
    `lev`% of NAV in the passive asset and `1-lev`% in cash.
    Handles different return kinds appropriately.

    inputs
    ------
    lev: float (e.g. 0.60 or 1.30), the amount of leverage.
    xr: float (e.g. -0.01 or 0.02), the passive-asset's excess-of-cash return.
        It's called `xr` to remind you that you can't lever for free:
        If you delever, you can deposit the excess cash in the bank and earn interest;
        Whereas if you uplever, you must pay a funding rate.
        (It's simplistic to assume the funding rate is the same as the deposit interest rate---it
        will usually be higher, since your private default risk is greater than the bank's---but ok.)
    kind: str (default 'log'), the kind of return ('geom', 'log', or 'arith').

    output
    ------
    levered_xr: float, the levered excess return.


    Proof (notice the math's elegant symmetry!):
    ```
    # principal_amount       =:              P
    ##  risked_amount        =        lev  * P
    ##  cash_balance         =     (1-lev) * P
    # if kind in ["geom", "arith"]:
        # final_amount       =        lev  * P * (1+xr)  +  (1-lev) * P * (1+0)
        # levered_xmult     :=                      final_amount                   /  principal_amount
                             =      [ lev  * P * (1+xr)  +  (1-lev) * P * (1+0) ]  /  P
                             =        lev      * (1+xr)  +  (1-lev)     * (1+0)
                             =        lev      * (1+xr)  +  (1-lev)
                             =        lev      *  M(xr)  +  (1-lev)
        # levered_xr = R(levered_xmult) =    levered_xmult - 1 = lev+lev*xr + 1-lev - 1 = lev*xr
    # elif kind == "log":
        # final_amount       =        lev  * P *  e^xr   +  (1-lev) * P *  e^0
        # levered_xmult     :=                      final_amount                   /  principal_amount
                             =      [ lev  * P *  e^xr   +  (1-lev) * P *  e^0  ]  /  P
                             =        lev      *  e^xr   +  (1-lev)     *  e^0
                             =        lev      *  e^xr   +  (1-lev)
                             =        lev      *  M(xr)  +  (1-lev)
        # levered_xr = R(levered_xmult) = ln(levered_xmult)    = ln(lev*e^xr + 1-lev)
    ```
    """
    levered_xmult = lev * __get_mult(r=xr, kind=kind)  +  (1-lev)
    levered_xr = __get_r_from_mult(mult=levered_xmult, kind=kind)
    return levered_xr


def _get_levered_xr(lev: FloatSeries, xr: FloatSeries, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    """Levered excess-of-cash return at a _single timestep_ for each asset."""
    levered_xr = [(ccy,
        __get_levered_xr(lev=lev.loc[ccy], xr=xr.loc[ccy], kind=kind)
    ) for ccy in lev.index]
    levered_xr = fsc.get_series(levered_xr, name=xr.name)
    return levered_xr


def get_levered_xr(lev: FloatDF, xr: FloatDF, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    """Levered excess-of-cash return at each timestep for each asset."""
    levered_xr = [(t,
        _get_levered_xr(lev=lev.loc[t, :], xr=xr.loc[t, :], kind=kind)
    ) for t in lev.index]
    levered_xr = fsc.get_df(levered_xr, values_are="rows")
    return levered_xr


def __get_agg_r(r_a: float=0, r_b: float=0, kind: str=DEFAULT_R_KIND) -> float:
    mult_a = __get_mult(r=r_a, kind=kind)
    mult_b = __get_mult(r=r_b, kind=kind)
    fluffed_agg_mult = mult_a + mult_b
    agg_mult = 1 + fluffed_agg_mult - 2
    agg_r = __get_r_from_mult(mult=agg_mult, kind=kind)
    return agg_r


def _get_agg_r(r: FloatSeries, kind: str=DEFAULT_R_KIND) -> float:
    """Aggregate returns across a single timestep.

    Proof (notice the math's elegant symmetry!):
    ```
    # principal_amount   =:    P
    # take e.g. r       :=                            xr
                         =:              [   a ,    b ,    c ,    d  ]
    # final_amount       =   {              principal amount                                         } +
                             { hypothetical collected returns   of fully investing in every asset    } +
                             {                        repayment of        borrowed          leverage }
    # if kind in ["geom", "arith"]:
        # final_amount   =     P  +  P * ( 1+a  + 1+b  + 1+c  + 1+d  )  -  P*4
        # agg_mult      :=                        final_amount                    /  principal_amount
                         =   [ P  +  P * ( 1+a  + 1+b  + 1+c  + 1+d  )  -  P*4 ]  /  P
                         =     1  +      ( 1+a  + 1+b  + 1+c  + 1+d  )  -    4
                         =     1  +      ( M(a) + M(b) + M(c) + M(d) )  -    4
                         =:    1  +             fluffed_agg_mult        -    N
        # agg_r = R(agg_mult) =    agg_mult - 1 = 1 + N+a+b+c+d - N  -  1     = a + b + c + d
    # if kind == "log":
        # final_amount   =     P  +  P * ( e^a  + e^b  + e^c  + e^d  )  -  P*4
        # agg_mult      :=                        final_amount                    /  principal_amount
                         =   [ P  +  P * ( e^a  + e^b  + e^c  + e^d  )  -  P*4 ]  /  P
                         =     1  +      ( e^a  + e^b  + e^c  + e^d  )  -    4
                         =     1  +      ( M(a) + M(b) + M(c) + M(d) )  -    4
                         =:    1  +             fluffed_agg_mult        -    N
        # agg_r = R(agg_mult) = ln(agg_mult)    = ln(1 + e^a+e^b+e^c+e^d - N)
    ```
    """
    mult = _get_mult(r=r, kind=kind)
    # this is now a float scalar
    fluffed_agg_mult = mult.sum()
    agg_mult = 1 + fluffed_agg_mult - len(r)
    agg_r = __get_r_from_mult(mult=agg_mult, kind=kind)
    return agg_r


def get_agg_r(r: FloatDF, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    """Aggregate returns across each timestep."""
    agg_r = r.apply(_get_agg_r, kind=kind, axis="columns")
    return agg_r


def _get_cum_r(r: FloatSeries, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    """Accumulate returns over time."""
    if kind == "geom":
        cum_mult = (1+r).cumprod()
        cum_r = cum_mult - 1
    elif kind == "log-of-geom":
        cum_mult = (1+r).cumprod()
        cum_r = np.log(cum_mult)
    elif kind == "log":
        cum_r = r.cumsum()
    elif kind == "arith":
        cum_r = r.cumsum()
    else:
        raise ValueError(kind)
    cum_r = cum_r.rename(r.name)
    return cum_r


def get_cum_r(r: FloatDF, kind: str=DEFAULT_R_KIND) -> FloatDF:
    """Accumulate returns over time."""
    cum_r = r.apply(_get_cum_r, kind=kind)
    return cum_r


def _get_px(r: FloatSeries, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    cum_r = _get_cum_r(r=r, kind=kind)
    if kind in ["geom", "arith"]:
        px = 1 + cum_r
    elif kind == "log":
        px = np.exp(cum_r)
    else:
        raise ValueError(kind)
    return px


def _get_refreqed_r(r: FloatSeries, kind: str=DEFAULT_R_KIND, freq: str=DEFAULT_DATETIME_FREQ) -> FloatSeries:
    if kind == "geom":
        px = _get_px(r=r, kind=kind)
        refreqed_px = px.asfreq(freq)
        refreqed_r = _get_r_from_px(px=refreqed_px, kind=kind)
    elif kind in ["log", "arith"]:
        refreqed_r = r.resample(freq).sum()
    return refreqed_r


def _get_pnl(
        w: FloatSeries, r: FloatSeries,
        kind: str=DEFAULT_R_KIND,
        agg: bool=True
    ) -> Union[float, FloatSeries]:
    """Active pnl at a single timestep."""
    # abuse of notation, but works for our purpose even if `r` is total (not excess)
    pnl = _get_levered_xr(lev=w, xr=r, kind=kind)
    pnl = _get_agg_r(pnl, kind=kind) if agg else pnl
    return pnl


def get_pnl(
        w: FloatDF, r: FloatDF,
        kind: str=DEFAULT_R_KIND,
        impl_lag: int=IMPL_LAG,
        agg: bool=True
    ) -> FloatSeriesOrDF:
    """Active pnl at each timestep."""
    w_ = w.shift(impl_lag)
    pnl = [
            (
                # key = timestep
                t,
                # value = pnl at that timestep
                _get_pnl(
                    w=w_.loc[t, :],
                    r=r.loc[t, :],
                    kind=kind,
                    agg=agg
                )
            )
        for t in w_.index
    ]
    pnl = fsc.get_series(pnl) if agg else \
        fsc.get_df(pnl, values_are="rows")
    return pnl


########################################################################################################################
## BACKWARD-LOOKING RETURN STATISTICS ##################################################################################
########################################################################################################################

def _get_est_er_of_r(
        r: FloatSeries,
        r_kind: str=DEFAULT_R_KIND,
        avg_kind: str=DEFAULT_AVG_KIND,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        est_horizon: int=DEFAULT_EST_HORIZON,
        annualizer: int=DAYCOUNTS["BY"]
    ) -> Floatlike:
    def _get_est_avg(ser: FloatSeries) -> float:
        return fsset._get_est_avg(
            ser=ser,
            avg_kind=avg_kind,
            est_window_kind=est_window_kind,
            est_horizon=est_horizon
        )
    if avg_kind.startswith("arith"):
        est_avg_r = _get_est_avg(ser=r)
    else:
        mult = _get_mult(r=r, kind=r_kind)
        est_avg_mult = _get_est_avg(ser=mult)
        est_avg_r = __get_r_from_mult(mult=est_avg_mult, kind=r_kind)
    ann_est_avg_r = annualizer * est_avg_r
    return ann_est_avg_r


def _get_est_vol_of_r(
        r: FloatSeries,
        de_avg_kind: Optional[str]=DEFAULT_DE_AVG_KIND,
        bessel_degree: Optional[int]=None,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        est_horizon: int=DEFAULT_EST_HORIZON,
        annualizer: int=DAYCOUNTS["BY"]
    ) -> Floatlike:
    """
    By the way, you'll often hear that a financial risk model should
        use a slightly longer-than-MSE-optimal estimation horizon, because:
    (a) Asset returns are kurtotic (susceptible to huge shocks), so
        using a longer lookback acts like a "floor" during periods of low volatility,
        reducing the risk of blowup in a tail event by
        "remembering" that markets weren't always so calm.
        |
        -> This is valid, since we often cop out of directly modeling kurtosis.
    (b) You don't want to constantly trade up and down to relever a
        volatility-targeted portfolio in response to
        the vacillations of your risk model.
        |
        -> This is stupid: If you don't want to be overly sensitive to
           short-term market fluctuations, use tcost aversion or turnover controls.
           Market noise isn't very informative to asset ER's, so the point is to
           try to "filter it out" when constructing trading signals;
           But when estimating risk, it's a different story:
           Volatility is, by definition, "just" market noise!
    """
    est_vol = fsset._get_est_std(
        ser=r,
        de_avg_kind=de_avg_kind,
        bessel_degree=bessel_degree,
        est_window_kind=est_window_kind,
        est_horizon=est_horizon
    )
    ann_est_vol = annualizer**0.5 * est_vol
    return ann_est_vol


def _get_est_sharpe_of_r(
        r: FloatSeries,
        de_avg_kind: Optional[str]=DEFAULT_DE_AVG_KIND,
        bessel_degree: Optional[int]=None,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        est_horizon: int=DEFAULT_EST_HORIZON,
        annualizer: int=DAYCOUNTS["BY"]
    ) -> Floatlike:
    est_er = _get_est_er_of_r(
        r=r,
        est_window_kind=est_window_kind,
        est_horizon=est_horizon,
        annualizer=annualizer
    )
    est_vol = _get_est_vol_of_r(
        r=r,
        de_avg_kind=de_avg_kind,
        bessel_degree=bessel_degree,
        est_window_kind=est_window_kind,
        est_horizon=est_horizon,
        annualizer=annualizer
    )
    est_sharpe = est_er / est_vol
    return est_sharpe


def _get_t_stat_of_r(
        r: FloatSeries,
        de_avg_kind: Optional[str]=DEFAULT_DE_AVG_KIND,
        window_kind: str=DEFAULT_EVAL_WINDOW_KIND
    ) -> Floatlike:
    # https://web.stanford.edu/~wfsharpe/art/sr/sr.htm
    granular_est_sharpe = _get_est_sharpe_of_r(
        r=r,
        de_avg_kind=de_avg_kind,
        est_window_kind=window_kind,
        annualizer=1
    )
    valid_timesteps = r.notna().sum()
    t_stat = granular_est_sharpe * valid_timesteps**0.5
    return t_stat


def _get_est_beta_of_r(
        of_r: FloatSeries,
        on_r: FloatSeries,
        de_avg_kind: Optional[str]=DEFAULT_DE_AVG_KIND,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND
    ) -> Floatlike:
    est_of_std = fsset._get_est_std(ser=of_r, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_on_std = fsset._get_est_std(ser=on_r, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_corr = fsset._get_est_corr(
        ser_a=of_r, ser_b=on_r,
        de_avg_kind=de_avg_kind,
        est_window_kind=est_window_kind
    )
    est_beta = (est_of_std / est_on_std) * est_corr
    return est_beta


def _get_alpha_t_stat_of_r(
        of_r: FloatSeries,
        on_r: FloatSeries,
        smoothing_horizon: int | None = DEFAULT_SMOOTHING_HORIZON,
    ) -> Floatlike:
    m = fsset.linreg(
        df=pd.DataFrame({"of_r": of_r, "on_r": on_r}),
        y="of_r",
        x=["on_r"],
        smoothing_horizon=smoothing_horizon,
    )
    alpha_t = m.tvalues["const"]
    return alpha_t


def get_alpha_t_stat_of_r(
        r: FloatDF,
        smoothing_horizon: int | None = DEFAULT_SMOOTHING_HORIZON,
    ) -> FloatDF:
    alpha_t_stat = {
        of_name: {
            on_name: 0 if on_name == of_name else _get_alpha_t_stat_of_r(
                of_r=of_r, on_r=on_r,
                smoothing_horizon=smoothing_horizon,
            )
        for (on_name, on_r) in r.items()}
    for (of_name, of_r) in r.items()}
    alpha_t_stat = pd.DataFrame(alpha_t_stat, columns=r.columns, index=r.columns)
    alpha_t_stat = alpha_t_stat.rename_axis(columns="of", index="on")
    return alpha_t_stat


########################################################################################################################
## FORWARD-LOOKING RETURN CONSTRUCTION #################################################################################
########################################################################################################################

def __get_exante_targeted_vol_xr(
        xr: FloatSeries,
        vol: FloatSeries,
        kind: str=DEFAULT_R_KIND,
        tgt_vol: float=DEFAULT_VOL
    ) -> FloatSeries:
    """Target-vol asset, treating `vol[t]` as its ground-truth passive vol at `t`."""
    # this is required portfolio leverage as $ notional / $ NAV
    req_lev = tgt_vol / vol
    levered_xr = _get_levered_xr(lev=req_lev, xr=xr, kind=kind)
    levered_xr = levered_xr.rename(xr.name)
    return levered_xr


def _get_fcast_targeted_vol_xr(
        xr: FloatSeries,
        r_kind: str=DEFAULT_R_KIND,
        est_window_kind: str=DEFAULT_EST_WINDOW_KIND,
        impl_lag: int=IMPL_LAG,
        tgt_vol: float=DEFAULT_VOL
    ) -> FloatSeries:
    """(Implementably) modulate volatility.

    With default settings, you get a implementably-targeted version of the xr, i.e.
        you could actually estimate its volatility then submit trades to lever it accordingly.
        On the other hand, if you pass est_window_kind = "full" and impl_lag = 0,
        you'll just get an expost-perfectly-vol-targeted version of `base_xr`,
        whose realized sample volatility will be exactly equal to `tgt_vol`.
    """
    est_vol = _get_est_vol_of_r(r=xr, est_window_kind=est_window_kind)
    # pad if this was a scalar (e.g. if est_window_kind == "full")
    est_vol = pd.Series(est_vol, index=xr.index)
    exante_vol = est_vol.shift(impl_lag)
    levered_xr = __get_exante_targeted_vol_xr(xr=xr, vol=exante_vol, kind=r_kind, tgt_vol=tgt_vol)
    return levered_xr


def __get_exante_hedged_xr(
        base_xr: FloatSeries,
        hedge_xr: FloatSeries,
        beta: FloatSeries,
        kind: str=DEFAULT_R_KIND
    ) -> FloatSeries:
    """Hedge out exposure to the hedge asset, treating `beta[t]` as the
    ground-truth beta of the base asset on the hedge asset at time `t`.
    """
    # FIRST, CALCULATE YOUR ACTIVE HEDGE EXCESS-OF-CASH PNL
    ##  this is required portfolio weight as $ notional / $ NAV,
    ##    which we can interpret as leverage :)
    hedge_lev = -beta
    hedge_xpnl = _get_levered_xr(lev=hedge_lev, xr=hedge_xr, kind=kind)
    del hedge_lev, hedge_xr
    # THEN, ADD IT TO THE PASSIVE BASE ASSET'S EXCESS-OF-CASH RETURNS
    xr = fsc.get_df([
        (base_xr.name, base_xr),
        (hedge_xpnl.name, hedge_xpnl)
    ], values_are="columns")
    del hedge_xpnl, base_xr
    hedged_base_xr = get_agg_r(r=xr, kind=kind)
    hedged_base_xr = hedged_base_xr.rename(xr.columns[0])
    return hedged_base_xr


def _get_fcast_hedged_xr(
        base_xr: FloatSeries,
        hedge_xr: FloatSeries,
        r_kind: str=DEFAULT_R_KIND,
        est_window_kind: str=DEFAULT_EST_WINDOW_KIND,
        impl_lag: int=IMPL_LAG
    ) -> FloatSeries:
    """(Implementably) short out base asset's exposure to hedge asset.

    Inputs should be excess-of-cash returns:
        If you delever, you can deposit the excess cash in the bank and earn interest;
        Whereas if you uplever, you must pay a funding rate.
        It's simplistic to assume the funding rate is the same as the deposit interest rate
        (it will usually be higher, since your default risk is greater than the bank's), but ok.

    With default settings, you get a implementably-hedged version of the base xr, i.e.
        you could actually estimate the beta then submit trades to short it out.
        On the other hand, if you pass est_window_kind = "full" and impl_lag = 0,
        you'll just get an expost-perfectly-hedged version of `base_xr`,
        whose realized sample beta will be exactly zero.
    """
    # at the end of each day, we submit an order to short this much of the hedge asset
    est_beta = _get_est_beta_of_r(of_r=base_xr, on_r=hedge_xr, est_window_kind=est_window_kind)
    # pad if this was a scalar (e.g. if est_window_kind == "full")
    est_beta = pd.Series(est_beta, index=base_xr.index)
    exante_beta = est_beta.shift(impl_lag)
    hedged_xr = __get_exante_hedged_xr(base_xr=base_xr, hedge_xr=hedge_xr, beta=exante_beta, kind=r_kind)
    return hedged_xr


########################################################################################################################
## PORTFOLIO MATH ######################################################################################################
########################################################################################################################

def __get_w_from_vw(vw: FloatSeries, vol_vector: FloatSeries) -> FloatSeries:
    """Get portfolio (notional) weights from portfolio risk (i.e. volatility) weights.

    inputs
    ------
    vw: FloatSeries, the portfolio's risk weight on each passive asset, e.g.
        `pd.Series({"US10Y": +10%, "CDX.NA.IG": -8%, "SPX": 0%})`.
    vol_vector: FloatSeries, volatility of each passive asset.

    output
    ------
    w: FloatSeries, the portfolio's implied notional weight on each passive asset.
    """
    w = vw / vol_vector
    return w


def _get_w_from_vw(vw: FloatSeries, cov_matrix: FloatDF) -> FloatSeries:
    var_vector = fsc.get_diag_of_df(df=cov_matrix)
    vol_vector = var_vector **0.5
    w = __get_w_from_vw(vw=vw, vol_vector=vol_vector)
    return w


def __get_exante_cov_of_w(w_a: FloatSeries, w_b: FloatSeries, cov_matrix: FloatDF) -> float:
    exante_cov = w_a @ cov_matrix @ w_b
    return exante_cov


def _get_exante_vol_of_w(w: FloatSeries, cov_matrix: FloatDF) -> float:
    exante_var = __get_exante_cov_of_w(w_a=w, w_b=w, cov_matrix=cov_matrix)
    exante_vol = exante_var **0.5
    return exante_vol


def _get_exante_corr_of_w(w_a: FloatSeries, w_b: FloatSeries, cov_matrix: FloatDF) -> float:
    exante_a_vol = _get_exante_vol_of_w(w=w_a, cov_matrix=cov_matrix)
    exante_b_vol = _get_exante_vol_of_w(w=w_b, cov_matrix=cov_matrix)
    exante_cov = __get_exante_cov_of_w(w_a=w_a, w_b=w_b, cov_matrix=cov_matrix)
    exante_corr = exante_cov / (exante_a_vol * exante_b_vol)
    return exante_corr


def _get_exante_beta_of_w(of_w: FloatSeries, on_w: FloatSeries, cov_matrix: FloatDF) -> float:
    """
    To get the beta of `of_w` on a single asset, pass
        `on_w=pd.Series({on_asset_name: 1})`.
    """
    exante_of_vol = _get_exante_vol_of_w(w=of_w, cov_matrix=cov_matrix)
    exante_on_vol = _get_exante_vol_of_w(w=on_w, cov_matrix=cov_matrix)
    exante_corr = _get_exante_corr_of_w(w_a=of_w, w_b=on_w, cov_matrix=cov_matrix)
    exante_beta = (exante_of_vol / exante_on_vol) * exante_corr
    return exante_beta


def _get_uncon_mvo_w(
        er_vector: FloatSeries,
        cov_matrix: FloatDF,
        vol_shkg_to_avg: float=0,
        corr_shkg_to_zero: float=0,
        vol_tgt: float=DEFAULT_VOL
    ) -> FloatSeries:
    # (\lambda \Sigma)^{-1} \mu
    raise NotImplementedError


def _get_exante_vol_targeted_w(w: FloatSeries, cov_matrix: FloatDF, tgt_vol: float=DEFAULT_VOL) -> FloatSeries:
    raise NotImplementedError


def _get_exante_hedged_w(of_w: FloatSeries, on_w: FloatSeries, cov_matrix: FloatDF) -> FloatSeries:
    raise NotImplementedError


########################################################################################################################
## SIMULATION ##########################################################################################################
########################################################################################################################

def __sim_r(
        ann_sharpe: float=0,
        ann_vol: float=DEFAULT_VOL,
        annualizer: int=DAYCOUNTS["BY"],
        kind: str=DEFAULT_R_KIND
    ) -> float:
    # first, simulate logarithmic returns
    single_timestep_sharpe = ann_sharpe / annualizer**0.5
    single_timestep_vol = ann_vol / annualizer**0.5
    single_timestep_er = single_timestep_sharpe * single_timestep_vol
    r = sps.norm.rvs(loc=single_timestep_er, scale=single_timestep_vol)
    # then, convert to user's desired return kind
    r = __rekind_r(r=r, from_kind="log", to_kind=kind)
    return r


def _sim_r(
        ann_sharpe: float=0,
        ann_vol: float=DEFAULT_VOL,
        sz_in_years: float=100,
        annualizer: int=DAYCOUNTS["BY"],
        kind: str=DEFAULT_R_KIND
    ) -> FloatSeries:
    sz_in_timesteps = int(sz_in_years * annualizer)
    r = [__sim_r(
        ann_sharpe=ann_sharpe,
        ann_vol=ann_vol,
        annualizer=annualizer,
        kind=kind)
    for _ in range(sz_in_timesteps)]
    dtx = fsc.get_dtx(periods=sz_in_timesteps)
    r = pd.Series(r, index=dtx)
    return r


########################################################################################################################
## VISUALIZATION #######################################################################################################
########################################################################################################################

def _get_est_perf_stats_of_r(
        r: FloatSeries,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        est_horizon: int=DEFAULT_EST_HORIZON
    ) -> FloatSeriesOrDF:
    est_perf_stats = [
        (
            "Sharpe",
            _get_est_sharpe_of_r(r=r, est_window_kind=est_window_kind, est_horizon=est_horizon)
        ), (
            "t-stat",
            _get_t_stat_of_r(r=r)
        ), (
            "ER",
            _get_est_er_of_r(r=r, est_window_kind=est_window_kind, est_horizon=est_horizon)
        ), (
            "Vol",
            _get_est_vol_of_r(r=r, est_window_kind=est_window_kind, est_horizon=est_horizon)
        )
    ]
    est_perf_stats = OrderedDict(est_perf_stats)
    # e.g. True if 'ewm' window, False if 'full' window
    values_are_seriess = isinstance(est_perf_stats["Sharpe"], pd.Series)
    # t-stat is at this point always simply a full-sample scalar, so we might need to pad it out
    if values_are_seriess:
        est_perf_stats["t-stat"] = pd.Series(
            est_perf_stats["t-stat"],
            index=est_perf_stats["Sharpe"].index,
            name="t-stat"
        )
    est_perf_stats = pd.DataFrame(est_perf_stats) if values_are_seriess else pd.Series(est_perf_stats)
    est_perf_stats.name = f"{est_horizon}-horizon {est_window_kind}-window {r.name}"
    return est_perf_stats


def get_est_perf_stats_of_r(
        r: FloatDF,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        est_horizon: int=DEFAULT_EST_HORIZON
    ) -> FloatDF:
    est_perf_stats = [(
        # this is a key e.g. 'ARF' or 'AQMNX' or 'SPY'
        colname,
        # this is either a FloatDF (e.g. if 'ewm' window) or a FloatSeries (e.g. if 'full' window)
        _get_est_perf_stats_of_r(r=col, est_window_kind=est_window_kind, est_horizon=est_horizon)
    ) for (colname, col) in r.items()]
    ####
    est_perf_stats = fsc.get_df(est_perf_stats, values_are="columns")
    # if each value was a DF, we'll get MultiIndex columns and want to flip stat names up to top-level
    if isinstance(est_perf_stats.columns, pd.MultiIndex):
        est_perf_stats = est_perf_stats.swaplevel(axis="columns")
        # sort stat names in my preferred order, then datacolnames in given order
        columns = est_perf_stats.columns.to_list()
        def _get_key(statname_colname: Tuple[str, str]) -> Tuple[int, int]:
            statname, colname = statname_colname
            del statname_colname
            primary_key = ROUND_DPS.index.get_loc(statname)
            secondary_key = r.columns.get_loc(colname)
            key = primary_key, secondary_key
            return key
        columns = sorted(columns, key=_get_key)
        est_perf_stats = est_perf_stats.reindex(columns=columns)
    # otherwise, we'll get regular columns and want to flip stat names up into columns
    else:
        est_perf_stats = est_perf_stats.T
    est_perf_stats.name = f"{est_horizon}-horizon {est_window_kind}-window"
    return est_perf_stats


def _round_perf_stats(perf_stats: pd.Series, round_: bool=True) -> pd.Series:
    if round_:
        for (k, dps) in ROUND_DPS.items():
            if k in perf_stats.index:
                perf_stats.loc[k] = np.round(perf_stats.loc[k], dps)
    return perf_stats


def _table_est_perf_stats_of_r(r: FloatSeries, rounded: bool=True) -> pd.Series:
    metadata = fsset._get_metadata(ser=r)
    est_perf_stats = _get_est_perf_stats_of_r(r=r)
    ####
    collected_stats = pd.concat([est_perf_stats, metadata])
    collected_stats = _round_perf_stats(collected_stats, round_=rounded)
    return collected_stats


def table_est_perf_stats_of_r(
        r: FloatDF, over_common_subsample: bool=True,
        rounded: bool=True
    ) -> Dict[str, pd.DataFrame]:
    """
    `{
        'alpha_t': t-stat of alpha of {row} over {column},
        'corr': ...,
        'standalone': ...,
    }`.
    """
    r = fsc.get_common_subsample(r) if over_common_subsample else r
    est_standalone_stats = r.apply(_table_est_perf_stats_of_r, axis="index", rounded=rounded)
    est_corr = fsset.get_est_corr(df=r)
    alpha_t_stat = get_alpha_t_stat_of_r(r=r)
    ####
    # flip stat names up into columns
    est_standalone_stats = est_standalone_stats.T
    collected_stats = [
        ("alpha_t", alpha_t_stat),
        ("corr", est_corr),
        ("standalone", est_standalone_stats)
    ]
    collected_stats = OrderedDict(collected_stats)
    return collected_stats


def _plot_cum_r(r: FloatSeries, kind: str=DEFAULT_PLOT_CUM_R_KIND, title: str="") -> fsc.PlotAxes:
    cum_r = _get_cum_r(r=r, kind=kind)
    return fsc.plot(
        cum_r,
        ypct=True,
        title=f"{title} {kind} CumRets"
    )


def plot_cum_r(r: FloatDF, kind: str=DEFAULT_PLOT_CUM_R_KIND, title: str="") -> fsc.PlotAxes:
    cum_r = get_cum_r(r=r, kind=kind)
    return fsc.plot(
        cum_r,
        ypct=True,
        title=f"{title} {kind} CumRets"
    )


def _chart_r(r: FloatSeries, plot_cum_r_kind: str=DEFAULT_PLOT_CUM_R_KIND, print_: bool=False) -> pd.Series:
    _plot_cum_r(r=r, kind=plot_cum_r_kind, title=r.name)
    est_perf_stats = _get_est_perf_stats_of_r(r=r)
    if print_:
        print(est_perf_stats)
    return est_perf_stats


def chart_r(r: FloatDF, plot_cum_r_kind: str=DEFAULT_PLOT_CUM_R_KIND, title: str="") -> None:
    #### plot cum r
    plot_cum_r(r=r, kind=plot_cum_r_kind, title=title)
    #### tables
    # TODO(sparshsah): split by early-mid-late third's then fullsample
    tables = table_est_perf_stats_of_r(r=r)
    # TODO(sparshsah): plot alpha t stat heatmap
    fsc.plot_corr_heatmap(tables["corr"], title="corr")
    #### plot rolling sr, er/vol
    fullsample_est_perf_stats = get_est_perf_stats_of_r(r=r)
    moving_est_perf_stats = get_est_perf_stats_of_r(r=r, est_window_kind="rolling", est_horizon=HORIZONS["super"])
    # setting sharex makes weird minor gridlines appear
    _, ax = plt.subplots(nrows=3)
    fsc.plot(
        moving_est_perf_stats["Sharpe"],
        axhline_locs=[0,] + list(fullsample_est_perf_stats["Sharpe"]),
        axhline_styles=["-",] + [":",]*len(r.columns),
        axhline_colors=["darkgrey",] + list(sns.color_palette()),
        title="Sharpe",
        ax=ax[0]
    )
    fsc.plot(
        moving_est_perf_stats["ER"],
        axhline_locs=[0,] + list(fullsample_est_perf_stats["ER"]),
        axhline_styles=["-",] + [":",]*len(r.columns),
        axhline_colors=["darkgrey",] + list(sns.color_palette()),
        ypct=True, title="ER",
        ax=ax[1]
    )
    fsc.plot(
        moving_est_perf_stats["Vol"],
        axhline_locs=[0,] + list(fullsample_est_perf_stats["Vol"]),
        axhline_styles=[":",],
        axhline_colors=["darkgrey",] + list(sns.color_palette()),
        ylim_bottom=0, ypct=True,
        title="Vol", ax=ax[2],
        # 2.5x the default height
        figsize=(fsc.FIGSIZE[0], 2.5*fsc.FIGSIZE[1])
    )
    plt.suptitle(moving_est_perf_stats.name, y=0.91)
    plt.show()
    #### diagnostic
    print(tables["standalone"].T)
