"""Financial programming utilities.

v0.3 alpha: API MAY (IN FACT, CERTAINLY WILL) BREAK AT ANY TIME!

author: [@sparshsah](https://github.com/sparshsah)


# Notes
* Each `get_est_{whatever}_of_r()` function estimates its
    specified market param based on the given data sample.
* Each `get_exante_{whatever}_of_w()` function calculates its
    specified portfolio stat taking as ground truth the given market params.
"""

from typing import Tuple, Dict, Union, Optional
from collections import OrderedDict
import pandas as pd
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
# https://github.com/sparshsah/foggy-lib/blob/main/util/foggy_pylib/core.py
import foggy_pylib.core as fc
# https://github.com/sparshsah/foggy-lib/blob/main/util/foggy_pylib/stats/tsa.py
import foggy_pylib.stats.est.tsa as fset

from foggy_pylib.core import FloatSeries, FloatDF, FloatSeriesOrDF, Floatlike
from foggy_pylib.stats.est.tsa import DEFAULT_AVG_KIND, DEFAULT_DE_AVG_KIND, \
    DEFAULT_EST_WINDOW_KIND, DEFAULT_EST_HORIZON, DEFAULT_EVAL_WINDOW_KIND


# market facts that we can't control
# approx duration of a 10Y US treasury note in a "normal" rates climate
DEFAULT_BOND_DUR: float = 7
# calendar, business
DAYCOUNTS: pd.Series = fc.get_series([
    ("d", 1), ("Bd", 1),
    ("W", 7), ("BW", 5),
    ("Cm", 30), ("Bm", 21),
    ("CQ", 91), ("BQ", 65),
    ("CY", 365), ("BY", 261)
])
# observe info at `t`, trade on it the course of `t+1`, earn at `t+2`
IMPL_LAG: int = 2

# analytical or portfolio-construction choices that we do control
DEFAULT_R_KIND: str = "geom"
DEFAULT_PLOT_CUM_R_KIND: str = "log-of-geom"
# nice standard number to target
DEFAULT_VOL: float = 0.10
CASH_NAME: str = "cash"

# smoothing, estimation, evaluation, etc
HORIZONS: pd.Series = fc.get_series([
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

ROUND_DPS: pd.Series = fc.get_series([
    ("alpha_t", 2),
    ("corr", 2),
    ("Sharpe", 2),
    ("t-stat", 2),
    ("ER", 4),
    ("Vol", 4),
    ("Frac valid timesteps", 3)
])


########################################################################################################################
## RETURN MANIPULATIONS ################################################################################################
########################################################################################################################

def __get_r_from_mult(mult: float, kind: str=DEFAULT_R_KIND) -> float:
    if kind in ["geom", "arith"]:
        r = mult-1
    elif kind == "log":
        r = np.log(mult)
    else:
        raise ValueError(kind)
    return r


def _get_r_from_mult(mult: FloatSeries, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    r = mult.map(__get_r_from_mult, kind=kind)
    r = r.rename(mult.name)
    return r


def __get_mult(r: float, kind: str=DEFAULT_R_KIND) -> float:
    if kind in ["geom", "arith"]:
        mult = 1+r
    elif kind == "log":
        mult = np.exp(r)
    else:
        raise ValueError(kind)
    return mult


def _get_mult(r: FloatSeries, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    mult = r.map(__get_mult, kind=kind)
    mult = mult.rename(r.name)
    return mult


def _get_r_from_px(px: FloatSeries, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    if kind == "geom":
        mult = px / px.shift()
        r = mult-1
    elif kind == "log":
        mult = px / px.shift()
        r = np.log(mult)
    elif kind == "arith":
        r = px.diff()
    else:
        raise ValueError(kind)
    r = r.rename(px.name)
    return r


def _get_r_from_yld(
        yld: FloatSeries,
        dur: Union[float, FloatSeries]=DEFAULT_BOND_DUR,
        annualizer: int=DAYCOUNTS["BY"]
    ) -> FloatSeries:
    """Approximation assuming log yields."""
    # TODO(sparshsah): prove and implement for every r_kind
    single_day_carry_exposed_return = yld.shift() / annualizer
    # remember: duration is in years, so we must use annualized yields
    dur = FloatSeries(dur, index=yld.index)
    delta_yld = yld - yld.shift()
    duration_exposed_return = -dur.shift() * delta_yld
    # combine the two effects
    r = single_day_carry_exposed_return + duration_exposed_return
    r = r.rename(name=yld.name)
    return r


def _get_agg_r(r: FloatSeries, kind: str=DEFAULT_R_KIND) -> float:
    """Aggregate returns across a single timestep."""
    if kind in ["geom", "arith"]:
        agg_r = r.sum()
    elif kind == "log":
        mult = np.exp(r)
        agg_mult = mult.sum()
        agg_r = np.log(agg_mult)
    return agg_r


def get_agg_r(r: FloatDF, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    """Aggregate returns across each timestep."""
    agg_r = r.apply(_get_agg_r, kind=kind, axis="columns")
    return agg_r


def _get_cum_r(r: FloatSeries, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    """Accumulate returns over time."""
    if kind == "geom":
        cum_r = (1+r).cumprod()
        cum_r = cum_r - 1
    elif kind == "log-of-geom":
        cum_r = (1+r).cumprod()
        cum_r = np.log(cum_r)
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


def _get_xr(r: FloatSeries, cash_r: FloatSeries) -> FloatSeries:
    """Excess-of-cash returns."""
    cash_r = cash_r.reindex(index=r.index)
    # obviously works for geom and arith r_kind
    # but, works for log too: ln(e^r / e^cash_r) = r - cash_r
    xr = r - cash_r
    xr = xr.rename(r.name)
    return xr


def get_xr(r: FloatDF, cash_r: FloatSeries) -> FloatSeries:
    """Excess-of-cash returns."""
    xr = r.apply(_get_xr, cash_r=cash_r)
    return xr


def __get_levered_xr(lev: float, xr: float, kind: str=DEFAULT_R_KIND) -> float:
    """Levered excess-of-cash return at a single timestep for a single asset."""
    if kind in ["geom", "arith"]:
        levered_xr = lev * xr
    elif kind == "log":
        # P := principal_amount
        #   risked_amount = lev * P
        #   cash_balance = (1-lev) * P
        # final_amount = lev * P * e^xr  +  (1-lev) * P
        # mult := final_amount / principal_amount = lev * e^xr  +  (1-lev)
        # lev_xr = ln(mult) = ln(lev * e^xr  +  1-lev)
        mult = lev*np.exp(xr) + 1-lev
        levered_xr = __get_r_from_mult(mult=mult, kind=kind)
    return levered_xr


def _get_levered_xr(lev: FloatSeries, xr: FloatSeries, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    """Levered excess-of-cash return at a single timestep for each asset."""
    levered_xr = [(ccy,
        __get_levered_xr(lev=lev.loc[ccy], xr=xr.loc[ccy], kind=kind)
    ) for ccy in lev.index]
    levered_xr = fc.get_series(levered_xr)
    return levered_xr


def get_levered_xr(lev: FloatDF, xr: FloatDF, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    """Levered excess-of-cash return at each timestep for each asset."""
    levered_xr = [(t,
        _get_levered_xr(lev=lev.loc[t, :], xr=xr.loc[t, :], kind=kind)
    ) for t in lev.index]
    levered_xr = fc.get_df(levered_xr, values_are="rows")
    return levered_xr


def _get_pnl(
        w: FloatSeries, r: FloatSeries,
        kind: str=DEFAULT_R_KIND,
        agg: bool=True
    ) -> Union[float, FloatSeries]:
    """Active pnl at a single timestep."""
    if not np.isclose(w.sum(), 1):
        raise ValueError(f"Active weighting\n{w}\nsums to {w.sum()}, did you forget cash balance?")
    if kind in ["geom", "arith"]:
        pnl = w * r
    elif kind == "log":
        # Proof:
        #   init_principal = P
        #   end_wealth = w_a*P*e^r_a + w_b*P*e^r_b
        #   mult = end_wealth / init_principal = w_a*e^r_a + w_b*e^r_b
        #   therefore, return = ln(w_a*e^r_a + w_b*e^r_b).
        # Note: You MUST, MUST include the allocation to cash!
        #   Otherwise imagine you have w_a = 0.60 and r_a = 0.05...
        #   ln(0.60*e^0.05) = -0.47 (!).
        #   You wanted,
        #   ln(0.60*e^0.05 + 0.40*e^0) = 0.03,
        #     which as a sanity check is close to the geometric version = 0.03.
        mult = np.exp(r)
        weighted_mult = w * mult
        pnl = np.log(weighted_mult)
        # TODO(sparshsah): uh, figure this out, i guess?
        if not agg:
            raise ValueError("Per-ccy active log returns act super-weird...")
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
            t,  # key
            _get_pnl(w=w_.loc[t, :], r=r.loc[t, :], kind=kind, agg=agg)  # value
        )
    for t in w_.index]
    pnl = fc.get_df(pnl, values_are="rows")
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
        return fset._get_est_avg(
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
    ----
    (a) Asset returns are kurtotic (susceptible to huge shocks), so
        using a longer lookback acts like a "floor" during periods of low volatility,
        reducing the risk of blowup in a tail event by
        "remembering" that markets weren't always so calm.
        L-> This is valid, since we often cop out of directly modeling kurtosis.
    |
    ~~~ BUT ALSO ~~~
    |
    (b) You don't want to constantly trade up and down to relever a
        volatility-targeted portfolio in response to
        the vacillations of your risk model.
        L-> This is stupid: If you don't want to be overly sensitive to
            short-term market fluctuations, use tcost aversion or turnover controls.
            Market noise isn't very informative to asset ER's, so
            it's good to filter it out when constructing trading signals;
            But when estimating risk, it's a different story:
            Volatility is, by definition, "just" market noise!
    """
    est_vol = fset._get_est_std(
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
    est_of_std = fset._get_est_std(ser=of_r, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_on_std = fset._get_est_std(ser=on_r, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_corr = fset._get_est_corr(
        ser_a=of_r, ser_b=on_r,
        de_avg_kind=de_avg_kind,
        est_window_kind=est_window_kind
    )
    est_beta = (est_of_std / est_on_std) * est_corr
    return est_beta


def _get_alpha_t_stat_of_r(
        of_r: FloatSeries,
        on_r: FloatSeries,
        de_avg_kind: Optional[str]=DEFAULT_DE_AVG_KIND,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND
    ) -> Floatlike:
    # TODO(sparshsah)
    _ = of_r, on_r, de_avg_kind, est_window_kind
    return np.nan


def get_alpha_t_stat_of_r(
        r: FloatDF,
        de_avg_kind: Optional[str]=DEFAULT_DE_AVG_KIND,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND
    ) -> FloatDF:
    alpha_t_stat = {
        of_name: {
            on_name: _get_alpha_t_stat_of_r(
                of_r=of_r, on_r=on_r,
                de_avg_kind=de_avg_kind,
                est_window_kind=est_window_kind
            )
        for (on_name, on_r) in r.items()}
    for (of_name, of_r) in r.items()}
    # we want "of" in rows and "on" in columns, so transpose output of DF constructor
    alpha_t_stat = pd.DataFrame(alpha_t_stat, columns=r.columns, index=r.columns).T
    return alpha_t_stat


########################################################################################################################
## FORWARD-LOOKING RETURN CONSTRUCTION #################################################################################
########################################################################################################################

def __get_exante_vol_targeted_xr(
        xr: FloatSeries,
        vol: FloatSeries,
        tgt_vol: float=DEFAULT_VOL
    ) -> FloatSeries:
    """Volatility-target the asset,
    treating `vol[t]` aa its ground-truth volatility at time t.
    """
    # this is portfolio leverage as $ notional / $ NAV
    req_leverage = tgt_vol / vol
    levered_xr = req_leverage * xr
    levered_xr = levered_xr.rename(xr.name)
    return levered_xr


def _get_fcast_vol_targeted_xr(
        xr: FloatSeries,
        est_window_kind: str=DEFAULT_EST_WINDOW_KIND,
        impl_lag: int=IMPL_LAG,
        tgt_vol: float=DEFAULT_VOL
    ) -> FloatSeries:
    """(Implementably) modulate volatility.

    Input should be excess-of-cash returns:
        If you delever, you can deposit the excess cash in the bank and earn interest;
        Whereas if you uplever, you must pay a funding rate.
        It's simplistic to assume the funding rate is the same as the deposit interest rate
        (it will usually be higher, since your default risk is greater than the bank's), but ok.

    With default settings, you get a implementably-vol-targeted version of the xr, i.e.
        you could actually estimate its volatility then submit trades to lever it accordingly.
        On the other hand, if you pass est_window_kind = "full" and impl_lag = 0,
        you'll just get an expost-perfectly-vol-targeted version of `base_xr`,
        whose realized sample volatility will be exactly equal to `tgt_vol`.
    """
    est_vol = _get_est_vol_of_r(r=xr, est_window_kind=est_window_kind)
    # At the end of each session (t), we review the data,
    # then trade up or down over the next session (t+1) to hit this much leverage,
    # finally earning the corresponding return over the next-next session (t+2).
    # Under this model, you get no execution during t+1, until the close when
    # you get all the execution at once. This seems pretty unrealistic,
    # but it's actually conservative: The alternative is to assume you trade fast
    # and start earning the return intraday during t+1.
    exante_vol = est_vol.shift(impl_lag)
    levered_xr = __get_exante_vol_targeted_xr(xr=xr, vol=exante_vol, tgt_vol=tgt_vol)
    return levered_xr


def __get_exante_hedged_xr(
        base_xr: FloatSeries,
        hedge_xr: FloatSeries,
        beta: FloatSeries,
    ) -> FloatSeries:
    """Hedge out exposure to the hedge asset, treating `beta[t]` as the
    ground-truth beta of the base asset on the hedge asset at time t.
    """
    # this is portfolio weight as $ notional / $ NAV
    hedge_pos = -beta
    hedge_xpnl = hedge_pos * hedge_xr
    hedged_base_xr = base_xr + hedge_xpnl
    hedged_base_xr = hedged_base_xr.rename(base_xr.name)
    return hedged_base_xr


def _get_fcast_hedged_xr(
        base_xr: FloatSeries,
        hedge_xr: FloatSeries,
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
    exante_beta = est_beta.shift(impl_lag)
    hedged_xr = __get_exante_hedged_xr(base_xr=base_xr, hedge_xr=hedge_xr, beta=exante_beta)
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
    var_vector = fc.get_diag_of_df(df=cov_matrix)
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

def _sim_r(
        ann_sharpe: float=0,
        ann_vol: float=DEFAULT_VOL,
        sz_in_years: float=100,
        annualizer: int=DAYCOUNTS["BY"]
    ) -> FloatSeries:
    single_timestep_sharpe = ann_sharpe / annualizer**0.5
    single_timestep_vol = ann_vol / annualizer**0.5
    single_timestep_er = single_timestep_sharpe * single_timestep_vol
    sz_in_timesteps = int(sz_in_years * annualizer)
    r = sps.norm.rvs(loc=single_timestep_er, scale=single_timestep_vol, size=sz_in_timesteps)
    r = pd.Series(r)
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
    est_perf_stats = fc.get_df(est_perf_stats, values_are="columns")
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
    metadata = fset._get_metadata(ser=r)
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
    r = fc.get_common_subsample(r) if over_common_subsample else r
    est_standalone_stats = r.apply(_table_est_perf_stats_of_r, axis="index", rounded=rounded)
    est_corr = fset.get_est_corr(df=r)
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


def _plot_cum_r(r: FloatSeries, kind: str=DEFAULT_R_KIND, title: str="") -> fc.PlotAxes:
    cum_r = _get_cum_r(r=r, kind=kind)
    return fc.plot(
        cum_r,
        ypct=True,
        title=f"{title} {kind} CumRets"
    )


def plot_cum_r(r: FloatDF, kind: str=DEFAULT_R_KIND, title: str="") -> fc.PlotAxes:
    cum_r = get_cum_r(r=r, kind=kind)
    return fc.plot(
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


def chart_r(r: FloatDF, plot_cum_r_kind: str= DEFAULT_PLOT_CUM_R_KIND, title: str="") -> None:
    #### plot cum r
    plot_cum_r(r=r, kind=plot_cum_r_kind, title=title)
    #### tables
    # TODO(sparshsah): split by early-mid-late third's then fullsample
    tables = table_est_perf_stats_of_r(r=r)
    # TODO(sparshsah): plot alpha t stat heatmap
    fc.plot_corr_heatmap(tables["corr"], title="corr")
    #### plot rolling sr, er/vol
    fullsample_est_perf_stats = get_est_perf_stats_of_r(r=r)
    moving_est_perf_stats = get_est_perf_stats_of_r(r=r, est_window_kind="rolling", est_horizon=HORIZONS["super"])
    # setting sharex makes weird minor gridlines appear
    _, ax = plt.subplots(nrows=3)
    fc.plot(
        moving_est_perf_stats["Sharpe"],
        axhline_locs=[0,] + list(fullsample_est_perf_stats["Sharpe"]),
        axhline_styles=["-",] + [":",]*len(r.columns),
        axhline_colors=["darkgrey",] + list(sns.color_palette()),
        title="Sharpe",
        ax=ax[0]
    )
    fc.plot(
        moving_est_perf_stats["ER"],
        axhline_locs=[0,] + list(fullsample_est_perf_stats["ER"]),
        axhline_styles=["-",] + [":",]*len(r.columns),
        axhline_colors=["darkgrey",] + list(sns.color_palette()),
        ypct=True, title="ER",
        ax=ax[1]
    )
    fc.plot(
        moving_est_perf_stats["Vol"],
        axhline_locs=[0,] + list(fullsample_est_perf_stats["Vol"]),
        axhline_styles=[":",],
        axhline_colors=["darkgrey",] + list(sns.color_palette()),
        ylim_bottom=0, ypct=True,
        title="Vol", ax=ax[2],
        # 2.5x the default height
        figsize=(fc.FIGSIZE[0], 2.5*fc.FIGSIZE[1])
    )
    plt.suptitle(moving_est_perf_stats.name, y=0.91)
    plt.show()
    #### diagnostic
    print(tables["standalone"])
