"""Financial programming utilities.

author: [@sparshsah](https://github.com/sparshsah)
"""

from typing import Dict, Union, Optional
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# https://github.com/sparshsah/foggy-lib/blob/main/util/foggy_pylib/core.py
import foggy_pylib.core as fc

FloatSeries = pd.Series
FloatDF = pd.DataFrame
Floatlike = Union[float, FloatSeries, FloatDF]

# market facts that we can't control
# approx duration of a 10Y US treasury note in a "normal" rates climate
DEFAULT_BOND_DUR: float = 7
# calendar, business
DAYCOUNTS: Dict[str, int] = {
    "d": 1, "Bd": 1,
    "W": 7, "BW": 5,
    "Cm": 30,"Bm": 21,
    "CQ": 91, "BQ": 65,
    "CY": 365, "BY": 261
}
# observe info at `t`, trade on it the course of `t+1`, earn at `t+2`
IMPL_LAG: int = 2

# analytical or portfolio-construction choices that we do control
DEFAULT_R_KIND: str = "log"
DEFAULT_AVG_KIND: str = "mean"
# nice standard number to target
DEFAULT_VOL: float = 0.10

# smoothing, estimation, evaluation, etc
HORIZONS: Dict[str, int] = {
    "micro": 3,
    "mini": DAYCOUNTS["BW"],
    "short": DAYCOUNTS["Bm"],
    "sweet": 42,
    "med": DAYCOUNTS["BQ"],
    "long": DAYCOUNTS["BY"],
    "ultra": 5 * DAYCOUNTS["BY"]
}
# smoothing e.g. to account for international trading-session async
DEFAULT_SMOOTHING_WINDOW_KIND: str = "rolling"
DEFAULT_SMOOTHING_HORIZON: int = HORIZONS["micro"]
# reality of market data-generating process is that it's non-stationary
DEFAULT_EST_WINDOW_KIND: str = "ewm"
DEFAULT_EST_HORIZON: int = HORIZONS["med"]
# evaluation, no need to specify horizon
DEFAULT_EVAL_WINDOW_KIND: str = "full"

########################################################################################################################
## RETURN MANIPULATIONS ################################################################################################
########################################################################################################################

def get_cum_r(r: FloatSeries, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    if kind == "arith":
        cum_r = r.cumsum()
    elif kind == "log":
        cum_r = r.cumsum()
    elif kind == "geom":
        cum_r = (1+r).cumprod() - 1
    else:
        raise ValueError(kind)
    return cum_r


def get_r_from_px(px: FloatSeries, kind: str=DEFAULT_R_KIND) -> FloatSeries:
    if kind == "arith":
        r = px.diff()
    elif kind == "log":
        r = px / px.shift()
        r = np.log(r)
    elif kind == "geom":
        r = px / px.shift()
        r = r - 1
    else:
        raise ValueError(kind)
    return r


def get_r_from_yld(yld: FloatSeries, dur: float=DEFAULT_BOND_DUR, annualizer: int=DAYCOUNTS["BY"]) -> FloatSeries:
    """Approximation for a constant-duration bond, assuming log yields."""
    single_day_carry_exposed_return = yld.shift() / annualizer
    # remember: duration is in years, so we must use annualized yields
    delta_yld = yld - yld.shift()
    duration_exposed_return = -dur * (delta_yld)
    r = single_day_carry_exposed_return + duration_exposed_return
    return r


def get_xr(r: FloatSeries, cash_r: FloatSeries) -> FloatSeries:
    """Excess-of-cash returns."""
    cash_r = cash_r.reindex(index=r.index).rename(r.name)
    xr = r - cash_r
    return xr


def get_vol_targeted(
        xr: FloatSeries,
        tgt_vol: float=DEFAULT_VOL,
        est_window_kind: str=DEFAULT_EST_WINDOW_KIND
    ) -> FloatSeries:
    """(Implementably) modulate volatility.

    Input should be excess-of-cash returns:
        If you delever, you can deposit the excess cash in the bank and earn interest;
        Whereas if you uplever, you must pay a funding rate.
        It's simplistic to assume the funding rate is the same as the deposit interest rate
        (it will usually be higher, since your default risk is greater than the bank's), but ok.
    """
    est_vol = get_est_vol(r=xr, est_window_kind=est_window_kind)
    # at the end of each session, we check the data,
    # then trade up or down to hit this much leverage...
    est_required_leverage = tgt_vol / est_vol
    # ... thence, over the next session, we earn this much return
    # (based on yesterday's estimate of required leverage)
    leverage_at_t = est_required_leverage.shift(IMPL_LAG)
    levered_xr_at_t = leverage_at_t * xr
    levered_xr_at_t = levered_xr_at_t.rename(xr.name)
    return levered_xr_at_t


def get_hedged(
        base_xr: FloatSeries,
        hedge_xr: FloatSeries,
        est_window_kind: str=DEFAULT_EST_WINDOW_KIND
    ) -> FloatSeries:
    """(Implementably) short out base asset's exposure to hedge asset.

    Inputs should be excess-of-cash returns:
        If you delever, you can deposit the excess cash in the bank and earn interest;
        Whereas if you uplever, you must pay a funding rate.
        It's simplistic to assume the funding rate is the same as the deposit interest rate
        (it will usually be higher, since your default risk is greater than the bank's), but ok.
    """
    # at the end of each day, we submit an order to short this much `out`
    est_beta = get_est_beta(of=base_xr, on=hedge_xr, est_window_kind=est_window_kind)
    # this is weight as $ notional / $ NAV
    hedge_pos_at_t = -est_beta.shift(IMPL_LAG)
    hedge_xpnl_at_t = hedge_pos_at_t * hedge_xr
    hedged_base_xr_at_t = base_xr + hedge_xpnl_at_t
    hedged_base_xr_at_t = hedged_base_xr_at_t.rename(base_xr.name)
    return hedged_base_xr_at_t


def smooth(
        r: FloatSeries,
        avg_kind: str=DEFAULT_AVG_KIND,
        window_kind: str=DEFAULT_SMOOTHING_WINDOW_KIND,
        horizon: int=DEFAULT_SMOOTHING_HORIZON,
        scale_up_pow: float=0
    ) -> FloatSeries:
    """Smooth returns, e.g. to account for international trading-session async.

    scale_up_pow: float,
        * ...
        * -1 -> scale down by 1/horizon (why would you do this? idk);
        * ...
        * 0 -> don't scale up;
        * ...
        * +0.5 -> scale up by horizon**0.5 (motivated by CLT -- STD of avg scales with inverse of N);
        * ....
    """
    est_avg = _get_est_avg(y=r, avg_kind=avg_kind, est_window_kind=window_kind, est_horizon=horizon)
    scaled = est_avg * horizon**scale_up_pow
    return scaled


########################################################################################################################
## STATISTICAL CALCULATIONS ############################################################################################
########################################################################################################################

def _get_window(
        ser: pd.Series,
        kind: str=DEFAULT_EVAL_WINDOW_KIND,
        horizon: int=HORIZONS["sweet"],
        min_periods: Optional[int]=None
    ) -> pd.core.window.Window:
    min_periods = fc.maybe(min_periods, ow=int(horizon/2))
    if kind == "full":
        window = ser
    elif kind == "expanding":
        window = ser.expanding(min_periods=min_periods)
    elif kind == "ewm":
        window = ser.ewm(com=horizon, min_periods=min_periods)
    elif kind == "rolling":
        window = ser.rolling(window=horizon, min_periods=min_periods)
    else:
        raise ValueError(kind)
    return window


def _get_est_avg(
        y: FloatSeries,
        avg_kind: str=DEFAULT_AVG_KIND,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        est_horizon: int=DEFAULT_EST_HORIZON
    ) -> Floatlike:
    window = _get_window(y, kind=est_window_kind, horizon=est_horizon)
    if avg_kind == "mean":
        est_avg = window.mean()
    elif avg_kind == "median":
        est_avg = window.median()
    else:
        raise ValueError(avg_kind)
    est_avg = est_avg
    return est_avg


def _get_est_deviations(
        y: FloatSeries,
        de_avg_kind: Optional[str]=None,
        avg_est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        avg_est_horizon: int=DEFAULT_EST_HORIZON,
    ) -> FloatSeries:
    avg = 0 if de_avg_kind is None else \
        _get_est_avg(y=y, avg_kind=de_avg_kind, est_window_kind=avg_est_window_kind, est_horizon=avg_est_horizon)
    est_deviations = y - avg
    return est_deviations


def get_est_cov(
        y: FloatSeries, x: FloatSeries,
        de_avg_kind: Optional[str]=DEFAULT_AVG_KIND,    
        smoothing_avg_kind: str=DEFAULT_AVG_KIND,
        smoothing_window_kind: str=DEFAULT_SMOOTHING_WINDOW_KIND,
        # pass `1` if you don't want to smooth
        smoothing_horizon: int=DEFAULT_SMOOTHING_HORIZON,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        est_horizon: int=DEFAULT_EST_HORIZON,
        # want this to be general (not necessarily finance-first)
        annualizer: int=1
    ) -> Floatlike:
    """Simple GARCH estimate of covariance.

    The estimate at time `t` incorporates information up to and including `t`.

    sources
    -------
    https://github.com/sparshsah/foggy-demo/blob/main/demo/stats/bias-variance-risk.ipynb.pdf
    https://faculty.fuqua.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf
    """
    df = pd.DataFrame(OrderedDict([("y", y), ("x", x)]))
    del x, y
    est_deviations = df.apply(
        lambda col: _get_est_deviations(
            col,
            de_avg_kind=de_avg_kind,
            avg_est_window_kind=est_window_kind,
            avg_est_horizon=est_horizon
        )
    )
    smoothed_est_deviations = est_deviations.apply(
        lambda col: smooth(
            r=col,
            avg_kind=smoothing_avg_kind,
            window_kind=smoothing_window_kind,
            horizon=smoothing_horizon,
            # account for CLT -> get the same STD
            scale_up_pow=0.5
        )
    )
    est_co_deviations = smoothed_est_deviations["y"] * smoothed_est_deviations["x"]
    est_cov = _get_window(est_co_deviations, kind=est_window_kind, horizon=est_horizon).mean()
    ann_est_cov = est_cov * annualizer
    return ann_est_cov


def _get_est_std(
        y: pd.Series,
        de_avg_kind: Optional[str]=None,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND
    ) -> Floatlike:
    est_var = get_est_cov(y=y, x=y, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_std = est_var **0.5
    return est_std


def get_est_corr(
        y: pd.Series,
        x: pd.Series,
        de_avg_kind: Optional[str]=DEFAULT_AVG_KIND,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND
    ) -> Floatlike:
    est_cov = get_est_cov(y=y, x=x, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_y_std = _get_est_std(y, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_x_std = _get_est_std(x, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_corr = est_cov / (est_y_std * est_x_std)
    return est_corr


########################################################################################################################
## FINANCIAL EVALUATIONS ###############################################################################################
########################################################################################################################

def get_est_er(
        r: FloatSeries,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        annualizer: int=DAYCOUNTS["BY"]
    ) -> Floatlike:
    est_avg = _get_est_avg(y=r, est_window_kind=est_window_kind)
    est_er = est_avg * annualizer
    return est_er


def get_est_vol(
        r: FloatSeries,
        de_avg_kind: Optional[str]=None,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        annualizer: int=DAYCOUNTS["BY"]
    ) -> Floatlike:
    est_std = _get_est_std(y=r, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_vol = est_std * annualizer**0.5
    return est_vol


def get_est_sharpe(
        r: FloatSeries,
        de_avg_kind: Optional[str]=None,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        annualizer: int=DAYCOUNTS["BY"]
    ) -> Floatlike:
    est_er = get_est_er(r=r, est_window_kind=est_window_kind, annualizer=annualizer)
    est_vol = get_est_vol(r=r, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind, annualizer=annualizer)
    est_sharpe = est_er / est_vol
    return est_sharpe


def get_t_stat(
        r: FloatSeries,
        de_avg_kind: Optional[str]=None,
        window_kind: str=DEFAULT_EVAL_WINDOW_KIND
    ) -> Floatlike:
    # https://web.stanford.edu/~wfsharpe/art/sr/sr.htm
    granular_est_sharpe = get_est_sharpe(r=r, de_avg_kind=de_avg_kind, est_window_kind=window_kind, annualizer=1)
    valid_timesteps = r.notna().sum()
    t_stat = granular_est_sharpe * valid_timesteps**0.5
    return t_stat


def get_est_beta(
        of: FloatSeries,
        on: FloatSeries,
        de_avg_kind: Optional[str]=None,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND
    ) -> Floatlike:
    est_corr = get_est_corr(y=of, x=on, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_of_std = _get_est_std(of, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_on_std = _get_est_std(on, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_beta = est_corr * (est_of_std / est_on_std)
    return est_beta


def get_est_perf_stats(r: FloatSeries, rounded: bool=True) -> FloatSeries:
    perf_stats = [
        ("Sharpe", get_est_sharpe(r=r)),
        ("ER", get_est_er(r=r)),
        ("Vol", get_est_vol(r=r)),
        ("t-stat", get_t_stat(r=r)),
        (
            "Frac valid timesteps",
            r.notna().sum() / len(r.index)
        ),
        ("Total valid timesteps", r.notna().sum()),
        ("Total timesteps", len(r.index)),
        ("First timestep", r.index[0]),
        ("First valid timestep", r.first_valid_index()),
        ("Last valid timestep", r.last_valid_index()),
        ("Last timestep", r.index[-1])
    ]
    perf_stats = OrderedDict(perf_stats)
    perf_stats = pd.Series(perf_stats)
    if rounded:
        round_dps = {"Sharpe": 2, "ER": 4, "Vol": 4, "t-stat": 2, "Frac valid timesteps": 3}
        for (k, dps) in round_dps.items():
            perf_stats.loc[k] = np.round(perf_stats.loc[k], dps)
    return perf_stats


########################################################################################################################
## VISUALIZATION #######################################################################################################
########################################################################################################################

def chart(r: FloatSeries, kind: str=DEFAULT_R_KIND) -> None:
    cum_r = get_cum_r(r=r)
    fc.plot(cum_r, ypct=True, title=f"{r.name} {kind} CumRets")
    est_perf_stats = get_est_perf_stats(r=r)
    print(est_perf_stats)
    # return est_perf_stats
