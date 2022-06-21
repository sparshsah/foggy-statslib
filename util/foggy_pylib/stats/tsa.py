"""Timeseries analysis.

v1.0 beta: API probably won't dramatically change, but
    implementations have not yet been thoroughly tested.

author: [@sparshsah](https://github.com/sparshsah)
"""

from typing import Union, Optional
import pandas as pd
# https://github.com/sparshsah/foggy-lib/blob/main/util/foggy_pylib/core.py
import foggy_pylib.core as fc

FloatSeries = pd.Series
FloatDF = pd.DataFrame
FloatSeriesOrDF = Union[FloatSeries, FloatDF]
Floatlike = Union[float, FloatSeriesOrDF]

DEFAULT_AVG_KIND: str = "mean"
DEFAULT_DE_AVG_KIND: Optional[str] = DEFAULT_AVG_KIND

DEFAULT_SMOOTHING_WINDOW_KIND: str = "rolling"
DEFAULT_SMOOTHING_HORIZON: int = 3  # overlap of yesterday, today, and tomorrow
# most real-world data-generating processes are non-stationary
DEFAULT_EST_WINDOW_KIND: str = "ewm"
DEFAULT_EST_HORIZON: int = 65  # inspired by number of days in a business quarter
# evaluation, no need to specify horizon
DEFAULT_EVAL_WINDOW_KIND: str = "full"
DEFAULT_EVAL_HORIZON: int = DEFAULT_EST_HORIZON  # doesn't matter since window is full


########################################################################################################################
## DATA MANIPULATIONS ##################################################################################################
########################################################################################################################

def ___get_window(
        ser: pd.Series,
        kind: str=DEFAULT_EVAL_WINDOW_KIND,
        horizon: int=DEFAULT_EVAL_HORIZON,
        min_periods: Optional[int]=None
    ) -> pd.core.window.Window:
    min_periods = fc.maybe(min_periods, ow=int(horizon/2))
    if kind == "full":
        window = ser
    elif kind == "expanding":
        window = ser.expanding(min_periods=min_periods)
    elif kind == "ewm":
        window = ser.ewm(span=horizon, min_periods=min_periods)
    elif kind == "rolling":
        window = ser.rolling(window=horizon, min_periods=min_periods)
    else:
        raise ValueError(kind)
    return window


def ___smooth(
        ser: FloatSeries,
        avg_kind: str=DEFAULT_AVG_KIND,
        window_kind: str=DEFAULT_SMOOTHING_WINDOW_KIND,
        horizon: int=DEFAULT_SMOOTHING_HORIZON,
        scale_up_pow: float=0
    ) -> FloatSeries:
    """E.g. Smooth passive-asset daily-returns data to account for international trading-session async.

    scale_up_pow: float,
        * ...
        * -1 -> scale down by 1/horizon (why would you do this? idk);
        * ...
        * 0 -> don't scale up;
        * ...
        * +0.5 -> scale up by horizon**0.5
            motivated by CLT: STD of avg scales with inverse of sqrt(N)
        * ....
    """
    est_avg = _get_est_avg(ser=ser, avg_kind=avg_kind, est_window_kind=window_kind, est_horizon=horizon)
    scaled = est_avg * horizon**scale_up_pow
    return scaled


########################################################################################################################
## DATA DESCRIPTION ####################################################################################################
########################################################################################################################

def _get_metadata(ser: FloatSeries) -> pd.Series:
    metadata = [
        (
            "Frac valid timesteps",
            ser.notna().sum() / len(ser.index)
        ), (
            "Total valid timesteps",
            ser.notna().sum()
        ), (
            "Total timesteps",
            len(ser.index)
        ), (
            "First timestep",
            ser.index[0]
        ), (
            "First valid timestep",
            ser.first_valid_index()
        ), (
            "Last valid timestep",
            ser.last_valid_index()
        ), (
            "Last timestep", ser.index[-1]
        )
    ]
    metadata = fc.get_series(metadata)
    return metadata


########################################################################################################################
## STATISTICAL CALCULATIONS ############################################################################################
########################################################################################################################

def _get_est_avg(
        ser: FloatSeries,
        avg_kind: str=DEFAULT_AVG_KIND,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        est_horizon: int=DEFAULT_EVAL_HORIZON
    ) -> Floatlike:
    window = ___get_window(ser, kind=est_window_kind, horizon=est_horizon)
    if avg_kind == "mean":
        est_avg = window.mean()
    elif avg_kind == "median":
        est_avg = window.median()
    else:
        raise ValueError(avg_kind)
    return est_avg


def __get_est_deviations(
        ser: FloatSeries,
        de_avg_kind: Optional[str]=DEFAULT_DE_AVG_KIND,
        avg_est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        avg_est_horizon: int=DEFAULT_EVAL_HORIZON,
    ) -> FloatSeries:
    # thing we're going to remove before calculating deviations
    avg = 0 if de_avg_kind is None else \
        _get_est_avg(
            ser=ser,
            avg_kind=de_avg_kind,
            est_window_kind=avg_est_window_kind,
            est_horizon=avg_est_horizon
        )
    est_deviations = ser - avg
    return est_deviations


def _get_est_cov(
        ser_a: FloatSeries, ser_b: FloatSeries,
        de_avg_kind: Optional[str]=DEFAULT_DE_AVG_KIND,
        smoothing_avg_kind: str=DEFAULT_AVG_KIND,
        smoothing_window_kind: str=DEFAULT_SMOOTHING_WINDOW_KIND,
        # pass `1` if you don't want to smooth
        smoothing_horizon: int=DEFAULT_SMOOTHING_HORIZON,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        est_horizon: int=DEFAULT_EVAL_HORIZON,
    ) -> Floatlike:
    """Simple GARCH estimate of covariance.
    The estimate at time `t` incorporates information up to and including `t`.

    You can get a "robust" estimate by setting smoothing_avg_kind = "median",
        which will tend to "filter out" highly-influential one-off outliers.
        If you do this, be sure to also explicitly model kurtosis!

    sources
    -------
    https://github.com/sparshsah/foggy-demo/blob/main/demo/stats/bias-variance-risk.ipynb.pdf
    https://faculty.fuqua.duke.edu/~charvey/Research/Published_Papers/P135_The_impact_of.pdf
    """
    df = fc.get_df([
        ("a", ser_a),
        ("b", ser_b)
    ])
    del ser_b, ser_a
    est_deviations = df.apply(
        lambda col: __get_est_deviations(
            col,
            de_avg_kind=de_avg_kind,
            avg_est_window_kind=est_window_kind,
            avg_est_horizon=est_horizon
        )
    )
    smoothed_est_deviations = est_deviations.apply(
        lambda col: ___smooth(
            ser=col,
            avg_kind=smoothing_avg_kind,
            window_kind=smoothing_window_kind,
            horizon=smoothing_horizon,
            # account for CLT -> get the same STD
            scale_up_pow=0.5
        )
    )
    est_co_deviations = smoothed_est_deviations["a"] * smoothed_est_deviations["b"]
    est_cov = ___get_window(
        est_co_deviations,
        kind=est_window_kind,
        horizon=est_horizon
    ).mean()
    # https://en.wikipedia.org/wiki/Bessel%27s_correction
    # If `de_avg_kind` is None: We treat the mean as known to be zero;
    # Otherwise: We should bias-correct by post-multiplying T/(T-1).
    sample_sz = ___get_window(
        est_co_deviations.notna(),  # 1 at every valid index
        kind=est_window_kind,
        horizon=est_horizon
    ).sum()
    bessel_degrees_of_freedom = bool(de_avg_kind)
    bessel_factor = sample_sz / ( sample_sz - bessel_degrees_of_freedom )
    besseled_est_cov = est_cov * bessel_factor
    return besseled_est_cov


def _get_est_std(
        ser: FloatSeries,
        de_avg_kind: Optional[str]=DEFAULT_DE_AVG_KIND,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        est_horizon: int=DEFAULT_EVAL_HORIZON
    ) -> Floatlike:
    est_var = _get_est_cov(
        ser_a=ser, ser_b=ser,
        de_avg_kind=de_avg_kind,
        est_window_kind=est_window_kind,
        est_horizon=est_horizon
    )
    est_std = est_var **0.5
    return est_std


def _get_est_corr(
        ser_a: FloatSeries, ser_b: FloatSeries,
        de_avg_kind: Optional[str]=DEFAULT_DE_AVG_KIND,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        round_dps: Optional[int]=None
    ) -> Floatlike:
    # important, else vol's could be calc'ed over inconsistent periods (violating nonnegative-definiteness)
    common_period = ser_a.dropna().index.intersection(ser_b.dropna().index)
    ser_a = ser_a.loc[common_period]
    ser_b = ser_b.loc[common_period]
    est_cov = _get_est_cov(ser_a=ser_a, ser_b=ser_b, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_a_std = _get_est_std(ser=ser_a, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_b_std = _get_est_std(ser=ser_b, de_avg_kind=de_avg_kind, est_window_kind=est_window_kind)
    est_corr = est_cov / (est_a_std * est_b_std)
    est_corr = round(est_corr, round_dps) if round_dps is not None else est_corr
    return est_corr


def get_est_corr(
        ser: FloatDF,
        de_avg_kind: Optional[str]=DEFAULT_DE_AVG_KIND,
        est_window_kind: str=DEFAULT_EVAL_WINDOW_KIND,
        round_dps: Optional[int]=None
    ) -> FloatDF:
    est_corr = {
        a: {
            b: _get_est_corr(
                ser_a=ser_a, ser_b=ser_b,
                de_avg_kind=de_avg_kind,
                est_window_kind=est_window_kind,
                round_dps=round_dps
            )
        for (b, ser_b) in ser.iteritems()}
    for (a, ser_a) in ser.iteritems()}
    est_corr = pd.DataFrame(est_corr).T
    return est_corr
