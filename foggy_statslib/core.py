"""Catch-all utility module.

v1.1 beta: API probably won't dramatically change, but
    implementations have not yet been thoroughly tested.

author: [@sparshsah](https://github.com/sparshsah)

made available under MIT license at https://github.com/sparshsah/foggy-statslib/blob/main/foggy_statslib/core.py
"""

from __future__ import annotations

from collections.abc import Callable, Collection
import datetime
import itertools
import operator
import os
import pickle
import random
import string
from typing import Any, Final, Iterable, TypeVar
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes as PlotAxes
import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker

T = TypeVar("T")
T_ = TypeVar("T_")
Data = pd.Series | pd.DataFrame
FloatSeries = pd.Series
FloatDF = pd.DataFrame
FloatSeriesOrDF = FloatSeries | FloatDF
Floatlike = float | FloatSeriesOrDF
Datelike = datetime.datetime | str

REASONABLE_FRACTION_OF_TOTAL: Final[float] = 0.95
# Unix epoch
DEFAULT_FIRST_DATETIME: Final[pd.Timestamp] = pd.to_datetime("1970-01-01")
DEFAULT_DATETIME_FREQ: Final[str] = "B"  # business day
TYPEFACE: Final[str] = "Arial"
FONT_SCALE: Final[int] = 1
LABEL_FONTSIZE: Final[int] = 12
LABEL_ROTATION: Final[int] = 0
LEGEND_LOC: Final[tuple[float, float]] = (1.04, 0.08)  # x, y
FIGSIZE: Final[tuple[float, float]] = (12, 8)  # width (x), height (y)

sns.set()


def maybe(
    v: T = None,
    ow: T_ = None,
) -> T | T_:
    """Maybe some value, otherwise some fill-in value."""
    return ow if v is None else v


########################################################################################################################
## DATA WRANGLING ######################################################################################################
########################################################################################################################


def flatten(
    lst: list[list],
) -> list:
    """
    >>> lst = [
    >>>     ["A0", "A1"],
    >>>     "B",
    >>>     ["C0"]
    >>> ]
    >>> flatten(lst)
    ["A0", "A1", "B", "C0"]
    """
    # sum(lst, start=[]) is so pretty but
    # works only if every element is a list...
    # so fails on the docstring example :(
    return list(itertools.chain(*lst))


def _rep_it_to_len(it: Iterable, len_: int) -> Iterable:
    """Repeat an iterable until it reaches the given length.
    E.g. [a, b], 5 -> [a, b, a, b, a].
    """
    num_reps = 1 + int(len_ * 1.0 / len(it))
    return (it * num_reps)[:len_]


def get_chunks(
    it: Iterable[T],
    sz: int,
    index_by_iloc: bool = False,
) -> Iterable[Iterable[T]]:
    """Break up `it` into `sz`-size chunks."""
    # this is a generator expression!
    return (
        it.iloc[i : i + sz] if index_by_iloc else it[i : i + sz]
        for i in range(0, len(it), sz)
    )


def get_dtx(
    periods: int | None = None,
    start: Datelike | None = None,
    end: Datelike | None = None,
    freq: str = DEFAULT_DATETIME_FREQ,
    name: str | None = None,
) -> pd.DatetimeIndex:
    """Helper for constructing toy index for simulated data."""
    if periods is None:
        start = maybe(start, DEFAULT_FIRST_DATETIME)
        end = maybe_date(end, freq=freq, granular=(freq != DEFAULT_DATETIME_FREQ))
    elif periods is not None and start is None:
        end = maybe_date(end, freq=freq, granular=(freq != DEFAULT_DATETIME_FREQ))
    # else, if necessary, let the constructor complain
    dtx = pd.date_range(start=start, periods=periods, end=end, freq=freq, name=name)
    return dtx


def get_union_of_many_pandas_indexes(
    ixs: Collection[pd.Index | pd.DatetimeIndex],
    dt: bool = False,
) -> pd.Index | pd.DatetimeIndex:
    ix = pd.DatetimeIndex([], freq="D") if dt else pd.Index([])
    for x in ixs:
        ix = ix.union(x)
    return ix


def get_series(
    data: list[tuple[Any, Any]],
    name: str | None = None,
) -> pd.Series:
    """Convert a list of (key, value) pairs into a pd.Series.

    The point of this was to use OrderedDict, so you'd preserve index sort in the Series.
    But then CPython started to preserve (insertion) order in default dicts.
    So I removed that and just retain this for backward compatibility.
    """
    data = dict(data)
    data = pd.Series(data, name=name)
    return data


def get_df(
    data: list[tuple[Any, pd.Series]],
    values_are: str = "rows",
) -> pd.DataFrame:
    """Convert orderly data `..., (rowname,row), ...` into a DataFrame.

    ```
    if values_are == "rows":
        if each value is a Series:
            return a DataFrame
        elif each value is a DataFrame:
            return a DataFrame with MultiIndex'ed index
    elif values_are == "columns":
        if each value is a Series:
            return a DataFrame
        elif each value is a DataFrame:
            return a DataFrame with MultiIndex'ed columns
    ```
    """
    data = dict(data)
    data = pd.concat(data.values(), axis="columns", keys=data.keys())
    if values_are == "rows":
        data = data.T
    elif values_are == "columns":
        pass
    else:
        raise ValueError(values_are)
    return data


def _describe_flag_ser(flag_ser: pd.Series[bool]) -> str:
    flag_avg = flag_ser.mean()
    flag_count = flag_ser.sum()
    tot_count = flag_ser.count()
    des = f"{flag_avg :.2%} ({flag_count} / {tot_count})"
    return des


def _construct_drop_warning_msg(
    is_drop: pd.Series[bool],
    reason: str = "unspecified reason",
) -> str:
    des = _describe_flag_ser(is_drop)
    msg = f"Dropping {des} of rows on account of {reason}!"
    return msg


def drop_rows(
    df: pd.DataFrame,
    is_drop: pd.Series[bool],
    reason: str = "unspecified reason",
) -> pd.DataFrame:
    if not df.index.equals(is_drop.index):
        raise ValueError("We don't know how to implicitly coerce conformation!")
    msg = _construct_drop_warning_msg(is_drop=is_drop, reason=reason)
    warnings.warn(msg)
    df = df.loc[~is_drop, :]
    return df


def set_index(
    df: pd.DataFrame,
    col_name: list[str] | str = "primary_key",
    force_integrity: bool = False,
    keep: str = "last",
    verify_integrity: bool = True,
) -> pd.DataFrame:
    if force_integrity:
        is_drop = df.duplicated(col_name, keep=keep)
        df = drop_rows(
            df=df,
            is_drop=is_drop,
            reason=f"duplicated {col_name}",
        )
    df = df.set_index(
        col_name,
        verify_integrity=verify_integrity,
    )
    return df


def decide_is_in(
    a: pd.Series | dict | Iterable | pd.Index,
    b: Any,
) -> pd.Series[bool]:
    """Decide whether each of a's values is present in b.

    Similar to `pd.isin()`, but more conveninent because
    you'll always get a pd.Series index'ed according to `a`:
    If `a` is a Series, the output will be indexed by `a.index`;
    If `a` is a dict, the output will be indexed by `a.keys()`;
    If `a` is a generic iterable, the output will be indexed by `[0, ..., len(a))`;
    If `a` is an Index, the output will be indexed by `a` itself.
    """
    a = pd.Series(a, index=a) if isinstance(a, pd.Index) else pd.Series(a)
    is_in = a.isin(b)
    return is_in


def fillna(
    df: Data,
    value: Any = 0,
    limit_consec: int = 0,
) -> Data:
    """Within each run of NaN's in the input DataFrame, fill
    up to the first `limit_consec` consecutive slots with `value`.

    Pandas's stock `bfill(limit=limit)` and `ffill(limit=limit)`
    methods correctly fill up to the first `limit` consecutive NaN's
    ---including if you invoke them as
    `fillna(method='bfill', limit=limit)` or
    `fillna(method='ffill', limit=limit)`---but
    `fillna(value=value, limit=limit)` bizarrely fills
    only up to the first `limit` *overall* NaN's...
    e.g.
    >>> pd.Series([NaN, NaN, NaN, 1, NaN]).fillna(value=0, limit=2)
    # NOT pd. Series([0, 0, NaN, 1, 0]), but rather
    pd.Series([0, 0, NaN, 1, NaN])
    """
    warnings.warn("This hasn't been thoroughly tested!")
    # NaN if df's original value was NaN, else just `value`...
    # seems like the opposite of what we want, but bear with me
    filler = df * 0 + value
    filler = filler.ffill(limit=limit_consec)
    # handle potential leading NaN's
    filler.iloc[:limit_consec] = value
    filled_df = df.combine_first(filler)
    return filled_df


def get_valid_subsample(df: Data) -> Data:
    """Discard everything before (rsp after)
    the first (rsp last) valid index.
    """
    first, last = df.first_valid_index(), df.last_valid_index()
    # if you don't check this, an all-NaN Series will slip through.
    # for fun, u can break this w/ e.g. [0:NaN, None:"B", 2:"C", 3:NaN].
    if first is None:
        # preserves Series's name, DataFrame's columns
        return df.iloc[:0]
    return df.loc[first:last]


def get_common_subsample(df: pd.DataFrame) -> pd.DataFrame:
    """The output won't start until every column has started,
    and will end as soon as any column ends.
    """
    first = df.apply(lambda ser: ser.first_valid_index()).max()
    last = df.apply(lambda ser: ser.last_valid_index()).min()
    if first is None:
        return df.iloc[:0]
    return df.loc[first:last]


def _check_diff(
    ser_a: pd.Series,
    ser_b: pd.Series,
) -> str:
    names_match = (
        (ser_a.name == ser_b.name)
        or (
            pd.isna(ser_a.name) and pd.isna(ser_b.name)
        )
    )
    names_diff = not names_match
    ix_diff_ab = ~decide_is_in(ser_a.index, ser_b.index)
    ix_diff_ab_des = _describe_flag_ser(ix_diff_ab)
    ix_diff_ba = ~decide_is_in(ser_b.index, ser_a.index)
    ix_diff_ba_des = _describe_flag_ser(ix_diff_ba)
    # arbitrarily, choose the first as the base
    ser_b = ser_b.rename(ser_a.name).reindex(index=ser_a.index)
    val_diff = ser_a != ser_b
    val_diff_des = _describe_flag_ser(val_diff)
    msg = (
        f"""A's and B's names {"DO" if names_diff else "don't"} differ."""
        + f" A contains {ix_diff_ab_des} labels over B."
        + f" B contains {ix_diff_ba_des} labels over A."
        + f" Coerced to A's schema, their values differ at {val_diff_des} of places."
    )
    return msg


def check_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.Series:
    # TODO(sparshsah): Use datacompy
    diff = pd.Series(dtype=float)
    diff.loc["_cols_missing_in_b"] = len(df_a.columns.difference(df_b.columns)) / len(
        df_a.columns
    )
    diff.loc["_cols_excess_in_b"] = len(df_b.columns.difference(df_a.columns)) / len(
        df_a.columns
    )
    diff.loc["_ix_missing_in_b"] = len(df_a.index.difference(df_b.index)) / len(
        df_a.index
    )
    diff.loc["_ix_excess_in_b"] = len(df_b.index.difference(df_a.index)) / len(
        df_a.index
    )
    # must reindex at this point to make pointwise comparisons work
    common_cols = df_a.columns.intersection(df_b.columns)
    common_ix = df_a.index.intersection(df_b.index)
    df_a = df_a.reindex(columns=common_cols, index=common_ix)
    df_b = df_b.reindex(columns=common_cols, index=common_ix)
    diff.loc["_dtypes"] = (df_a.dtypes != df_b.dtypes).mean()
    for colname in df_a.columns:
        try:
            defined_equal = np.isclose(df_a[colname], df_b[colname])
        except TypeError:
            # isclose doesn't handle strings, etc
            defined_equal = df_a[colname] == df_b[colname]
        undefined_equal = df_a[colname].isna() & df_b[colname].isna()
        equal = defined_equal | undefined_equal
        unequal = ~equal
        diff.loc[colname] = unequal.mean()
    return diff


########################################################################################################################
## DB ##################################################################################################################
########################################################################################################################


def _filter_like(
    it: Iterable[str],
    like: str,
    fn: Callable = lambda x: x.upper(),
    not_: bool = False,
) -> list[str]:
    fn = (lambda x: x) if fn is None else fn
    cond = operator.not_ if not_ else bool
    result = [x for x in it if cond(fn(like) in fn(x))]
    return result


def filter_like(
    it: Iterable[str],
    like_ands: Iterable[str] = ("",),
    like_ors: Iterable[str] = ("",),
    fn: Callable = lambda x: x.upper(),
    not_: bool = False,
) -> list[str]:
    # TODO(sparshsah): Could probably use a binary indicator for each element...
    #     Then collect at end. To preserve legitimate mutiplicitous elements.
    for like_and in like_ands:
        # overwrite
        it = _filter_like(it=it, like=like_and, fn=fn, not_=not_)
    # now we've narrowed down
    like_or_results = [
        _filter_like(it=it, like=like_or, fn=fn, not_=not_) for like_or in like_ors
    ]
    result = flatten(like_or_results)
    # remove dupe's introduced in `or` step, while preserving order...
    # unfortunately, also removes legitimate multiples found in the original...
    result = list(dict.fromkeys(result))
    return result


def get_nearest_value(
    df: Data,
    key: float,
    axis: str = "index",
) -> tuple[float, Any]:
    """
    For all those times you have a float-indexed Series or DataFrame
    like `ser = pd.Series({0.1: "a", 0.2: "b", 0.3: "c"})`,
    and you try to access `ser.loc[2/10]`,
    but get a KeyError because of floating-point imprecision.

    Just do `get_nearest_value(ser, 2./10.)`,
    and get back "b" as desired.
    Or, you could do `get_nearest_value(ser, 0.22)`,
    and it will again return "b".

    If you want to search along the columns, pass `axis="columns"`.

    OUTPUT
    ------
    (loc, v): (float, Any), the nearest key and its value.
        For most use cases, you'll just want the value, i.e.
        `_, v = get_nearest_value(df, key)`.
    """
    if axis != "index":
        raise NotImplementedError(f"axis={axis}")
    index = df.index
    distance = np.abs(index - key)
    distance = pd.Series(distance, index=index)
    loc = distance.idxmin()
    v = df.loc[loc]
    return loc, v


########################################################################################################################
## CALCULATIONS ########################################################################################################
########################################################################################################################


def ceil(x: float, increment: float = 1.0) -> float:
    """E.g. Get the next 10% mark for a chart axis:
    >>>> ceil(x=0. 1337, increment=0.10)
    0.20
    """
    return np.ceil(x / increment) * increment


def get_diag_of_ser(ser: pd.Series) -> pd.DataFrame:
    diag_df = np.diag(ser)
    diag_df = pd.DataFrame(diag_df, index=ser.index, columns=ser.index)
    return diag_df


def get_diag_of_df(df: pd.DataFrame) -> pd.DataFrame:
    diag_df = np.diag(df)
    diag_df = pd.DataFrame(diag_df, index=df.index, columns=df.columns)
    return diag_df


def get_inv_of_df(df: FloatDF) -> FloatDF:
    inv_df = np.linalg.inv(df)
    inv_df = pd.DataFrame(inv_df, index=df.index, columns=df.columns)
    return inv_df


def compose_cov_matrix(vol_vector: FloatSeries, corr_matrix: FloatDF) -> FloatDF:
    vol_matrix = get_diag_of_ser(ser=vol_vector)
    cov_matrix = vol_matrix @ corr_matrix @ vol_matrix
    return cov_matrix


# weighted averages


def __get_wa(
    df: pd.DataFrame,
    y: str = "y",
    y_dropna: bool = False,
    y_isna: bool = False,
    w: str = "__unit__",
    w_fillna_val: float = np.nan,
) -> float:
    # values of interest
    y_col = df[y]
    if y_dropna:
        y_col = y_col.dropna()
    if y_isna:
        y_col = y_col.isna()
    # weights
    if w == "__unit__":
        w_col = pd.Series(1, index=y_col.index)
    else:
        w_col = df[w]
    w_col = w_col.reindex(index=y_col.index)
    w_col = w_col.fillna(w_fillna_val)
    # calc
    if w_col.sum():
        wa = np.average(y_col, weights=w_col)
    else:
        # either w is all zero's, OR even worse y (and therefore w) is altogether empty
        wa = np.nan
    return wa


def _get_wa(
    df: pd.DataFrame,
    y: str = "y",
    y_dropna: bool = False,
    y_isna: bool = False,
    w: str = "__unit__",
    w_fillna_val: float = np.nan,
    x: Iterable[str] = ("__unit__",),
) -> pd.Series:
    # splits
    x = list(x)
    # calc
    gb = df.groupby(x)
    wa = gb.apply(
        lambda sub_df: __get_wa(
            df=sub_df,
            y=y,
            y_dropna=y_dropna,
            y_isna=y_isna,
            w=w,
            w_fillna_val=w_fillna_val,
        )
    )
    wa.name = y
    return wa


def get_wa(
    df: pd.DataFrame,
    ys: Iterable[str] = ("y",),
    y_dropna: bool = False,
    y_isna: bool = False,
    w: str = "__unit__",
    w_fillna_val: float = np.nan,
    x: Iterable[str] = ("__unit__",),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            y: _get_wa(
                df=df,
                y=y,
                y_dropna=y_dropna,
                y_isna=y_isna,
                w=w,
                w_fillna_val=w_fillna_val,
                x=x,
            )
            for y in ys
        }
    )


########################################################################################################################
## DATA REPRESENTATION #################################################################################################
########################################################################################################################


def _abbrfstr(s: str, mangle: bool = False) -> str:
    """Abbreviate a string.

    >>> _abbrfstr("a")
    "a"

    >>> _abbrfstr("ab")
    "ab"

    >>> _abbrfstr("abc")
    "ac"

    >>> _abbrfstr("abcd")
    "ad"
    """
    head = s[0]
    tail = s[1:]
    tail = tail[-1] if tail else ""
    mangler = random.choice(string.ascii_lowercase) if mangle else ""
    abbr = head + tail + mangler
    return abbr


def strfpct(pct: float, dps: int = 1) -> str:
    """E.g. 0.1337 -> 13.4%."""
    pct = pct * 100
    pct = round(pct, dps)
    out = "{pct:." + str(dps) + "f}%"
    out = out.format(pct=pct)
    return out


def strfccy(amt: float, ccy: str = "$") -> str:
    """Pretty-format a dollar amount."""
    return f"{ccy}{amt:,.2f}"


def strftime_scalar(
    dt: Datelike,
    fmt: str = "%Y%q",
) -> str:
    """
    Tries to take the best-of-both-worlds:
    * C89 / Python: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    * Pandas: https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases
    """
    dt = pd.to_datetime(dt)
    if fmt == "%Y%q":
        base = dt.strftime("%YQ")
        q = str(dt.quarter)
        ret = base + q
    else:
        ret = dt.strftime(fmt)
    return ret


def strftime_vector(
    dt: pd.Series,
    fmt: str = "%Y%q",
) -> pd.Series:
    """
    Tries to take the best-of-both-worlds:
    * C89 / Python: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    * Pandas: https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases
    """
    dt = pd.to_datetime(dt)
    if fmt == "%Y%q":
        base = dt.dt.strftime("%YQ")
        q = dt.dt.quarter.astype("string")
        ret = base + q
    else:
        ret = dt.dt.strftime(fmt)
    return ret


def strfdate(date: Datelike = "now", granular: bool = False) -> str:
    """Pretty-formate a date."""
    date = datetime.datetime.now() if date == "now" else date
    date = pd.to_datetime(date)
    fmt = "%Y-%m-%d"
    fmt = fmt + "-%H-%M-%S" if granular else fmt
    out = date.strftime(fmt)
    return out


########################################################################################################################
## DATA I/O ############################################################################################################
########################################################################################################################


def _validate_dirpath(dirpath: str, raise_: bool = False) -> str:
    if not dirpath.endswith("/"):
        msg = f"Directory path '{dirpath}' doesn't end with '/'!"
        if raise_:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
    return dirpath


def _validate_ext(ext: str, raise_: bool = False) -> str:
    if not ext.startswith("."):
        msg = f"Extension '{ext}' doesn't start with '.'!"
        if raise_:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
    return ext


def _extract_ext(fpath: str, validate: bool = True) -> str:
    """E.g. 'path/to/data.csv' -> '.csv'."""
    # everything after the last dot
    ext = fpath.split(".")[-1]
    ext = f".{ext}"
    if validate:
        ext = _validate_ext(ext=ext)
    return ext


def _get_qualified_ext(qualifier: str = "", ext: str = ".txt") -> str:
    """E.g.
    >>> _get_qualified_ext('some_description', '.csv')
    # this is now ready to be prefixed with e.g. a timestamp 't2019-01-26'
    '_some_description.csv'
    """
    ext = _validate_ext(ext=ext)
    ext = f"_{qualifier}{ext}"
    return ext


def _gen_tmp_filename(ext: str = ".txt") -> str:
    ext = _validate_ext(ext=ext)
    # timestamp
    t = maybe_date(granular=True, as_str=True)
    # random seq of 16 capital letters
    alphabet = [chr(ascii_code) for ascii_code in range(65, 65 + 26)]
    seq = [random.choice(alphabet) for _ in range(16)]
    seq = "".join(seq)
    # hopefully, the timestamp plus the random sequence
    # are entropic enough to avoid collisions
    fname = f"t{t}_{seq}{ext}"
    return fname


def gen_tmp_filepath(dirpath: str = "data/", ext: str = ".txt") -> str:
    dirpath = _validate_dirpath(dirpath)
    ext = _validate_ext(ext=ext)
    fname = _gen_tmp_filename(ext=ext)
    fpath = os.path.join([dirpath, fname])
    return fpath


def _from_pickle(fpath: str, **kwargs) -> Any:
    ext = _extract_ext(fpath=fpath)
    if ext in (".pkl", ".pickle"):
        with open(fpath, "rb") as pkl:
            data = pickle.load(pkl)
    elif ext == ".csv":
        data = pd.read_csv(fpath, **kwargs)
    elif ext in (".xls", ".xlsx"):
        data = pd.read_excel(fpath, **kwargs)
    else:
        raise NotImplementedError(ext)
    return data


def from_pickle(
    description: str = "data",
    dirpath: str = "data/",
    ext: str = ".pkl",
    **kwargs,
) -> Any:
    """Convenience function to find and load
    the most recent pickle (or CSV, etc)
    of the specified kind in the given folder.
    Assumes that filenames contain timestamps.

    E.g.
    >>> from_pickle()
    <some object loaded from './data/t2019-01-26_data.pkl'>
    """
    dirpath = _validate_dirpath(dirpath=dirpath)
    ext = _validate_ext(ext=ext)
    ext = _get_qualified_ext(qualifier=description, ext=ext)
    pkls = os.listdir(dirpath)
    pkls = [pkl for pkl in pkls if pkl.endswith(ext)]
    if not pkls:
        raise RuntimeError(
            f"from_pickle: dir '{dirpath}' contains no such files '*.{ext}'!"
        )
    # if you pre- or suf-fixed by timestamp, here's your payoff
    pkls = sorted(pkls)
    pkl = pkls[-1]
    pkl = os.path.join(dirpath, pkl)
    return _from_pickle(fpath=pkl, **kwargs)


def _to_pickle(data: T, fpath: str) -> T:
    """`fpath` is e.g. 'path/to/file.csv'."""
    ext = _extract_ext(fpath=fpath)
    if ext in (".pkl", ".pickle"):
        with open(fpath, "wb") as pkl:
            pickle.dump(data, pkl, pickle.HIGHEST_PROTOCOL)
    elif ext == ".csv":
        data.to_csv(fpath)
    elif ext in (".xls", ".xlsx"):
        data.to_excel(fpath)
    else:
        raise NotImplementedError(ext)
    return data


def to_pickle(
    data: T,
    description: str = "data",
    dirpath: str = "data/",
    ext: str = ".pkl",
    **kwargs,
) -> T:
    ext = _validate_ext(ext=ext)
    ext = _get_qualified_ext(qualifier=description, ext=ext)
    fpath = gen_tmp_filepath(dirpath=dirpath, ext=ext)
    _ = _to_pickle(data=data, fpath=fpath)
    # just in case you want to validate that this worked
    data = _from_pickle(fpath=fpath, **kwargs)
    return data


########################################################################################################################
## DATES ###############################################################################################################
########################################################################################################################


def __maybe_date(date: Datelike | None = None) -> Datelike:
    date = maybe(date, datetime.datetime.now())
    return date


def __get_offset_date(
    date: Datelike,
    offset: int = 0,
    freq: str = DEFAULT_DATETIME_FREQ,
) -> Datelike:
    offset = offset * pd.tseries.frequencies.to_offset(freq)
    offset_date = date + offset
    return offset_date


def _get_offset_date(
    date: Datelike | None = None,
    offset: int = 0,
    freq: str = DEFAULT_DATETIME_FREQ,
):
    """Like `__get_offset_date()`, but makes sure `offset_date` is not after `date`.
    E.g. On Saturday, gives you yesterday (Friday) instead of day-after-tomorrow (Monday).
    """
    if offset > 0:
        raise ValueError(
            f"Use `__get_offset_date(offset={offset})`! This one is designed to enforce a lag."
        )
    date = __maybe_date(date=date)
    offset_date = __get_offset_date(date=date, offset=offset, freq=freq)
    if offset_date > date:
        offset_date = __get_offset_date(date=date, offset=offset - 1, freq=freq)
    assert not offset_date > date, (date, offset, freq, offset_date)
    return offset_date


def maybe_date(
    date: Datelike | None = None,
    ow_offset: int = 0,
    freq: str | None = None,
    granular: bool = False,
    as_str: bool = False,
) -> Datelike:
    freq = maybe(
        # if the user has specified, then respect that
        freq,
        # otherwise, go with Daily if granular, else go with Business
        "D" if granular else DEFAULT_DATETIME_FREQ,
    )
    date = maybe(date, _get_offset_date(offset=ow_offset, freq=freq))
    date = strfdate(date, granular=granular)
    date = date if as_str else pd.to_datetime(date)
    return date


########################################################################################################################
## DATA VISUALIZATION ##################################################################################################
########################################################################################################################


def __iprint_val(val: Any, flush: bool = False) -> str:
    bookend = "'" if isinstance(val, str) else ""
    out = f"{bookend}{val}{bookend}"
    if flush:
        print(out)
    return out


def _iprint_ser(ser: pd.Series, tab_sz: int = 0, flush: bool = False) -> str:
    out = tab_sz * " " + "pd.Series(" + "\n"
    out += (tab_sz + 4) * " " + "dict([" + "\n"
    for (k, v) in ser.items():
        out += (tab_sz + 8) * " " + "(" + "\n"
        out += (tab_sz + 12) * " " + f"{__iprint_val(k)}," + "\n"
        out += (tab_sz + 12) * " " + f"{__iprint_val(v)}" + "\n"
        out += (tab_sz + 8) * " " + ")," + "\n"
    out += (tab_sz + 4) * " " + "])," + "\n"
    out += tab_sz * " " + f"name={__iprint_val(ser.name)}," + "\n"
    # quoted so i can use pandas's native dtype parser instead of importing names
    out += tab_sz * " " + f"dtype='{ser.dtype}'," + "\n"
    out += tab_sz * " " + ")"
    if flush:
        print(out)
    return out


def iprint_df(df: pd.DataFrame, flush: bool = False) -> str:
    """Print a DataFrame to stdout in a format you can copy-paste into a REPL.
    Useful when you have access to the console but not the filesystem.
    """
    out = "pd.DataFrame(dict([" + "\n"
    for colname, col in df.items():
        out += 4 * " " + "(" + "\n"
        out += 8 * " " + f"{__iprint_val(colname)}," + "\n"
        out += _iprint_ser(col, tab_sz=8) + "\n"
        out += 4 * " " + ")," + "\n"
    out += "]))"
    if flush:
        print(out)
    return out


def _describe_series(ser: pd.Series, bool_is_numeric: bool = True) -> pd.Series:
    if bool_is_numeric:
        if ser.dtype == bool:
            ser = ser.astype(int)
    d = ser.describe()
    return d


def describe_df(df: pd.DataFrame, bool_is_numeric: bool = True) -> pd.DataFrame:
    if bool_is_numeric:
        caster = {
            colname: int if dtype == bool else dtype for (colname, dtype) in df.items()
        }
        df = df.astype(caster)
    d = df.describe()
    return d


def plot(
    df: Data,
    kind: str = "line",
    # THEME
    typeface: str = TYPEFACE,
    font_scale: float = FONT_SCALE,
    mimic_excel: bool = False,
    # BARPLOT OPTIONS
    ## if `True`, then use height, else use provided labels
    bar_labels: pd.Series | bool = False,
    bar_labels_fmt: str = "%g",
    # SCATTERPLOT OPTIONS
    ## if `True`, then use `df.index`, else use provided labels
    scatter_labels: pd.Series | bool = False,
    label_fontsize: float = LABEL_FONTSIZE,
    label_rotation: float = LABEL_ROTATION,
    # GRIDLINES
    ## horizontal (y = ...)
    axhline_locs: Iterable[float] = (0,),
    axhline_zs: Iterable[float] = (0,),
    axhline_styles: Iterable[str] = ("-",),
    axhline_colors: Iterable[str] = ("gray",),
    ## vertical (x = ...)
    axvline_locs: Iterable[float] = tuple(),
    axvline_zs: Iterable[float] = (0,),
    axvline_styles: Iterable[str] = ("-",),
    axvline_colors: Iterable[str] = ("gray",),
    # AXIS LIMITS
    ## x
    xlim: tuple[float, float] | None = None,
    xlim_left: float | None = None,
    xlim_right: float | None = None,
    ## y
    ylim: tuple[float, float] | None = None,
    ylim_bottom: float | None = None,
    ylim_top: float | None = None,
    # AXIS DIRECTION
    invert_xaxis: bool = False,
    invert_yaxis: bool = False,
    # AXIS TICK INTERVALS
    ## x (major, minor)
    xtick_intervals: tuple[float, float] | None = None,
    xtick_major_interval: float | None = None,
    # can set this to `None` to carry over value from `df.plot()`
    xtick_minor_interval: float | None = None,
    ## y (major, minor)
    ytick_intervals: tuple[float, float] | None = None,
    ytick_major_interval: float | None = None,
    ytick_minor_interval: float | None = None,
    # AXIS TICK LABELS
    pct_dps: int = 1,
    granular_dates: bool = False,
    ## x
    xticklabels: bool = True,
    xticklabels_fmt: Callable[[Any, Any], str] | None = None,
    xint: bool = False,
    xpct: bool = False,
    xdollar: bool = False,
    xdates: bool = False,
    xticklabels_rotation: int | str = "auto",
    ## y
    yticklabels: bool = True,
    yticklabels_fmt: Callable[[Any, Any], str] | None = None,
    yint: bool = False,
    ypct: bool = False,
    ydollar: bool = False,
    ydates: bool = False,
    yticklabels_rotation: int | str = "auto",
    # AXIS TITLES
    xlabel: str | None = None,
    ylabel: str | None = None,
    # LEGEND
    legend: bool | None = None,
    legend_title: str | None = None,
    legend_title_fontsize: str | None = None,  # e.g. 'xx-large'
    legend_loc: tuple[float, float] | str = "best",
    # (SUB)PLOT TITLE
    title: str | None = None,
    titley: float = 1.0,
    # SIDE EFFECTS
    ax: PlotAxes | None = None,
    figsize: tuple[float, float] = FIGSIZE,  # width (x), height (y)
    plt_show: bool | None = None,
    savefig_path: str | None = None,
    plt_close: bool | None = None,
    **kwargs,
) -> PlotAxes:

    # helpers
    def _raise_before_override(description: str, val: Any = None) -> None:
        if val is not None:
            msg = f"plot: {description} {val} will be overriden!"
            raise ValueError(msg)
        return val

    ## axis tick formatters (2nd arg, tick_position, is unused)
    def _strfint(tick_val, _):
        return f"{tick_val :,.0f}"
    def _strfpct(tick_val, _):
        return strfpct(tick_val, dps=pct_dps)
    def _strfdollar(tick_val, _):
        return strfccy(tick_val)
    def _strfdate(tick_val, _):
        return strfdate(tick_val, granular=granular_dates)

    # convenience handling of plot kind
    if kind == "histpct":
        kind = "hist"
        kwargs["weights"] = np.ones(len(df.index)) * 1.0 / len(df.index)
        ypct = True
    if kind == "scatter":
        if "x" not in kwargs:
            kwargs["x"] = df.columns[0]
        if "y" not in kwargs:
            kwargs["y"] = df.columns[-1]
        if "c" not in kwargs:
            # warnings.warn("plot: default scatter-point color is c='black'!")
            kwargs["c"] = "black"
        if xlabel is None:
            xlabel = kwargs["x"]
        if ylabel is None:
            ylabel = kwargs["y"]
    # implicit default side effects
    ## user provided `ax` OR wants to savefig -> don't show plot;
    ## user didn't provide `ax` AND doesn't want to savefig -> show plot.
    plt_show = maybe(v=plt_show, ow=(ax is None) and (savefig_path is None))
    ## user doesn't want to savefig OR wants to show -> leave open;
    ## user wants to savefig AND doesn't want to show -> close.
    plt_close = maybe(v=plt_close, ow=(savefig_path is not None) and (not plt_show))

    _ = typeface  # ignore
    sns.set(font_scale=font_scale)
    ax = df.plot(kind=kind, ax=ax, figsize=figsize, **kwargs)

    # BARPLOT OPTIONS
    if kind == "bar":
        if bar_labels is not False:
            ax.bar_label(
                container=ax.containers[0],
                labels=None if bar_labels is True else bar_labels,
                fmt=bar_labels_fmt,
            )
    # SCATTERPLOT OPTIONS
    if kind == "scatter":
        ## could be `False`, `True`, or pd.Series
        if scatter_labels is not False:
            for i in df.index:
                ax.text(
                    x=df.loc[i, kwargs["x"]],
                    y=df.loc[i, kwargs["y"]],
                    s=(
                        i if scatter_labels is True
                        else scatter_labels[i]
                    ),
                    fontsize=label_fontsize,
                    rotation=label_rotation,
                )
                del i

    # GRIDLINES
    if mimic_excel:
        ax.set_axis_bgcolor("white")
        ax.yaxis.grid(
            True,
            which="major",
            color="black",
            linestyle="-",
            linewidth=0.25,
            alpha=0.50,
            zorder=0,
        )
    else:
        # `which` could be 'major', 'minor', or 'both'
        # ax.grid(True, axis="both", which="both")
        ax.xaxis.grid(True, which="both")
        ax.yaxis.grid(True, which="both")
    ax.set_axisbelow(True)
    ## horizontal
    axline_locs = axhline_locs
    num_axlines = len(axline_locs)
    axline_zs = _rep_it_to_len(it=axhline_zs, len_=num_axlines)
    axline_styles = _rep_it_to_len(it=axhline_styles, len_=num_axlines)
    axline_colors = _rep_it_to_len(it=axhline_colors, len_=num_axlines)
    for n in range(num_axlines):
        ax.axhline(
            axline_locs[n],
            zorder=axline_zs[n],
            linestyle=axline_styles[n],
            linewidth=None,
            color=axline_colors[n],
            alpha=1,
        )
    del axline_colors, axline_styles, axline_zs, num_axlines, axline_locs
    ## vertical
    axline_locs = axvline_locs
    num_axlines = len(axline_locs)
    axline_zs = _rep_it_to_len(it=axvline_zs, len_=num_axlines)
    axline_styles = _rep_it_to_len(it=axvline_styles, len_=num_axlines)
    axline_colors = _rep_it_to_len(it=axvline_colors, len_=num_axlines)
    for n in range(num_axlines):
        ax.axvline(
            axline_locs[n],
            zorder=axline_zs[n],
            linestyle=axline_styles[n],
            linewidth=None,
            color=axline_colors[n],
            alpha=1,
        )
    del axline_colors, axline_styles, axline_zs, num_axlines, axline_locs

    # AXIS LIMITS
    ## x
    if xlim is not None:
        _raise_before_override(
            description="individually-specified `xlim_left`", val=xlim_left
        )
        _raise_before_override(
            description="individually-specified `xlim_right`", val=xlim_right
        )
        xlim_left, xlim_right = xlim
    # sensible default
    if kind == "line" and xlim_left is None:
        xlim_left = df.index[0]
        if isinstance(xlim_left, str):
            # https://github.com/matplotlib/matplotlib/blob/0e3a261/lib/matplotlib/axes/_base.py#L3722-L3723
            # For god-knows-what reason, Matplotlib has decided that
            # if you set `xlim_left, xlim_right == some_iterable, None`,
            # then you really meant to set `xlim = some_iterable`,
            # and so it will "helpfully" reassign `xlim_left, xlim_right = some_iterable`.
            # Obviously, str is iterable, so
            # if you set `xlim_left="a"` then it will fail with
            #     `ValueError: not enough values to unpack (expected 2) (got 1)`,
            # if you set `xlim_left="abc"` then it will fail with
            #     `ValueError: too many values to unpack (expected 2) (got 3)`,
            # and if you're unlucky enough to set `xlim_left="ab"` then it will blithely press forward with
            #     `xlim_left, xlim_right = "a", "b"`.
            # The insidious problem with the last case is that str is also comparable --
            # so Matplotlib will lexicographically sort your index,
            # then display only labels inside the interval ["a", "b"].
            # The entire thing is not worth the hassle, so we just null it out.
            xlim_left = None
    if kind == "line" and xlim_right is None:
        xlim_right = df.index[-1]
        if isinstance(xlim_right, str):
            xlim_right = None
    # set
    if xlim_left is not None:
        ax.set_xlim(left=xlim_left)
    if xlim_right is not None:
        ax.set_xlim(right=xlim_right)
    ## y
    if ylim is not None:
        _raise_before_override(
            description="individually-specified `ylim_bottom`", val=ylim_bottom
        )
        _raise_before_override(
            description="individually-specified `ylim_top`", val=ylim_top
        )
        ylim_bottom, ylim_top = ylim
    # set
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    if ylim_top is not None:
        ax.set_ylim(top=ylim_top)

    # AXIS DIRECTION
    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()

    # AXIS TICK INTERVALS
    ## x
    if xtick_intervals is not None:
        _raise_before_override(
            description="individually-specified `xtick_major_interval`",
            val=xtick_major_interval,
        )
        _raise_before_override(
            description="individually-specified `xtick_minor_interval`",
            val=xtick_minor_interval,
        )
        xtick_major_interval, xtick_minor_interval = xtick_intervals
    if xtick_major_interval is not None:
        ax.xaxis.set_major_locator(plt.MultipleLocator(xtick_major_interval))
    if xtick_minor_interval is not None:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(xtick_minor_interval))
    ## y
    if ytick_intervals is not None:
        _raise_before_override(
            description="individually-specified `ytick_major_interval`",
            val=ytick_major_interval,
        )
        _raise_before_override(
            description="individually-specified `ytick_minor_interval`",
            val=ytick_minor_interval,
        )
        ytick_major_interval, ytick_minor_interval = ytick_intervals
    if ytick_major_interval is not None:
        ax.yaxis.set_major_locator(plt.MultipleLocator(ytick_major_interval))
    if ytick_minor_interval is not None:
        ax.yaxis.set_minor_locator(plt.MultipleLocator(ytick_minor_interval))

    # AXIS TICK LABELS
    ## x
    if sum([xticklabels_fmt is not None, xpct, xdollar, xdates]) > 1:
        msg = f"plot: conflict in {xticklabels_fmt}, {xpct}, {xdollar}, {xdates}!"
        raise ValueError(msg)
    fmt = None
    if xticklabels_fmt is not None:
        fmt = xticklabels_fmt
    elif xint:
        fmt = _strfint
    elif xpct:
        fmt = _strfpct
    elif xdollar:
        fmt = _strfdollar
    elif xdates:
        fmt = _strfdate
    if fmt is not None:
        ax.xaxis.set_major_formatter(plt_ticker.FuncFormatter(fmt))
    del fmt
    ax.xaxis.set_tick_params(rotation=xticklabels_rotation)
    if not xticklabels:
        # plt.setp(ax.get_xticklabels(), visible=False)
        ax.xaxis.set_visible(False)
    ## y
    if sum([yticklabels_fmt is not None, ypct, ydollar, ydates]) > 1:
        msg = f"plot: conflict in {yticklabels_fmt}, {ypct}, {ydollar}, {ydates}!"
    fmt = None
    if yticklabels_fmt is not None:
        fmt = yticklabels_fmt
    elif yint:
        fmt = _strfint
    elif ypct:
        fmt = _strfpct
    elif ydollar:
        fmt = _strfdollar
    elif ydates:
        fmt = _strfdate
    if fmt is not None:
        ax.yaxis.set_major_formatter(plt_ticker.FuncFormatter(fmt))
    del fmt
    ax.yaxis.set_tick_params(rotation=yticklabels_rotation)
    if not yticklabels:
        # plt.setp(ax.get_yticklabels(), visible=False)
        ax.yaxis.set_visible(False)
    # AXIS TITLES
    ## x
    xlabel = maybe(xlabel, df.index.name)
    xlabel = maybe(xlabel, False)
    if xlabel:
        ax.set_xlabel(xlabel)
        # ax.get_xaxis().get_label().set_visible(True)
        ax.xaxis.label.set_visible(True)
    else:
        # ax.get_xaxis().get_label().set_visible(False)
        ax.xaxis.label.set_visible(False)
    ## y
    ylabel = maybe(ylabel, False)
    if ylabel:
        ax.set_ylabel(ylabel)
        # ax.get_yaxis().get_label().set_visible(True)
        ax.yaxis.label.set_visible(True)
    else:
        # ax.get_yaxis().get_label().set_visible(False)
        ax.yaxis.label.set_visible(False)

    # LEGEND
    # if legend is None, ignore
    if legend is False:
        # ax.get_legend().set_visible(legend)
        ax.legend_.set_visible(legend)
    if legend:
        ax.legend(title=legend_title, fontsize=legend_title_fontsize, loc=legend_loc)

    # (SUB)PLOT TITLE
    if title is None and hasattr(df, "name"):
        title = df.name
    if title is not None:
        ax.set_title(title, y=titley)

    # SIDE EFFECTS
    if savefig_path is not None:
        plt.savefig(savefig_path)
    if plt_show:
        plt.show()
    if plt_close:
        plt.close()
    return ax


def get_y_equals_x_plot_base(
    lim: float = 1,
    style: str = "--",
    color: str = "black",
    **kwargs,
) -> PlotAxes:
    """Usage, e.g.
    >>> ax = get_y_equals_x_plot_base()
    >>> # the actual data we want to plot over this 45degree line
    >>> plot(df, kind="scatter", ax=ax, plt_show=True)
    """
    df = pd.Series({-lim: -lim, lim: lim})
    ax = plot(df=df, style=style, color=color, plt_show=False, **kwargs)
    return ax


def plot_t_stat(t_stat: Data, **kwargs) -> PlotAxes:
    return plot(
        t_stat,
        axhline_locs=(-1.96, 0, +1.96),
        axhline_styles=("--", "-", "--"),
        **kwargs,
    )


def plot_corr_heatmap(
    corr_matrix: pd.DataFrame,
    cmap: str = "RdYlGn",
    title: str | None = None,
    figsize: tuple[float, float] = (8, 8),  # width (x), height (y)
    **kwargs,
) -> None:
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        cmap=cmap,
        cbar=False,
        square=True,
        **kwargs,
    )
    plt.title(title, size="x-large")
    plt.show()


def plot_eda(
    df: pd.DataFrame,
    col_names: Any = ...,
    height: float = 2.5,
    aspect: int = 1,
    alpha: float = 0.50,
    title: str | None = None,
    savefig_path: str | None = None,
) -> None:
    """Exploratory data analysis."""
    df = df.rename(
        columns=lambda c: f"{c} ({df[c].isna().mean()*100:.2f}% na)",
    )
    if col_names == ...:
        col_names = df.columns
    sns.pairplot(
        df,
        vars=col_names,
        height=height,
        aspect=aspect,
        plot_kws={
            "alpha": alpha,
        },
    )
    if title is not None:
        plt.suptitle(title)
    if savefig_path is not None:
        plt.savefig(savefig_path)
    plt.show()
