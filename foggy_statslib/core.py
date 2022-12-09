"""Catch-all utility module.

v1.1 beta: API probably won't dramatically change, but
    implementations have not yet been thoroughly tested.

author: [@sparshsah](https://github.com/sparshsah)

made available under MIT license at https://github.com/sparshsah/foggy-statslib/blob/main/foggy_statslib/core.py
"""

from __future__ import annotations

# standard syntax utils:
from typing import Tuple, List, Iterable, Callable, Union, Optional, Any
import operator
# standard data structures:
from collections import OrderedDict
# standard calculations:
import itertools
# standard sys:
import os
from warnings import warn
# standard data i/o:
import datetime as dt
import random
import pickle
# standard visualization: None
# extensions for syntax utils: None
# extensions for data structures:
import pandas as pd
# extensions for calculations:
import numpy as np
# extensions for sys: None
# extensions for data i/o: None
# extensions for visualization:
import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker
from matplotlib.axes import Axes as PlotAxes
import seaborn as sns
sns.set()

# `T` is used to indicate a generic type, but is NOT just a synonym for `Any`.
# For example, in Caml, we can have a function `foo: 'a -> 'b`,
#         which accepts any type, returning some unspecified type.
#         We might write this in Python as `foo: Any -> Any`.
#     But, we can also have a function `bar: 'a -> 'a`,
#         which accepts any type, returning the SAME type.
#         We might write this in Python as `bar: T -> T`.
#
# Example 1: `x:Any in lst:list[Any]` means that the object `x`, of some arbitrary type,
#         is a member of list `lst`, whose members have no specified type,
#         and aren't even guaranteed to all share the same type as each other.
#     But, `x:T in lst:list[T]` means that the object `x`, of some generic type,
#         is a member of list `lst`, every member of whom has that SAME type.
#
# Example 2: `fun: Any -> Any` accepts any type, returning some unspecified type;
#     But, `fun: T -> T` accepts any type, returning the SAME type.
T = Any
T_ = Any
Data = Union[pd.Series, pd.DataFrame]
FloatSeries = pd.Series
FloatDF = pd.DataFrame
FloatSeriesOrDF = Union[FloatSeries, FloatDF]
Floatlike = Union[float, FloatSeriesOrDF]
Datelike = Union[dt.datetime, str]

REASONABLE_FRACTION_OF_TOTAL: float = 0.95
# Unix epoch
DEFAULT_FIRST_DATETIME = pd.to_datetime("1970-01-01")
DEFAULT_DATETIME_FREQ = "B"  # business day
TYPEFACE = "Arial"
FONT_SCALE = 1
LABEL_FONTSIZE = 12
LABEL_ROTATION = 0
LEGEND_LOC = (1.04, 0.08)  # x, y
FIGSIZE = (12, 8)  # width (x), height (y)


def maybe(v: T=None, ow: T_=None) -> Union[T, T_]:
    """Maybe some value, otherwise some fill-in value."""
    return ow if v is None else v


########################################################################################################################
## DATA WRANGLING ######################################################################################################
########################################################################################################################

def get_dtx(
        periods: Optional[int]=None,
        start: Optional[Datelike]=None,
        end: Optional[Datelike]=None,
        freq: str=DEFAULT_DATETIME_FREQ
    ) -> pd.DatetimeIndex:
    if (periods is None):
        start = maybe(start, DEFAULT_FIRST_DATETIME)
        end = maybe_date(end, freq=freq, granular=(freq!=DEFAULT_DATETIME_FREQ))
    elif periods is not None and (start is None):
        end = maybe_date(end, freq=freq, granular=(freq!=DEFAULT_DATETIME_FREQ))
    # else, if necessary, let the constructor complain
    dtx = pd.date_range(start=start, periods=periods, end=end, freq=freq)
    dtx = pd.DatetimeIndex(dtx)
    return dtx


def get_series(data: List[Tuple[Any, Any]], name: Optional[str]=None) -> pd.Series:
    """Convert a list of (key, value) pairs into a pd.Series."""
    data = OrderedDict(data)
    data = pd.Series(data, name=name)
    return data


def get_df(data: List[Tuple[Any, pd.Series]], values_are: str="rows") -> pd.DataFrame:
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
    data = OrderedDict(data)
    data = pd.concat(data.values(), axis="columns", keys=data.keys())
    if values_are == "rows":
        data = data.T
    elif values_are == "columns":
        pass
    else:
        raise ValueError(values_are)
    return data


def flatten(lst: List[List[T]]) -> List[T]:
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
    num_reps = 1 + int(len_ *1./ len(it))
    return (it * num_reps)[:len_]


def get_chunks(
        it: Iterable[T], sz: int, index_by_iloc=False
    ) -> Iterable[Iterable[T]]:
    """Break up `it` into `sz`-size chunks."""
    # this is a generator expression!
    return (
        it.iloc[i:i+sz] if index_by_iloc else \
        it[i:i+sz]
    for i in range(0, len(it), sz))


def fillna(df: Data, value: Any=0, limit_consec: int=0) -> Data:
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
    warn("This hasn't been thoroughly tested!")
    # NaN if df's original value was NaN, else just `value`...
    # seems like the opposite of what we want, but bear with me
    filler = df*0 + value
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
    first = df.apply(
        lambda ser: ser.first_valid_index()
    ).max()
    last = df.apply(
        lambda ser: ser.last_valid_index()
    ).min()
    if first is None:
        return df.iloc[:0]
    return df.loc[first:last]


def get_diag_of_df(df: pd.DataFrame) -> pd.DataFrame:
    diag_df = np.diag(df)
    diag_df = pd.DataFrame(diag_df, index=df.index, columns=df.columns)
    return diag_df


def get_inv_of_df(df: pd.DataFrame) -> pd.DataFrame:
    inv_df = np.linalg.inv(df)
    inv_df = pd.DataFrame(inv_df, index=df.index, columns=df.columns)
    return inv_df


def check_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.Series:
    diff = pd.Series(dtype=float)
    diff.loc["_cols_missing_in_b"] = len(df_a.columns.difference(df_b.columns)) / len(df_a.columns)
    diff.loc["_cols_excess_in_b"] = len(df_b.columns.difference(df_a.columns)) / len(df_a.columns)
    diff.loc["_ix_missing_in_b"] = len(df_a.index.difference(df_b.index)) / len(df_a.index)
    diff.loc["_ix_excess_in_b"] = len(df_b.index.difference(df_a.index)) / len(df_a.index)
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
        like: str, fn: Callable=lambda x: x.upper(),
        not_: bool=False
    ) -> List[str]:
    fn = (lambda x: x) if fn is None else fn
    cond = operator.not_ if not_ else bool
    result = [x for x in it if cond( fn(like) in fn(x) )]
    return result


def filter_like(
        it: Iterable[str],
        like_ands: Iterable[Any]=("",),
        like_ors: Iterable[Any]=("",),
        fn: Callable=lambda x: x.upper(),
        not_: bool=False
    ) -> List[str]:
    for like_and in like_ands:
        # overwrite
        it = _filter_like(it=it, like=like_and, fn=fn, not_=not_)
    # now we've narrowed down
    like_or_results = [
        _filter_like(it=it, like=like_or, fn=fn, not_=not_)
    for like_or in like_ors]
    result = flatten(like_or_results)
    return result


def get_nearest_value(
        df: Data, key: float, axis: str="index"
    ) -> Tuple[float, Any]:
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

def ceil(x: float, increment: float=1.) -> float:
    """E.g. Get the next 10% mark for a chart axis:
    >>>> ceil(x=0. 1337, increment=0.10)
    0.20
    """
    return np.ceil(x / increment) * increment


########################################################################################################################
## DATA REPRESENTATION #################################################################################################
########################################################################################################################

def strfpct(pct: float, dps: int=1) -> str:
    """E.g. 0.1337 -> 13.4%."""
    pct = pct * 100
    pct = round(pct, dps)
    out = "{pct:." + str(dps) + "f}%"
    out = out.format(pct=pct)
    return out


def strfccy(amt: float, ccy: str="$") -> str:
    """Pretty-format a dollar amount."""
    return f"{ccy}{round(amt, 2):.2f}"


def strfdate(date: Datelike="now", granular: bool=False) -> str:
    """Pretty-formate a date."""
    date = dt.datetime.now() if date == "now" else date
    date = pd.to_datetime(date)
    fmt = "%Y-%m-%d"
    fmt = fmt + "-%H-%M-%S" if granular else fmt
    out = date.strftime(fmt)
    return out


########################################################################################################################
## DATA I/O ############################################################################################################
########################################################################################################################

def _validate_dirpath(dirpath: str, raise_: bool=False) -> str:
    if not dirpath.endswith("/"):
        msg = f"Directory path '{dirpath}' doesn't end with '/'!"
        if raise_:
            raise ValueError(msg)
        else:
            warn(msg)
    return dirpath


def _validate_ext(ext: str, raise_: bool=False) -> str:
    if not ext.startswith("."):
        msg = f"Extension '{ext}' doesn't start with '.'!"
        if raise_:
            raise ValueError(msg)
        else:
            warn(msg)
    return ext


def _extract_ext(fpath: str, validate: bool=True) -> str:
    """E.g. 'path/to/data.csv' -> '.csv'."""
    # everything after the last dot
    ext = fpath.split(".")[-1]
    ext = f".{ext}"
    if validate:
        ext = _validate_ext(ext=ext)
    return ext


def _get_qualified_ext(qualifier: str="", ext: str=".txt") -> str:
    """E.g.
    >>> _get_qualified_ext('some_description', '.csv')
    # this is now ready to be prefixed with e.g. a timestamp 't2019-01-26'
    '_some_description.csv'
    """
    ext = _validate_ext(ext=ext)
    ext = f"_{qualifier}{ext}"
    return ext


def _gen_tmp_filename(ext: str=".txt") -> str:
    ext = _validate_ext(ext=ext)
    # timestamp
    t = maybe_date(granular=True, as_str=True)
    # random seq of 16 capital letters
    alphabet = [chr(ascii_code) for ascii_code in range(65, 65 + 26)]
    seq = [random.choice(alphabet) for _ in range (16)]
    seq = "".join(seq)
    # hopefully, the timestamp plus the random sequence
    # are entropic enough to avoid collisions
    fname = f"t{t}_{seq}{ext}"
    return fname


def gen_tmp_filepath(dirpath: str="data/", ext: str=".txt") -> str:
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
        description: str="data",
        dirpath: str="data/", ext: str=".pkl",
        **kwargs
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
        data: T, description: str="data",
        dirpath: str="data/", ext: str=".pkl",
        **kwargs
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

def __maybe_date(date: Optional[Datelike]=None) -> Datelike:
    date = maybe(date, dt.datetime.now())
    return date


def __get_lagged_date(date: Datelike, lags: int=0, freq: str=DEFAULT_DATETIME_FREQ) -> Datelike:
    offset = - lags * pd.tseries.frequencies.to_offset(freq)
    lagged_date = date + offset
    return lagged_date


def _get_lagged_date(date: Optional[Datelike]=None, lags: int=0, freq: str=DEFAULT_DATETIME_FREQ):
    """Like `__get_lagged_date()`, but makes sure `lagged_date` is not after `date`.
    E.g. On Saturday, gives you yesterday (Friday) instead of day-after-tomorrow (Monday).
    """
    date = __maybe_date(date=date)
    lagged_date = __get_lagged_date(date=date, lags=lags, freq=freq)
    if lagged_date > date:
        lagged_date = __get_lagged_date(date=date, lags=lags+1, freq=freq)
    assert not lagged_date > date, (date, lags, freq, lagged_date)
    return lagged_date


def maybe_date(
        date: Datelike=None,
        ow_lags: int=0, freq=DEFAULT_DATETIME_FREQ,
        granular: bool=False, as_str: bool=False
    ) -> Datelike:
    date = maybe(date, _get_lagged_date(lags=ow_lags, freq=freq))
    date = strfdate(date, granular=granular)
    date = date if as_str else pd.to_datetime(date)
    return date


########################################################################################################################
## DATA VISUALIZATION ##################################################################################################
########################################################################################################################

def __iprint_val(val: Any, flush: bool=False) -> str:
    bookend = '\'' if isinstance(val, str) else ''
    out = f"{bookend}{val}{bookend}"
    if flush:
        print(out)
    return out


def _iprint_ser(ser: pd.Series, tab_sz: int=0, flush: bool=False) -> str:
    out = tab_sz*" " + "pd.Series(" + "\n"
    out += (tab_sz+4)*" " + "OrderedDict([" + "\n"
    for (k, v) in ser.items():
        out += (tab_sz+8)*" " + "(" + "\n"
        out += (tab_sz+12)*" " + f"{__iprint_val(k)}," + "\n"
        out += (tab_sz+12)* " " + f"{__iprint_val(v)}" + "\n"
        out += (tab_sz+8)*" " + ")," + "\n"
    out += (tab_sz+4)*" " + "])," + "\n"
    out += tab_sz*" " + f"name={__iprint_val(ser.name)}," + "\n"
    # quoted so i can use pandas's native dtype parser instead of importing names
    out += tab_sz*" " + f"dtype='{ser.dtype}'," + "\n"
    out += tab_sz*" " + ")"
    if flush:
        out = "from collections import OrderedDict" + "\n" + out
        print(out)
    return out


def iprint_df(df: pd.DataFrame, flush: bool=False) -> str:
    """Print a DataFrame to stdout in a format you can copy-paste into a REPL.
    Useful when you have access to the console but not the filesystem.
    """
    out = "pd.DataFrame(OrderedDict([" + "\n"
    for colname, col in df.items():
        out += 4*" " + "(" + "\n"
        out += 8*" " + f"{__iprint_val(colname)}," + "\n"
        out += _iprint_ser(col, tab_sz=8) + "\n"
        out += 4* " " + ")," + "\n"
    out += "]))"
    if flush:
        out = "from collections import OrderedDict" + "\n" + out
        print(out)
    return out


def _describe_series(ser: pd.Series, bool_is_numeric: bool=True) -> pd.Series:
    if bool_is_numeric:
        if ser.dtype == bool:
            ser = ser.astype(int)
    d = ser.describe()
    return d


def describe_df(df: pd.DataFrame, bool_is_numeric: bool=True) -> pd.DataFrame:
    if bool_is_numeric:
        caster = {colname:
            int if dtype == bool else dtype
        for (colname, dtype) in df.items()}
        df = df.astype(caster)
    d = df.describe()
    return d


def plot(
        df: Data, kind: str="line",
        # THEME
        typeface: str=TYPEFACE, font_scale: float=FONT_SCALE,
        mimic_excel: bool=False,
        # SCATTERPLOT OPTIONS
        scatter_labels: bool=False,
        label_fontsize: float=LABEL_FONTSIZE,
        label_rotation: float=LABEL_ROTATION,
        # GRIDLINES
        ## horizontal (y = ...)
        axhline_locs: Iterable[float]=(0,),
        axhline_zs: Iterable[float]=(0,),
        axhline_styles: Iterable[str]=("-",),
        axhline_colors: Iterable[str]=("gray",),
        ## vertical (x = ...)
        axvline_locs: Iterable[float]=tuple(),
        axvline_zs: Iterable[float]=(0,),
        axvline_styles: Iterable[str]=("-",),
        axvline_colors: Iterable[str]=("gray",),
        # AXIS LIMITS
        ## x
        xlim: Optional[Tuple[float, float]]=None,
        xlim_left: Optional[float]=None, xlim_right: Optional[float]=None,
        ## y
        ylim: Optional[Tuple[float, float]]=None,
        ylim_bottom: Optional[float]=None, ylim_top: Optional[float]=None,
        # AXIS DIRECTION
        invert_xaxis: bool=False,
        invert_yaxis: bool=False,
        # AXIS TICK INTERVALS
        ## x (major, minor)
        xtick_intervals: Optional[Tuple[float, float]]=None,
        xtick_major_interval: Optional[float]=None,
        # can set this to `None` to carry over value from `df.plot()`
        xtick_minor_interval: Optional[float]=None,
        ## y (major, minor)
        ytick_intervals: Optional[Tuple[float, float]]=None,
        ytick_major_interval: Optional[float]=None,
        ytick_minor_interval: Optional[float]=None,
        # AXIS TICK LABELS
        pct_dps: int=1, granular_dates: bool=False,
        ## x
        xticklabels: bool=True,
        xpct: float=False, xdollar: bool=False, xdates: bool=False,
        ## y
        yticklabels: bool=True,
        ypct: float=False, ydollar: bool=False, ydates: bool=False,
        # AXIS TITLES
        xlabel: Optional[str]=None, ylabel: Optional[str]=None,
        # LEGEND
        legend: bool=False,
        legend_title: Optional[str]=None,
        legend_title_fontsize: Optional[str]=None,  # e.g. 'xx-large'
        legend_loc: Union[Tuple[float, float], str]="best",
        # (SUB)PLOT TITLE
        title: Optional[str]=None, titley: float=1.0,
        # SIDE EFFECTS
        ax: Optional[PlotAxes]=None,
        figsize: Tuple[float, float]=FIGSIZE,  # width (x), height (y)
        plt_show: Optional[bool]=None,
        savefig_path: Optional[str]=None,
        plt_close: Optional[bool]=None,
        **kwargs
    ) -> PlotAxes:

    # helpers
    def _raise_before_override(description: str, val: Any=None) -> None:
        if val is not None:
            msg = f"plot: {description} {val} will be overriden!"
            raise ValueError(msg)
        return val
    ## axis tick formatters (2nd arg, tick_position, is unused)
    def _strfpct(tick_val, _):
        return strfpct(tick_val, dps=pct_dps)
    def _strfdollar(tick_val, _):
        return strfccy(tick_val)
    def _strfdate(tick_val, _):
        return strfdate(tick_val, granular=granular_dates)

    # convenience handling of plot kind
    if kind == "histpct":
        kind = "hist"
        kwargs["weights"] = np.ones(len(df.index)) *1./ len(df.index)
        ypct = True
    if kind == "scatter":
        if "x" not in kwargs:
            kwargs["x"] = df.columns[0]
        if "y" not in kwargs:
            kwargs["y"] = df.columns[-1]
        if "c" not in kwargs:
            # warn("plot: default scatter-point color is c='black'!")
            kwargs["c"] = "black"
        if xlabel is None:
            xlabel = kwargs["x"]
        if ylabel is None:
            ylabel = kwargs["y"]
    # implicit default side effects
    ## user provided `ax` OR wants to savefig -> don't show plot;
    ## user didn't provide `ax` AND doesn't want to savefig -> show plot.
    plt_show = maybe(
        v=plt_show,
        ow = (ax is None) and (savefig_path is None)
    )
    ## user doesn't want to savefig OR wants to show -> leave open;
    ## user wants to savefig AND doesn't want to show -> close.
    plt_close = maybe(
        v=plt_close,
        ow = (savefig_path is not None) and (not plt_show)
    )

    sns.set(font=typeface, font_scale=font_scale)
    ax = df.plot(kind=kind, ax=ax, figsize=figsize, **kwargs)

    # SCATTERPLOT OPTIONS
    if scatter_labels:
        for i in df.index:
            ax.text(
                x=df.loc[i, kwargs["x"]], y=df.loc[i, kwargs["y"]], s=i,
                family=typeface, fontsize=label_fontsize,
                rotation=label_rotation
            )
            del i

    # GRIDLINES
    if mimic_excel:
        ax.set_axis_bgcolor("white")
        ax.yaxis.grid(
            True, which="major", color="black", linestyle="-",
            linewidth=0.25, alpha=0.50, zorder=0
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
            axline_locs[n], zorder=axline_zs[n],
            linestyle=axline_styles[n], linewidth=None,
            color=axline_colors[n], alpha=1
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
            axline_locs[n], zorder=axline_zs[n],
            linestyle=axline_styles[n], linewidth=None,
            color=axline_colors[n], alpha=1
        )
    del axline_colors, axline_styles, axline_zs, num_axlines, axline_locs

    # AXIS LIMITS
    ## x
    if xlim is not None:
        _raise_before_override(
            description="individually-specified `xlim_left`",
            val=xlim_left
        )
        _raise_before_override(
            description="individually-specified `xlim_right`",
            val=xlim_right
        )
        xlim_left, xlim_right = xlim
    # sensible default
    if kind == "line" and xlim_left is None:
        xlim_left = df.index[0]
    if kind == "line" and xlim_right is None:
        xlim_right = df.index[-1]
    # set
    if xlim_left is not None:
        ax.set_xlim(left=xlim_left)
    if xlim_right is not None:
        ax.set_xlim(right=xlim_right)
    ## y
    if ylim is not None:
        _raise_before_override(
            description="individually-specified `ylim_bottom`",
            val=ylim_bottom
        )
        _raise_before_override(
            description="individually-specified `ylim_top`",
            val=ylim_top
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
            val=xtick_major_interval
        )
        _raise_before_override(
            description="individually-specified `xtick_minor_interval`",
            val=xtick_minor_interval
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
            val=ytick_major_interval
        )
        _raise_before_override(
            description="individually-specified `ytick_minor_interval`",
            val=ytick_minor_interval
        )
        ytick_major_interval, ytick_minor_interval = ytick_intervals
    if ytick_major_interval is not None:
        ax.yaxis.set_major_locator(plt.MultipleLocator(ytick_major_interval))
    if ytick_minor_interval is not None:
        ax.yaxis.set_minor_locator(plt.MultipleLocator(ytick_minor_interval))

    # AXIS TICK LABELS
    ## x
    if sum([xpct, xdollar, xdates]) > 1:
        msg = f"plot: conflict in {xpct}, {xdollar}, {xdates}!"
        raise ValueError(msg)
    fmt = None
    if xpct:
        fmt = _strfpct
    elif xdollar:
        fmt = _strfdollar
    elif xdates:
        fmt = _strfdate
    if fmt is not None:
        ax.xaxis.set_major_formatter(plt_ticker.FuncFormatter(fmt))
    del fmt
    if not xticklabels:
        plt.setp(ax.get_xticklabels(), visible=False)
    ## y
    if sum([ypct, ydollar, ydates]) > 1:
        msg = f"plot: conflict in {ypct}, {ydollar}, {ydates}!"
    fmt = None
    if ypct:
        fmt = _strfpct
    elif ydollar:
        fmt = _strfdollar
    elif ydates:
        fmt = _strfdate
    if fmt is not None:
        ax.yaxis.set_major_formatter(plt_ticker.FuncFormatter(fmt))
    del fmt
    if not yticklabels:
        plt.setp(ax.get_xticklabels(), visible=False)
    # AXIS TITLES
    ## x
    xlabel = maybe(xlabel, False)
    if xlabel:
        ax.set_xlabel(xlabel)
        ax.get_xaxis().get_label().set_visible(True)
    else:
        ax.get_xaxis().get_label().set_visible(False)
    ## y
    ylabel = maybe(ylabel, False)
    if ylabel:
        ax.set_ylabel(ylabel)
        ax.get_yaxis().get_label().set_visible(True)
    else:
        ax.get_yaxis().get_label().set_visible(False)

    # LEGEND
    if legend:
        ax.legend(
            title=legend_title, fontsize=legend_title_fontsize,
            loc=legend_loc
        )

    # (SUB)PLOT TITLE
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
        lim: float=1, style: str="--", color: str="black", **kwargs
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
        **kwargs
    )


def plot_corr_heatmap(
        corr_matrix: pd.DataFrame,
        cmap: str="RdYlGn",
        title: Optional[str]=None,
        figsize: Tuple[float, float]=(8, 8),  # width (x), height (y)
        **kwargs
    ) -> None:
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True, fmt=".2f",
        vmin=-1, vmax=1, center=0, cmap=cmap, cbar=False,
        square=True, **kwargs
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
        columns = lambda c: f"{c} ({df[c].isna().mean()*100:.2f}% na)",
    )
    if col_names ==  ...:
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
