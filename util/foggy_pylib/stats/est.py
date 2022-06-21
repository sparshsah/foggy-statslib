"""Statistical estimation.

v1.0 beta: API probably won't dramatically change, but
    implementations have not yet been thoroughly tested.

author: [@sparshsah](https://github.com/sparshsah)


# Notes
* Currently, each estimator adheres to a frequentist paradigm,
    using the realized (observed) sample stat directly as its point estimate of the parameter.
    But in the future, we could implement a more Bayesian approach,
    using the data to instead inform our posterior distribution for the parameter.
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


########################################################################################################################
## DATA DESCRIPTION ####################################################################################################
########################################################################################################################

def _get_metadata(ser: FloatSeries) -> pd.Series:
    metadata = [
        (
            "Frac valid obs",
            ser.notna().sum() / len(ser.index)
        ), (
            "Total valid obs",
            ser.notna().sum()
        ), (
            "Total obs",
            len(ser.index)
        ), (
            "First obs",
            ser.index[0]
        ), (
            "First valid obs",
            ser.first_valid_index()
        ), (
            "Last valid obs",
            ser.last_valid_index()
        ), (
            "Last obs", ser.index[-1]
        )
    ]
    metadata = fc.get_series(metadata)
    return metadata
