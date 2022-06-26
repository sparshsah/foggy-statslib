"""Core statistical estimation.

v1.1 beta: API probably won't dramatically change, but
    implementations have not yet been thoroughly tested.

author: [@sparshsah](https://github.com/sparshsah)


# Notes
* Currently, each estimator adheres to a frequentist paradigm,
    using the realized (observed) sample stat directly as its point estimate of the parameter.
    But in the future, we could implement a more Bayesian approach,
    using the data to instead inform our posterior distribution for the parameter.
"""

from typing import Optional
# pylint: disable=unused-import
from numpy import mean as get_amean
# pylint: disable=unused-import
from scipy.stats import gmean as get_gmean, hmean as get_hmean
from foggy_statslib.core import FloatSeries

DEFAULT_AVG_KIND: str = "arith_mean"
DEFAULT_DE_AVG_KIND: Optional[str] = DEFAULT_AVG_KIND


def get_qmean(a: FloatSeries) -> float:
    square = a**2
    mean_square = square.mean()
    root_mean_square = mean_square**0.5
    return root_mean_square
