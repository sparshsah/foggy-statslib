# some preprocessing
import pandas as pd
from foggy_pylib.stats.est.core import get_qmean, get_gmean, get_hmean
# fill them in if necessary
if not hasattr(pd.Series, "qmean"):
    pd.Series.qmean = get_qmean
if not hasattr(pd.Series, "gmean"):
    pd.Series.gmean = get_gmean
if not hasattr(pd.Series, "hmean"):
    pd.Series.hmean = get_hmean
# clean up
del get_hmean, get_gmean, get_qmean, pd
