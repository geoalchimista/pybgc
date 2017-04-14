"""Regression functions."""
from collections import namedtuple
import numpy as np
from scipy import stats


def linregress_no_intercept(x, y):
    """
    Do a linear regression without intercept for two sets of measurements.

    by Wu Sun <wu.sun@ucla.edu>, 4 Nov 2016

    Parameters
    ----------
    x, y : array_like, one-dimensional
        Two sets of measurements. Both arrays should have the same length.

    Returns
    -------
    slope : float
        Slope of the regression line.
    intercept : float
        Intercept of the regression line, forced to be zero.
    r_value : float
        Pearson correlation coefficient.
    p_value : float
        Two-sided p-value for a hypothesis test whose null hypothesis is
        that the slope is zero.
    stderr_slope : float
        Standard error of the estimated slope.

    """
    x = np.asarray(x)
    y = np.asarray(y)
    # making copies of only finite values
    x_copy = np.copy(x[np.isfinite(x + y)])
    y_copy = np.copy(y[np.isfinite(x + y)])

    n = len(x_copy)
    df = n - 2  # degree of freedom

    slope = np.sum(y_copy * x_copy) / np.sum(x_copy ** 2.)
    intercept = 0.  # intercept is forced to be zero by definition
    y_pred = slope * x_copy
    r_value, _ = stats.pearsonr(y_copy, y_pred)
    stderr_slope = np.sqrt(np.sum((y_pred - y_copy) ** 2.) /
                           (df * np.sum((x_copy - np.mean(x_copy))**2)))
    p_value = 2 * stats.distributions.t.sf(np.abs(slope / stderr_slope), df)

    LinregressNoInterceptResult = namedtuple('LinregressNoInterceptResult',
                                             ('slope', 'intercept', 'r_value',
                                              'p_value', 'stderr_slope'))
    return LinregressNoInterceptResult(slope, intercept, r_value,
                                       p_value, stderr_slope)
