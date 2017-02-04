"""
Functions for summary statistics.

(c) Wu Sun <wu.sun@ucla.edu> 25 Jan 2017

"""
import numpy as np


def resist_mean(x, IQR_range=1.5):
    """
    Calculate outlier-resistant mean of the sample using Tukey's outlier test.

    Caveat: Does support calculation along an axis, unlike `numpy.mean()`.

    Parameters
    ----------
    x : array_like
        The sample.
    IQR_range : float, optional
        Parameter to control the inlier range defined by
            [ Q_1 - IQR_range * (Q_3 - Q_1), Q_3 - IQR_range * (Q_3 - Q_1) ]
        By default the parameter is 1.5, the original value used by John Tukey.

    Returns
    -------
    x_rmean : float
        The resistant mean of the sample with outliers removed.

    References
    ----------
    .. [T77] John W. Tukey (1977). Exploratory Data Analysis. Addison-Wesley.

    """
    x = np.array(x)
    if np.sum(np.isfinite(x)) <= 1:
        return np.nanmean(x)
    else:
        x_q1, x_q3 = np.nanpercentile(x, [25, 75])
        x_iqr = x_q3 - x_q1
        x_uplim = x_q3 + IQR_range * x_iqr
        x_lolim = x_q1 - IQR_range * x_iqr
        x_rmean = np.nanmean(x[(x >= x_lolim) & (x <= x_uplim)])
        return x_rmean


def resist_std(x, IQR_range=1.5):
    """
    Calculate outlier-resistant standard deviation of the sample using
    Tukey's outlier test.

    Caveat: Does support calculation along an axis, unlike `numpy.std()`.

    Parameters
    ----------
    x : array_like
        The sample.
    IQR_range : float, optional
        Parameter to control the inlier range defined by
            [ Q_1 - IQR_range * (Q_3 - Q_1), Q_3 - IQR_range * (Q_3 - Q_1) ]
        By default the parameter is 1.5, the original value used by John Tukey.

    Returns
    -------
    x_rstd : float
        The resistant standard deviation of the sample with outliers removed.
        Degree of freedom = 1 is enforced for the sample standard deviation.

    References
    ----------
    .. [T77] John W. Tukey (1977). Exploratory Data Analysis. Addison-Wesley.

    """
    x = np.array(x)
    if np.sum(np.isfinite(x)) <= 1:
        return(np.nanstd(x, ddof=1))
    else:
        x_q1, x_q3 = np.nanpercentile(x, [25, 75])
        x_iqr = x_q3 - x_q1
        x_uplim = x_q3 + IQR_range * x_iqr
        x_lolim = x_q1 - IQR_range * x_iqr
        x_rstd = np.nanstd(x[(x >= x_lolim) & (x <= x_uplim)], ddof=1)
        return x_rstd


def IQR_func(x, axis=None):
    """
    Calculate the interquartile range of an array.

    Parameters
    ----------
    x : array_like
        The sample.
    axis : int, optional
        Axis along which the percentiles are computed. Default is to ignore
        and compute the flattened array.
        (Same as the `axis` argument in `numpy.nanpercentile()`.)

    Returns
    -------
    IQR : float or array_like
        The interquartile range of an array.

    """
    if np.sum(np.isfinite(x)) > 0:
        q1, q3 = np.nanpercentile(x, [25., 75.], axis=axis)
        return q3 - q1
    else:
        return np.nan
