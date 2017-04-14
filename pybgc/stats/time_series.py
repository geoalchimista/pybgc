"""
Time series functions.

by Wu Sun <wu.sun@ucla.edu>, 4 Nov 2016

"""
from collections import namedtuple
import numpy as np


def running_std(series, window_size):
    """
    Running standard deviation function.

    Parameters
    ----------
    series : array_like
        Input 1D time series.
    window_size : int
        Size of the moving window.

    Returns
    -------
    rnstd_series : array_like
        Time series of running standard deviation, same length to the input.

    Raises
    ------
    ValueError
        If the `window_size` is larger than the size of the input time series.

    See Also
    --------
    `running_zscore` : Running Z-score function.

    """
    if len(series) < window_size:
        raise ValueError('Window size larger than data size. ')

    left_size = (int(window_size) - 1) / 2
    right_size = window_size - 1 - left_size
    rnstd_series = np.zeros(len(series))
    for i in range(series.size):
        if i <= left_size:
            extracted_window = np.concatenate(([series[0]] * (left_size - i),
                                               series[0: right_size + i + 1]))
        elif i + right_size + 1 > len(series):
            extracted_window = np.concatenate(
                (series[i - left_size:],
                 [series[-1]] * (right_size + i + 1 - len(series))))
        else:
            extracted_window = series[i - left_size: i + right_size + 1]
        rnstd_series[i] = np.nanstd(extracted_window, ddof=1)
    return rnstd_series


def running_zscore(series, window_size, modified_zscore=False):
    """
    Running Z-score function.

    Parameters
    ----------
    series : array_like
        Input 1D time series.
    window : int
        Size of the moving window.
    modified_zscore : bool, optional
        If `True`, use the Iglewicz-Hoaglin modified Z-score [IH93]_
        instead of the original Z-score.

    Returns
    -------
    zscore_series : array_like
        The calculated moving Z-score of the series.

    Raises
    ------
    ValueError
        If the `window_size` is larger than the size of the input time series.

    See Also
    --------
    `running_std` : Running standard deviation function.

    References
    ----------
    .. [IH93] Iglewicz, Boris and Hoaglin, David C. (1993). How to Detect and
        Handle Outliers. ASQC Quality Press, Milwaukee, WI, 1993.

    """
    def _mad(x):
        """Median absolute deviation. """
        return np.nanmedian(np.abs(x - np.nanmedian(x)))

    if len(series) < window_size:
        raise ValueError('Window size larger than data size. ')

    left_size = (int(window_size) - 1) / 2
    right_size = window_size - 1 - left_size
    zscore_series = np.zeros(len(series))
    for i in range(series.size):
        if i <= left_size:
            extracted_window = np.concatenate(([series[0]] * (left_size - i),
                                               series[0: right_size + i + 1]))
        elif i + right_size + 1 > len(series):
            extracted_window = np.concatenate(
                (series[i - left_size:],
                 [series[-1]] * (right_size + i + 1 - len(series))))
        else:
            extracted_window = series[i - left_size: i + right_size + 1]

        if modified_zscore:
            zscore_series[i] = (series[i] - np.nanmean(extracted_window)) / \
                np.nanstd(extracted_window, ddof=1)
        else:
            zscore_series[i] = 0.6745 * \
                (series[i] - np.nanmedian(extracted_window)) / \
                _mad(extracted_window)

    return zscore_series


def hourly_median(hours, obs, full_hour_levels=True):
    """
    Hourly median function.

    Parameters
    ----------
    hours : array_like
        Input array of hour numbers. Must be of the same length as `obs`.
    obs : array_like
        Input array of observation values.
    full_hour_levels : bool, optional
        Default is True to include all 24 hours. If not, only consider the
        hours that are in the `hours` array.

    Returns
    -------
    hour_level : array_like
        Unique hour levels.
    median : array_like
        Median values by hour, same in length as `hour_level`.
    q1 : array_like
        First quartile values by hour, same in length as `hour_level`.
    q3 : array_like
        Third quartile values by hour, same in length as `hour_level`.

    See Also
    --------
    `hourly_avg` : Hourly average function.

    """
    hours = np.round(np.array(hours))  # force to be numpy array
    obs = np.array(obs)  # force to be numpy array
    if full_hour_levels:
        unique_hour_levels = np.arange(24)
    else:
        unique_hour_levels = np.unique(hours)
    obs_hrmed = np.zeros(unique_hour_levels.size) * np.nan
    obs_q1 = np.zeros(unique_hour_levels.size) * np.nan
    obs_q3 = np.zeros(unique_hour_levels.size) * np.nan
    for loop_num in range(unique_hour_levels.size):
        obs_hrmed[loop_num] = np.nanmedian(
            obs[hours == unique_hour_levels[loop_num]])
        obs_q1[loop_num] = np.nanpercentile(
            obs[hours == unique_hour_levels[loop_num]], 25)
        obs_q3[loop_num] = np.nanpercentile(
            obs[hours == unique_hour_levels[loop_num]], 75)

    HourlyMedianResult = namedtuple('HourlyMedianResult',
                                    ('hour_level', 'median', 'q1', 'q3'))
    return HourlyMedianResult(unique_hour_levels, obs_hrmed, obs_q1, obs_q3)


def hourly_avg(hours, obs, full_hour_levels=True, std_ddof=1):
    """
    Hourly average function.

    Parameters
    ----------
    hours : array_like
        Input array of hour numbers. Must be of the same length as `obs`.
    obs : array_like
        Input array of observation values.
    full_hour_levels : bool, optional
        Default is True to include all 24 hours. If not, only consider the
        hours that are in the `hours` array.

    Returns
    -------
    hour_level : array_like
        Unique hour levels.
    avg : array_like
        Average values by hour, same in length as `hour_level`.
    std : array_like
        Standard deviation values by hour, same in length as `hour_level`.

    See Also
    --------
    `hourly_median` : Hourly median function.

    """
    hours = np.round(np.array(hours))  # force to be numpy array
    obs = np.array(obs)  # force to be numpy array
    if full_hour_levels:
        unique_hour_levels = np.arange(24)
    else:
        unique_hour_levels = np.unique(hours)
    obs_avg = np.zeros(unique_hour_levels.size) * np.nan
    obs_std = np.zeros(unique_hour_levels.size) * np.nan
    for loop_num in range(unique_hour_levels.size):
        obs_avg[loop_num] = np.nanmean(
            obs[hours == unique_hour_levels[loop_num]])
        obs_std[loop_num] = np.nanstd(
            obs[hours == unique_hour_levels[loop_num]], ddof=std_ddof)
    HourlyAverageResult = namedtuple(
        'HourlyAverageResult', ('hour_level', 'avg', 'std'))
    return HourlyAverageResult(unique_hour_levels, obs_avg, obs_std)
