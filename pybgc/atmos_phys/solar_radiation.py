import numpy as np
import datetime
from collections import namedtuple
from .. import constants


def solar_angle(dt, lat, lon, timezone=0.):
    """
    Calculate solar angle and sunrise/sunset times.

    Original program by `NOAA Global Radiation Group
    <http://www.esrl.noaa.gov/gmd/grad/solcalc/calcdetails.html>`_.

    Translated to Python by Wu Sun <wu.sun "at" ucla.edu> on 23 Nov 2015.

    Parameters
    ----------
    dt : datetime.datetime
        Time variable packaged in the built-in datetime format.
    lat : float
        Latitude (-90 to 90).
    lon : float
        Longitude (-180 to 180).
    timezone : float, optional
        Time zone with respect to UTC (-12 to 12), by default set to 0.

    Returns
    -------
    solar_angle_result : namedtuple
        Unpack the namedtuple to get results.

    Raises
    ------
    TypeError
        If `dt` is not a `datetime.datetime` instance.

    """
    if not isinstance(dt, datetime.datetime):
        raise TypeError('Input `dt` is not a `datetime.datetime` instance.')

    # current eccentricity of earth orbit
    ecc_earth = constants.ecc_earth

    # use NASA truncated Julian Date method to calculate Julian Date number
    # JD = 2440000.5 at 00:00 on May 24, 1968
    time_delta = dt - datetime.datetime(1968, 5, 24, 0, 0, 0)
    julian_date = 2440000.5 + time_delta.days + \
        time_delta.seconds / 86400. - timezone / 24.
    julian_century = (julian_date - 2451545.) / 36525.

    # geometric mean longitude of the sun, in degree
    geom_mean_lon_sun = np.mod(
        280.46646 + julian_century *
        (36000.76983 + julian_century * 0.0003032), 360)
    # geometric mean anomaly of the sun, in degree
    geom_mean_anom_sun = 357.52911 + julian_century * \
        (35999.05029 - 0.0001537 * julian_century)
    # solar equator of center
    sun_eq_ctr = np.sin(np.radians(geom_mean_anom_sun)) * \
        (1.914602 - julian_century * (0.004817 + 1.4e-5 * julian_century)) + \
        np.sin(np.radians(geom_mean_anom_sun) * 2) * \
        (0.019993 - 0.000101 * julian_century) + \
        np.sin(np.radians(geom_mean_anom_sun) * 3) * 0.000289
    # solar true longitude
    sun_true_lon = geom_mean_lon_sun + sun_eq_ctr
    # solar true anomaly
    sun_true_anom = geom_mean_anom_sun + sun_eq_ctr
    # solar rad vector in AUs
    sun_rad_vector = (1.000001018 * (1. - ecc_earth**2)) / \
        (1. + ecc_earth * np.cos(np.radians(sun_true_anom)))
    # solar apparent longitude
    sun_app_lon = sun_true_lon - 0.00569 - \
        0.00478 * np.sin(np.radians(125.04 - 1934.136 * julian_century))

    # in degree
    mean_obliq_ecliptic = 23. + (26. + (21.448 - julian_century * (
        46.815 + julian_century * (0.00059 - julian_century * 0.001813))
    ) / 60.) / 60.

    obliq_corr = mean_obliq_ecliptic + 0.00256 * \
        np.cos(np.radians(125.04 - 1934.136 * julian_century))
    sun_rt_ascen = np.degrees(np.arctan2(
        np.cos(np.radians(obliq_corr)) * np.sin(np.radians(sun_app_lon)),
        np.cos(np.radians(sun_app_lon))))
    sun_declin = np.degrees(np.arcsin(
        np.sin(np.radians(obliq_corr)) * np.sin(np.radians(sun_app_lon))))

    var_y = np.tan(np.radians(obliq_corr / 2.)) ** 2
    eq_time = 4. * np.degrees(
        var_y * np.sin(2. * np.radians(geom_mean_lon_sun)) -
        2. * ecc_earth * np.sin(np.radians(geom_mean_anom_sun)) +
        4. * ecc_earth * var_y * np.sin(np.radians(geom_mean_anom_sun)) *
        np.cos(2. * np.radians(geom_mean_lon_sun)) -
        0.5 * var_y**2 * np.sin(4. * np.radians(geom_mean_lon_sun)) -
        1.25 * ecc_earth**2 * np.sin(2. * np.radians(geom_mean_anom_sun)))
    # in minutes

    HA_sunrise = np.degrees(np.arccos(
        np.cos(np.radians(90.833)) /
        (np.cos(np.radians(lat)) * np.cos(np.radians(sun_declin))) -
        np.tan(np.radians(lat)) * np.tan(np.radians(sun_declin))))

    solar_noon_local = (720. - 4. * lon - eq_time +
                        timezone * 60.) / 1440.  # in day
    sunrise_local = solar_noon_local - HA_sunrise * 4. / 1440.  # in day
    sunset_local = solar_noon_local + HA_sunrise * 4. / 1440.  # in day
    sunlight_duration = 8. * HA_sunrise / 1440.  # in day
    sunlight_duration_min = 8. * HA_sunrise  # in minutes

    fractional_day = dt.hour / 24. + dt.minute / 1440. + \
        (dt.second + dt.microsecond * 1e-6) / 86400.
    # true solar time in minutes
    true_solar_time_min = np.mod(
        fractional_day * 1440. + eq_time + 4. * lon - 60. * timezone, 1440)

    # calculate hour angle
    if(true_solar_time_min / 4. < 0.):
        hour_angle = true_solar_time_min / 4. + 180.
    else:
        hour_angle = true_solar_time_min / 4. - 180.

    solar_zenith_angle = np.degrees(np.arccos(
        np.sin(np.radians(lat)) * np.sin(np.radians(sun_declin)) +
        np.cos(np.radians(lat)) * np.cos(np.radians(sun_declin)) *
        np.cos(np.radians(hour_angle))))

    solar_elev_angle = 90. - solar_zenith_angle

    # calculate solar azimuth angle
    if hour_angle > 0.:
        solar_azimuth_angle = np.mod(np.degrees(np.arccos(
            ((np.sin(np.radians(lat)) *
                np.cos(np.radians(solar_zenith_angle))) -
                np.sin(np.radians(sun_declin))) /
            (np.cos(np.radians(lat)) *
                np.sin(np.radians(solar_zenith_angle))))) + 180, 360)
    else:
        solar_azimuth_angle = np.mod(540. - np.degrees(np.arccos(
            ((np.sin(np.radians(lat)) *
                np.cos(np.radians(solar_zenith_angle))) -
                np.sin(np.radians(sun_declin))) /
            (np.cos(np.radians(lat)) *
                np.sin(np.radians(solar_zenith_angle))))), 360)

    if solar_elev_angle > 85.:
        approx_atmos_refrac = 0.
    elif 5. < solar_elev_angle <= 85.:
        approx_atmos_refrac = \
            (58.1 / np.tan(np.radians(solar_elev_angle)) -
                0.07 / np.tan(np.radians(solar_elev_angle)) ** 3 +
                0.000086 / np.tan(np.radians(solar_elev_angle)) ** 5) / 3600.
    elif -0.575 < solar_elev_angle <= 5.:
        approx_atmos_refrac = \
            (1735. - 518.2 * solar_elev_angle +
                103.4 * solar_elev_angle ** 2 -
                12.79 * solar_elev_angle ** 3 +
                0.711 * solar_elev_angle ** 4) / 3600.
    elif solar_elev_angle <= -0.575:
        approx_atmos_refrac = \
            -20.774 / 3600. / np.tan(np.radians(solar_elev_angle))

    solar_elev_angle_corr = solar_elev_angle + approx_atmos_refrac

    SolarAngleResult = namedtuple(
        'SolarAngleResult', ('solar_noon', 'sunrise', 'sunset', 'hour_angle',
                             'solar_zenith_angle', 'solar_elevation_angle',
                             'solar_azimuth_angle', 'atmospheric_refraction',
                             'solar_elevation_angle_corrected'))
    solar_angle_result = SolarAngleResult(
        solar_noon_local, sunrise_local, sunset_local, hour_angle,
        solar_zenith_angle, solar_elev_angle, solar_azimuth_angle,
        approx_atmos_refrac, solar_elev_angle_corr)

    # solar_angle_result = dict([
    #     ('solar noon', solar_noon_local),
    #     ('sunrise', sunrise_local),
    #     ('sunset', sunset_local),
    #     ('hour angle', hour_angle),
    #     ('solar zenith angle', solar_zenith_angle),
    #     ('solar elevation angle', solar_elev_angle),
    #     ('solar azimuth angle', solar_azimuth_angle),
    #     ('atmospheric refraction', approx_atmos_refrac),
    #     ('solar elevation angle corrected', solar_elev_angle_corr), ])

    return(solar_angle_result)


def blackbody_radiation():
    """TODO: blackbody radiation as a function of wavelength or wavenumber."""
    pass
