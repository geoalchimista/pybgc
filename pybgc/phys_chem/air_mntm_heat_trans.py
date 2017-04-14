"""
Calculate heat and momentum transfer coefficients of air, including
kinematic viscosity, dynamic viscosity, thermal diffusivity, and
thermal conductivity.

by Wu Sun <wu.sun@ucla.edu>, 01 Sep 2015

References
----------
.. [M99] Massman, W. J. (1999). Molecular diffusivities of Hg vapor in air,
   O2 and N2 near STP and the kinematic viscosity and thermal diffusivity
   of air near STP. *Atmos. Environ.* 33, 453-457.

"""
import numpy as np
from .. import constants


def kin_visc_air(temp, pressure=constants.p_std, kelvin=False):
    """
    Calculate the kinematic viscosity of air.

    Parameters
    ----------
    temp : float or array_like
        Temperature, by default in Celsius unless `kelvin` is set `True`.
    pressure : float or array_like, optional
        Pressure in Pascal.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.

    Returns
    -------
    nu_air : float or array_like
        Kinematic viscosity of air (m^2 s^-1).

    """
    p_std = constants.p_std
    T_0 = constants.T_0
    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin

    nu_0 = 1.327e-5  # at STP, m^2 s^-1
    nu_air = nu_0 * p_std / pressure * (T_k / T_0) ** 1.81

    return nu_air


def dyn_visc_air(temp, kelvin=False):
    """
    Calculate the dynamic viscosity of air.

    Parameters
    ----------
    temp : float or array_like
        Temperature, by default in Celsius unless `kelvin` is set `True`.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.

    Returns
    -------
    eta_air : float or array_like
        Dynamic viscosity of air (kg m^-1 s^-1, or equivalently, Pa s)

    """
    T_0 = constants.T_0
    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin

    eta_0 = 1.714e-5  # at STP, kg m^-1 s^-1
    eta_air = eta_0 * (T_k / T_0) ** 0.81

    return eta_air


def therm_diff_air(temp, pressure=constants.p_std, kelvin=False):
    """
    Calculate the thermal diffusivity of air.

    Parameters
    ----------
    temp : float or array_like
        Temperature, by default in Celsius unless `kelvin` is set `True`.
    pressure : float or array_like, optional
        Pressure in Pascal.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.

    Returns
    -------
    kappa_air : float or array_like
        Thermal diffusivity of air (m^2 s^-1).
    """
    p_std = constants.p_std
    T_0 = constants.T_0
    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin

    kappa_0 = 1.869e-5  # at STP, m^2 s^-1
    kappa_air = kappa_0 * p_std / pressure * (T_k / T_0) ** 1.81

    return kappa_air


def therm_cond_air(temp, kelvin=False):
    """
    Calculate the thermal conductivity of air.

    Parameters
    ----------
    temp : float or array_like
        Temperature, by default in Celsius unless `kelvin` is set `True`.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.

    Returns
    -------
    lambda_air : float or array_like
        Thermal conductivity of air (W m^-1 K^-1).

    """
    T_0 = constants.T_0
    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin

    lambda_0 = 2.432e-5  # at STP, W m^-1 K^-1
    lambda_air = lambda_0 * (T_k / T_0) ** 0.81

    return lambda_air
