import numpy as np
from .. import constants


def p_sat_h2o(temp, ice=False, kelvin=False, method='gg'):
    """
    Calculate saturation vapor pressure over water or ice at a temperature.

    by Wu Sun <wu.sun "at" ucla.edu>, 14 Sep 2014

    Parameters
    ----------
    temp : float or array_like
        Temperature, in Celsius degree by default.
    ice : bool, optional
        Calculate saturation vapor pressure on ice if enabled.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.
    method : str, optional
        Method used to evaluate saturation vapor pressure.
        'gg': default, Goff-Gratch equation (1946). [GG46]_
        'buck': Buck Research Instruments L.L.C. (1996). [B96]_
        'cimo': CIMO Guide (2008). [WMO]_

    Returns
    -------
    e_sat : float or array_like
        Saturation vapor pressure in Pascal.

    Raises
    ------
    ValueError
        If keyword 'ice' is enabled but temperature is above 0 C or 273.15 K.

    References
    ----------
    .. [GG46] Goff, J. A., and Gratch, S. (1946). Low-­pressure properties of
       water from -160 to 212 F, in Transactions of the American Society of
       Heating and Ventilating Engineers, pp 95-­122, presented at the 52nd
       Annual Meeting of the American Society of Heating and Ventilating
       Engineers, New York.
    .. [B96] Buck Research Instruments L.L.C. (1996). *Buck Research CR-1A
       User's Manual*, Appendix 1.
    .. [WMO] World Meteorological Organization. (2008). *Guide to
       Meteorological Instruments and Methods of Observation*, Appendix 4B,
       WMO-No. 8 (CIMO Guide), Geneva.

    Examples
    --------
    >>> print(p_sat_h2o(25))
    3165.19563338

    >>> print(p_sat_h2o([0, 5, 15, 25]))
    [  610.33609993   871.31372986  1703.28100711  3165.19563338]

    >>> print(p_sat_h2o(25, method='buck'))
    3168.53141228

    >>> print(p_sat_h2o(273.15, kelvin=True))
    610.336099933

    >>> print(p_sat_h2o(-15, ice=True))
    165.014773924

    >>> print(p_sat_h2o(258.15, kelvin=True, ice=True, method='cimo'))
    165.287132017

    """
    T_k = np.array(temp, dtype='d') + (not kelvin) * constants.T_0
    # force temperature to be in Kelvin

    if (np.sum(T_k > 273.16) and ice):
        # The triple point of water is 273.16 K
        raise ValueError('Temperature error, no ice exists.')

    if not ice:
        if method == 'buck':
            T_c = T_k - 273.15  # temperature in Celsius degree
            e_sat = 6.1121 * np.exp((18.678 - T_c / 234.5) *
                                    T_c / (257.14 + T_c)) * 100
        elif method == 'cimo':
            T_c = T_k - 273.15  # temperature in Celsius degree
            e_sat = 6.112 * np.exp(17.62 * T_c / (243.12 + T_c)) * 100
        else:
            # Goff-Gratch equation by default
            u_T = 373.16 / T_k
            v_T = T_k / 373.16
            e_sat = (- 7.90298 * (u_T - 1) + 5.02808 * np.log10(u_T) -
                     1.3816e-7 * (10**(11.344 * (1 - v_T)) - 1) +
                     8.1328e-3 * (10**(- 3.49149 * (u_T - 1)) - 1) +
                     np.log10(1013.246))
            e_sat = 10**e_sat * 100
    else:
        if method == 'buck':
            T_c = T_k - 273.15  # temperature in Celsius degree
            e_sat = 6.1115 * np.exp((23.036 - T_c / 333.7) *
                                    T_c / (279.82 + T_c)) * 100
        elif method == 'cimo':
            T_c = T_k - 273.15  # temperature in Celsius degree
            e_sat = 6.112 * np.exp(22.46 * T_c / (272.62 + T_c)) * 100
        else:
            # Goff-Gratch equation by default
            u_T = 273.16 / T_k
            v_T = T_k / 273.16
            e_sat = (- 9.09718 * (u_T - 1) - 3.56654 * np.log10(u_T) +
                     0.876793 * (1 - v_T) + np.log10(6.1071))
            e_sat = 10**e_sat * 100

    return e_sat
