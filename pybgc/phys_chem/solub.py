"""
Calculate solubilities of common gases in pure water or seawater.

Note: If a gas species has its own function for the calculation, do not use
the general function.

| Gas species          | CO2 | O2 | COS | Others |
|----------------------|-----|----|-----|--------|
| Having own function? |  Y  |  Y |  Y  |   N    |
| Seawater solubility? |  Y  |  Y |  N  |   N    |

by Wu Sun <wu.sun@ucla.edu>, 14 Sep 2014

@TODO: add CH4, N2O, H2, H2S, ...

"""

import numpy as np
from .. import constants
from ..phys_chem.water_properties import seawater_density


p_std = constants.p_std  # standard atmospheric pressure in Pascal
T_0 = constants.T_0
R_gas = constants.R_gas  # universal gas constant, in J K^-1 mol^-1
M_w = constants.M_w  # molar weight of water in kg mol^-1
calorie = constants.calorie  # calorie in J


def solub_co2(temp, salinity=0., pressure=p_std, kelvin=False, bunsen=True):
    """
    Calculate CO2 solubility in natural water.

    Parameters
    ----------
    temp : float or array_like
        Temperature, in Celsius degree by default.
    pressure : float or array_like, optional
        Ambient pressure in Pascal. Default is standard atmospheric pressure.
        Note this is not the partial pressure.
    salinity : float, optional
        Salinity, in per mil mass fraction (g kg^-1 seawater).
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.
    bunsen : bool, optional
        Calculated solubility is reported as Bunsen solubility coefficient by
        default. If this argument is set `False`, calculated solubility is
        expressed in mol L^-1 atm^-1.

    Returns
    -------
    kcc_co2 : float or array_like
        Bunsen solubility coefficient of CO2, a dimensionless quantity.
        kcc_co2 = [CO2]_aq / [CO2]_gas
    *or*
    kcp_co2 : float or array_like
        Solubility of CO2, with the unit mol L^-1 atm^-1.

    Raises
    ------
    ValueError
        If `pressure` is zero or negative.
        If `salinity` is negative or exceeds saturation.

    See also
    --------
    `solub_cos` : Calculate COS solubility in pure water. Seawater unsupported.
    `solub_o2` : Calculate O2 solubility in natural water.
    `solub_general` : Calculate the solubility of a gas in pure water.

    References
    ----------
    .. [MR71] Murray, C. N. and Riley, J. P. (1971). The solubility of gases
       in distilled water and seawater--IV. Carbon dioxide. *Deep-Sea Res.*,
       18, 533-541.
    .. [W74] Weiss, R. F. (1974). Carbon dioxide in water and seawater:
       the solubility of a non-ideal gas. *Mar. Chem.*, 2, 203-215.

    Examples
    --------
    >>> print(solub_co2(25))
    0.83100457308301234

    >>> print(solub_co2([0, 5, 15, 25]))
    [ 1.73886979  1.46250769  1.07645952  0.83100457]

    >>> print(solub_co2(25, salinity=0., pressure=10e5))
    0.084201538367636233

    >>> print(solub_co2(25, salinity=35.))
    0.71093841674810188

    """
    if (np.sum(pressure <= 0)):
        raise ValueError('Pressure error, cannot use zero or negative value.')
    if (np.sum(salinity < 0.)):
        raise ValueError('Salinity error, cannot use negative value.')
    if (np.sum(salinity > 360)):
        # The saturation concentration of NaCl in water is 360 g/L
        raise ValueError('Salinity error, oversaturation reached.')

    # coefficients from Weiss (1974) fitted to Murray and Riley (1971) data
    A_1 = - 58.0931
    A_2 = 90.5069
    A_3 = 22.2940
    B_1 = 0.027766
    B_2 = -0.025888
    B_3 = 0.0050578

    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin
    pressure = np.array(pressure, dtype='d')

    ln_kcp_co2 = (
        A_1 + A_2 * (100. / T_k) + A_3 * np.log(T_k / 100.) +
        salinity * (B_1 + B_2 * (T_k / 100.) + B_3 * (T_k / 100.) ** 2))

    kcp_co2 = np.exp(ln_kcp_co2)
    # 'cp' means kcp = c_aq / p_i, in mol L^-1 atm^-1
    # p_i is the partial pressure of this gas

    if bunsen:
        air_conc = pressure / R_gas / T_k
        # air molar concentration at temperature T_k, in mol m^-3
        kcc_co2 = kcp_co2 * 1e3 / air_conc
        return kcc_co2
    else:
        return kcp_co2


def solub_cos(temp, pressure=p_std, kelvin=False, bunsen=True):
    """
    Calculate COS solubility in pure water.

    Currently there is no equation available for COS solubility in seawater.

    Parameters
    ----------
    temp : float or array_like
        Temperature, in Celsius degree by default.
    pressure : float or array_like, optional
        Ambient pressure in Pascal. Default is standard atmospheric pressure.
        Note this is not the partial pressure.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.
    bunsen : bool, optional
        Calculated solubility is reported as Bunsen solubility coefficient by
        default. If this argument is set `False`, calculated solubility is
        expressed in mol L^-1 atm^-1.

    Returns
    -------
    kcc_cos : float or array_like
        Bunsen solubility coefficient of COS, a dimensionless quantity.
        kcc_cos = [COS]_aq / [COS]_gas
    *or*
    kcp_cos : float or array_like
        Solubility of COS, with the unit mol L^-1 atm^-1.

    Raises
    ------
    ValueError
        If `pressure` is zero or negative.

    See also
    --------
    `solub_co2` : Calculate CO2 solubility in natural water.
    `solub_o2` : Calculate O2 solubility in natural water.
    `solub_general` : Calculate the solubility of a gas in pure water.

    References
    ----------
    .. [E89] Elliott, S., Lu, E., and Rowland, F. S. (1989). Rates and
       mechanisms for the hydrolysis of carbonyl sulfide in natural waters.
       *Environ. Sci. Tech*. 23(4), 458-461.

    Examples
    --------
    >>> print(solub_cos(25))
    0.487598265444

    >>> print(solub_cos([0, 5, 15, 25]))
    [ 1.54884692  1.20816169  0.75508217  0.48759827]

    >>> print(solub_cos(25, pressure=10e5))
    0.0494058942461

    >>> print(solub_cos(25, bunsen=False))
    0.0199301144534

    """
    if (np.sum(pressure <= 0)):
        raise ValueError('Pressure error, cannot use zero or negative value.')

    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin
    pressure = np.array(pressure, dtype='d')

    # coefficients from fitting Elliott et al. (1989) data
    kcc_cos = T_k * np.exp(4050.32 / T_k - 20.0007)

    if bunsen:
        return kcc_cos
    else:
        air_conc = pressure / R_gas / T_k
        # air molar concentration at temperature T_k, in mol m^-3
        kcp_cos = kcc_cos * air_conc * 1e-3
        # 'cp' means kcp = c_aq / p_i, in mol L^-1 atm^-1
        # p_i is the partial pressure of this gas
        return kcp_cos


def solub_o2(temp, salinity=0., pressure=p_std, kelvin=False, bunsen=True):
    """
    Calculate O2 solubility in natural water.

    Parameters
    ----------
    temp : float or array_like
        Temperature, in Celsius degree by default.
    pressure : float or array_like, optional
        Ambient pressure in Pascal. Default is standard atmospheric pressure.
        Note this is not the partial pressure.
    salinity : float, optional
        Salinity, in per mil mass fraction (g kg^-1 seawater).
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.
    bunsen : bool, optional
        Calculated solubility is reported as Bunsen solubility coefficient by
        default. If this argument is set `False`, calculated solubility is
        expressed in mol L^-1 atm^-1.

    Returns
    -------
    kcc_o2 : float or array_like
        Bunsen solubility coefficent of O2, a dimensionless quantity.
        kcc_o2 = [O2]_aq / [O2]_gas
    *or*
    kcp_o2 : float or array_like
        Solubility of O2, with the unit mol L^-1 atm^-1.

    Raises
    ------
    ValueError
        If `pressure` is zero or negative.
        If `salinity` is negative or exceeds saturation.

    See also
    --------
    `solub_co2` : Calculate CO2 solubility in natural water.
    `solub_cos` : Calculate COS solubility in pure water. Seawater unsupported.
    `solub_general` : Calculate the solubility of a gas in pure water.

    References
    ----------
    .. [GG92] Garcia, H. E. and Gordon, I. L. (1992). Oxygen solubility in
       seawater: Better fitting equations. *Limnol. Oceanogr.*, 37(6),
       1307-1312.

    Examples
    --------
    >>> print(solub_o2(25))
    0.83100457308301234

    >>> print(solub_o2([0, 5, 15, 25]))
    [ 1.73886979  1.46250769  1.07645952  0.83100457]

    >>> print(solub_o2(25, salinity=0., pressure=10e5))
    0.084201538367636233

    >>> print(solub_o2(25, salinity=35.))
    0.71093841674810188

    """
    if (np.sum(pressure <= 0)):
        raise ValueError('Pressure error, cannot use zero or negative value.')
    if (np.sum(salinity < 0.)):
        raise ValueError('Salinity error, cannot use negative value.')
    if (np.sum(salinity > 360)):
        # The saturation concentration of NaCl in water is 360 g/L
        raise ValueError('Salinity error, oversaturation reached.')

    # coefficients from Garcia and Gordon (1992)
    # fitted to Benson and Krause (1984) data
    # modified from the MATLAB code by Roberta Hamme <rhamme@ucsd.edu> (2005)
    A_0 = 5.80871
    A_1 = 3.20291
    A_2 = 4.17887
    A_3 = 5.10006
    A_4 = -9.86643e-2
    A_5 = 3.80369
    B_0 = -7.01577e-3
    B_1 = -7.70028e-3
    B_2 = -1.13864e-2
    B_3 = -9.51519e-3
    C_0 = -2.75915e-7

    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin
    pressure = np.array(pressure, dtype='d')

    temp_S = np.log((298.15 - T_k + T_0) / T_k)   # scaled temperature

    conc_o2 = np.exp(
        A_0 + A_1 * temp_S + A_2 * temp_S ** 2 + A_3 * temp_S ** 3 +
        A_4 * temp_S ** 4 + A_5 * temp_S ** 5 + salinity * (
            B_0 + B_1 * temp_S + B_2 * temp_S ** 2 + B_3 * temp_S ** 3) +
        C_0 * salinity ** 2)
    # conc_o2 in umol kg-1, under dry mole fraction of O2 = 0.20946

    air_conc = pressure / R_gas / T_k
    # air molar concentration at temperature T_k, in mol m^-3

    kcc_o2 = conc_o2 / \
        (air_conc * 0.20946 * seawater_density(T_k, salinity, kelvin=True))

    if bunsen:
        return kcc_o2
    else:
        kcp_o2 = kcc_o2 * air_conc * 1e-3
        # 'cp' means kcp = c_aq / p_i, in mol L^-1 atm^-1
        # p_i is the partial pressure of this gas
        return kcp_o2


def solub_general(gas_name, temp, pressure=p_std, kelvin=False, bunsen=False):
    """
    Calculate the solubility of a gas in pure water.

    Supported gases include He, Ne, Ar, Kr, Xe, Rn, H2, N2, O2, O3, CO, CO2,
    CH4, COS, NH3, N2O, NO, H2S and SO2.

    Parameters
    ----------
    gas_name : str
        Chemical name of the gas to be calculated for. Must be in lower case.
    temp : float or array_like
        Temperature, in Celsius degree by default.
    pressure : float or array_like, optional
        Ambient pressure in Pascal. Default is standard atmospheric pressure.
        Note this is not the partial pressure.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.
    bunsen : bool, optional
        Calculated solubility is reported as Bunsen solubility coefficient by
        default. If this argument is set `False`, calculated solubility is
        expressed in mol L^-1 atm^-1.

    Returns
    -------
    k_xp : float or array_like
        Solubility in mol L^-1 atm^-1.
    *or*
    k_cc : float or array_like
        Bunsen solubility coefficient, a dimensionless quantity.
        k_cc = [X]_aq / [X]_gas

    Raises
    ------
    ValueError
        If `gas_name` is not a string.
        If `gas_name` is not a supported gas species.
    ValueError
        If `pressure` is zero or negative.

    See also
    --------
    `solub_co2` : Calculate CO2 solubility in natural water.
    `solub_cos` : Calculate COS solubility in pure water. Seawater unsupported.
    `solub_o2` : Calculate O2 solubility in natural water.

    References
    ----------
    .. [W77] Wilhelm, E., Battino, R. and Wilcock, R. J. (1977). Low-pressure
       solubility of gases in liquid water. *Chem. Rev.*, 77(2), 219-262.

    """
    # original coefficients in caloric units
    coef_solub_vs_temp = {}
    coef_solub_vs_temp['he'] = [-233.163, 8737.84, 32.2652, -0.0119726]
    coef_solub_vs_temp['ne'] = [-310.827, 12766.8, 43.6185, -0.0127534]
    coef_solub_vs_temp['ar'] = [-336.760, 16170.1, 46.2117, -0.00608793]
    coef_solub_vs_temp['kr'] = [-270.967, 15992.9, 33.2892, 0.0260485]
    coef_solub_vs_temp['xe'] = [-360.119, 18744.6, 49.0332, -0.00311323]
    coef_solub_vs_temp['rn'] = [-499.309, 25864.2, 69.3241, 0.00101227]
    coef_solub_vs_temp['h2'] = [-357.602, 13897.5, 52.2871, -0.0298936]
    coef_solub_vs_temp['n2'] = [-327.850, 16757.6, 42.8400, 0.0167645]
    coef_solub_vs_temp['o2'] = [-286.942, 15450.6, 36.5593, 0.0187662]
    coef_solub_vs_temp['o3'] = [-29.7374, 3905.44, 0., 0.]
    coef_solub_vs_temp['co'] = [-341.325, 16487.3, 46.3757, 0.]
    coef_solub_vs_temp['co2'] = [-317.658, 17371.2, 43.0607, -0.00219107]
    coef_solub_vs_temp['ch4'] = [-365.183, 18106.7, 49.7554, -0.000285033]
    coef_solub_vs_temp['cos'] = [-439.589, 23896.1, 60.3429, 0.]
    coef_solub_vs_temp['nh3'] = [-162.446, 2179.59, 32.9085, -0.119722]
    coef_solub_vs_temp['n2o'] = [-180.950, 13205.8, 20.0399, 0.0238544]
    coef_solub_vs_temp['no'] = [-333.515, 16358.8, 45.3253, -0.0519354]
    coef_solub_vs_temp['h2s'] = [-297.158, 16347.7, 40.2024, 0.00257153]
    coef_solub_vs_temp['so2'] = [-29.8850, 5709.15, 0.601884, 0.]

    if not isinstance(gas_name, str):
        raise ValueError('Not a proper gas name.')
    if gas_name not in coef_solub_vs_temp:
        raise ValueError('Gas species "%s" is not support.' % gas_name)

    if (np.sum(pressure <= 0)):
        raise ValueError('Pressure error, cannot use zero or negative value.')

    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin
    pressure = np.array(pressure, dtype='d')

    # extract coefficients for the current gas and convert to SI units
    coefs = np.array(coef_solub_vs_temp[gas_name]) * calorie

    k_xp = np.exp(
        (coefs[0] + coefs[1] / T_k + coefs[2] * np.log(T_k) +
         coefs[3] * T_k) / R_gas)  # atm^-1

    if not bunsen:
        return k_xp  # in atm^-1
    else:
        k_cc = k_xp * 1e3 / M_w * R_gas * T_k / pressure
        # Bunsen solubility, dimensionless
        return k_cc
