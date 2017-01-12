"""
A collection of functions of gas diffusivities in air, water, and soil media.

by Wu Sun <wu.sun "at" ucla.edu>, 14 Sep 2014

"""
import numpy as np
from .. import constants


p_std = constants.p_std
R_gas = constants.R_gas
T_0 = constants.T_0

# gas diffusivity in air at STP condition, m^2 s^-1
diffus_in_air_stp = {
    'h2o': 2.178e-5, 'co2': 1.381e-5, 'ch4': 1.952e-5, 'co': 1.807e-5,
    'so2': 1.089e-5, 'o3': 1.444e-5, 'nh3': 1.978e-5, 'n2o': 1.436e-5,
    'no': 1.802e-5, 'no2': 1.361e-5, 'n2': 1.788e-5, 'o2': 1.820e-5,
    'cos': 1.381e-5 / 1.21, }

# gas diffusivity in water at STP condition, m^2 s^-1
diffus_in_water_pre_exp = {
    'he': 818e-9, 'ne': 1608e-9, 'kr': 6393e-9, 'xe': 9007e-9, 'rn': 15877e-9,
    'h2': 3338e-9, 'ch4': 3047e-9, 'co2': 5019e-9, 'cos': 10 ** -1.3246 * 1e-4,
    'co': 0.407e-4, 'no': 39.8e-4, }

# activation energy for gas diffusivity in water as in Arrhenius equation
diffus_in_water_E_act = {
    'he': 11.70e3, 'ne': 14.84e3, 'kr': 20.20e3, 'xe': 21.61e3, 'rn': 23.26e3,
    'h2': 16.06e3, 'ch4': 18.36e3, 'co2': 19.51e3,
    'cos': np.log(10) * 1010 * R_gas,
    'co': 5860 * 4.184, 'no': 8360 * 4.184}

# typical shape parameters for tortuosity effect from soil survey [CH78]_
typical_shape_params = {
    'sand': 4.05, 'loamy sand': 4.38, 'sandy loam': 4.9, 'silt loam': 5.30,
    'loam': 5.39, 'sandy clay loam': 7.12, 'silty clay loam': 7.75,
    'clay loam': 8.52, 'sandy clay': 10.4, 'silty clay': 10.4, 'clay': 11.4}


def diffus_in_air(gas_name, temp, pressure=p_std, kelvin=False):
    """
    Calculate the diffusivity of a gas in air (see the list below).

    Supported gases include H2O, CO2, CH4, CO, SO2, O3, NH3, N2O, NO, NO2,
    N2, O2 [M98]_ and COS [S10]_.

    Parameters
    ----------
    gas_name : str
        Chemical name of the gas to be calculated for. Must be in lower case.
    temp : float or array_like
        Temperature, in Celsius degree by default.
    pressure : float or array_like, optional
        Ambient pressure in Pascal. Default is standard atmospheric pressure.
        Note this is not the partial pressure of the gas.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.

    Returns
    -------
    D_air : float or array_like
        Diffusivity (m^2 s^-1) in air.

    Raises
    ------
    ValueError
        If `gas_name` is not a string.
        If `gas_name` is not a supported gas species.

    See also
    --------
    `diffus_in_water` : Calculate the diffusivity of a gas in water.
    `diffus_in_soil_air` : Calculate the diffusivity of a gas in soil air.
    `diffus_in_soil_water` : Calculate the diffusivity of a gas in soil water.

    References
    ----------
    .. [M98] Massman, W. J. (1998). A review of the molecular diffusivities of
       H2O, CO2, CH4, CO, O3, SO2, NH3, N2O, NO, and NO2 in air, O2 and N2
       near STP. *Atmos. Environ.*, 32(6), 1111-1127.
    .. [S10] Seibt, U. et al. (2010). A kinetic analysis of leaf uptake of COS
       and its relation to transpiration, photosynthesis and carbon isotope
       fractionation. *Biogeosci.*, 7, 333–341.

    """
    if not isinstance(gas_name, str):
        raise ValueError('Not a proper gas name.')
    if gas_name not in diffus_in_air_stp:
        raise ValueError('Gas species "%s" is not support.' % gas_name)

    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin
    pressure = np.array(pressure, dtype='d')

    D_air = diffus_in_air_stp[gas_name] * (p_std / pressure) * \
        (T_k / T_0) ** 1.81

    return D_air


def diffus_in_soil_air(gas_name, temp, total_poros, air_poros, pressure=p_std,
                       shape_param=4.9, texture_name=None, kelvin=False):
    """
    Calculate the diffusivity of a gas in soil air (see the list below).

    Supported gases include H2O, CO2, CH4, CO, SO2, O3, NH3, N2O, NO, NO2,
    N2, O2 [M98]_ and COS [S10]_.

    Parameters
    ----------
    gas_name : str
        Chemical name of the gas to be calculated for. Must be in lower case.
    temp : float or array_like
        Temperature, in Celsius degree by default.
    total_poros : float or array_like
        Total porosity of soil, in m^3 m^-3.
    air_poros : float or array_like
        Air-filled porosity of soil, in m^3 m^-3.
    pressure : float or array_like, optional
        Ambient pressure in Pascal. Default is standard atmospheric pressure.
        Note this is not the partial pressure.
    shape_param : float, optional
        A shape parameter for soil water retention curve that determines the
        tortuosity of soil gas diffusion. It depends on soil texture [CH78]_.
        The default value 4.9 is for sandy loam soil.
    texture_name : str, optional
        Soil texture name, in lower case. When not given properly, sandy loam
        soil is assumed as the default.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.

    Returns
    -------
    D_soil_air : float or array_like
        Diffusivity (m^2 s^-1) in soil air.

    Raises
    ------
    ValueError
        If `gas_name` is not a string.
        If `gas_name` is not a supported gas species.
        If soil `texture_name` is not a proper one in the predefined list.

    See also
    --------
    `diffus_in_air` : Calculate the diffusivity of a gas in air.
    `diffus_in_water` : Calculate the diffusivity of a gas in water.
    `diffus_in_soil_water` : Calculate the diffusivity of a gas in soil water.

    References
    ----------
    .. [M98] Massman, W. J. (1998). A review of the molecular diffusivities of
       H2O, CO2, CH4, CO, O3, SO2, NH3, N2O, NO, and NO2 in air, O2 and N2
       near STP. *Atmos. Environ.*, 32(6), 1111-1127.
    .. [S10] Seibt, U. et al. (2010). A kinetic analysis of leaf uptake of COS
       and its relation to transpiration, photosynthesis and carbon isotope
       fractionation. *Biogeosci.*, 7, 333–341.
    .. [CH78] Clapp, R. B. and Hornberger, G. M. (1978). Empirical equations
       for some soil hydraulic properties. *Water Resources Res.*,
       14(4), 601-604.

    """
    if not isinstance(gas_name, str):
        raise ValueError('Not a proper gas name.')
    if gas_name not in diffus_in_air_stp:
        raise ValueError('Gas species "%s" is not support.' % gas_name)

    if texture_name is not None:
        if texture_name not in typical_shape_params:
            raise ValueError(
                'Not a proper soil texture name. Allowed names are: "%s"' %
                ', '.join(typical_shape_params))
        else:
            shape_param = typical_shape_params[texture_name]

    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin
    pressure = np.array(pressure, dtype='d')
    total_poros = np.array(total_poros, dtype='d')
    air_poros = np.array(air_poros, dtype='d')

    D_soil_air = diffus_in_air_stp[gas_name] * (p_std / pressure) * \
        (T_k / T_0) ** 1.81 * air_poros ** 2 * \
        (air_poros / total_poros) ** (3. / shape_param)

    return D_soil_air


def diffus_in_water(gas_name, temp, kelvin=False):
    u"""
    Calculate the diffusivity of a gas in water (see the list below).

    Supported gases include He, Ne, Kr, Xe, Rn, H2, CH4, CO2 [J87]_,
    CO, NO [WH68]_, and COS [UFUA96]_.

    Parameters
    ----------
    gas_name : str
        Chemical name of the gas to be calculated for. Must be in lower case.
    temp : float or array_like
        Temperature, in Celsius degree by default.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.

    Returns
    -------
    D_aq : float or array_like
        Diffusivity (m^2 s^-1) in water.

    Raises
    ------
    ValueError
        If `gas_name` is not a string.
        If `gas_name` is not a supported gas species.

    See also
    --------
    `diffus_in_air` : Calculate the diffusivity of a gas in air.
    `diffus_in_soil_air` : Calculate the diffusivity of a gas in soil air.
    `diffus_in_soil_water` : Calculate the diffusivity of a gas in soil water.

    References
    ----------
    .. [J87] Jähne, B. et al. (1987). Measurement of the diffusion
       coefficients of sparingly soluble gases in water. *J. Geophys. Res.*,
       92(C10), 10767-10776.
    .. [WH68] Wise, D. L. and Houghton, G. (1968). Diffusion coefficients of
       neon, krypton, xenon, carbon monoxide and nitric oxide in water
       at 10-6O C. *Chem. Eng. Sci.*, 23, 1211-1216.
    .. [UFUA96] Ulshöfer, V. S., Flöck, O. R., Uher, G., and Andreae, M. O.
       (1996). Photochemical production and air-sea exchange of sulfide in
       the eastern Mediterranean Sea. *Mar. Chem.*, 53, 25-39.

    """
    if not isinstance(gas_name, str):
        raise ValueError('Not a proper gas name.')
    if gas_name not in diffus_in_air_stp:
        raise ValueError('Gas species "%s" is not support.' % gas_name)

    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin

    D_aq = diffus_in_water_pre_exp[gas_name] * \
        np.exp(- diffus_in_water_E_act[gas_name] / R_gas / T_k)

    return D_aq


def diffus_in_soil_water(gas_name, temp, total_poros, air_poros,
                         shape_param=4.9, texture_name=None, kelvin=False):
    u"""
    Calculate the diffusivity of a gas in soil water (see the list below).

    Supported gases include He, Ne, Kr, Xe, Rn, H2, CH4, CO2 [J87]_,
    CO, NO [WH68]_, and COS [U96]_.

    Parameters
    ----------
    gas_name : str
        Chemical name of the gas to be calculated for. Must be in lower case.
    temp : float or array_like
        Temperature, in Celsius degree by default.
    total_poros : float or array_like
        Total porosity of soil, in m^3 m^-3.
    air_poros : float or array_like
        Air-filled porosity of soil, in m^3 m^-3.
    shape_param : float, optional
        A shape parameter for soil water retention curve that determines the
        tortuosity of soil gas diffusion. It depends on soil texture [CH78]_.
        The default value 4.9 is for sandy loam soil.
    texture_name : str, optional
        Soil texture name, in lower case. When not given properly, sandy loam
        soil is assumed as the default.
    kelvin : boolean, optional
        Temperature input is in Kelvin if enabled.

    Returns
    -------
    D_soil_aq : float or array_like
        Diffusivity (m^2 s^-1) in soil water.

    Raises
    ------
    ValueError
        If `gas_name` is not a string.
        If `gas_name` is not a supported gas species.
        If soil `texture_name` is not a proper one in the predefined list.

    See also
    --------
    `diffus_in_air` : Calculate the diffusivity of a gas in air.
    `diffus_in_water` : Calculate the diffusivity of a gas in water.
    `diffus_in_soil_air` : Calculate the diffusivity of a gas in soil air.

    References
    ----------
    .. [J87] Jähne, B. et al. (1987). Measurement of the diffusion
       coefficients of sparingly soluble gases in water. *J. Geophys. Res.*,
       92(C10), 10767-10776.
    .. [WH68] Wise, D. L. and Houghton, G. (1968). Diffusion coefficients of
       neon, krypton, xenon, carbon monoxide and nitric oxide in water
       at 10-6O C. *Chem. Eng. Sci.*, 23, 1211-1216.
    .. [UFUA96] Ulshöfer, V. S., Flöck, O. R., Uher, G., and Andreae, M. O.
       (1996). Photochemical production and air-sea exchange of sulfide in
       the eastern Mediterranean Sea. *Mar. Chem.*, 53, 25-39.
    .. [CH78] Clapp, R. B. and Hornberger, G. M. (1978). Empirical equations
       for some soil hydraulic properties. *Water Resources Res.*,
       14(4), 601-604.

    """
    if not isinstance(gas_name, str):
        raise ValueError('Not a proper gas name.')
    if gas_name not in diffus_in_air_stp:
        raise ValueError('Gas species "%s" is not support.' % gas_name)

    if texture_name is not None:
        if texture_name not in typical_shape_params:
            raise ValueError(
                'Not a proper soil texture name. Allowed names are: "%s"' %
                ', '.join(typical_shape_params))
        else:
            shape_param = typical_shape_params[texture_name]

    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin
    total_poros = np.array(total_poros, dtype='d')
    water_poros = total_poros - np.array(air_poros, dtype='d')

    D_soil_aq = diffus_in_water_pre_exp[gas_name] * \
        np.exp(-diffus_in_water_E_act[gas_name] / R_gas / T_k) * \
        water_poros ** 2 * \
        (water_poros / total_poros) ** (shape_param / 3. - 1.)

    return D_soil_aq
