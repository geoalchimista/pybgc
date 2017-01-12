"""
List of physical constants.

by Wu Sun <wu.sun "at" ucla.edu>, 23 Sep 2015

References
----------
.. [TS05] Trenberth, K. E. and Smith, L. (2005). The Mass of the Atmosphere:
   A Constraint on Global Analyses. *Journal of Climate*, 18, 864-875.
.. [OW02] Or, D. and Wraith, J. M. (2002). Soil Water Content and Water
   Potential Relationships, in Warrick, A. W. (eds.) *Soil Physics Companion*,
   pp 81-82., CRC Press, Boca Raton, FL, USA.

"""
import scipy.constants.constants as sci_const
import numpy as np

# a dictionary for named constants
named_constants = {}


# constants imported from scipy
named_constants['molar gas constant'] = R_gas = sci_const.R  # J mol^-1 K^-1
named_constants['standard atmosphere'] = p_std = sci_const.atm  # Pascal
named_constants['zero Celsius'] = T_0 = sci_const.zero_Celsius  # Kelvin
named_constants['standard acceleration of gravity'] = g = sci_const.g  # m s^-2
named_constants['calorie'] = calorie = sci_const.calorie  # calorie in Joule


# gas constants
named_constants['dry air molar weight'] = M_d = 28.97e-3  # kg mol^-1
named_constants['water molar weight'] = M_w = 18.015268e-3  # kg mol^-1
named_constants['air concentration at STP'] = air_conc_stp = \
    p_std / R_gas / T_0  # mol m^-3
named_constants['molar volume at STP'] = 1. / air_conc_stp  # m^3 mol^-1

named_constants['specific gas constant of dry air'] = \
    R_spec_d = R_gas / M_d  # J kg^-1 K^-1
named_constants['specific gas constant of water vapor'] = \
    R_spec_w = R_gas / M_w  # J kg^-1 K^-1

# dry air heat capacity values at room temperature; may change with temperature
named_constants['isobaric specific heat capacity of dry air'] = \
    cp_spec_d = 1.004e3  # J kg^-1 K^-1
named_constants['isobaric molar heat capacity of dry air'] = \
    cp_d = cp_spec_d * M_d  # J mol^-1 K^-1


# soil constants
soil_textures = ['sand', 'loamy sand', 'sandy loam', 'loam',
                 'silt', 'silt loam', 'sandy clay loam', 'clay loam',
                 'silty clay loam', 'sandy clay', 'silty clay', 'clay']
named_constants['soil texture list'] = soil_textures


# properties of the earth
named_constants['eccentricity of the earth orbit'] = ecc_earth = 0.016704232

# properties of the atmosphere
named_constants['total mass of the atmosphere'] = m_atm = 5.1480e18  # kg
named_constants['dry mass of the atmosphere'] = m_atm_d = 5.1352e18  # kg
named_constants['dry lapse rate'] = Gamma_d = g / cp_spec_d  # K m^-1
named_constants['mean lapse rate'] = Gamma_mean = 6.5e-3   # K m^-1
named_constants['von Karman constant'] = kappa = 0.40  # dimensionless
