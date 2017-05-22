"""
Hydrolysis rate constants for gas species in natural water

Wu Sun <wu.sun@ucla.edu> on 19 May 2017

"""
import numpy as np
from .. import constants
from ..phys_chem.water_properties import water_dissoc

T_0 = constants.T_0
params_hydrol_cos = {
    'e89': [2.11834513803e-5, 10418.3722377, 14.16881179, 6469.11889197],
    'e89_original': [2.145674000903277e-5, 10450., 12.701762133763484, 6040.],
    'rc94': [1.63838819502e-5, 6444.02904777, 11.7115101829, 2427.27401921],
    'rc94_original': [1.6805937699342022e-5, 6468., 2.3359419797800687, 4428.],
    'k03': [9.60317126325e-6, 12110., 19.1176322696, 11580.],
}


def hydrolysis_cos(temp, pH=7, kelvin=False, seawater=False, method='e89'):
    """
    COS hydrolysis rate in natural waters.
    Applicable range: temperature 5--30 C and pH 4--10.

    Wu Sun <wu.sun@ucla.edu> on 19 May 2017

    Parameters
    ----------
    temp : float or array_like
        Temperature, in Celsius degree by default.
    pH : float or array_like, optional
        pH value; default is 7.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.
    seawater : bool, optional
        If enabled, use the equation for COS hydrolysis in seawater. Equivalent
        to setting method as 'rc94'. This option will override `method`.
    method : str, optional
        Select the parameter sets:

        - 'e89': refitted to Elliott et al. (1989) using updated pK_w
        - 'e89_original': original parameters in Elliott et al. (1989)
        - 'k03': Kamyshny et al. (2003)
        - 'rc94': refitted to Radford-Knoery & Cutter (1994) using updated pK_w
        - 'rc94_original': original parameters in
                           Radford-Knoery & Cutter (1994)

        If the method name is invalid, falls back to the 'e89' method.

    Return
    ------
    k_hyd : float or array_like
        The combined first order hydrolysis rate constant of COS [s^-1].

    References
    ----------
    .. [E89] Elliott, S., Lu, E., and Rowland, F. S. (1989). Rates and
       mechanisms for the hydrolysis of carbonyl sulfide in natural waters.
       Environ. Sci. Tech., 23(4), 458-461.
    .. [K03] Kamyshny, A., Goifman, A., Rizkov, D., and Lev, O. (2003).
       Formation of carbonyl sulfide by the reaction of carbon monoxide and
       inorganic polysulfides. Environ. Sci. Tech., 37(9), 1865-1872.
    .. [RC94] Radford-Knoery, J., and Cutter, G. A. (1993). Determination of
       carbonyl sulfide and hydrogen sulfide species in natural waters using
       specialized collection procedures and gas chromatography with flame
       photometric detection. Anal. Chem., 65(8), 976-982.

    """
    T_ref = 298.15
    if seawater:
        method = 'rc94'

    params = params_hydrol_cos.get(method, params_hydrol_cos['e89'])

    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin

    c_OH = 10. ** (pH - water_dissoc(T_k, kelvin=True))
    # OH concentration [mol L^-1]

    k_hyd = params[0] * np.exp(-params[1] * (1. / T_k - 1. / T_ref)) + \
        params[2] * np.exp(-params[3] * (1. / T_k - 1. / T_ref)) * c_OH
    return k_hyd
