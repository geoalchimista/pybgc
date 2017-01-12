import numpy as np
from .. import constants


def retention_curve(theta_w, texture_name, theta_sat=None, theta_res=None,
                    pascal=True):
    """
    Calculate soil matric potential (or matric head) and hydraulic
    conductivity.

    by Wu Sun <wu.sun "at" ucla.edu>, 09 Sep 2015

    Parameters
    ----------
    theta_w : float or array_like
        Soil volumetric water content (m^3 m^-3).
    texture_name : str
        Name of soil texture.
    theta_sat : float or array_like, optional
        Saturated soil water content (m^3 m^-3).
    theta_res : float or array_like, optional
        Residual soil water content (m^3 m^-3)
    pascal : bool, optional
        Used to indicate the unit of matric potential. By default is `True`,
        and returns matric potential in Pascal. If set to `False`, returns
        matric head in meter.

    Returns
    -------
    psi_mat or h_mat : float or array_like
        If `pascal` is `True`, returns `psi_mat`, matric potential in Pascal.
        If `pascal` is `False`, returns `h_mat`, matric head in meter.
    K_unsat : float or array_like
        Unsaturated hydraulic conductivity. The unit is m^2 s^-1 Pa^-1
        if `pascal` is `True`, and m s^-1 if `pascal` is `False`.

    Raises
    ------
    ValueError
        If soil `texture_name` is not a proper one in the predefined list.

    References
    ----------
    .. [1] van Genuchten, M. Th. (1980). A closed-form equation for predicting
       the hydraulic conductivity of unsaturated soils.
       *Soil Sci. Soc. Am. J.*, 44, 892-898.
    .. [2] Or, D. and Wraith, J. M. (2002). Soil Water Content and Water
       Potential Relationships, in Warrick, A. W. (eds.)
       *Soil Physics Companion*, pp 81-82, CRC Press, Boca Raton, FL, USA.

    """
    g = constants.g  # Earth's gravity
    soil_textures = constants.soil_textures
    retention_curve_params = constants.retention_curve_params

    if texture_name not in soil_textures:
        raise ValueError(
            'Not a proper soil texture name. Allowed names are: %s' %
            ', '.join(soil_textures))

    texture_id = soil_textures.index(texture_name)

    if theta_sat is None:
        theta_sat = retention_curve_params['theta_sat'][texture_id]
    if theta_res is None:
        theta_res = retention_curve_params['theta_res'][texture_id]

    n = retention_curve_params['n'][texture_id]
    m = 1. - 1. / n
    alpha = retention_curve_params['alpha'][texture_id]
    K_sat = retention_curve_params['K_sat'][texture_id]

    h_mat = (((theta_w - theta_res) / (theta_sat - theta_res)) **
             (-1. / m) - 1.) ** (1. / n) / alpha

    theta_rel = (theta_w - theta_res) / (theta_sat - theta_res)
    K_rel = theta_rel ** 0.5 * (1. - (1 - theta_rel ** (1. / m)) ** m) ** 2.
    # relative hydraulic conductivity

    K_unsat = K_rel * K_sat  # unsaturated hydraulic conductivity (m s^-1)

    if not pascal:
        return h_mat, K_unsat
    else:
        psi_mat = - h_mat * 1e3 * g
        K_unsat = K_unsat / g * 1e-3  # convert to m^2 s^-1 Pa^-1
        return psi_mat, K_unsat
