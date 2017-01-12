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
    # retention curve parameters from NRCS database
    # alpha converted to m^-1; K_sat converted to m s^-1
    retention_curve_params = np.zeros(
        (len(soil_textures), ),
        dtype=[('theta_res', 'f8'), ('theta_sat', 'f8'), ('alpha', 'f8'),
               ('n', 'f8'), ('K_sat', 'f8')])
    retention_curve_params['theta_res'] = \
        np.array([0.045, 0.057, 0.065, 0.078, 0.034, 0.067,
                  0.100, 0.095, 0.089, 0.1, 0.07, 0.068])
    retention_curve_params['theta_sat'] = \
        np.array([0.43, 0.41, 0.41, 0.43, 0.46, 0.45,
                  0.39, 0.41, 0.43, 0.38, 0.36, 0.38])
    retention_curve_params['alpha'] = \
        np.array([0.145, 0.124, 0.075, 0.036, 0.016, 0.020,
                  0.059, 0.019, 0.01, 0.027, 0.005, 0.008]) * 100
    retention_curve_params['n'] = \
        np.array([2.68, 2.28, 1.89, 1.56, 1.37, 1.41,
                  1.48, 1.31, 1.23, 1.23, 1.09, 1.09])
    retention_curve_params['K_sat'] = \
        np.array([712.8, 350.2, 106.1, 25.0, 6.00, 10.8,
                  31.4, 6.24, 1.68, 2.88, 0.48, 4.80]) / 8.64e4 * 1e-2

    if texture_name not in soil_textures:
        raise ValueError(
            'Not a proper soil texture name. Allowed names are: "%s"' %
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
