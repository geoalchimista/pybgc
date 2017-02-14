import numpy as np


def hlr_func(p, x):
    """
    Hyperbolic light response function.

    Parameters
    ----------
    p : array_like
        A list or array of four parameters.
        - p[0]: theta (0 to 1), curvature parameter
        - p[1]: alpha, apparent quantum yield
        - p[2]: P_m (>0), maximum gross photosynthetic rate
        - p[3]: R_d (>0), dark respiration rate
    x : array_like
        Photosynthetic photon flux density (umol m^-2 s^-1).

    Return
    ------
    Net photosynthetic assimilation rate (umol m^-2 s^-1).

    References
    ----------
    .. [1] Ögren, E. and Evans, J. R. (1993). Photosynthetic light-response
       curves: I. The influence of CO2 partial pressure and leaf inversion.
       Planta, 189, 182-190.

    """
    theta, alpha, P_m, R_d = p
    if np.isclose(theta, 0.):
        return alpha * P_m * x / (alpha * x + P_m) - R_d
    else:
        return (alpha * x + P_m -
                np.sqrt((alpha * x + P_m)**2 -
                        4. * theta * alpha * x * P_m)) * 0.5 / theta - R_d


def resid_hlr_func(p, x, y):
    """
    Residual function for the hyperbolic light response.

    Parameters
    ----------
    p : array_like
        A list or array of four parameters.
        - p[0]: theta (0 to 1), curvature parameter
        - p[1]: alpha, apparent quantum yield
        - p[2]: P_m (>0), maximum gross photosynthetic rate
        - p[3]: R_d (>0), dark respiration rate
    x : array_like
        Photosynthetic photon flux density (umol m^-2 s^-1).
    y : array_like
        Observed photosynthetic assimilation rate (umol m^-2 s^-1).

    Return
    ------
    Residual of predicted photosynthetic rate (umol m^-2 s^-1).

    """
    return y - hlr_func(p, x)
