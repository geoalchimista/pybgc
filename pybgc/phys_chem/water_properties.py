import numpy as np
from .. import constants


T_0 = constants.T_0
M_w = constants.M_w  # molar weight of water vapor, kg mol^-1


def water_density(temp, kelvin=False):
    u"""
    Calculate water density as a function of temperature.

    by Wu Sun <wu.sun "at" ucla.edu>, 29 Sep 2015

    Parameters
    ----------
    temp : float or array_like
        Temperature, in Celsius degree by default.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.

    Returns
    -------
    rho_w : float or array_like
        Water density (kg m^-3).

    Raises
    ------
    None

    References
    ----------
    .. [WP02] Wagner, W. and Pruß, A. (2002). The IAPWS Formulation 1995 for
       the Thermodynamic Properties of Ordinary Water Substance for
       General and Scientific Use. *J. Phys. Chem. Ref. Data*, 31, 387.

    """
    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin

    # critical temperature and density
    T_crit = 647.096
    rho_crit = 322.

    # parameters
    theta = 1 - T_k / T_crit
    b = np.array([1.99274064, 1.09965342, -0.510839303, -1.7549349,
                  -45.5170352, -6.74694450e5])
    exponents = np.array([1., 2., 5., 16., 43., 110.]) / 3.

    rho_ratio = np.ones_like(theta)  # initialize with 1
    for m in range(b.size):
        rho_ratio += b[m] * theta ** exponents[m]

    rho_w = rho_crit * rho_ratio
    return rho_w


def seawater_density(temp, salinity, kelvin=False):
    """
    Calculate seawater density as a function of temperature and salinity.

    Applicable range: 1 atm, 0-40 C, 0.5-43 g kg^-1 salinity.

    by Wu Sun <wu.sun "at" ucla.edu>, 29 Sep 2015

    Parameters
    ----------
    temp : float or array_like
        Temperature, in Celsius degree by default.
    salinity : float or array_like
        Salinity, in per mil mass fraction (g kg^-1 seawater).
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.

    Returns
    -------
    rho_sw : float or array_like
        Sea water density (kg m^-3).

    Raises
    ------
    ValueError
        If salinity is negative or exceeds saturation.

    References
    ----------
    .. [MP81] Millero, F. J. and Poisson, A. (1981). International
       one-atmosphere equation of state of seawater. *Deep Sea Res.*,
       28A(6), 625-629.

    """
    if (np.sum(salinity < 0.)):
        raise ValueError('Salinity error, cannot use negative value.')
    if (np.sum(salinity > 360.)):
        # The saturation concentration of NaCl in water is 360 g/L
        raise ValueError('Salinity error, oversaturation reached.')

    T_c = np.array(temp, dtype='d') - kelvin * T_0
    # force temperature to be in Celsius
    salinity = np.array(salinity, dtype='d')

    # parameters
    rho_0_coefs = np.array([999.842594, 6.793952e-2, -9.095290e-3,
                            1.001685e-4, -1.120083e-6, 6.536336e-9, ])
    A_coefs = np.array([8.24493e-1, -4.0899e-3, 7.6438e-5,
                        8.2467e-7, 5.3875e-9, ])
    B_coefs = np.array([-5.72466e-3, 1.0227e-4, -1.6546e-6, ])

    # calculate each term
    rho_0 = rho_0_coefs[0] + rho_0_coefs[1] * T_c + \
        rho_0_coefs[2] * T_c ** 2 + rho_0_coefs[3] * T_c ** 3 + \
        rho_0_coefs[4] * T_c ** 4 + rho_0_coefs[5] * T_c ** 5
    A_term = A_coefs[0] + A_coefs[1] * T_c + A_coefs[2] * T_c ** 2 + \
        A_coefs[3] * T_c**3 + A_coefs[4] * T_c**4
    B_term = B_coefs[0] + B_coefs[1] * T_c + B_coefs[2] * T_c ** 2
    C = 4.8314e-4

    rho_sw = rho_0 + A_term * salinity + B_term * salinity**1.5 + C * salinity
    return rho_sw


def water_dissoc(temp, kelvin=False):
    """
    Calculate water dissociation constant (pK_w) as a function of temperature.

    by Wu Sun <wu.sun "at" ucla.edu>, 29 Sep 2015

    Parameters
    ----------
    temp : float or array_like
        Temperature, in Celsius degree by default.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.

    Returns
    -------
    pK_w : float or array_like.
        Water dissociation constant.

    Raises
    ------
    None

    References
    ----------
    .. [BL06] Bandura, A. V. and Lvov, S. N. (2006). The Ionization Constant
       of Water over Wide Ranges of Temperature and Density.
       *J. Phys. Chem. Ref. Data*, 35, 15.

    """

    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin
    n = 6.
    alpha_0 = -0.864671
    alpha_1 = 8659.19
    alpha_2 = -22786.2
    beta_0 = 0.642044
    beta_1 = -56.8534
    beta_2 = -0.375754
    rho_w = water_density(T_k, kelvin=True) * 1e-3  # in g cm^-3 here

    Z = rho_w * np.exp(alpha_0 + alpha_1 * T_k ** -1 +
                       alpha_2 * T_k ** -2 * rho_w ** (2. / 3.))
    pK_w_G = 0.61415 + 48251.33 * T_k ** -1 - 67707.93 * T_k ** -2 + \
        10102100 * T_k ** -3

    pK_w = -2 * n * (np.log10(1 + Z) - Z / (Z + 1) * rho_w *
                     (beta_0 + beta_1 * T_k ** -1 + beta_2 * rho_w)) + \
        pK_w_G + 2 * np.log10(M_w)

    return pK_w


def latent_heat(temp, kelvin=False, ice=False):
    """
    Calculate the latent heat of vaporization of liquid water, or the latent
    heat of fusion of ice at standard atmospheric pressure.

    Warning: Do not use this function for latent heat of vaporization above
    boiling point.

    by Wu Sun <wu.sun "at" ucla.edu>, 02 Sep 2015

    Parameters
    ----------
    temp : float or array_like
        Temperature, by default in Celsius unless `kelvin` is set `True`.
    kelvin : bool, optional
        Temperature input is in Kelvin if enabled.
    ice : bool, optional
        Calculate the latent heat of fusion if enabled.

    Returns
    -------
    L_v: float or array_like
        Latent heat of vaporization in J mol^-1 (return this if `ice` is
        set `False`).
    L_f : float or array_like
        Latent heat of fusion (melting) in J mol^-1 (return this if `ice` is
        set `True`).

    """
    T_k = np.array(temp, dtype='d') + (not kelvin) * T_0
    # force temperature to be in Kelvin

    if not ice:
        L_v = 1.91846e6 * (T_k / (T_k - 33.91)) ** 2 * M_w
        L_v_sc = 56759 - 42.212 * T_k + \
            np.exp(0.1149 * (281.6 - T_k))  # for supercooled water
        L_v = L_v * (T_k >= 273.15) + L_v_sc * (T_k < 273.15)
        return L_v
    else:
        L_f = 46782.5 + 35.8925 * T_k - 0.07414 * \
            T_k ** 2 + 541.5 * np.exp(- (T_k / 123.75) ** 2)
        return L_f
