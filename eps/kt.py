import numpy as np

def kt_evolutionary_power_spectrum(freq: float, t: float, S0: float, omega_g: float, zeta_g: float, c: float = 0.2, t_p: float = 1.0):
    """
    Nonstationary Kanai-Tajimi power spectrum.

    This function implements the time-modulated Kanai-Tajimi evolutionary 
    power spectrum model, where the stationary spectrum is shaped by 
    ground frequency and damping parameters and modulated in time by 
    an exponential envelope.

    Parameters
    ----------
    freq : float or ndarray
        Frequency (rad/s). Can be scalar or array.
    t : float or ndarray
        Time (s). Can be scalar, array, or broadcastable with `freq`.
    S0 : float
        Intensity scale.
    omega_g : float
        Ground dominant frequency (rad/s).
    zeta_g : float
        Damping ratio.
    c : float, optional
        Modulating coefficient (default 0.2).
    t_p : float, optional
        Peak time (default 1.0).

    Returns
    -------
    Sw : ndarray
        Nonstationary Kanai-Tajimi spectrum, with shape compatible with inputs.

    Notes
    -----
    The functional form is:
        S_s(ω) = S0 * (ω_g^4 + 4 * ζ_g^2 * ω_g^2 * ω^2) / ((ω_g^2 - ω^2)^2 + 4 * ζ_g^2 * ω_g^2 * ω^2)
        A(t) = exp(-c * |t - t_p| / t_p)
        S(ω, t) = A(t)^2 * S_s(ω)
    """
    freq = np.asarray(freq)
    t = np.asarray(t)
    A = np.exp(-c * np.abs(t - t_p) / t_p)
    num = omega_g**4 + 4 * zeta_g**2 * omega_g**2 * freq**2
    den = (omega_g**2 - freq**2)**2 + 4 * zeta_g**2 * omega_g**2 * freq**2
    S_stat = S0 * num / den
    return A**2 * S_stat