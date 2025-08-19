import numpy as np

def kt_evolutionary_power_spectrum(freq, t, S0, omega_g, zeta_g, b0=0.2):
    """
    Nonstationary (evolutionary) Kanai–Tajimi power spectrum.

    Parameters
    ----------
    freq : float or ndarray
        Frequency (rad/s).
    t : float or ndarray
        Time (s).
    S0 : float
        Intensity scale.
    omega_g : float
        Ground dominant frequency (rad/s).
    zeta_g : float
        Damping ratio.
    b0 : float, optional
        Temporal decay coefficient (default 0.2).

    Returns
    -------
    Sw : ndarray
        Nonstationary Kanai–Tajimi spectrum evaluated at given freq and time.
    """
    # Stationary Kanai–Tajimi PSD
    denom = (omega_g**2 - freq**2)**2 + (2 * zeta_g * omega_g * freq)**2
    S_static = S0 * omega_g**4 / denom

    # Nonstationary envelope
    envelope = t**2 * np.exp(-b0 * t)

    Sw = S_static * envelope
    return Sw