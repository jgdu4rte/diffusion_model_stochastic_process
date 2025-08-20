import numpy as np

def nonseparable_evolutionary_power_spectrum(freq: float, t: float, S0: float, b0: float):
    """
    Nonseparable evolutionary power spectrum.

    This function implements the time-modulated type 
    evolutionary power spectrum model, where the stationary spectrum 
    is shaped by two frequency-dependent filters and modulated in time 
    by an exponential-decay envelope.

    Parameters
    ----------
    freq : float or ndarray
        Frequency (rad/s). Can be scalar or array.
    t : float or ndarray
        Time (s). Can be scalar, array, or broadcastable with `freq`.
    S0 : float
        Intensity parameter (controls overall amplitude).
    b0 : float
        Evolutionary decay constant (controls rate of temporal decay).

    Returns
    -------
    Sw : ndarray
        Nonseparable evolutionary power spectrum values, with the same 
        broadcasted shape as the inputs.

    Notes
    -----
    The functional form is:

        Sw(ω, t) = S0 * ( (ω / (5π))^2 ) * ( exp(-b0 * t) * t^2 ) * exp(-(ω / (10π))^2 * t)

    This corresponds to a filtered white-noise spectrum modulated in time.
    """
    aux1 = (freq / (5 * np.pi)) ** 2
    aux2 = (freq / (10 * np.pi)) ** 2
    Sw = S0 * aux1 * (np.exp(-b0 * t) * t**2) * np.exp(-aux2 * t)

    return Sw
