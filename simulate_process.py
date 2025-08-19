import numpy as np

from eps.cp import cp_evolutionary_power_spectrum
from eps.kt import kt_evolutionary_power_spectrum

def simulate_process(time, S0, frequency_max, omega_g: float, zeta_g: float, b0=0.2, m=800, random_state=None, eps_type='cp'):
    """
    Simulate nonstationary base excitation process (ndof=1).

    Parameters
    ----------
    time : ndarray
        Time vector (1D array).
    S0 : float
        Intensity parameter for the EPS.
    frequency_max : float
        Maximum frequency (rad/s).
    b0 : float, optional
        Decay constant for EPS (default 0.2).
    m : int, optional
        Number of frequency steps for discretization (default 800).
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    W : ndarray, shape (len(time),)
        Simulated nonstationary base excitation process.
    """
    rng = np.random.default_rng(random_state)

    N = len(time)
    dw = frequency_max / m

    W = np.zeros(N)

    # Nonstationary base excitation
    for i in range(m):
        freq = i * dw
        if (eps_type == 'cp'):
            eps_fun = cp_evolutionary_power_spectrum(freq, time, S0, b0)  # shape (N,)
        else:
            eps_fun = kt_evolutionary_power_spectrum(freq, time, S0, omega_g, zeta_g, b0)  # shape (N,)
        rand_phase = 2 * np.pi * rng.random()
        W += np.sqrt(4 * dw * eps_fun) * np.cos(freq * time - rand_phase)

    return W