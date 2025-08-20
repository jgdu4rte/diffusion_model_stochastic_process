import numpy as np
from scipy.signal import find_peaks

def ghw_band_options(x, fs, bins_per_band=4, spike_prominence=0.1):
    """
    Generate two types of GHW bands for a signal:
      1. Uniform bands (continuous)
      2. Adaptive bands based on FFT energy spikes (continuous, covers full spectrum)
    
    Parameters
    ----------
    x : ndarray
        Input 1D signal
    fs : float
        Sampling frequency (Hz)
    bins_per_band : int
        Number of FFT bins per uniform band
    spike_prominence : float
        Prominence threshold for spike detection in spectrum

    Returns
    -------
    uniform_bands : ndarray, shape (num_bands, 2)
        [(flo, fhi), ...] uniform-width, contiguous bands
    adaptive_bands : ndarray, shape (num_bands, 2)
        [(flo, fhi), ...] contiguous bands aligned with FFT peaks
    freqs : ndarray
        FFT frequency grid (up to Nyquist)
    """
    N = len(x)
    nyquist = fs / 2
    freqs = np.fft.rfftfreq(N, 1/fs)
    n_rfft = len(freqs)

    # ---- 1. Uniform bands (continuous) ----
    uniform_bands = []
    n = 0
    while n < n_rfft:
        m = min(n + bins_per_band - 1, n_rfft - 1)
        flo = freqs[n]
        fhi = freqs[m]
        uniform_bands.append((flo, fhi))
        n = m + 1

    # Fix contiguity: next flo = previous fhi
    for i in range(1, len(uniform_bands)):
        uniform_bands[i] = (uniform_bands[i-1][1], uniform_bands[i][1])

    # ---- 2. Adaptive bands based on energy spikes (continuous) ----
    fx = np.fft.rfft(x)
    spectrum = np.abs(fx)
    
    # Detect peaks in FFT magnitude
    peaks, _ = find_peaks(spectrum, prominence=spike_prominence * np.max(spectrum))
    
    # Include 0 Hz and Nyquist
    spike_freqs = [0] + freqs[peaks].tolist() + [nyquist]
    spike_freqs = sorted(list(set(spike_freqs)))
    
    # Build continuous bands from consecutive spike frequencies
    adaptive_bands = []
    for i in range(len(spike_freqs)-1):
        adaptive_bands.append((spike_freqs[i], spike_freqs[i+1]))

    # Fix contiguity
    for i in range(1, len(adaptive_bands)):
        adaptive_bands[i] = (adaptive_bands[i-1][1], adaptive_bands[i][1])

    return np.array(uniform_bands), np.array(adaptive_bands), np.array(freqs)