import numpy as np
from typing import List, Tuple, Dict, Union

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def ghw_transform(
    x: Union[np.ndarray, List[float]],
    fs: float,
    bands_hz: List[Tuple[float, float]],
    analytic: bool = True,
    nfft: Union[int, None] = None,
    return_downsampled: bool = True,
) -> Dict[str, List[np.ndarray]]:
    """
    Generalized Harmonic Wavelet transform via frequency-domain masking.
    Supports uniform and adaptive spike-based bands.
    
    Parameters
    ----------
    x : array-like, shape (N,) or (M,N)
        Signal(s). If 2D, each row is a separate signal.
    fs : float
        Sampling rate (Hz).
    bands_hz : list of (flo, fhi)
        Non-overlapping frequency bands, in Hz. 0 <= flo < fhi <= fs/2.
    analytic : bool
        If True, produce analytic complex signals (positive freqs doubled internally)
    nfft : int or None
        FFT length. Defaults to next power of 2 >= N.
    return_downsampled : bool
        If True, critically downsample each band by Nyquist of band.

    Returns
    -------
    out : dict
        {
            "complex": [c_band0, c_band1, ...],
            "real": [(Re, Im) per band],
            "bands": bands_hz,
            "fs_band": [effective_fs per band],
            "reconstruct_fullrate": callable
        }
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    M, N = x.shape

    if nfft is None:
        nfft = _next_pow2(N)

    # FFT frequency grid
    freqs = np.fft.rfftfreq(nfft, d=1/fs)
    X = np.fft.rfft(x, n=nfft, axis=1)

    complex_bands = []
    real_pairs = []
    fs_bands = []

    X_sum = np.zeros_like(X, dtype=np.complex128)

    for (flo, fhi) in bands_hz:
        if flo < 0 or fhi > fs/2 + 1e-9 or fhi <= flo:
            raise ValueError(f"Bad band [{flo},{fhi}] for fs={fs}")

        # Mask for positive FFT bins
        band_mask = (freqs >= flo) & (freqs < fhi)
        mask = np.zeros_like(freqs)
        mask[band_mask] = 1.0

        # Analytic doubling of interior bins
        X_band = X * mask
        if analytic:
            dbl = mask.copy()
            dbl[0] = 0.0  # DC
            if nfft % 2 == 0:
                nyq_bin = nfft // 2
                if nyq_bin < len(dbl):
                    dbl[nyq_bin] = 0.0
            interior = np.ones_like(freqs, dtype=bool)
            interior[0] = False
            if nfft % 2 == 0:
                interior[nyq_bin] = False
            X_band[:, interior] *= 2.0

        # Accumulate for full reconstruction
        X_sum += X_band

        # Build full complex spectrum for IFFT
        full_spec = np.zeros((M, nfft), dtype=np.complex128)
        full_spec[:, :len(freqs)] = X_band
        band_time = np.fft.ifft(full_spec, axis=1)[:, :N]

        # Optional critical downsampling
        if return_downsampled:
            bw = max(fhi - flo, 1e-12)
            dec = int(max(1, np.floor(fs / (2 * bw))))
            band_time_ds = band_time[:, ::dec]
            complex_bands.append(band_time_ds.squeeze() if M == 1 else band_time_ds)
            fs_bands.append(fs / dec)
            real_pairs.append((band_time_ds.real.squeeze() if M == 1 else band_time_ds.real,
                               band_time_ds.imag.squeeze() if M == 1 else band_time_ds.imag))
        else:
            complex_bands.append(band_time.squeeze() if M == 1 else band_time)
            fs_bands.append(fs)
            real_pairs.append((band_time.real.squeeze() if M == 1 else band_time.real,
                               band_time.imag.squeeze() if M == 1 else band_time.imag))

    def reconstruct_fullrate() -> np.ndarray:
        x_rec = np.fft.irfft(X_sum, n=nfft, axis=1)[:, :N]
        return x_rec.squeeze() if M == 1 else x_rec

    return {
        "complex": complex_bands,
        "real": real_pairs,
        "bands": bands_hz,
        "fs_band": fs_bands,
        "reconstruct_fullrate": reconstruct_fullrate,
    }