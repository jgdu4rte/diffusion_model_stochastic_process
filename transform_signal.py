import numpy as np
import pandas as pd
import os
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from wavelet.get_band_options import ghw_band_options
from wavelet.ghw_transform import ghw_transform
from tqdm import tqdm

def add_missing_values(x, missing_rate=0.2, block=False, block_size=25):
    x_missing = x.copy()
    n = len(x)
    x_max = np.max(x)
    mask_time = np.zeros(n, dtype=bool)  # <-- record which time indices are dropped

    if block:
        n_blocks = int(missing_rate * n / block_size)
        for _ in range(n_blocks):
            start = np.random.randint(0, n - block_size + 1)
            end = start + block_size
            x_missing[start:end] = np.random.uniform(0, x_max, size=block_size)
            mask_time[start:end] = True
    else:
        mask_time = (np.random.rand(n) < missing_rate)
        x_missing[mask_time] = 0

    return x_missing, mask_time

def add_noise(x, snr_db=10):
    """Add Gaussian noise at a given SNR (in dB)."""
    signal_power = np.mean(np.abs(x)**2)
    snr_linear = 10**(snr_db/10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, noise_power, size=x.shape)
    return x + noise

def get_files_in_directory(directory_path, num_files=100):
    all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                 if os.path.isfile(os.path.join(directory_path, f))]
    return np.random.choice(all_files, size=min(num_files, len(all_files)), replace=False).tolist()

def select_columns(df, num_cols=200, cp_cols=100, kt_cols=100):
    cols = df.columns[:-1]  # Exclude last column (time)
    cp_columns = [col for col in cols if 'cp' in col.lower()]
    kt_columns = [col for col in cols if 'kt' in col.lower()]
    
    cp_selected = np.random.choice(cp_columns, size=min(cp_cols, len(cp_columns)), replace=False).tolist()
    kt_selected = np.random.choice(kt_columns, size=min(kt_cols, len(kt_columns)), replace=False).tolist()
    
    remaining_cols = [col for col in cols if col not in cp_selected and col not in kt_selected]
    remaining_needed = num_cols - (len(cp_selected) + len(kt_selected))
    other_selected = np.random.choice(remaining_cols, size=min(remaining_needed, len(remaining_cols)), replace=False).tolist()
    
    return cp_selected + kt_selected + other_selected

def process_column(df: pd.DataFrame, col_name: str, file_name: str, snr_db=10, missing_rate=0.2, block=False):
    t = df["time"].values
    x = df[col_name].values

    dt = np.diff(t)
    fs = 1.0 / np.median(dt)

    # Bands
    uniform_bands, _, _ = ghw_band_options(x, fs, bins_per_band=4, spike_prominence=1e-1)

    x_noisy = add_noise(x, snr_db=snr_db)
    x_missing, mask_time = add_missing_values(x_noisy, missing_rate=missing_rate, block=block)

    def run_and_save(signal, suffix, snr_db=None, missing_rate=None, block=False, mask_time=None):
        out_uniform = ghw_transform(signal, fs, uniform_bands, analytic=True, return_downsampled=False)
        coeffs = out_uniform["complex"]          # list of bands -> shape (H, W)
        bands = out_uniform["bands"]
        freq_centers = np.array([(b[0]+b[1])/2 for b in bands], dtype='float32')

        df_result = pd.DataFrame(
            data=np.array(coeffs),
            index=[f'band_{i}' for i in range(len(coeffs))],
            columns=[f'time_{i}' for i in range(len(coeffs[0]))],
            dtype='complex64'
        )
        df_result['freq_centers'] = pd.Series(freq_centers, index=df_result.index, dtype='float32')

        # ---- Metadata ----
        df_result.attrs["file_name"] = file_name
        df_result.attrs["column"] = col_name
        df_result.attrs["suffix"] = suffix
        df_result.attrs["snr_db"] = snr_db
        df_result.attrs["missing_rate"] = missing_rate
        df_result.attrs["block_dropout"] = block
        if mask_time is not None:
            df_result.attrs["mask_time"] = mask_time.astype(np.uint8)  # save as bytes

        output_file = f'data/transformations/{file_name}_{col_name}{suffix}.pkl'
        df_result.to_pickle(output_file)

    # Original
    run_and_save(x, "")

    # Noisy + missing (now includes mask_time in attrs)
    run_and_save(x_missing, "_missing", missing_rate=missing_rate, block=block, mask_time=mask_time)

def process_file(file_path):
    file_name = os.path.basename(file_path).split('.csv')[0]
    df = pd.read_csv(file_path)
    
    # Select columns
    selected_columns = select_columns(df)
    
    # Process each selected column
    for col in selected_columns:
        process_column(df, col, file_name)

files_list = get_files_in_directory('data/realizations/', num_files=300)
print(f'{len(files_list)} files to process...')

if not os.path.exists('data/transformations'):
    os.makedirs('data/transformations', exist_ok=True)

# Run in parallel
with tqdm_joblib(tqdm(total=len(files_list), desc="Files processed")):
    Parallel(n_jobs=-1)(delayed(process_file)(f) for f in files_list)