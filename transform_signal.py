import numpy as np
import pandas as pd
import os
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from wavelet.get_band_options import ghw_band_options
from wavelet.ghw_transform import ghw_transform
from tqdm import tqdm

def get_files_in_directory(directory_path):
    """
    Returns a list of all files in the specified directory (non-recursive).
    """
    return [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

def process_column(df: pd.DataFrame, col_name: str, file_name: str):
    t = df["time"].values
    x = df[col_name].values

    dt = np.diff(t)
    fs = 1.0 / np.median(dt)

    # bands
    uniform_bands, _, _ = ghw_band_options(x, fs, bins_per_band=4, spike_prominence=1e-1)

    # Apply GHW
    out_uniform = ghw_transform(x, fs, uniform_bands, analytic=True, return_downsampled=False)

    # Extract coefficients and bands from ghw_transform output
    coeffs = out_uniform["complex"]    # list of arrays per band
    bands = out_uniform["bands"]       # (flo, fhi) Hz

    freq_centers = np.array([(b[0]+b[1])/2 for b in bands])

    df_complex = pd.DataFrame(coeffs)
    df_real = df_complex.apply(np.real).add_prefix('real_')
    df_imag = df_complex.apply(np.imag).add_prefix('imag_')
    df_result = pd.concat([df_real, df_imag], axis=1)
    df_result['freq_centers'] = freq_centers
    df_result = df_result.astype('float32')

    # Save results
    output_file = f'data/transformations/{file_name}_{col_name}.pkl'
    df_result.to_pickle(output_file)

def process_file(file_path):
    file_name = os.path.basename(file_path).split('.csv')[0]
    df = pd.read_csv(file_path)
    
    # Process each column in the file
    for col in df.columns[:-1]:
        process_column(df, col, file_name)

# Get list of files
files_list = get_files_in_directory('data/realizations/')[:50]

print(f'{len(files_list)} files to process...')

# Run in parallel
with tqdm_joblib(tqdm(total=len(files_list), desc="Files processed")):
    Parallel(n_jobs=-1)(delayed(process_file)(f) for f in files_list)