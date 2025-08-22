import numpy as np
import pandas as pd
import os
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from wavelet.get_band_options import ghw_band_options
from wavelet.ghw_transform import ghw_transform
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

def get_files_in_directory(directory_path, num_files=100):
    """
    Returns a list of randomly selected files in the specified directory (non-recursive).
    """
    all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                 if os.path.isfile(os.path.join(directory_path, f))]
    return np.random.choice(all_files, size=min(num_files, len(all_files)), replace=False).tolist()

def select_columns(df, num_cols=200, cp_cols=100, kt_cols=100):
    """
    Selects specified number of columns, prioritizing cp and kt columns.
    """
    cols = df.columns[:-1]  # Exclude last column (time)
    cp_columns = [col for col in cols if 'cp' in col.lower()]
    kt_columns = [col for col in cols if 'kt' in col.lower()]
    
    # Randomly select specified number of cp and kt columns
    cp_selected = np.random.choice(cp_columns, size=min(cp_cols, len(cp_columns)), replace=False).tolist()
    kt_selected = np.random.choice(kt_columns, size=min(kt_cols, len(kt_columns)), replace=False).tolist()
    
    # Fill remaining columns if needed
    remaining_cols = [col for col in cols if col not in cp_selected and col not in kt_selected]
    remaining_needed = num_cols - (len(cp_selected) + len(kt_selected))
    other_selected = np.random.choice(remaining_cols, size=min(remaining_needed, len(remaining_cols)), replace=False).tolist()
    
    return cp_selected + kt_selected + other_selected

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

    freq_centers = np.array([(b[0]+b[1])/2 for b in bands], dtype='float32')

    # Create DataFrame with timesteps as columns and frequency bands as rows
    df_result = pd.DataFrame(
        data=np.array(coeffs),  # coeffs is list of arrays, each array is a band
        index=[f'band_{i}' for i in range(len(coeffs))],
        columns=[f'time_{i}' for i in range(len(coeffs[0]))],
        dtype='complex64'
    )
    # Add freq_centers as the last column
    df_result['freq_centers'] = pd.Series(freq_centers, index=df_result.index, dtype='float32')

    # Save results
    output_file = f'data/transformations/{file_name}_{col_name}.pkl'
    df_result.to_pickle(output_file)

def process_file(file_path):
    file_name = os.path.basename(file_path).split('.csv')[0]
    df = pd.read_csv(file_path)
    
    # Select columns
    selected_columns = select_columns(df)
    
    # Process each selected column
    for col in selected_columns:
        process_column(df, col, file_name)

# Get list of 250 random files
files_list = get_files_in_directory('data/realizations/', num_files=222)

print(f'{len(files_list)} files to process...')

if not os.path.exists('data/transformations'):
    os.makedirs('data/transformations', exist_ok=True)

# Run in parallel
with tqdm_joblib(tqdm(total=len(files_list), desc="Files processed")):
    Parallel(n_jobs=-1)(delayed(process_file)(f) for f in files_list)