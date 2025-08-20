import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from simulate_process import simulate_process

def process_combination(s0, b0, omega_g, zeta_g, time, freq):
    w_cp = np.zeros((len(time), len(time)))
    w_kt = np.zeros((len(time), len(time)))
    
    for i in range(w_cp.shape[0]):
        w_cp[i,:] = simulate_process(time, s0, max(freq)/2, omega_g, zeta_g, b0)
        w_kt[i,:] = simulate_process(time, s0, omega_g, omega_g, zeta_g, b0, eps_type='kt')
    
    w_cp_df = pd.DataFrame(w_cp.T).add_prefix('cp_')
    w_kt_df = pd.DataFrame(w_kt.T).add_prefix('kt_')
    df_results = pd.concat([w_cp_df, w_kt_df], axis=1)
    df_results['time'] = time
    df_results.to_csv(f'data/realizations/stochastic_process_{s0}_{b0}_{omega_g}_{zeta_g}_freq_cp_{max(freq)/2}_freq_kt_{omega_g}.csv', index=False)

time = np.linspace(0, 40, 1024)
freq = np.linspace(0, 50, 2)
S0 = np.linspace(0.5, 2, 4)
b0 = np.linspace(0.15, 0.55, 4)
omega_g = np.linspace(10, 45, 4)
zeta_g = np.linspace(0.1, 0.5, 4)

# Create all combinations of parameters
combinations = [(_S0, _b0, _omega_g, _zeta_g) 
                for _S0 in S0 
                for _b0 in b0 
                for _omega_g in omega_g 
                for _zeta_g in zeta_g]

# Parallel execution
Parallel(n_jobs=-1)(delayed(process_combination)(s0, b0, omega_g, zeta_g, time, freq) 
                    for s0, b0, omega_g, zeta_g in tqdm(combinations))