import numpy as np
import pandas as pd
from simulate_process import simulate_process

time = np.linspace(0, 40, 1000)
freq = np.linspace(0, 50, 10)

#EPSs parameters
S0 = np.linspace(0.1,1,10)
b0 = np.linspace(0.15,0.55,5)

#KT parameter
omega_g = np.linspace(10,45,5)
zeta_g = np.linspace(0.1,0.5,5)

w_cp = np.zeros((len(time), len(time)))
w_kt = np.zeros((len(time), len(time)))

for s0 in S0:
    for _b0 in b0:
        for _omega_g in omega_g:
            for _zeta_g in zeta_g:
                for i in range(w_cp.shape[0]):
                    w_cp[i,:] = simulate_process(time, s0, max(freq)/2, _omega_g, _zeta_g, _b0)
                    w_kt[i,:] = simulate_process(time, s0, _omega_g, _omega_g, _zeta_g, _b0, eps_type='kt')

                w_cp_df = pd.DataFrame(w_cp).add_prefix('cp_')
                w_kt_df = pd.DataFrame(w_kt).add_prefix('kt_')

                df_results = pd.concat([w_cp_df, w_kt_df], axis=1)
                df_results['time'] = time
                df_results.to_csv(f'data/realizations/stochastic_process_{s0}_{_b0}_{_omega_g}_{_zeta_g}.csv', index=False)