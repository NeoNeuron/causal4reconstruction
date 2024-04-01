#%%
from subprocess import run 
from multiprocessing import Pool
import numpy as np

fv = np.arange(0.01,0.05,0.002)
f  = np.arange(0.05,0.2-0.0001,0.01)
FV, F = np.meshgrid(fv, f)
# %%
V = FV/F
#%%
pool = Pool(50)
results = [
    pool.apply_async(
        func = run, 
        args=([
            './a.out',
            '0.250',
            '0.02', '0.02',
            '%.3f'%f_val,
            '%.3f'%u_val,
        ],),
        ) for f_val, u_val in zip(F.flatten(), V.flatten())
    ]
pool.close()
pool.join()
# %%
