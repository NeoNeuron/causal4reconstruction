#%%
from multiprocessing import Pool
from subprocess import run
#%%
import numpy as np
# import matplotlib.pyplot as plt

# ====================
# generate config string for adj matrix
# ====================
conn_mat = np.zeros((30,30))
motif = np.array([[0,0,0],[1,0,1],[0,0,0]])
for i in range(10):
    conn_mat[i*3:(i+1)*3,:][:,i*3:(i+1)*3] = motif
# fig, ax = plt.subplots(1,1)
# ax.pcolormesh(conn_mat, cmap='RdBu', edgecolor='w', lw=0.1)
# ax.invert_yaxis()
# ax.axis('scaled')
#%%

def child_proc(j):
    cml_options_list=["./a.out", 
        "--NE=30",
        "--full_mode=1",
        "--conn_matrix=" + ' '.join([f'{val:.0f}' for val in conn_mat.flatten()]),
        "--f=" + ' '.join([f'{val:.1f}' for val in np.ones(30)*0.2]),
        "--Nu=0.05",
        "--S=%.1e %.1e %.1e %.1e"%(j,j,j,j), 
        "--T_Max=1e9",
    ]
    result = run(cml_options_list, capture_output=True, universal_newlines=True)
    [print(line) for line in result.stdout.splitlines()]
    if result.returncode != 0:
        print(result.stderr)


ss= np.arange(21)*0.001+0.01
p = Pool(len(ss))
results = [p.apply_async(func=child_proc, args=(j,)) for j in ss]
p.close()
p.join()
