#%%
import multiprocessing
num_cores = multiprocessing.cpu_count()
print("Number of CPU cores:", num_cores)
from multiprocessing import Pool
from subprocess import run
import numpy as np
from causal4.utils import save2bin
import os
from pathlib import Path
REPO_PATH = Path(os.path.dirname(__file__))
#%%
def child_proc(N, path, id, T, s=0.02, P=0.25, CP=0.0):
    cml_options_list=["../bin/simHH", 
        "--NE=%d"%N,
        "--S=%.3f %.3f %.3f %.3f"%(s, s, s, s), 
        "--T_Max=%e"%T,
        "--full_mode=0",
        "--seed=11 11",
        "--TrialID=%d"%id,
        "--fE=%.3f"%(0.10),
        "--Nu=%.3f"%(0.10),
        "--P_c=%.2f"%(P),
        "--random_S=0",
        "--record_path=%s"%path,
        "--CP=%.2f"%(CP),
    ]
    result = run(cml_options_list, capture_output=True, universal_newlines=True)
    for line in result.stdout.splitlines():
        print(line)
        if line.startswith('file:'):
            fname = line.split('\\')[-1]
    if result.returncode != 0:
        print(result.stderr)
    return fname

def load_spike_train(path, fname, time_offset=0):
    spike_train = np.fromfile(path+fname+'_spike_train.dat', dtype=float).reshape(-1,2)
    spike_train[:,0] += time_offset
    os.remove(path+fname+'_spike_train.dat')
    return spike_train

if __name__ == '__main__':
    import sys
    N = 100
    T = 1e7
    path = '../HH/data/'
    p = Pool(num_cores)
    T_single_process = T / num_cores
    results = [p.apply_async(
        func=child_proc,
        args=(N, path, seed, T_single_process, 0.02, float(sys.argv[1]), 0.)) for seed in range(num_cores)]
    p.close()
    p.join()
    path += 'EE/N=%d/'%N
    target_file = path+results[0].get()+'_spike_train.dat'
    for i, res in enumerate(results[1:]):
        spike = load_spike_train(path, res.get(), (i+1)*T_single_process)
        save2bin(target_file, spike, fmode='ab', clearcache=True)
    [os.remove(path+res.get()+'_state.dat') for res in results]
