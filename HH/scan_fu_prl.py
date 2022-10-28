from multiprocessing import Pool
from subprocess import run
import numpy as np

def child_proc(f, fu):
    u = int(fu * 1000. / f+0.5) / 1000.0
    cml_options_list=["./a.out", 
        "--NE=3",
        "--seed=100 100",
        "--S=0.02 0.02 0.02 0.02", 
        "--T_Max=4e7",
        "--full_mode=1",
        "--conn_matrix=0 1 0 0 0 1 0 0 0",
        "--f=%.3f %.3f %.3f"%(f,f,f),
        "--Nu=%.3f"%(fu/f),
        "--record_v=1",
        "--record_vlim=0 1e6",
    ]
    result = run(cml_options_list, capture_output=True, universal_newlines=True)
    [print(line) for line in result.stdout.splitlines()]
    if result.returncode != 0:
        print(result.stderr)

f_list = np.arange(16)*0.01 + 0.05
fu_list = np.arange(21)*0.002 + 0.01

ff, fufu = np.meshgrid(f_list, fu_list)

p = Pool(60)
results = [p.apply_async(func=child_proc, args=(f, fu)) for f, fu in zip(ff.flatten(),fufu.flatten())]
p.close()
p.join()

# ! Note: if TE ratio less than 1e2, pickup those parameter pairs and try other random seeds.