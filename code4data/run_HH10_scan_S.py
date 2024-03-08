from multiprocessing import Pool
from subprocess import run
import numpy as np
def child_proc(j):
    cml_options_list=["./bin/simHH", 
        "--NE=10",
        "--S=%.1e %.1e %.1e %.1e"%(j,j,j,j), 
        "--T_Max=1e8",
        "--fE=0.1",
        "--Nu=0.1",
        "--record_path=../HH/data/",
        "--record_v=0",
    ]
    result = run(cml_options_list, capture_output=True, universal_newlines=True)
    [print(line) for line in result.stdout.splitlines()]
    if result.returncode != 0:
        print(result.stderr)


N = 31
ss = np.arange(N)*1e-3
p = Pool(len(ss))
results = [p.apply_async(func=child_proc, args=(j,)) for j in ss]
p.close()
p.join()
