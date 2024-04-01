from multiprocessing import Pool
from subprocess import run
import numpy as np

def child_proc(f, fu):
    u = int(fu * 1000. / f+0.5) / 1000.0;
    cml_options_list=["./a.out", 
        "--NE=10",
        "--S=0.02 0.02 0.02 0.02", 
        "--T_Max=1e7",
        "--full_mode=0",
        "--fE=%.3f"%f,
        "--Nu=%.3f"%u,
    ]
    result = run(cml_options_list, capture_output=True, universal_newlines=True)
    [print(line) for line in result.stdout.splitlines()]
    if result.returncode != 0:
        print(result.stderr)

f_list = np.arange(16)*0.01 + 0.05
fu_list = np.arange(21)*0.002 + 0.01

ff, fufu = np.meshgrid(f_list, fu_list)

p = Pool(30)
results = [p.apply_async(func=child_proc, args=(f, fu)) for f, fu in zip(ff.flatten(),fufu.flatten())]
p.close()
p.join()
