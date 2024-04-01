from multiprocessing import Pool
from subprocess import run

def child_proc(j):
    cml_options_list=["./a.out", 
        "--NE=100",
        "--NI=0",
        "--S=%.1e %.1e %.1e %.1e"%(j,j,j,j), 
        "--P_c=0.25",
        "--T_Max=1.7e6",
        "--T_step=0.1",
        "--record_v=1",
    ]
    result = run(cml_options_list, capture_output=True, universal_newlines=True)
    [print(line) for line in result.stdout.splitlines()]
    if result.returncode != 0:
        print(result.stderr)


ss = [0.1, 0.08, 0.06,0.05,0.04, 0.02, 0.01,]
p = Pool(len(ss))
results = [p.apply_async(func=child_proc, args=(j,)) for j in ss]
p.close()
p.join()
