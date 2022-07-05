from multiprocessing import Pool
from subprocess import run

def child_proc(j):
    cml_options_list=["./a.out", 
        "--NE=100",
        "--T_Max=1e7",
        "--random_S=4",
        "--Nu=%.3f"%j,
    ]
    result = run(cml_options_list, capture_output=True, universal_newlines=True)
    [print(line) for line in result.stdout.splitlines()]
    if result.returncode != 0:
        print(result.stderr)


Nus = [0.09, 0.10, 0.11, 0.12]
p = Pool(len(Nus))
results = [p.apply_async(func=child_proc, args=(j,)) for j in Nus]
p.close()
p.join()
