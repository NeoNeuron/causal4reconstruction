from multiprocessing import Pool
from subprocess import run
def child_proc(j):
    cml_options_list=["./bin/simHH", 
        "--NE=100",
        "--S=0.02 0.02 0.02 0.02", 
        "--T_Max=1e7",
        "--fE=0.1",
        "--Nu=0.1",
        "--random_S=%d"%j,
        "--record_path=../HH/data/",
        "--record_v=0",
    ]
    result = run(cml_options_list, capture_output=True, universal_newlines=True)
    [print(line) for line in result.stdout.splitlines()]
    if result.returncode != 0:
        print(result.stderr)


random_opts = [0, 1, 2, 3, 4]
p = Pool(len(random_opts))
results = [p.apply_async(func=child_proc, args=(j,)) for j in random_opts]
p.close()
p.join()
