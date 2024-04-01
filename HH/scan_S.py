from multiprocessing import Pool
from subprocess import run

def child_proc(j):
    if j>=1e-3:
        state_path = f'HHp=0.25s={j:.3f}f=0.100u=0.100_state.dat'
    else:
        state_path = f'HHp=0.25s={j:.5f}f=0.100u=0.100_state.dat'
    cml_options_list=["./a.out", 
        "--S=%.1e %.1e %.1e %.1e"%(j,j,j,j), 
        "--save_mode=a", 
        "--state_path="+state_path,
        "--T_Max=5e9",
        "--full_mode=1",
    ]
    result = run(cml_options_list, capture_output=True, universal_newlines=True)
    [print(line) for line in result.stdout.splitlines()]
    if result.returncode != 0:
        print(result.stderr)


ss = [3e-2,2.5e-2,2e-2,1.5e-2,1e-2,7e-3,5e-3,3e-3,2e-3,1e-3,7e-4,5e-4,3e-4,1e-4,7e-5,5e-5,3e-5,1e-5]
p = Pool(len(ss))
results = [p.apply_async(func=child_proc, args=(j,)) for j in ss]
p.close()
p.join()
