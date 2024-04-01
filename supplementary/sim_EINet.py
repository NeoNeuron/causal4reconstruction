# ## References
# 
# 1. van Vreeswijk, C., & Sompolinsky, H. (1996). Chaos in neuronal networks with balanced excitatory and inhibitory activity. Science (New York, N.Y.), 274(5293), 1724–1726. https://doi.org/10.1126/science.274.5293.1724
# %%
import brainpy as bp
import brainpy.math as bm
import causal4.utils as utils
bm.set_platform('cpu')
from pathlib import Path, PosixPath
from typing import Union
import warnings
warnings.filterwarnings('ignore')
import numpy as np
def get_conn_matrix(num_pre:int, num_post:int, K:int,
                    mode:str='FixedPre', seed:int=0,
                    include_self:bool=True, sparse:bool=True):
    np.random.seed(seed)
    if sparse:
        conn_mat = None
        if mode == 'FixedPre':
            conn_mat = [np.vstack(
                (
                    np.random.choice(num_pre, size=K, replace=False),
                    i*np.ones(K, dtype=int)
                    )
                ) for i in range(num_post)]
        elif mode == 'FixedPost':
            conn_mat = [np.vstack(
                (
                    i*np.ones(K, dtype=int),
                    np.random.choice(num_post, size=K, replace=False)
                    )
                ) for i in range(num_pre)]
        else:
            raise NotImplementedError(f"mode {mode} not implemented!")
        conn_mat = np.hstack(conn_mat)
        if not include_self and num_pre == num_post:
            conn_mat = conn_mat[:, conn_mat[0]!=conn_mat[1]]
    else:
        if mode == 'FixedPre':
            conn_mat = np.zeros((num_pre, num_post), dtype=bool)
            for i in range(num_post):
                conn_mat[np.random.choice(num_pre, K, replace=False), i] = True
        elif mode == 'FixedPost':
            conn_mat = np.zeros((num_pre, num_post), dtype=bool)
            for i in range(num_pre):
                conn_mat[i, np.random.choice(num_post, K, replace=False)] = True
        else:
            raise NotImplementedError(f"mode {mode} not implemented!")
        if not include_self and num_pre == num_post:
            conn_mat[np.eye(num_pre, dtype=bool)] = False
    return conn_mat
# %%
class EINet(bp.Network):
    def __init__(self, num_e:int, num_i:int, K:int,
                 mu0:float, conn_path:Union[str, PosixPath],
                 poisson_seed=0, delay_step:int=0,
                 method='exp_auto', **kwargs):
        super(EINet, self).__init__(**kwargs)
        pars = dict(
            V_rest=0., V_reset=0., tau=20., tau_ref=2., R=20.,
            method=method, V_initializer=bp.init.Uniform(0., 0.7),
            mode = bm.NonBatchingMode(), ref_var=True,
            )
        E = bp.neurons.LIF(num_e, V_th=1.0, **pars)
        I = bp.neurons.LIF(num_i, V_th=0.7, **pars)
        self.num_e = num_e
        self.num_i = num_i

        # synapses
        w_e2e =  1.0/bm.sqrt(K)  # excitatory synaptic weight
        w_e2i =  1.0/bm.sqrt(K)  # excitatory synaptic weight
        w_i2e = -2.0/bm.sqrt(K)  # inhibitory synaptic weight
        w_i2i = -1.8/bm.sqrt(K)  # inhibitory synaptic weight

        sparse = True
        syn_kws = dict(delay_step=delay_step, post_ref_key='refractory')
        if sparse:
            conn_mat = np.fromfile(conn_path, dtype=float).reshape(2,-1)
            _conn = conn_mat[:, (conn_mat[0]<num_e) * (conn_mat[1]<num_e)]
            self.E2E = bp.synapses.Delta(E, E, bp.conn.IJConn(_conn[0], _conn[1]), g_max=w_e2e, **syn_kws)
            _conn = conn_mat[:, (conn_mat[0]<num_e) * (conn_mat[1]>=num_e)]
            _conn[1] -= num_e
            self.E2I = bp.synapses.Delta(E, I, bp.conn.IJConn(_conn[0], _conn[1]), g_max=w_e2i, **syn_kws)
            _conn = conn_mat[:, (conn_mat[0]>=num_e) * (conn_mat[1]<num_e)]
            _conn[0] -= num_e
            self.I2E = bp.synapses.Delta(I, E, bp.conn.IJConn(_conn[0], _conn[1]), g_max=w_i2e, **syn_kws)
            _conn = conn_mat[:, (conn_mat[0]>=num_e) * (conn_mat[1]>=num_e)]
            _conn -= num_e
            self.I2I = bp.synapses.Delta(I, I, bp.conn.IJConn(_conn[0], _conn[1]), g_max=w_i2i, **syn_kws)
            _conn = None
            conn_mat = None
        else:
            conn_mat = np.fromfile(conn_path, dtype=float).reshape(N,N)
            self.E2E = bp.synapses.Delta(E, E, bp.conn.MatConn(conn_mat[:num_e, :num_e]), g_max=w_e2e, **syn_kws)
            self.E2I = bp.synapses.Delta(E, I, bp.conn.MatConn(conn_mat[:num_e, num_e:]), g_max=w_e2i, **syn_kws)
            self.I2E = bp.synapses.Delta(I, E, bp.conn.MatConn(conn_mat[num_e:, :num_e]), g_max=w_i2e, **syn_kws)
            self.I2I = bp.synapses.Delta(I, I, bp.conn.MatConn(conn_mat[num_e:, num_e:]), g_max=w_i2i, **syn_kws)
            conn_mat = None

        # connectioni from noise neurons to excitatory and inhibitory neurons
        f2e = 1.0/bm.sqrt(K)  # excitatory synaptic weight
        f2i = 0.8/bm.sqrt(K)  # excitatory synaptic weight
        # self.ffwd_E = bp.synapses.PoissonInput(E.V, 1, freq=mu0*K, weight=f2e, mode=bm.NonBatchingMode())
        # self.ffwd_I = bp.synapses.PoissonInput(I.V, 1, freq=mu0*K, weight=f2i, mode=bm.NonBatchingMode())
        self.ffwd_E = bp.neurons.PoissonGroup(num_e, freqs=mu0*K, seed=poisson_seed)
        self.ffwd_I = bp.neurons.PoissonGroup(num_i, freqs=mu0*K, seed=poisson_seed*1000)
        self.ffwd2E = bp.synapses.Delta(self.ffwd_E, E, bp.conn.One2One(), g_max=f2e, post_ref_key='refractory')
        self.ffwd2I = bp.synapses.Delta(self.ffwd_I, I, bp.conn.One2One(), g_max=f2i, post_ref_key='refractory')

        self.E = E
        self.I = I
        self.FE = self.ffwd_E
        self.FI = self.ffwd_I
        self.w_e2e =  1.0/bm.sqrt(K)
        self.w_e2i =  1.0/bm.sqrt(K)
        self.w_i2e = -2.0/bm.sqrt(K)
        self.w_i2i = -1.8/bm.sqrt(K)
        self.f2e   =  1.0/bm.sqrt(K)
        self.f2i   =  0.8/bm.sqrt(K)

def run_model(num_e:int, num_i:int, K:int, mu:float,
              conn_path:Union[str, PosixPath],
              poisson_seed:int, T:float, delay:float=2.):
    dt = 0.02
    model = EINet(num_e, num_i, K, mu, conn_path, poisson_seed, delay_step=int(delay/dt))
    runner = bp.DSRunner(model,
                        monitors=['E.spike', 'I.spike',],# 'FE.spike', 'FI.spike'],
                        dt=dt)
    runner.run(5.)
    runner.reset_state()
    n_epoch = 50
    for i in range(n_epoch):
        runner.run(T/n_epoch)
        print(f"{poisson_seed:d} th copy: E {runner.mon['E.spike'].sum()/num_e/T_single*1e3*n_epoch:f} Hz, I {runner.mon['I.spike'].sum()/num_i/T_single*1e3*n_epoch:f} Hz")
        if i == 0:
            spk_time = utils.prepare_save_spikes(runner.mon.ts, runner.mon['E.spike'], runner.mon['I.spike'])
        else:
            spk_time_buff = utils.prepare_save_spikes(runner.mon.ts, runner.mon['E.spike'], runner.mon['I.spike'])
            spk_time = np.concatenate((spk_time, spk_time_buff), axis=1)
        # runner.mon = None        # clear cached memory in monitors
        # runner._monitors = None  # clear cached memory in monitors
    return spk_time
#%%
if __name__ == '__main__':
    #%%
    num_e = 32000
    num_i = 8000
    N = num_e+num_i
    K = 4000
    mu = 50
    conn_seed = 0
    T_total = 1e1
    delay = 0.0
    regen_conn = False

    folder = Path(f"../EINet/data/EE/N={num_e+num_i:d}")
    folder.mkdir(parents=True, exist_ok=True)
    fname = f"EINet-K={K:d}delay={delay:.1f}mu={mu:d}_T={T_total:.2e}_spike_train.dat"
    # generate and save sparse conn_matrix
    conn_path = folder / f"connect_matrix-p={K*2./N:.3f}.dat"
    if regen_conn or not conn_path.exists():
        conn_E2E = get_conn_matrix(num_e, num_e, K, mode='FixedPre', seed=conn_seed+1, include_self=False)
        conn_E2I = get_conn_matrix(num_e, num_i, K, mode='FixedPre', seed=conn_seed+2)
        conn_I2E = get_conn_matrix(num_i, num_e, K, mode='FixedPre', seed=conn_seed+3)
        conn_I2I = get_conn_matrix(num_i, num_i, K, mode='FixedPre', seed=conn_seed+4, include_self=False)
        conn_E2I[1] += num_e
        conn_I2E[0] += num_e
        conn_I2I += num_e
        conn_mat = np.hstack([conn_E2E, conn_E2I, conn_I2E, conn_I2I])
        conn_E2E = conn_E2I = conn_I2E = conn_I2I = None
        utils.save2bin(conn_path, conn_mat, clearcache=True)
        conn_mat = None

    # define all parameter values need to explore
    import ray
    from ray.util.multiprocessing import Pool
    if not ray.is_initialized():
        ray.init()
    num_total_cpus = ray.cluster_resources()['CPU']
    n_cpus_per_process = 4         # number of cpus to use for each process
    n_processes = int(num_total_cpus/n_cpus_per_process)     # number of processes to use
    T_single = T_total/n_processes

    pool = Pool(ray_address="auto",
                processes=n_processes,
                ray_remote_args={"num_cpus": n_cpus_per_process})
    processes = []
    for i in range(n_processes):
        processes += [pool.apply_async(
            func=run_model, 
            args=(num_e, num_i, K, mu, conn_path, i, T_single, delay),
            )]
    pool.close()
    pool.join()
    results = [r.get() for r in processes]

    # save data
    for i in range(n_processes):
        if i == 0:
            utils.save2bin(folder/fname, results[i].T, clearcache=True, verbose=True)
        else:
            results_buffer = results[i].copy()
            results_buffer[0] += i*T_single
            utils.save2bin(folder/fname, results_buffer.T, 'ab', clearcache=True, verbose=False)
    results = None
# %%
