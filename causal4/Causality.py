# -*- coding:utf-8 -*-
# Author: Kai Chen
# Description: API for Causality calculation;
# %%
import numpy as np
import os

def get_fname(dtype:str, midterm:str, order:tuple, bin:float, delay:float, 
              spk_fname:str, T:float=None, p:float=None, s:float=None, 
              f:float=None, u:float=None, **kwargs)->str:
    """Generate filename of causal values.

    Args:
        dtype (str): type of dynamical systems.
        midterm (str): midterm of path of data files.
        order (tuple): conditional order in causal measures, (k, l)
        bin (float): bin size for binarization of spike trains
        delay (float): delay length, m.
        spk_fname (str): filename of spike trains.
        T (float): time duration of recorded data.
        p (float, optional): wiring probability of network. Defaults to None.
        s (float, optional): coupling strength. Defaults to None.
        f (float, optional): FFWD Poisson strength. Defaults to None.
        u (float, optional): FFWD Poisson frequency. Defaults to None.

    Returns:
        : str of filename
    """

    DIRPATH = f"./{dtype:s}/" + midterm
    for kw in ('LN-', 'U-', 'G-', 'E-'):
        DIRPATH=DIRPATH.replace(kw, '')
    prefix = f"TGIC2.0-K={order[0]:d}_{order[1]:d}" \
        + f"bin={bin:.2f}delay={delay:.2f}"
    if T is not None:
        prefix += f"T={T:.2e}"
    prefix += "-"
    if spk_fname is not None:
        spk_name_new = spk_fname[:-4] if spk_fname.endswith('.dat') else spk_fname
    else:
        if dtype == 'Lorenz':
            dtype='L'
        elif dtype == 'Logistic':
            dtype='Log'
        if s < 1e-3 and s>0:
            spk_name_new = f"{dtype:s}p={p:.2f}s={s:.5f}f={f:.3f}u={u:.3f}"
        else:
            spk_name_new = f"{dtype:s}p={p:.2f}s={s:.3f}f={f:.3f}u={u:.3f}"
    return DIRPATH + prefix + f"{spk_name_new:s}.dat"


class CausalityIO(object):
    """ ! Original data structure
        dat[0]  : TE value
        dat[1]  :	x index 
        dat[2]  :	y index 
        p = p(x_{n+1}, x-, y-)
        dat[3]  : p(x_{n+1}=1)
        dat[4]  : p(y_{n}=1)
        dat[5]  : p0 = p(0,0,0) + p(1,0,0)
        dat[6]  : dpl(or more) = p(x=1|x-, y-_l = 1) - p(x=1|x-, y-_l = 0)
        dat[7]  : \Delta p_m := p(x = 1, y- = 1)/p(x = 1)/p(y- = 1) - 1
        dat[7+order]  : TE(l=5)
        dat[8+order]  : GC
        dat[9+order] : \sum{TDMI}
        dat[10+order] : \sum{NCC^2}
        dat[11+order] : approx for 2*\sum{TDMI}
        y-->x.  Total N*N*(k+12).
    """

    def __init__(self, dtype, N=2, order=(1,1), bin=0.5, delay=0, 
                 T=None, **kwargs) -> None:

        self.dtype=dtype
        if isinstance(N, tuple):
            assert len(N) == 2
            self.N = N[0]+N[1]
            self.midterm = 'data/'
            if N[0]*N[1] > 0:
                self.midterm += f"EI/N={self.N:d}/"
            elif N[0] > 0:
                self.midterm += f"EE/N={self.N:d}/"
            elif N[1] > 0:
                self.midterm += f"II/N={self.N:d}/"
            else:
                raise ValueError('No neuron!')
        else:
            self.N=N
            self.midterm = f"data/EE/N={self.N:d}/"
        self.T = T
        self._order=order
        self.bin=bin
        self.delay=delay
        self.causal_map = dict(
            TE = 0, 
            GC = self.order[1]+8, 
            MI = self.order[1]+9, 
            CC = self.order[1]+10,
            dp = 6,
            Delta_p = self.order[1]+6,
            px = 3,
            py = 4,
        )
    
    @property
    def order(self):
        return self._order
    
    @order.setter
    def order(self, order):
        self._order = order
        self.causal_map['GC'] = order[1]+8
        self.causal_map['MI'] = order[1]+9
        self.causal_map['CC'] = order[1]+10

    def scan_pm(self, pm_name:str, val_range:np.ndarray, pre_id:int, 
        post_id:int, causal_type:str, spk_fname:str=None
        )->np.ndarray:
        """Scan causal values across given parameter names.

        Args:
            pm_name (str): parameter name.
            val_range (np.ndarray): value range of given parameter.
            pre_id (int): index of pre-synaptic node.
            post_id (int): index of post-synaptic node.
            causal_type (str): name of causal values.
            spk_fname (str, optional): filename of data files.
                Override class defined system parameters if not None.
                Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            np.ndarray: array containing causal values.
        """

        if pm_name not in ('delay', 'order', 'bin'):
            raise ValueError("'pm_name' must be one of ('delay', 'order', 'bin').")
        
        id = pre_id + post_id * self.N  # matrix[post_id, pre_id]

        result = np.zeros_like(val_range).astype(float)
        for i, val in enumerate(val_range):
            # print(f'Load data from {self.cat_fname(spk_fname=spk_fname, **{pm_name:val},):s} ...')
            dat = np.fromfile(self.get_fname(spk_fname=spk_fname, **{pm_name:val}), dtype=np.float64)
            if dat.shape[0] % (self.order[1]+12) != 0:
                first = dat[:-9].reshape((-1, self.order[1]+12))
            else:
                first = dat.reshape((-1, self.order[1]+12))
            result[i] = first[id, self.causal_map[causal_type]]
        return result

    def get_fname(self, spk_fname:str, **kwargs):
        pm_buff = dict(
            dtype=self.dtype, 
            midterm=self.midterm, 
            T=self.T,
            order=self.order,
            bin=self.bin,
            delay=self.delay,
            # properties about the name of spike train data file, 
            #  which will be overrided if spk_fname is not None.
        )
        for key in kwargs:
            if key in pm_buff:
                pm_buff[key] = kwargs[key]

        return get_fname(spk_fname=spk_fname, **pm_buff)

    def load_from_single(self, spk_name, causal_type):
        """Local causality data from fixed parameters, and scan over all pairs of nodes.

        Args:
            spk_name (str): Filename of spike train data.
            causal_type (str): name of causal values, CC, MI, GC, TE.

        Returns:
            np.ndarray: (N,N) matrix of causal values.
        """
        fname = self.get_fname(spk_name)
        dat = np.fromfile(fname, dtype=np.float64)
        if causal_type in ('th', 'acc'):
            if dat.shape[0] % (self.order[1]+12) == 0:
                raise ValueError(f"causal_type '{causal_type}' is not supported for latest version datafile.")
            if causal_type == 'th':
                return dat[-8:-4] # TE, GC, MI, CC
            elif causal_type == 'acc':
                return dat[-4:] # TE, GC, MI, CC
        else:
            if dat.shape[0] % (self.order[1]+12) != 0:
                dat = dat[:-9].reshape((-1, self.order[1]+12))
            else:
                dat = dat.reshape((-1, self.order[1]+12))
            if dat.shape[0] != self.N*self.N:
                return dat[:, self.causal_map[causal_type]]
            else:
                return dat[:, self.causal_map[causal_type]].reshape(self.N, self.N)


class CausalityAPI(CausalityIO):
    """
    dat[0]  : TE value
    dat[1]  :	x index 
    dat[2]  :	y index 
      p = p(x_{n+1}, x-, y-)
    dat[3]  : p(x_{n+1}=1)
    dat[4]  : p(y_{n}=1)
    dat[5]  : p0 = p(0,0,0) + p(1,0,0)
    dat[6]  : dpl(or more) = p(x=1|x-, y-_l = 1) - p(x=1|x-, y-_l = 0)
    dat[7]  : \Delta p_m := p(x = 1, y- = 1)/p(x = 1)/p(y- = 1) - 1
    dat[7+order]  : TE(l=5)
    dat[8+order]  : GC
    dat[9+order] : \sum{TDMI}
    dat[10+order] : \sum{NCC^2}
    dat[11+order] : approx for 2*\sum{TDMI}
    y-->x.  Total N*N*(k+12)+9.
    """

    def __init__(self, dtype='HH', N=2, order=(1,1), bin=0.5,
                 delay=0, T=None, p=0.25, s=0.1,
                 f=0.1, u=0.1, **kwargs) -> None:

        super().__init__(dtype=dtype, N=N, order=order, bin=bin, delay=delay, T=T)
        self.p=p
        self.s=s
        self.f=f
        self.u=u
    
    def scan_pm(self, pm_name:str, val_range:np.ndarray, pre_id:int, 
        post_id:int, causal_type:str, spk_fname:str=None
        )->np.ndarray:
        """Scan causal values across given parameter names.

        Args:
            pm_name (str): parameter name.
            val_range (np.ndarray): value range of given parameter.
            pre_id (int): index of pre-synaptic node.
            post_id (int): index of post-synaptic node.
            causal_type (str): name of causal values.
            spk_fname (str, optional): filename of data files.
                Override class defined system parameters if not None.
                Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            np.ndarray: array containing causal values.
        """

        if pm_name not in ('s', 'delay', 'f', 'u', 'order', 'bin'):
            raise ValueError("'pm_name' must be one of ('s', 'delay', 'f', 'u', 'order', 'bin').")
        
        id = pre_id + post_id * self.N  # matrix[post_id, pre_id]

        result = np.zeros(len(val_range), dtype=float)
        for i, val in enumerate(val_range):
            # print(f'Load data from {self.cat_fname(spk_fname=spk_fname, **{pm_name:val},):s} ...')
            dat = np.fromfile(self.get_fname(spk_fname=spk_fname, **{pm_name:val}), dtype=np.float64)
            if pm_name == 'order':
                self.order = val
            if dat.shape[0] % (self.order[1]+12) != 0:
                first = dat[:-9].reshape((-1, self.order[1]+12))
            else:
                first = dat.reshape((-1, self.order[1]+12))
            result[i] = first[id, self.causal_map[causal_type]]
        return result

    def get_fname(self, spk_fname:str=None, **kwargs):
        pm_buff = dict(
            dtype=self.dtype, 
            midterm=self.midterm, 
            T=self.T,
            order=self.order,
            bin=self.bin,
            delay=self.delay,
            # properties about the name of spike train data file, 
            #  which will be overrided if spk_fname is not None.
            p=self.p,
            s=self.s,
            f=self.f,
            u=self.u,
        )
        for key in kwargs:
            if key in pm_buff:
                pm_buff[key] = kwargs[key]

        return get_fname(spk_fname=spk_fname, **pm_buff)

# %%
class CausalityAPI3(CausalityAPI):
    def __init__(self, dtype, N, order, bin, delay, 
                 T, p, s1, s2, f, u, **kwargs) -> None:
        super().__init__(dtype=dtype, N=N, order=order, bin=bin, delay=delay, 
                         T=T, p=p, s=s1, f=f, u=u, **kwargs)
        self.s1 = s1
        self.s2 = s2
    
    def get_fname(self, **kwargs):
        pm_buff = dict(
            dtype=self.dtype, 
            midterm=self.midterm, 
            order=self.order,
            bin=self.bin,
            delay=self.delay,
            T=self.T,
            p=self.p,
            s1=self.s1,
            s2=self.s2,
            f=self.f,
            u=self.u,
        )
        for key in kwargs:
            if key in pm_buff:
                pm_buff[key] = kwargs[key]
        return f"./{pm_buff['dtype']:s}/"+pm_buff['midterm'] \
               + f"TGIC2.0-K={pm_buff['order'][0]:d}_{pm_buff['order'][1]:d}" \
               + f"bin={pm_buff['bin']:.2f}delay={pm_buff['delay']:.2f}T={pm_buff['T']:.2e}-" \
               + f"{pm_buff['dtype']:s}p={pm_buff['p']:.2f}" \
               + f"s={pm_buff['s1']:.3f}s={pm_buff['s2']:.3f}" \
               + f"f={pm_buff['f']:.3f}u={pm_buff['u']:.3f}.dat"

    
    def scan_pm(self, pm_name, val_range, pre_id, post_id, causal_type):

        if pm_name not in ('s1', 's2', 'delay', 'f', 'u', 'order', 'bin'):
            raise ValueError("'pm_name' must be one of ('s1', 's2', 'delay', 'f', 'u', 'order', 'bin').")
        
        id = pre_id + post_id * self.N  # matrix[post_id, pre_id]

        result = np.zeros_like(val_range).astype(float)
        for i, val in enumerate(val_range):
            # print(f'Load data from {self._cat_fname(**{pm_name:val}, **kwargs):s} ...')
            dat = np.fromfile(self.get_fname(**{pm_name:val}), dtype=np.float64)
            if dat.shape[0] % (self.order[1]+12) != 0:
                first = dat[:-9].reshape((-1, self.order[1]+12))
            else:
                first = dat.reshape((-1, self.order[1]+12))
            # second = dat[-9:]
            result[i] = first[id, self.causal_map[causal_type]]
        return result

#%% 
import subprocess as sp
def run(verbose=False, shuffle=False, **kwargs):
    '''
    Run calculation of causal values.

    Parameters
    ----------

    '''
    flag_map = dict(
        fname = "-f %s",
        Ne = "--NE %d",
        Ni = "--NI %d",
        order = '--order %d,%d', # Trick: use comma to separate two order numbers;
        T = "--T_Max %f",
        DT = "--DT %f",
        auto_T_max = "--auto_T_max %d",
        bin = "--bin %f",
        delay = "--sample_delay %f",
        con_mat = "--matrix_name %s",
        path_input = "--path_input %s",
        path_output = "--path_output %s",
        mask_file = "--mask_file %s",
        n_thread = "-j %d",
    )
    cml_options = './bin/calCausality -c ./Causality/NetCau_parameters.ini '
    for key in kwargs:
        if key in flag_map:
            cml_options += flag_map[key]%kwargs[key] + ' '
    # for section in self.config.sections():
    #     for option in list(self.config[section]):
    #         cml_options += '--'+section+'.'+option + ' ' + (self.config[section][option]) + ' '
    if verbose:
        cml_options += '-v '
    if shuffle:
        cml_options += '-s '
    # Trick: replace comma with space;
    cml_options_list = [item.replace(',', ' ') if ',' in item else item for item in cml_options.split(' ')]
    result = sp.run(cml_options_list, capture_output=True, universal_newlines=True)
    if verbose:
        [print(line) for line in result.stdout.splitlines()]
    if result.returncode != 0:
        print(result.stderr)
# %%
from multiprocessing import Pool
def scan_pm_single(pm:str, val_range:np.ndarray, verbose=False, mp=None, **kwargs):
    """ Scan causality based on the single data file.
        Call C/C++ interface to calculate causal values.
        Parameters: delay, order, dt, ; 

    Args:
        pm (str): key name of parameter to be scanned.
        val_range (np.ndarray): array of val_range value to be scanned.
        verbose (bool, optional): Defaults to False.
        mp (int, optional): number of processes, None for single process. Defaults to None.

    Returns:
        list: list of results from subprocess.run()
    """
    if pm in kwargs:
        del kwargs[pm]
    if mp is None:
        result = [run(verbose, **{pm:val}, **kwargs) for val in val_range]
    else:
        p = Pool(mp)
        result = [
            p.apply_async(
            func = run, args=(verbose,), kwds = dict(**{pm:val},**kwargs),
            ) for val in val_range
        ]
        p.close()
        p.join()
    return result

def scan_delay(force_regen:bool, pm_causal:dict, delay:np.ndarray, mp:int=30):
    """scan over delay values for specific setting of systems.
        spk_fname mode used in get_fname system, and pm_causal['fname'] are used.

    Args:
        force_regen (bool): whether regenerate all data forcely.
        pm_causal (dict): dict of parametes of causal measures.
        delay (np.ndarray): array of delays to be scanned.
        mp (int, optional): number of processes used for calculation. Defaults to 30.
    """

    delay_not_gen = []
    if 'delay' in pm_causal:
        pm = pm_causal.copy()
        del pm['delay']
    else:
        pm = pm_causal
    spk_fname = pm['fname']
    dtype = spk_fname.split('p=')[0]
    N = pm['Ne']+pm['Ni']
    midterm = 'data/'
    if pm['Ne']*pm['Ni'] > 0:
        midterm += f"EI/N={N:d}/"
    elif pm['Ne'] > 0:
        midterm += f"EE/N={N:d}/"
    elif pm['Ni'] > 0:
        midterm += f"II/N={N:d}/"
    else:
        raise ValueError('No neuron!')
    for val in delay:
        fname_buff = get_fname(dtype, midterm, spk_fname=spk_fname, delay=val, **pm)
        if not os.path.isfile(fname_buff):
            # print('[WARNING]: ' + fname_buff + ' not exist.')
            delay_not_gen.append(val)
    delay_not_gen = np.array(delay_not_gen)
    ##%%
    # run cal_causality to calculate causalities
    if force_regen:
        _ = scan_pm_single('delay', delay, True, mp=mp, **pm)
    else:
        if delay_not_gen.shape[0] > 0:
            _ = scan_pm_single('delay', delay_not_gen, True, mp=mp, **pm)

# %%
def scan_pm_multi(pm:str, val_range:np.ndarray, verbose=False, mp=None, fname_kws={}, **kwargs):
    """ Scan causality based on the single data file.
        Call C/C++ interface to calculate causal values.
        Parameters: p, s, f, u; 

    Args:
        pm (str): key name of parameter to be scanned.
        val_range (np.ndarray): array of val_range value to be scanned.
        verbose (bool, optional): Defaults to False.
        mp (int, optional): number of processes, None for single process. Defaults to None.
        kw

    Returns:
        list: list of results from subprocess.run()
    """
    #TODO: check kwargs works or not, by varying u=0.1
    def cat_name(dtype='HH', p=0.25, s=0.02, f=0.1, u=0.1):
        if s>=0.001:
            return f'{dtype:s}p={p:.2f}s={s:.3f}f={f:.3f}u={u:.3f}'
        else:
            return f'{dtype:s}p={p:.2f}s={s:.5f}f={f:.3f}u={u:.3f}'
    if 'fname' in kwargs:
        del kwargs['fname']
    if mp is None:
        result = [run(verbose, fname=cat_name(**{pm:val}, **fname_kws), **kwargs) for val in val_range]
    else:
        p = Pool(mp)
        result = [
            p.apply_async(
            func = run, args=(verbose,), 
            kwds = dict(fname=cat_name(**{pm:val}, **fname_kws), **kwargs,),
            ) for val in val_range
        ]
        p.close()
        p.join()
    return result

def scan_pm_multi3(pm:str, val_range:np.ndarray, verbose=False):
    """ Scan causality based on the single data file.
        Call C/C++ interface to calculate causal values.
        Parameters: s1, s2, f, u; 

    Args:
        pm (str): key name of parameter to be scanned.
        val_range (np.ndarray): array of val_range value to be scanned.
        verbose (bool, optional): Defaults to False.

    Returns:
        list: list of results from subprocess.run()
    """
    def cat_name(dtype='HH', p=0.25, s1=0.02, s2=0.02, f=0.1, u=0.1):
        return f'{dtype:s}p={p:.2f}s={s1:.3f}s2={s2:.3f}f={f:.3f}u={u:.3f}'

    result = [run(verbose, NE=3, fname=cat_name(pm=val)) for val in val_range]
    return result
# %%
