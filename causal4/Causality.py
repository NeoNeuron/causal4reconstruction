# -*- coding:utf-8 -*-
# Author: Kai Chen
# Description: API for Causality calculation;
# %%
import numpy as np
import os
import pandas as pd
from . import c_program
from tqdm import tqdm

class CausalityFileName(str):
    """
        Typical example: TGIC2.0-K=1_1bin=0.50delay=0.00T=6.35e+03-minnie65_v661.dat
    """
    def __new__(cls, value):
        inst = super().__new__(cls, value)
        items = value.split('-')[1:]
        for key in ['K', 'bin', 'delay', 'T']:
            items[0] = items[0].replace(key, '')
        items[0] = items[0].split('=')
        inst.__dict__.update({
            'order': tuple([int(ii) for ii in items[0][1].split('_')]),
            'dt':    float(items[0][2]),
            'delay': float(items[0][3]),
            'T':     float(items[0][4]),
            'fname': items[1].split('.')[0],
        })
        return inst

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

class CausalityEstimator(object):
    """ #! Original data structure
        dat[0]  : TE value
        dat[1]  : pre-synaptic neuron (y) index 
        dat[2]  : post-synaptic neuron (x) index 
        #! Note : in following notation: p( , , ) = p(x_{n+1}, x-, y-)
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
    def __init__(self, path:str, spk_fname:str, N:int,
                 order=(1,1), dt=0.5, delay=None, T=None, **kwargs) -> None:
        self.path = path    # folder path of spk_data file
        self.spk_fname = spk_fname  # fname of spk_data file
        self.N=N
        self.T = T
        self.order=order
        self.dt=dt
        self.delay=delay
        self.DT = kwargs['DT'] if 'DT' in kwargs else 1e3
        self.n_thread = kwargs['n_thread'] if 'n_thread' in kwargs else 10
    
    def save_ini(self, fname:str='causality.ini'):
        with open(self.path+fname, "w") as file:
            if self.T is None:
                myDict={'N': self.N, 'DT': 0, 'dt': self.dt, 'auto_Tmax': 1,
                        'order': self.order, 'delay': self.delay, 
                        'spike_train_filename': self.spk_fname,
                        'path': self.path, 'n_thread': self.n_thread}
            else:
                myDict={'N': self.N, 'Tmax': self.T, 'DT': self.DT, 'dt': self.dt, 'auto_Tmax': 0,
                        'order': self.order, 'delay': self.delay, 
                        'spike_train_filename': self.spk_fname,
                        'path': self.path, 'n_thread': self.n_thread}
            for key, val in myDict.items():
                if isinstance(val, tuple):
                    file.write(f"{key:s} = {val[0]} {val[1]}\n")
                else:
                    file.write(f"{key:s} = {val}\n")

    def get_optimal_delay(self, delay_range:list, dry_run:int=False) -> int:
        dfs = []
        print('>> start searching the optimal delay ...')
        if dry_run:
            print('Dry run, no new estimation.')
            missing = []
            for delay in tqdm(delay_range):
                if not self._check_exist(delay):
                    missing.append(delay)
            if len(missing) == 0:
                print('All files exist.')
            else:
                print(f"Delay not estimated: {missing}")
                return None
        for delay in tqdm(delay_range):
            self._run_estimation(delay, regen=False, verbose=False)
            df = self.fetch_data(delay, new_run=True)
            # using sum(MI) as the estimation standards
            dfs.append(df['sum(MI)'].to_list())
        dfs = np.asarray(dfs)   # shape (len(delay_range), Num_paris)
        self.delay = delay_range[np.argmax(dfs.mean(1))]
        return self.delay

    def causality_fname(self, delay:float=None):
        if delay is None and self.delay is None:
            raise ValueError("delay is not specified.")
        else:
            delay = self.delay if delay is None else delay
            return f"TGIC2.0-K={self.order[0]:d}_{self.order[1]:d}" \
                    + f"bin={self.dt:.2f}delay={delay:.2f}" \
                    + f"T={self.T:.2e}-{self.spk_fname:s}.dat"

    def _check_exist(self, delay:float=None):
        causality_full_fname = self.path + '/' + self.causality_fname(delay)
        if os.path.exists(causality_full_fname):
            return causality_full_fname
        else:
            return False

    def _run_estimation(self, delay:float=None, regen:bool=False, verbose:bool=False):
        """ Run the estimation process.

        Args:
            delay (float, optional): The delay parameter. If not provided, the self.delay value will be used.
            regen (bool, optional): Whether to regenerate the output files. Defaults to False.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.

        Returns:
            str: The output filename of causality data.
        """
        if delay is None:
            delay = self.delay
        pm = dict(
            N = self.N,
            order = self.order,
            T = self.T if self.T is not None else 0,
            DT = self.DT,
            dt = self.dt,
            delay = delay,
            path = self.path,
            spk_fname = self.spk_fname,
            n_thread = self.n_thread,
        )
        if self.T is None:
            pm['auto_Tmax'] = 1 
            output_fname = run(verbose=verbose, force_regen=regen, **pm)
            self.T = CausalityFileName(output_fname).T
            return output_fname
        else:
            return run(verbose=verbose, force_regen=regen, **pm)

    def fetch_data(self, delay:float=None, new_run:bool=False):
        """ Fetches the causality data from a file.

        Args:
            delay (float, optional): The delay parameter. If not specified, the optimal delay will be used.
            new_run (bool, optional): Flag indicating whether to run new estimation if causality file not exist.
                                      Defaults to False.

        Returns:
            DataFrame: A pandas DataFrame containing the fetched causality data.

        Raises:
            FileNotFoundError: If the causality file does not exist.

        """
        # if (shuffle_flag)
            # strcat(str, "_shuffle");
        # check the delay parameter
        if delay is None:
            if self.delay is None:
                print("INFO: delay is not specified. " +
                      "Use CausalityEstimator.get_optimal_delay to " +
                      "search the optimal delay.")
                print("INFO: default searching range: np.arange(0, 11*self.dt, dt).")
                self.get_optimal_delay(np.arange(11)*self.dt)
            delay = self.delay
        # check existence of causality data file
        fname = self._check_exist(delay)
        
        if fname:
            data = np.fromfile(fname, dtype=np.float64)
            if data.shape[0] % (self.order[1]+12) != 0:
                data = data[:-9].reshape((-1, self.order[1]+12))
                print(f"this is an out-of-dated datafile, containing accuracy and threshold, won't be loaded.")
            else:
                data = data.reshape((-1, self.order[1]+12))
            
            columns = ['TE', 'pre_id', 'post_id', 'px', 'py', 'p0'] + \
                      [f'dp{i}' for i in range(1, self.order[1]+1)] + \
                      ['Delta_p', 'TE(l=5)', 'GC', 'sum(MI)', 'sum(CC2)', '2sum(MI)']
            df = pd.DataFrame(data, columns=columns)
            df = df.astype({'pre_id':int, 'post_id':int})
            return df
        else:
            if new_run:
                print("Initialize new run ...")
                self._run_estimation(delay, verbose=True)
                return self.fetch_data(delay, new_run=False)
            else:
                raise FileNotFoundError("Causality file does not exist.")

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
from pathlib import Path
def run(verbose=False, shuffle=False, force_regen=False, **kwargs) -> str:
    '''
    Run estimation of causal values.

    Parameters
    ----------

    '''
    flag_map = dict(
        cfg_file = "-c %s",
        spk_fname = "-f %s",
        N = "-N %d",
        order = '--order %d,%d', # Trick: use comma to separate two order numbers;
        T = "--Tmax %f",
        DT = "--DT %f",
        auto_Tmax = "--auto_Tmax",
        dt = "--dt %f",
        delay = "--delay %f",
        path = "-p %s",
        mask_file = "--mask_file %s",
        n_thread = "-j %d",
    )
    cml_options = str(c_program)
    for key in kwargs:
        if key in flag_map:
            if '%' in flag_map[key]:
                cml_options += ' ' + flag_map[key]%kwargs[key]
            else:
                cml_options += ' ' + flag_map[key]
    cml_options += ' -v'
    if shuffle:
        cml_options += ' -s'
    output_file = kwargs['path'] + '/'
    output_file += f"TGIC2.0-K={kwargs['order'][0]:d}_{kwargs['order'][1]:d}" \
                   + f"bin={kwargs['dt']:.2f}delay={kwargs['delay']:.2f}" \
                   + f"T={kwargs['T']:.2e}-{kwargs['spk_fname']:s}.dat"
    if force_regen or not Path(output_file).exists():
        #* Trick: replace comma with space
        cml_options_list = [item.replace(',', ' ') if ',' in item else item for item in cml_options.split(' ')]
        result = sp.run(cml_options_list, capture_output=True, universal_newlines=True)
        for line in result.stdout.splitlines():
            if verbose:
                print('>>', line)
            if 'save to file' in line:
                output_file = line.split(':')[-1]
        if result.returncode != 0:
            print(result.stderr)
    else:
        if verbose:
            print(f'File {output_file:s} exists.')
    return output_file
# %%