# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator
@FuncFormatter
def sci_formatter(x, pos):
    return r'$10^{%d}$'%x

plt.rcParams['font.size']=20
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False

from causal4.figrc import *
from causal4.Causality import CausalityEstimator
from causal4.utils import polyR2
key_map = {'TE':'TE', 'MI':'sum(MI)', 'GC':'GC', 'CC':'sum(CC2)'}

def Convergence(ax, data:pd.DataFrame, x, neu_pair):
    key_pairs = (('MI', 'CC'), ('GC', 'CC'), ('TE', 'MI'), ('GC', 'TE'))
    mk = ('o', 'o', '+', 'x')
    color=('b', 'orange', 'g', 'r')
    alphas = (0.4, 0.4, 1, 1)
    data_plot = data[(data['pre_id']==neu_pair[0]) & (data['post_id']==neu_pair[1])].copy().sort_values(by=x)
    data_plot['TE'] *= 2
    data_plot['sum(MI)'] *= 2
    dx = data_plot[x].abs().to_numpy()
    residues = np.zeros((4,len(data_plot)))
    count = 0
    for kp, mk_, alpha_, color_ in zip(key_pairs, mk, alphas, color):
        residues[count,:] = (data_plot[key_map[kp[0]]] - data_plot[key_map[kp[1]]]).abs().to_numpy()
        # pval=np.polyfit(np.log10(plot_var), np.log10(residues[count,:]), deg=1)
        # print(polyR2(np.log10(dp), np.log10(residues[count,:]), pval))
        ax.plot(np.log10(data_plot[x].abs()), np.log10(residues[count,:]), 
            ls='', c=color_, marker=mk_, ms=12, alpha=alpha_, 
            label='-'.join([line_rc[key]['label'] for key in kp]))
        count+=1
    
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.xaxis.set_major_formatter(sci_formatter)
    ax.yaxis.set_major_formatter(sci_formatter)
    return dx, residues

def Convergence_dp(ax, data:pd.DataFrame, neu_pair):
    dp, residues = Convergence(ax, data, 'Delta_p', neu_pair)
    ax.plot(np.log10(dp), 3*(np.log10(dp)-np.log10(dp[0]))+np.log10(residues[0,0]),'k-',alpha=1,linewidth=2)
    ax.plot(np.log10(dp), 2*(np.log10(dp)-np.log10(dp[0]))+np.log10(residues[1,0]),'k--',alpha=1,linewidth=2)
    return ax, dp, residues

def Convergence_dt(ax, data:pd.DataFrame, neu_pair):
    dt, residues = Convergence(ax, data, 'dt', neu_pair)
    ax.plot(np.log10(dt), 2*(np.log10(dt)-np.log10(dt[0]))+np.log10(residues[0,0]),'k--',alpha=1,linewidth=2)
    ax.plot(np.log10(dt), 3*(np.log10(dt)-np.log10(dt[0]))+np.log10(residues[1,0]),'k-',alpha=1,linewidth=2)
    return ax, dt, residues

# %%[markdown]
# Figure 2 in maintext
# %%
pm_causal = dict(
        N           = 10,
        T           = 1e8,  # ms,
        DT          = 2e5,  # ms,
        dt          = 0.5,   #  ms,
        order       = (5,5), # x, y,
        delay       = 3,   # ms,
        spk_fname   = 'HHp=0.25s=0.020f=0.100u=0.100',
        conn_file   = 'connect_matrix-p=0.250.dat',
        path        = '../HH/data/N=10_scan_pm/',
    )
dts = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
ss = [3e-2, 2.5e-2, 2e-2, 1.5e-2, 1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
# ss = [3e-2, 2.5e-2, 2e-2, 1.5e-2, 1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3, 7e-4, 5e-4, 3e-4, 1e-4, 7e-5, 5e-5, 3e-5, 1e-5] # order=(25,1)
#%%
data_scan_dt = []
estimator = CausalityEstimator(**pm_causal, n_thread=60)
for dt in dts:
    estimator.dt = dt
    buffer = estimator.fetch_data(new_run=False)
    buffer = buffer[['TE', 'pre_id', 'post_id', 'Delta_p', 'GC', 'sum(MI)', 'sum(CC2)', 'dp1']]
    buffer['dt'] = dt
    data_scan_dt.append(buffer)
data_scan_dt = pd.concat(data_scan_dt)
# %%
data_scan_dp = []
estimator = CausalityEstimator(**pm_causal, n_thread=60)
for s in ss:
    if s <1e-3:
        estimator.spk_fname = f'HHp=0.25s={s:.5f}f=0.100u=0.100'
    else:
        estimator.spk_fname = f'HHp=0.25s={s:.3f}f=0.100u=0.100'
    buffer = estimator.fetch_data(new_run=True)
    buffer = buffer[['TE', 'pre_id', 'post_id', 'Delta_p', 'GC', 'sum(MI)', 'sum(CC2)', 'dp1']]
    buffer['s'] = s
    data_scan_dp.append(buffer)
data_scan_dp = pd.concat(data_scan_dp)
# %%
#!========================================
#! Convergence order : dt
#!========================================

dat = np.fromfile(pm_causal['path']+pm_causal['conn_file'], dtype=float).reshape(10,10)
neu_pairs = np.array(np.nonzero(dat))
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
fig, axes = plt.subplots(4,5,figsize=(20,17), gridspec_kw=dict(hspace=0.4, wspace=0.2))
for neu_pair, ax in zip(neu_pairs.T, axes.flatten()):
    ax, _, _= Convergence_dt(ax, data_scan_dt, neu_pair)
    ax.set_title(r'%d$\rightarrow$%d'%tuple(neu_pair))

[axi.set_xlabel('$\Delta t$',fontsize=18) for axi in axes[-1,:]]
[axi.set_ylabel('remainder',fontsize=18) for axi in axes[:,0]]
fig.savefig(fig_path/f"scan_dt_all_pairs_K={pm_causal['order'][0]:d}-{pm_causal['order'][1]:d}.pdf")

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
fig, ax = plt.subplots(1,1,figsize=(7,6))
sample_pair = [1,2]
ax, dt, residues = Convergence_dt(ax, data_scan_dt, sample_pair)
ax.set_xlabel('$\Delta t$',fontsize=18)
ax.set_ylabel('remainder',fontsize=18)
ax.set_xlim(-2,0)
ax.set_ylim(-11,-5)
ax.legend(fontsize=18,loc='lower right')
fig.tight_layout()
fig.savefig(fig_path/f"scan_dt_pair_{sample_pair[0]:d}-{sample_pair[1]:d}_K={pm_causal['order'][0]:d}-{pm_causal['order'][1]:d}.pdf")
np.savez(data_path/f"HH10_K={pm_causal['order'][0]:d}-{pm_causal['order'][1]:d}_dt.npz", dt=dt, res=residues)

# %%
#!========================================
#! Independence between Delta_p and dt
#!========================================
sample_pair = [4,8]
data_plot = data_scan_dt[(data_scan_dt['pre_id']==sample_pair[0]) & (data_scan_dt['post_id']==sample_pair[1])].copy().sort_values(by='dt')
fig, ax = plt.subplots(figsize=(5,4))
ax.plot(np.log10(data_plot['dt']), np.log10(data_plot['Delta_p']), '-o')
ax.set_ylim(-2,0)
ax.set_xlim(-2,0)
ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
ax.xaxis.set_major_locator(MaxNLocator(2, integer=True))
ax.xaxis.set_major_formatter(sci_formatter)
ax.yaxis.set_major_formatter(sci_formatter)
ax.set_xlabel(r'$\Delta t$ (ms)')
ax.set_ylabel(r'$\Delta p_m$')
np.savez(data_path/f"HH10_K={pm_causal['order'][0]:d}-{pm_causal['order'][1]:d}_dt-Deltap.npz", dt=dts, res=residues)
# %%
#!========================================
#! Independence between dp and dt
#!========================================
sample_pair = [4,8]
data_plot = data_scan_dt[(data_scan_dt['pre_id']==sample_pair[0]) & (data_scan_dt['post_id']==sample_pair[1])].copy().sort_values(by='dt')
fig, ax = plt.subplots(figsize=(5,4))
ax.plot(data_plot['dt'], data_plot['dp1'], '-o')
ax.set_ylim(0)
ax.set_xlim(0)
ax.ticklabel_format(style='sci', scilimits=(0,0),)
ax.set_xlabel(r'$\Delta t$ (ms)')
ax.set_ylabel(r'$\delta p_{Y\to X}$')
np.savez(data_path/f"HH10_K={pm_causal['order'][0]:d}-{pm_causal['order'][1]:d}_dt-dp.npz", dt=dts, res=residues)
plt.tight_layout()
fig.savefig(fig_path/'HH10_dp-dt.pdf')
# %%
#!========================================
#! Convergence order : dp
#!========================================
dat = np.fromfile(pm_causal['path']+pm_causal['conn_file'], dtype=float).reshape(10,10)
neu_pairs = np.array(np.nonzero(dat))

plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
fig, axes = plt.subplots(4,5,figsize=(20,17), gridspec_kw=dict(hspace=0.4, wspace=0.2))
for neu_pair, ax in zip(neu_pairs.T, axes.flatten()):
    ax,_,_ = Convergence_dp(ax, data_scan_dp, neu_pair)
    ax.set_title(r'%d$\rightarrow$%d'%tuple(neu_pair))

[axi.set_xlabel('$\Delta p_m$',fontsize=18) for axi in axes[-1,:]]
[axi.set_ylabel('remainder',fontsize=18) for axi in axes[:,0]]
fig.savefig(fig_path/f"scan_dp_all_pairs_K={pm_causal['order'][0]:d}-{pm_causal['order'][1]:d}.pdf")

plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
fig, ax = plt.subplots(1,1,figsize=(7,6))
sample_pair=(5,4)
ax, dp, residues = Convergence_dp(ax, data_scan_dp, sample_pair)
ax.set_xlabel('$\Delta p_m$',fontsize=18)
ax.set_ylabel('remainder',fontsize=18)
ax.legend(loc='lower right', fontsize=14)
fig.tight_layout()
fig.savefig(fig_path/f"scan_dp_pair_{sample_pair[0]:d}-{sample_pair[1]:d}_K={pm_causal['order'][0]:d}-{pm_causal['order'][1]:d}.pdf")
np.savez(data_path/f"HH10_K={pm_causal['order'][0]:d}-{pm_causal['order'][1]:d}_dp.npz", dp=dp, res=residues)

# %%[markdown]
# Figure 3A-3B in maintext
#%%
#!====================
#! Config
#!====================
pm_causal = dict(
        N         = 10,
        T         = 1e8,  # ms,
        DT        = 2e5,  # ms,
        dt        = 0.5,   #  ms,
        order     = (1,1), # x, y,
        delay     = 3,   # ms,
        spk_fname = 'HHp=0.25s=0.020f=0.100u=0.100',
        conn_file = 'connect_matrix-p=0.250.dat',
        path      = '../HH/data/N=10_scan_pm/',
    )
l_order = np.arange(1, 29)
k_order = np.arange(1, 29)
#%%
#!====================
#! Calculate causal values
#!====================
L = [(1, i) for i in l_order]
K = [(i, 1) for i in k_order[1:]]
data_scan = []
estimator = CausalityEstimator(**pm_causal, n_thread=60)
for l in L:
    estimator.order = l
    buffer = estimator.fetch_data(new_run=True)
    buffer = buffer[['TE', 'pre_id', 'post_id', 'Delta_p', 'GC', 'sum(MI)', 'sum(CC2)']]
    buffer['k'] = l[0]
    buffer['l'] = l[1]
    data_scan.append(buffer)

for k in K:
    estimator.order = k
    buffer = estimator.fetch_data(new_run=True)
    buffer = buffer[['TE', 'pre_id', 'post_id', 'Delta_p', 'GC', 'sum(MI)', 'sum(CC2)']]
    buffer['k'] = k[0]
    buffer['l'] = k[1]
    data_scan.append(buffer)

data_scan = pd.concat(data_scan)
data_scan.to_pickle(data_path/'HH10_scan_kl.pkl')
# %%
#!====================
#! Plotting figure
#!====================
dat = np.fromfile(pm_causal['path']+pm_causal['conn_file'], dtype=float).reshape(10,10)
node_pairs = np.array(np.nonzero(dat))

plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
fig, axes = plt.subplots(4,5,figsize=(20,16), gridspec_kw=dict(
    hspace=0.4, wspace=0.2, left=0.08, right=0.95, bottom=0.08, top=0.95))
key_map = {'TE':'TE', 'MI':'sum(MI)', 'GC':'GC', 'CC':'sum(CC2)'}
# significant levels
threshold = data_scan[(data_scan['pre_id']==1) & (data_scan['post_id']==0) & data_scan['k']==1].copy().sort_values(by='l')['sum(CC2)']
for pair, axi in zip(node_pairs.T, axes.flatten()):
    buff = data_scan[(data_scan['pre_id']==pair[0]) & (data_scan['post_id']==pair[1]) & data_scan['k']==1].copy().sort_values(by='l')
    for key in ('CC', 'MI', 'GC', 'TE'):
        if key in ('TE', 'MI'):
            buff[key_map[key]] *= 2 
        axi.plot(buff['l'], buff[key_map[key]], '-', **line_rc[key], alpha=1)
    axi.plot(buff['l'], threshold, ls='--', lw=3, c='grey',label='threshold', clip_on=False)
    axi.grid()
    axi.set_xlim(0,25)
    axi.set_ylim(0,2.65e-5)
    axi.set_title(f'{pair[0]:d} '+r'$\rightarrow$'+f' {pair[1]:d}', fontsize=30)
    axi.legend(fontsize=10)

[axi.set_xlabel(r'order $l$', fontsize=30) for axi in axes[-1,:]]
[axi.set_ylabel('causal value', fontsize=30) for axi in axes[:,0]]
plt.savefig(fig_path/'HH10_scan_l.pdf')
#%%
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
fig, axes = plt.subplots(4,5,figsize=(20,16), gridspec_kw=dict(
    hspace=0.4, wspace=0.2, left=0.08, right=0.95, bottom=0.08, top=0.95))
key_map = {'TE':'TE', 'MI':'sum(MI)', 'GC':'GC', 'CC':'sum(CC2)'}
# significant levels
threshold = data_scan[(data_scan['pre_id']==1) & (data_scan['post_id']==0) & (data_scan['l']==1)].copy().sort_values(by='k')['sum(CC2)']
for pair, axi in zip(node_pairs.T, axes.flatten()):
    buff = data_scan[(data_scan['pre_id']==pair[0]) & (data_scan['post_id']==pair[1]) & (data_scan['l']==1)].copy().sort_values(by='k')
    for key in ('CC', 'MI', 'GC', 'TE'):
        if key in ('TE', 'MI'):
            buff[key_map[key]] *= 2 
        axi.plot(buff['k'], buff[key_map[key]], '-', **line_rc[key], alpha=1)
    axi.plot(buff['k'], threshold, ls='--', lw=3, c='grey',label='threshold', clip_on=False)
    axi.grid()
    axi.set_xlim(0,25)
    axi.set_ylim(0,6e-6)
    axi.set_title(f'{pair[0]:d} '+r'$\rightarrow$'+f' {pair[1]:d}', fontsize=30)
    axi.legend(fontsize=10)


[axi.set_xlabel(r'order $k$', fontsize=30) for axi in axes[-1,:]]
[axi.set_ylabel('causal value', fontsize=30) for axi in axes[:,0]]
plt.savefig(fig_path/'HH10_scan_k.pdf')
# %%[markdown]
# Figure 3C in maintext
#%%
#!====================
#! Config
#!====================
pm_causal = dict(
        N         = 10,
        T         = 1e8,  # ms,
        DT        = 2e5,  # ms,
        dt        = 0.5,   #  ms,
        order     = (1,1), # x, y,
        # delay   = 3,   # ms,
        spk_fname = 'HHp=0.25s=0.020f=0.100u=0.100',
        conn_file = 'connect_matrix-p=0.250.dat',
        path      = '../HH/data/N=10_scan_pm/',
    )
delay = np.arange(81)*0.5
l_order = [1,5,10,20]
#%%
#!====================
#! Calculate causal values
#!====================
estimator = CausalityEstimator(**pm_causal, n_thread=60)
for l in l_order:
    estimator.order = (1, l)
    print(estimator.get_optimal_delay(delay))
#%%
data_scan = []
estimator = CausalityEstimator(**pm_causal, n_thread=60)
for l in l_order:
    estimator.order = (1,l)
    for d_ in delay:
        buffer = estimator.fetch_data(d_, new_run=True)
        buffer = buffer[['TE', 'pre_id', 'post_id', 'Delta_p', 'GC', 'sum(MI)', 'sum(CC2)']]
        buffer['k'] = 1
        buffer['l'] = l
        buffer['delay'] = d_
        data_scan.append(buffer)

data_scan = pd.concat(data_scan)
# %%
#!====================
#! Plotting scan_delay_figure across different pairs
#!====================
from causal4.utils import match_features
fig_data=pd.DataFrame({"l":[], "delay":[], "CC":[], "MI":[], "GC":[], "TE":[]})
fig, ax_right = plt.subplots(1,4, figsize=(16,4.5))
data_scan = match_features(data_scan, N=10, conn_file=pm_causal['path']+pm_causal['conn_file'])
delay = np.arange(81)*0.5
delay_full = np.hstack((-np.flip(delay[1:]), delay))
for idx, l in enumerate((1, 5, 10, 20)):
    for pre_id, post_id in ((0,1), (0,2)):
        df_buff = {"l":l, "pre_id": pre_id, "post_id": post_id}
        for key in ('CC', 'MI', 'GC', 'TE',):
            cau_val1 = data_scan[(data_scan['pre_id']==pre_id)
                                & (data_scan['post_id']==post_id)
                                & (data_scan['l']==l)].copy().sort_values(by='delay')[key_map[key]].to_numpy()
            cau_val2 = data_scan[(data_scan['pre_id']==post_id)
                                & (data_scan['post_id']==pre_id)
                                & (data_scan['l']==l)].copy().sort_values(by='delay')[key_map[key]].to_numpy()
            cau_val = np.hstack((np.flip(cau_val2[1:]), cau_val1))
            if key in ('TE', 'MI'):
                cau_val *= 2
            ax_right[idx].plot(delay_full, cau_val, **line_rc[key])
            df_buff['delay'] = delay_full
            df_buff[key] = cau_val
        ax_right[idx].set_ylim(0)
        ax_right[idx].set_xlim(-20,20)
        ax_right[idx].set_xticks([-20, -10, 0, 10, 20])
        # ax_right[idx].set_yticks([1e-5, 2e-5, 3e-5])
        ax_right[idx].set_title(r'$l=%d$'%l)
        ax_right[idx].axvline(0, ls='--', color='grey')
        ax_right[idx].set_xlabel('Delay (ms)')
        ax_right[idx].legend(fontsize=13, loc='upper left')
        fig_data=fig_data.append(df_buff, ignore_index=True)
ax_right[0].set_ylabel('causal Value')
ax_right[-1].set_xticks([0, 10, 20])

plt.tight_layout()
plt.savefig(fig_path/'HH_scan_delay_multi-l.pdf')
fig_data.to_pickle(data_path/'HH10_scan_delay.pkl')

# %%[markdown]
# Figure 3D in maintext
# %%
plt.rcParams['font.size']=20
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
pm_causal = dict(
        N         = 10,
        T         = 1e8,  # ms,
        DT        = 2e5,  # ms,
        dt        = 0.5,   #  ms,
        order     = (1,1), # x, y,
        delay     = 3,   # ms,
        spk_fname = 'HHp=0.25s=0.020f=0.100u=0.100',
        conn_file = 'connect_matrix-p=0.250.dat',
        path      = '../HH/data/N=10_scan_pm/',
    )
N = 31
s_list = np.arange(N)*1e-3
#%%
estimator = CausalityEstimator(**pm_causal, n_thread=60)
data_scan = []
for s in s_list:
    estimator.spk_fname = f"HHp=0.25s={s:.3f}f=0.100u=0.100"
    buff = estimator.fetch_data(new_run=True)
    buff['s'] = s
    data_scan.append(buff)
data_scan = pd.concat(data_scan)
#%%
data_plot = data_scan[(data_scan['pre_id']==0) & (data_scan['post_id']==1)].copy().sort_values('s')
fig, ax = plt.subplots(1,1, figsize=(6,5))
for key in ('CC', 'MI', 'GC', 'TE'):
    if key in ('MI', 'TE'):
        data_plot[key_map[key]] *=2
    ax.plot(data_plot['s'], data_plot[key_map[key]], ls='-',  **line_rc[key], clip_on=False)
data_threshold = data_scan[(data_scan['pre_id']==0) & (data_scan['post_id']==2)].copy().sort_values('s')
ax.plot(data_threshold['s'], data_threshold[key_map['CC']], ls='--', lw = 4, color='grey', label='threshold', clip_on=False)
ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
ax.legend(fontsize=14)
ax.set_xlim(0, 0.03)
ax.set_xticks([0,0.01, 0.02, 0.03])
ax.set_ylim(0, 1.2e-5)
ax.set_yticks([0,5e-6, 1e-5])
ax.set_xlabel(r'$S$')
ax.set_ylabel('causal Value')
plt.tight_layout()
fig.savefig(fig_path/'HH_scan_S.pdf')
#%%
fig, ax = plt.subplots(1,1, figsize=(6,5))
from scipy.optimize import curve_fit
func = lambda x, a: a*x
pval,_ = curve_fit(func, data_plot['s'], data_plot['dp1'], p0=[0,])
ax.plot(data_plot['s'], func(data_plot['s'], pval), 'k')
ax.plot(data_plot['s'][::2], data_plot['dp1'][::2], 'o', color='#3F47BB', ms=10, clip_on=False)
ax.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
ax.set_xlim(0, 0.03)
ax.set_xticks([0,0.01, 0.02, 0.03])
ax.set_ylim(0, 3.5e-3)
ax.set_yticks([0,1e-3, 2e-3, 3e-3])
ax.set_xlabel(r'$S$')
ax.set_ylabel(r'$\delta p$')
plt.tight_layout()
fig.savefig(fig_path/'HH_dp_scan_S.pdf')
data_scan.to_pickle(data_path/'HH10_scan_s_dp.pkl')


# %%
