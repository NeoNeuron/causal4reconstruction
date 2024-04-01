# %%
from causal4.figrc import *
from causal4.utils import Linear_R2
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
key_pairs, mk, color, alphas = get_conv_config()

#%% convergence test dt & dp
fig, ax = plt.subplots(1, 2, figsize=(12.5,4.5),
                       subplot_kw=dict(xscale='log', yscale='log'),
                       gridspec_kw=dict(wspace =0.5, hspace =0))

data=np.load(data_path/'HH10_K=5-5_dt.npz', allow_pickle=True)
dt=data['dt']
res=data['res']

for i in range(4):
    ax[0].plot(dt, res[i,:], 
        ls='', c=color[i], marker=mk[i], ms=12, alpha=alphas[i], 
        label='-'.join([line_rc[key]['label'] for key in key_pairs[i]]))
ax[0].plot(dt, (dt/dt[0])**2*res[0,0],'k--', alpha=1, lw=2)
ax[0].plot(dt, (dt/dt[0])**3*res[1,0],'k-',  alpha=1, lw=2)
ax[0].set_xlabel('$\Delta t$')
ax[0].set_ylabel('remainder')
ax[0].set_yticks([1e-11,1e-9,1e-7,1e-5])
ax[0].set_xlim(1e-2,1)
ax[0].set_ylim(1e-11,1e-5)
ax[0].legend(fontsize=14,loc=(0.5,0.05))

# print fit R2
print('R^2:')
for i, name in enumerate(('dmi','dgc', 'dte', 'dtegc')):
    pval = np.polyfit(np.log10(dt), np.log10(res[i,:]), deg=1)
    print(name+' '+r'%6.3f'%Linear_R2(np.log10(dt), np.log10(res[i,:]), pval))

data = np.load(data_path/'HH10_K=5-5_dp.npz', allow_pickle=True)
dp=data['dp'][3:]
res=data['res'][:,3:]
print(dp)

for i in range(4):
    ax[1].plot(dp, res[i,:], 
        ls='', c=color[i], marker=mk[i], ms=12, alpha=alphas[i], 
        label='-'.join([line_rc[key]['label'] for key in key_pairs[i]]))
ax[1].plot(dp, (dp/dp[0])**3*res[0,0],'k-',  alpha=1, lw=2)
ax[1].plot(dp, (dp/dp[0])**2*res[1,0],'k--', alpha=1, lw=2)

ax[1].set_xlabel('$\Delta p_m$')
ax[1].set_ylabel('remainder')
ax[1].set_yticks([1e-11,1e-9,1e-7,1e-5])
ax[1].set_xlim(5e-3,1)
ax[1].set_ylim(1e-11,1.2e-5)
ax[1].legend(fontsize=14,loc=(0.5,0.05))

## set y ticks and label indices
for i, letter in enumerate('AB'):
    add_log_minor_tick(ax[i], 'y')
    ax[i].text(-0.2, 1.05, letter,fontsize=24,weight='normal', transform=ax[i].transAxes)
#plt.savefig(path_out+'conv10.eps',format='eps',bbox_inches = 'tight')
plt.savefig(fig_path/'fig2_conv10.pdf',bbox_inches = 'tight',dpi=300, transparent=True)

# print fit R2
print('R^2:')
for i, name in enumerate(('dmi','dgc', 'dte', 'dtegc')):
    pval = np.polyfit(np.log10(dp), np.log10(res[i,:]), deg=1)
    print(name+' '+r'%6.3f'%Linear_R2(np.log10(dp), np.log10(res[i,:]), pval))

# %%