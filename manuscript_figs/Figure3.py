# author: Kai Chen
#%%
from causal4.figrc import *
from scipy.optimize import curve_fit
import pandas as pd
key_map = {'TE':'TE', 'MI':'sum(MI)', 'GC':'GC', 'CC':'sum(CC2)'}
#%% scan parameters order k,l,S,delay
data = pd.read_pickle(data_path/'HH10_scan_kl.pkl')

fig,ax=plt.subplots(2,2,figsize=(12,10))
plt.subplots_adjust(hspace=0.5, wspace =0.3)

#scan order k
threshold = data[(data['pre_id']==1) & (data['post_id']==0) & (data['l']==1)].copy().sort_values(by='k')['sum(CC2)']
buff = data[(data['pre_id']==0) & (data['post_id']==1) & (data['l']==1)].copy().sort_values(by='k')
for key in ('CC', 'MI', 'GC', 'TE'):
	if key in ('TE', 'MI'):
		buff[key_map[key]] *= 2 
	ax[0,0].plot(buff['k'], buff[key_map[key]], '-', **line_rc[key], alpha=1)
ax[0,0].plot(buff['k'], threshold, ls='--', lw=3, c='grey',label='Threshold', clip_on=False)

ax[0,0].set_xlim(0,30)
ax[0,0].set_ylim(0,5e-6)
ax[0,0].ticklabel_format(useMathText=True)
ax[0,0].set_xlabel('order $k$')
ax[0,0].set_ylabel('causal value')
ax[0,0].legend(fontsize=14,loc='lower right')
ax[0,0].legend(fontsize=14,loc=(0.58,0.08))

#scan order l
# significant levels
threshold = data[(data['pre_id']==1) & (data['post_id']==0) & data['k']==1].copy().sort_values(by='l')['sum(CC2)']
buff = data[(data['pre_id']==0) & (data['post_id']==1) & data['k']==1].copy().sort_values(by='l')
for key in ('CC', 'MI', 'GC', 'TE'):
	if key in ('TE', 'MI'):
		buff[key_map[key]] *= 2 
	ax[0,1].plot(buff['l'], buff[key_map[key]], '-', **line_rc[key], alpha=1)
ax[0,1].plot(buff['l'], threshold, ls='--', lw=3, c='grey',label='Threshold', clip_on=False)

ax[0,1].set_xlim(0,30)
ax[0,1].set_ylim(0,3e-5)
ax[0,1].ticklabel_format(useMathText=True)
# ax[0,1].set_yticks(list(range(4)))
ax[0,1].set_xlabel('order $l$')
ax[0,1].set_ylabel('causal value')
# ax[0,1].legend(fontsize=14,loc='lower right')
ax[0,1].legend(fontsize=14,loc=(0.58, 0.08))

#scan delay
df=pd.read_pickle(data_path/'HH10_scan_delay.pkl')

mask = (df['l']==1)&(df['pre_id']==0)&(df['post_id']==1)
for key in ('CC', 'MI', 'GC', 'TE',):
	delay_full = df.loc[mask, 'delay'].values[0]
	cau_val = df.loc[mask, key].values[0]
	ax[1,0].plot(delay_full, cau_val, **line_rc[key])
mask = (df['l']==1)&(df['pre_id']==0)&(df['post_id']==2)
delay_full = df.loc[mask, 'delay'].values[0]
cau_val = df.loc[mask, 'TE'].values[0]
ax[1,0].plot(delay_full[np.abs(delay_full)<=20], cau_val[np.abs(delay_full)<=20], 
		 lw=3, ls='--', color='gray', label='Threshold')[0].set_clip_on(False)
ax[1,0].set_xlim(-40,40)
ax[1,0].set_ylim(0,4.05*1e-6)
ax[1,0].set_xticks(np.arange(-4,5,2)*10)
ax[1,0].ticklabel_format(useMathText=True)
ax[1,0].set_xlabel('time delay (ms)')
ax[1,0].set_ylabel('causal value')
ax[1,0].legend(fontsize=14,loc='upper left')

#scan S
df=pd.read_pickle(data_path/'HH10_scan_s_dp.pkl')
# neuron 0->1 is connected, neuron 0->2 is disconnected
threshold = df[(df['pre_id']==0) & (df['post_id']==2)].copy().sort_values('s')
df_plot = df[(df['pre_id']==0) & (df['post_id']==1)].copy().sort_values('s')
causal_mean = (df_plot['TE']*2 + df_plot['sum(MI)']*2 + df_plot['GC'] + df_plot['sum(CC2)'])/4

func = lambda x, a, b: a*x**2+b
pval,_ =curve_fit(func, df_plot['s'][:20], causal_mean[:20],p0=[1e-2,0])

for key in ('CC', 'MI', 'GC', 'TE'):
	if key in ('TE', 'MI'):
		df_plot[key_map[key]] *= 2
	ax[1,1].plot(df_plot['s'][:26], df_plot[key_map[key]][:26], **line_rc[key], clip_on=False)
ax[1,1].plot(df_plot['s'][:26], func(df_plot['s'][:26], pval[0], pval[1]), '-*', lw=1.5, zorder=0, ms=10, label='quadratic fit', color='grey')[0].set_clip_on(False)
ax[1,1].set_xlim(0,2.5e-2)
ax[1,1].set_ylim(0,7.5e-6)
ax[1,1].ticklabel_format(useMathText=True)
ax[1,1].set_xticks([0,0.01,0.02])
ax[1,1].set_xticklabels(['0','0.01','0.02'])
ax[1,1].set_yticks(np.arange(0,7,2)*1e-6)
ax[1,1].set_xlabel('$S$ (mS${\cdot}$cm$^{-2}$)')
ax[1,1].set_ylabel('causal value')
ax[1,1].legend(fontsize=14,loc='upper left')

for axi, letter in zip(ax.flatten(), 'ABCD'):
	axi.text(-0.2,1.08,letter,fontsize=24,weight='normal', transform=axi.transAxes)

plt.savefig(fig_path/'fig3_scan_kl_s_delay10.pdf', bbox_inches='tight', dpi=300, transparent=True) 
# %%
