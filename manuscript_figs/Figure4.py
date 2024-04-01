# %%
from causal4.figrc import *
import pandas as pd
import pickle
from matplotlib.ticker import FuncFormatter
@FuncFormatter
def sci_formatter(x, pos):
    return r'$10^{%d}$'%x

#%% Density HH N=100 & N=20
fig,ax=plt.subplots(1,2,figsize=(13,4.5))
plt.subplots_adjust(wspace=0.4, hspace=0)

with open(data_path/'HH100_conn_types_data_fig.pkl', 'rb') as f:
    data_fig = pd.DataFrame(pickle.load(f)['HH100'])
    edges= data_fig['edges']
    counts  = data_fig['hist']
for key in ('CC', 'MI', 'GC', 'TE'):
	ax[0].plot(edges[key], counts[key], **line_rc[key])
ax[0].set_xlabel('causal value')
ax[0].set_ylabel('probability density')
ax[0].legend(fontsize=14,loc='upper left')
ax[0].set_xlim(-10,-4)
ax[0].set_ylim(0,1.2)
ax[0].xaxis.set_major_formatter(sci_formatter)


with open(data_path/'HH100_conn_types_data_recon.pkl', 'rb') as f:
    data_recon = pd.DataFrame(pickle.load(f)['HH100'])
data_recon = data_recon[(data_recon['pre_id']<20)&(data_recon['post_id']<20)].copy()
for key in ('CC', 'MI', 'GC', 'TE'):
	data_buff = data_recon['log-'+key]
	counts, edges = np.histogram(data_buff, bins=50, range=(-10,-4), density=True)
	ax[1].plot(edges[:-1], counts, **line_rc[key])
ax[1].set_xlabel('causal value')
ax[1].set_ylabel('probability density')
ax[1].legend(fontsize=14,loc='upper left')
ax[1].set_yticks(np.arange(0,1.3,0.2))
ax[1].set_xlim(-10,-4)
ax[1].set_ylim(0,1.2)
ax[1].xaxis.set_major_formatter(sci_formatter)

for axi, letter in zip(ax, 'AB'):
	axi.text(-0.15,1.05,letter,fontsize=24,weight='normal', transform=axi.transAxes)
fig.savefig(fig_path/'fig4_Density_HH_100.pdf',bbox_inches='tight', transparent=True)
