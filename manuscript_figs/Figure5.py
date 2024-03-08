# author: Kai Chen
# %%
import pickle
from causal4.figrc import *
from causal4.utils import Gaussian, Double_Gaussian
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] =False
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator
@FuncFormatter
def sci_formatter(x, pos):
    return r'$10^{%d}$'%x
import pandas as pd

#%%
fig = plt.figure(figsize=(24, 9.5), dpi=300,)
gs = fig.add_gridspec(2,1, hspace=0.35, left=0.05, right=0.21, bottom=0.08, top=0.95)
ax1 = np.array([fig.add_subplot(gsi) for gsi in gs])

# density fit illustration HH
with open(data_path/'HH100_conn_types_data_fig.pkl', 'rb') as f:
    data = pd.DataFrame(pickle.load(f)['HH100-LN'])
    TE = data.loc['TE']
    edges= TE['edges']
    nte  = TE['hist']
    nte1 = TE['hist_conn']
    nte2 = TE['hist_disconn']
    pval = TE['log_norm_fit_pval']
    fte = Double_Gaussian(edges, *pval)
    fte1 = Gaussian(edges, pval[1], pval[3], pval[5])
    fte2 = Gaussian(edges, pval[0], pval[2], pval[4])
    opt_th=TE['th_gauss']
	# connectivity data
    conn = data.loc['conn']
    bins = conn['edges']
    counts = conn['hist']


# data=loadmat(path+'FitDensity_HH_100P30nu0.09', squeeze_me=True)

ax1[0].plot(edges,nte,color=(65/255,65/255,194/255),lw=2,label='simulate')
ax1[0].plot(edges,fte,color='r',lw=2,label='fit')
ax1[0].set_xlabel('TE value')
ax1[0].set_ylabel('probability density')
ax1[0].xaxis.set_major_formatter(sci_formatter)
xticks=[-9,-6,-3,0]
ax1[0].set_xticks(xticks)
# ax1[0].set_yticks(np.arange(5))
ax1[0].set_xlim(xticks[0],xticks[-1])
ax1[0].set_ylim(0,0.8)
ax1[0].legend(fontsize=16,loc="upper right", bbox_to_anchor=(1.05,1))

axins = inset_axes(ax1[0], width="100%", height="100%",
                   bbox_to_anchor=(.55, .25, .5, .35),
                   bbox_transform=ax1[0].transAxes, loc='center')
axins.spines['left'].set_visible(False)

axins.bar(bins, counts, width=bins[1]-bins[0], align='edge')
axins.set_xticks([-2,-1])
axins.set_xticklabels(axins.get_xticks(), fontsize=12)
axins.set_title('Log-Normal', fontsize=14)
axins.set_xlabel(r'$\mathrm{mS\cdot cm}^{-2}$', fontsize=12)
axins.set_yticks([])
axins.xaxis.set_major_formatter(sci_formatter)

##two gaussions
ax1[1].plot(edges,nte1,color='r',lw=2,label='connected')
ax1[1].plot(edges,fte1,color=(65/255,65/255,194/255),lw=2,label='fit connected')
ax1[1].plot(edges,nte2,color='orange',lw=2,label='unconnected')
ax1[1].plot(edges,fte2,color='darkgreen',lw=2,label='fit unconnected')

ax1[1].axvline(opt_th, ymax=1, color='k', ls='-', lw=2,label='threshold')
ax1[1].xaxis.set_major_locator(MaxNLocator(4, integer=True))
ax1[1].xaxis.set_major_formatter(sci_formatter)

ax1[1].set_xlabel('TE value')
ax1[1].set_ylabel('probability density')
ax1[1].set_xticks(xticks)
# ax1[1].set_yticks(np.arange(5))
ax1[1].set_xlim(xticks[0], xticks[-1])
ax1[1].set_ylim(0,0.8)
ax1[1].legend(fontsize=16,loc="upper right", bbox_to_anchor=(1.05,1))

# ROC curve
stimulus_names = (
	'drifting_gratings-with-gray',
	'static_gratings',
	'natural_scenes',
	'natural_movie',
)

with open(data_path/'allen_data.pkl', 'rb') as f:
    data = pickle.load(f)

gs = fig.add_gridspec(2,2, wspace=0.48, hspace=0.35, left=0.29, right=0.70, bottom=0.08, top=0.95)
ax2 = np.array([fig.add_subplot(gsi) for gsi in gs]).reshape(2,2)

for axi, fname in zip(ax2.T.flatten(), stimulus_names):
	roc=data[fname]['roc']

	##roc
	for key in ('CC', 'MI', 'GC', 'TE'):
		axi.plot(roc[key][0,:], roc[key][1,:], **line_rc[key])[0].set_clip_on(False)
	axi=roc_formatter(axi)
	axi.set_ylabel(axi.get_ylabel(), y=0.4)
	axi.legend(fontsize=13,loc='upper right')

# inset

for axi, fname in zip(ax2.T.flatten(), stimulus_names):
	hist=data[fname]['hist']
	axins = inset_axes(axi, width="55%", height="66%",
                   bbox_to_anchor=(.28, .2, .85, .85),
                   bbox_transform=axi.transAxes, loc=3)
	# axins.set_facecolor(None)

	for key in ('CC', 'MI', 'GC', 'TE'):
		axins.plot(data['edges'], hist[key], **line_rc[key])
	axins.set_xlabel('causal value',fontsize=12)
	axins.set_ylabel('probability density',fontsize=12)
	axins.set_yticks(np.arange(0,0.7,0.2))
	axins.set_yticklabels(['0','0.2','0.4','0.6'],fontsize=12)
	axins.set_xticks(np.arange(-8,-1, 2)) #fontsize=14)
	axins.set_xticklabels([r"$10^{%.0f}$"%val for val in axins.get_xticks()], fontsize=12)
	axins.set_xlim(-8,-2)
	axins.set_ylim(0)

# add stimuli figure
for axi, fname in zip(ax2.T.flatten(), stimulus_names):
    if '-' in fname:
        arr_image = plt.imread(fig_path/(fname.split('-')[0]+'.png'), format='png')
    else:
        arr_image = plt.imread(fig_path/(fname+'.png'), format='png')
    axins = inset_axes(axi, width="75%", height="75%",
                   bbox_to_anchor=(-.33, .375, .3, 1.3),
                   bbox_transform=axi.transAxes, loc=3)

    axins.imshow(arr_image)
    axins.axis('off')

#! Draw the heatmap of correlation coefficient matrix
gs = fig.add_gridspec(2,1, left=0.78, right=0.98, bottom=0.25, top=0.87, height_ratios=[1,10], hspace=0.05)
ax_cb = fig.add_subplot(gs[0])
ax3 = fig.add_subplot(gs[1])

key = 'TE'
heatmap_data = data['consistency']['TE']
mask = np.triu(np.ones_like(heatmap_data, dtype=bool),k=1)
ax3 = sns.heatmap(heatmap_data, mask=mask,
	vmin=0, vmax=1, 
	cmap=plt.cm.OrRd, 
	cbar_kws={'ticks':[0,0.5,1], "orientation": "horizontal"},
	square=True,
	lw=.5,
	ax=ax3,
	cbar_ax=ax_cb,
	annot=True,
	annot_kws={"fontsize":20}
	)

ax_cb.xaxis.set_ticks_position('top')
ax3.set_xticklabels([])
ax3.set_yticklabels([])

length = 1./heatmap_data.shape[0]-0.01
# draw x-axis
for i in range(len(stimulus_names)):
    if '-' in stimulus_names[i]:
        arr_image = plt.imread(fig_path/(stimulus_names[i].split('-')[0]+'.png'), format='png')
    else:
        arr_image = plt.imread(fig_path/(stimulus_names[i]+'.png'), format='png')
    axins = inset_axes(ax3, width="100%", height="100%",
                    bbox_to_anchor=(.005+i*(length+0.01), -length-0.01, length, length),
                    bbox_transform=ax3.transAxes, loc='center')

    axins.imshow(arr_image)
    axins.axis('off')

# draw y-axis
for i in range(len(stimulus_names)):
    if '-' in stimulus_names[i]:
        arr_image = plt.imread(fig_path/(stimulus_names[i].split('-')[0]+'.png'), format='png')
    else:
        arr_image = plt.imread(fig_path/(stimulus_names[i]+'.png'), format='png')
    axins = inset_axes(ax3, width="100%", height="100%",
                    bbox_to_anchor=(-length-0.01, 1-length-0.005-i*(length+0.01), length, length),
                    bbox_transform=ax3.transAxes, loc='center')

    axins.imshow(arr_image)
    axins.axis('off')

print(f"Minimum coincidence rate : {heatmap_data.min():6.3f}")
print(f"Maximum coincidence rate : {np.sort(np.unique(heatmap_data))[-2]:6.3f}")
print(heatmap_data)

for axi, letter in zip(ax1, 'AB'):
	axi.text(-0.15,1.05,letter,fontsize=24,weight='normal', transform=axi.transAxes)
for axi, letter in zip(ax2.T.flatten(), 'CDEF'):
	axi.text(-0.38,1.05,letter,fontsize=24,weight='normal', transform=axi.transAxes)
ax3.text(-0.3,1.18,'G',fontsize=24,weight='normal', transform=ax3.transAxes)

plt.savefig(fig_path/'fig5_Allen_density_roc_illus_horizon.pdf',dpi=300, transparent=True)

# %%
# AUCs: CC, MI, GC, TE
# print(f'{np.mean([0.942,0.958,0.942,0.957]):.4f}')
# print(f'{np.mean([0.948,0.955,0.948,0.954]):.4f}')
# print(f'{np.mean([0.938,0.966,0.940,0.966]):.4f}')
# print(f'{np.mean([0.955,0.978,0.957,0.977]):.4f}')
