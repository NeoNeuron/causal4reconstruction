# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
@FuncFormatter
def sci_formatter(x, pos):
    return r'$10^{%.1f}$'%x
plt.rcParams['axes.spines.top']=False
plt.rcParams['axes.spines.right']=False
from causal4.Causality import CausalityEstimator
import os
REPO_PATH = os.path.dirname(__file__)
from causal4.figrc import *
from causal4.utils import *
import causal4.myplot as mplt
import pickle
import yaml

with open(REPO_PATH+'simulation_parameters_PNAS.yaml', 'r') as yamlfile:
    configs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print(configs)
# %%
# create causality estimators
estimators = {key:CausalityEstimator(**val, n_thread=60) for key, val in configs.items()}
#%%
conn_types = ['HH100', 'HH100-G', 'HH100-U', 'HH100-E', 'HH100-LN',]# 'HH100-LN-Song2005']
fitting_pms = {
    'HH100':    [(-10, -4), (0.5, 0.5, -5.8, -4.9, 1, 1)],
    'HH100-G':  [(-10, -4), (0.5, 0.5, -5.8, -4.9, 1, 1)],
    'HH100-U':  [(-10, -3), (0.5, 0.5, -7.0, -4.3, 1, 1)],
    'HH100-E':  [(-10, -2), (0.5, 0.5, -5.8, -4.5, 1, 1)],
    'HH100-LN': [(-10, -2), (0.5, 0.5, -6.5, -4.9, 1, 1)],
    'HH100-LN-Song2005': [(-7, -3), (0.5, 0.5, -5, -4.3, 1, 1)],
    }
#%%
data_recon = {}
data_fig = {}
for key, estimator in estimators.items():
    print(estimator.causality_fname())
    cfg = configs[key]
    data = estimator.fetch_data()
    data_matched = match_features(data, N=cfg['N'],
                                       conn_file=cfg['path']+cfg['conn_file'])
    if key in conn_types:
        weight_hist_type = 'log' if 'LN' in key else 'linear'
        data_recon[key], data_fig[key] = reconstruction_analysis(
            data_matched, hist_range=fitting_pms[key][0], fit_p0=fitting_pms[key][1],
            weight_hist_type=weight_hist_type)
with open(data_path/'HH100_conn_types_data_recon.pkl', 'wb') as f:
    pickle.dump(data_recon, f)
with open(data_path/'HH100_conn_types_data_fig.pkl', 'wb') as f:
    pickle.dump(data_fig, f)
# %% ========================================
# plot raster
for key, cfg in configs.items():
    fig = mplt.plot_raster(cfg);
# %% ========================================
#! Blind test for HH100 networks
# ====================
# Draw histogram of causal values
# ====================
import pandas as pd
with open(data_path/'HH100_conn_types_data_fig.pkl', 'rb') as f:
    data = pickle.load(f)

fig, ax = plt.subplots(2,3, figsize=(23,14))
for conn, axi in zip(conn_types, ax.flatten()):
    data_ = pd.DataFrame(data[conn])
    if conn != 'HH100':
        _, ax_hist, axins, _ = mplt.ReconstructionFigure(data_, True, False, axi)
    else:
        _, ax_hist, axins, _ = mplt.ReconstructionFigure(data_, False, False, axi)
    # plot the distribution of connectivity weight
    if axins is not None:
        if 'LN' in conn:
            axins.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            axins.xaxis.set_major_formatter(sci_formatter)
        axins.set_title('Structural Conn', fontsize=14)

plt.tight_layout()
plt.savefig(fig_path/"histogram_of_HH-all.pdf", transparent=True)

# %%
