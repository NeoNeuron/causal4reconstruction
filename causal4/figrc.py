import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.io import loadmat
import networkx as nx
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.labelsize'] = 18

# setup line colors
line_rc = {
    'CC':{'color':'#008000', 'lw':4, 'label':'$\Sigma$TDCC$^2$'},
    'MI':{'color':'#FF0000', 'lw':3, 'label':'$2\Sigma$TDMI'},
    'GC':{'color':'#FFA500', 'lw':3, 'label':'GC'},
    'TE':{'color':'#4141C2', 'lw':2, 'label':'2TE'},
}
c_inv = {
    '#008000': '#CB3800', 
    '#FF0000': '#6D4B9A', 
    '#FFA500': '#00826E', 
    '#4141C2': '#FF4684', 
    '#CB3800': '#008000',
    '#6D4B9A': '#FF0000',
    '#00826E': '#FFA500',
    '#FF4684': '#4141C2',
}

def get_conv_config():
    key_pairs = (('MI', 'CC'), ('GC', 'CC'), ('TE', 'MI'), ('GC', 'TE'))
    mk = ('o', 'o', '+', 'x')
    color=('b', 'orange', 'g', 'r')
    alphas = (0.4, 0.4, 1, 1)
    return key_pairs, mk, color, alphas

def add_log_minor_tick(ax, axis='x'):
    minor = LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
    if axis in ['x', 'both']:
        ax.xaxis.set_minor_locator(minor)
        ax.xaxis.set_minor_formatter(NullFormatter())
    elif axis in ['y', 'both']:
        ax.yaxis.set_minor_locator(minor)
        ax.yaxis.set_minor_formatter(NullFormatter())
    return ax
    

def roc_formatter(ax:plt.Axes)->plt.Axes:
    for line in ax.get_lines():
        # turn off axis clip
        line.set_clip_on(False)
    # set axis lim and label
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([0,0.5,1])
    ax.set_xticklabels(['0','0.5','1'])
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels(['0','0.5','1'])
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')

    # make right and top axis invisible
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    return ax

def net_graph(axins, adjmat:np.ndarray, coord:np.ndarray=None, nodesize:float=1000., fontsize:float=10., nodelabels:np.ndarray=None):
    """Generate network diagram using networkx.

    Args:
        axins (plt.Axes): axis to plot. 
        adjmat (np.ndarray): adjacent matrix of graph.
        coord (np.ndarray, optional): (n, 2) 2-D coordinates of nodes. Defaults to None.
        nodesize (float, optional): Defaults to 1000.
        fontsize (float, optional): Defaults to 10.

    Returns:
        axins: axis to plot.
    """

    G = nx.DiGraph(adjmat)
    if coord is None:
        axis_n = np.ceil(np.sqrt(adjmat.shape[0])).astype(int)
        coord = np.array(np.meshgrid(np.arange(axis_n), np.arange(axis_n))).reshape((2,-1))
        coord = coord[:,:adjmat.shape[0]].T
        
    pos = {n: coordinate for n, coordinate in zip(G,coord)}
    nx.draw_networkx_nodes(
        G, pos=pos, ax=axins,
        node_color='white',
        edgecolors='k',
        node_size=nodesize,
        )

    if nodelabels is None:
        labels = {n:n for n in G}
    else:
        labels = {n:label for n, label in zip(G, nodelabels)}
    nx.draw_networkx_labels(
        G, pos=pos, ax=axins,
        labels=labels,
        font_size=fontsize,
        )
    nx.draw_networkx_edges(
    G, pos=pos, ax=axins,
        width=2,
        node_size=nodesize,
        arrowsize=nodesize/100,
        arrows=True,
        )
    axins.axis('equal')
    axins.axis('off')
    return axins