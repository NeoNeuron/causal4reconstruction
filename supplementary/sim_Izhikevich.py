# Izhikevich 2003
# simulate (exponential) current based Izhikevich networks using Brian2
# author: Kai Chen
# %%
import matplotlib.pyplot as plt
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top']   = False
from causal4.utils import save2bin
import numpy as np
import subprocess as sp
from brian2 import *
import os, sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-N",    type=int,   default=2,   help="number of neurons", )
parser.add_argument("-T",    type=float, default=100, help="simulation period, s", )
parser.add_argument("-s",    type=float, default=0.4, help="Coupling strength", )
parser.add_argument("-f",    type=float, default=1.4, help="Poisson strength, mV", )
parser.add_argument("-u",    type=float, default=0.5, help="Poisson rate, in kHz", )
parser.add_argument("--recordv", action='store_true', help="save voltage data", )
parser.add_argument("--verbose", action='store_true', help="show progress bar", )
parser.add_argument("--suffix", type=str, default='', help="suffix of output spk filename", )
args = parser.parse_args()
#%%
class ProgressBar(object):
    def __init__(self, toolbar_width=40):
        self.toolbar_width = toolbar_width
        self.ticks = 0

    def __call__(self, elapsed, complete, start, duration):
        if complete == 0.0:
            # setup toolbar
            sys.stdout.write("[%s]" % (" " * self.toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (self.toolbar_width + 1)) # return to start of line, after '['
        else:
            ticks_needed = int(round(complete * self.toolbar_width))
            if self.ticks < ticks_needed:
                sys.stdout.write("=" * (ticks_needed-self.ticks))
                sys.stdout.flush()
                self.ticks = ticks_needed
        if complete == 1.0:
            sys.stdout.write("\n")

report_func = '''
        int remaining = (int)((1-completed)/completed*elapsed+0.5);
        if (completed == 0.0)
        {
            std::cout << "Starting simulation at t=" << start << " s for duration " << duration << " s"<<std::flush;
        }
        else
        {
            int barWidth = 70;
            std::cout << "\\r[";
            int pos = barWidth * completed;
            for (int i = 0; i < barWidth; ++i) {
                    if (i < pos) std::cout << "=";
                    else if (i == pos) std::cout << ">";
                    else std::cout << " ";
            }
            std::cout << "] " << int(completed * 100.0) << "% completed. | "<<int(remaining) <<"s remaining"<<std::flush;
        }
    '''
pid = os.getpid()
directory = f"standalone{pid}"
set_device('cpp_standalone', directory=directory) #, build_on_run=False)
# prefs.devices.cpp_standalone.extra_make_args_unix = ['-j1']
#%%
start_scope()
tfinal = args.T * second
N      = args.N
ffwd_J = args.f
freq   = args.u*kHz
tauI   = 5*ms
w      = args.s

eqs = """dv/dt = (0.04*v**2 + 5*v + 140 - u + I)/ms : 1
         du/dt = (a*(b*v - u))/ms  : 1
         dI/dt = -I/tauI : 1
         a : 1
         b : 1
         c : 1
         d : 1
       """

group = NeuronGroup(N, eqs, threshold="v>=30", reset="v=c; u+=d", method="euler")
group.v = -65

spikemon = SpikeMonitor(group)
if args.recordv:
    statemon = StateMonitor(group, 'v', record=True, when='after_thresholds')
group.a = 0.02
group.b = 0.2
group.c = -65
group.d = 8
group.u = 1

p=0.25
S = Synapses(group, group, on_pre="I += w",)
np.random.seed(2022)
conn_mat = np.zeros((N,N))
conn_mat[~np.eye(N, dtype=bool)] = (np.random.rand(N*N-N)<p).astype(float)
src, tar = conn_mat.nonzero()
S.connect(i=src, j=tar)

P = PoissonGroup(N, freq, name='ffwd_Poisson')
ffwd_conn = Synapses(P, group, on_pre='I += ffwd_J')
ffwd_conn.connect('i==j')

if args.verbose:
    # run(tfinal, report=ProgressBar(), report_period=1*second)
    run(tfinal, report=report_func, report_period=1*second)
else:
    run(tfinal)

try:
    assert spikemon.i.shape[0] > 0
    print('\nMfr : ', spikemon.i.shape[0]*1.0/N/tfinal)
except AssertionError:
    print('\nNo spike is generated.')

# %%
data_out = np.vstack((spikemon.t/ms, spikemon.i)).T
fname_pfx = f"Izhikevich/data/EE/N={N:d}/"\
            + f"Izhikevichp={p:.2f}s={w:.3f}"\
            + f"f={ffwd_J:.3f}u={freq/kHz:.3f}"
save2bin(fname_pfx + f"_spike_train.dat", data_out, verbose=True)
save2bin(f"../Izhikevich/data/EE/N={N:d}/connect_matrix-p={p:.3f}.dat", conn_mat)
if args.recordv:
    data_out = np.vstack((statemon.t/ms, statemon.v)).T
    save2bin(fname_pfx + f"_voltage.dat", data_out, verbose=True)
# %%
sp.run(['rm', '-rf', directory])
