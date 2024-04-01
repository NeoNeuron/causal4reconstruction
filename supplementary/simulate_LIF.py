# %%
# simulate current based Leaky integrate-and-fire networks using Brian2
# author: Kai Chen

if __name__ == '__main__':
    # %%
    import numpy as np
    import sys, os
    import subprocess as sp
    from causal4.utils import save2bin
    from brian2 import *
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-N",    type=int,   default=2, help="number of neurons", )
    parser.add_argument("-T",    type=float, default=100, help="simulation period, ms", )
    parser.add_argument("--ref", type=float, default=2, help="refractory period", )
    parser.add_argument("-s",    type=float, default=0.4, help="Coupling strength", )
    parser.add_argument("-f",    type=float, default=0.5, help="Poisson strength, mV", )
    parser.add_argument("-u",    type=float, default=0.4, help="Poisson rate, in kHz", )
    parser.add_argument("--recordv", action='store_true', help="save voltage data", )
    parser.add_argument("--verbose", action='store_true', help="show progress bar", )
    parser.add_argument("--suffix", type=str, default='', help="suffix of output spk filename", )
    args = parser.parse_args()

    # progress bar for python mode
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
    
    # progress bar for cpp_standalone mode
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
    N        = args.N
    Vr       = -65.*mV
    theta    = -40.*mV
    tau_v    = 20.*ms
    ref      = args.ref*ms
    duration = args.T*second
    J        = args.s*mV
    ffwd_J   = args.f*mV
    freq     = args.u*kHz
    p        = 0.25

    eqs = """
    dV/dt = -(V-Vr)/tau_v : volt (unless refractory)
    """
    # %%
    start_scope()
    group = NeuronGroup(N, eqs, threshold='V>theta',
                        reset='V=Vr', refractory=ref, method='euler')
    group.V = Vr
    conn = Synapses(group, group, on_pre='V += J')
    np.random.seed(2022)
    conn_mat = np.zeros((N,N))
    conn_mat[~np.eye(N, dtype=bool)] = (np.random.rand(N*N-N)<p).astype(float)
    src, tar = conn_mat.nonzero()
    conn.connect(i=src, j=tar)
    P = PoissonGroup(N, freq, name='ffwd_Poisson')
    ffwd_conn = Synapses(P, group, on_pre='V += ffwd_J')
    ffwd_conn.connect('i==j')

    M = SpikeMonitor(group)
    if args.recordv:
        V = StateMonitor(group, 'V', record=True)

    if args.verbose:
        # run(duration, report=ProgressBar(), report_period=1*second)
        run(duration, report=report_func, report_period=1*second)
    else:
        run(duration)
    try:
        assert M.i.shape[0] > 0
        print('\nMfr : ', M.i.shape[0]*1.0/N/duration)
    except AssertionError:
        print('\nNo spike is generated.')

    spk_data = np.vstack((M.t/ms, M.i)).T
    # %%
    data_out = np.vstack((M.t/ms, M.i)).T
    fname_pfx = f"LIF/data/EE/N={N:d}/" +\
                f"LIFp={p:.2f}s={J/mV:.3f}" +\
                f"ref={ref/ms:.1f}" +\
                f"f={ffwd_J/mV:.3f}u={freq/kHz:.3f}"
    save2bin(fname_pfx+f"_spike_train{args.suffix:s}.dat", data_out, verbose=True)
    save2bin(f"../LIF/data/EE/N={N:d}/connect_matrix-p={p:.3f}.dat", conn_mat)
    if args.recordv:
        data_out = np.vstack((V.t/ms, V.V/mV)).T
        save2bin(fname_pfx+f"_voltage{args.suffix:s}.dat", data_out, verbose=True)

    # %%
    # delete tmp folder
    sp.run(['rm', '-rf', directory])

    
