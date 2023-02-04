# %%
import numpy as np
import matplotlib.pyplot as plt
from struct import pack, unpack, _clearcache
from pathlib import Path
# %%
def downsample(fname:str, ofname:str, path:str, ra:float):
    """downsample spike train data

    Args:
        fname (str): filename of raw spike train data
        ofname (str): filename of downsampled spike train data
        path (str): path of spike train data
        ra (float): downsample threshold
    """
    binsize = 10
    n_spk_buff = 100_000
    total_spk_counts = 0
    del_spk_counts = 0
    ifpath = Path(path+fname)
    ofpath = Path(path+ofname)
    if ofpath.exists():
        total_spk_counts = int(ifpath.stat().st_size/8/2)
        del_spk_counts = int((ifpath.stat().st_size-ofpath.stat().st_size)/8/2)
        print(f">> {ofpath} already exists ...")
        print(f">> raw size :      {total_spk_counts:12.0f} spikes")
        print(f">> filtered size : {del_spk_counts:12.0f} spikes")
        print(f">> {100*del_spk_counts/total_spk_counts:.2f}% spikes have been deleted!")
        return 0
    of = open(ofpath, 'wb')
    with open(ifpath, 'rb') as f:
        buff_char = f.read(n_spk_buff*8*2)
        buff_len = int(len(buff_char)/8)
        threshold = None
        slices = None
        current_end_edge = 0.0

        while buff_len>0:
            # load data
            spk_buff = np.array(unpack('d'*buff_len, buff_char)).reshape(-1,2)
            tmax = spk_buff[-1,0]
            edges = current_end_edge + \
                np.arange(int((tmax-current_end_edge)/binsize)+1)*binsize
            current_end_edge = edges[-1]
            # histogram exclude (current_end_edge, tmax]

            # calculate temporal histogram
            if slices is not None and isinstance(slices, np.ndarray):
                spk_buff = np.vstack([slices, spk_buff])
            counts, _ = np.histogram(spk_buff[:,0], bins=edges)
            if threshold is None:
                threshold = counts.mean()+ra*counts.std()

            # filter bins and save to file
            sync_bin_idx = counts>threshold
            slices = np.array(np.vsplit(spk_buff, np.cumsum(counts)))
            new_spks = np.vstack(slices[:-1][~sync_bin_idx])
            of.write(pack("d"*new_spks.shape[0]*new_spks.shape[1], *new_spks.flatten()))
            _clearcache()

            # post-process
            slices = slices[-1]
            del_spk_counts += counts[sync_bin_idx].sum()
            total_spk_counts += counts.sum()

            buff_char = f.read(n_spk_buff*8*2)
            buff_len = int(len(buff_char)/8)

        # process last slices
        if slices.shape[0] <= threshold:
            of.write(pack("d"*slices.shape[0]*slices.shape[1], *slices.flatten()))
        else:
            del_spk_counts += slices.shape[0]
        total_spk_counts += slices.shape[0]
    of.close()
    print(f">> {total_spk_counts:d} spikes in total ...")
    print(f">> {del_spk_counts:d} spikes to delete ...")
    print(f">> {100*del_spk_counts/total_spk_counts:.2f}% spikes have been deleted!")
#%%
# * visualize the difference after filtering
def visualize_downsample(path, fname, ofname, n_spks=100_000, xlim=None):
    with open(path+fname, 'rb') as f:
        buff_char = f.read(n_spks*8*2)
        buff_len = int(len(buff_char)/8)
        spk_raw = np.array(unpack('d'*buff_len, buff_char)).reshape(-1,2)
    with open(path+ofname, 'rb') as f:
        buff_char = f.read(n_spks*8*2)
        buff_len = int(len(buff_char)/8)
        spk_filtered = np.array(unpack('d'*buff_len, buff_char)).reshape(-1,2)
    spk_data = {'raw':spk_raw, 'filtered':spk_filtered}

    plt.figure(figsize=(10,4))
    plt.plot(spk_raw[:,0], spk_raw[:,1],'|',color='navy',ms=10, label='raw')
    plt.plot(spk_filtered[:,0], spk_filtered[:,1], 'o', 
         markeredgecolor='orange', markerfacecolor='none',
         label='after downsampling')
    if xlim is not None:
        plt.xlim(*xlim)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuronal Indices')
    plt.legend()
    return spk_data
# %%
if __name__ == '__main__':
# %%
    PATH = 'HH/data/EE/N=100/'
    fname  = 'P0.35HHp=0.25s=0.020f=0.100u=0.100_spike_train.dat'
    ofname = 'HHp=0.25s=0.020f=0.100u=0.100P=0.35_spike_train.dat'
    ra = 0.5
    downsample(fname, ofname, PATH, ra)

    visualize_downsample(PATH, fname, ofname)
