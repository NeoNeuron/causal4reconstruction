# ## Direct download via api.brain-map.org
# Some people have reported issues downloading the files via the AllenSDK (the connection is extremely slow, or gets interrupted frequently). If this applies to you, you can try downloading the files via HTTP requests sent to **api.brain-map.org**. This approach is not recommended, because you will have to manually keep track of the file locations. But if you're doing analysis that doesn't depend on the AllenSDK (e.g., in Matlab), this may not matter to you.
# 
# You can follow the steps below to retrieve the URLs for all of the NWB files in this dataset.

# %%
import os
from allensdk.brain_observatory.ecephys.ecephys_project_api.utilities import build_and_execute
from allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine import RmaEngine
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from subprocess import run

rma_engine = RmaEngine(scheme="http", host="api.brain-map.org")

def retrieve_link(session_id):
    
    well_known_files = build_and_execute(
        (
        "criteria=model::WellKnownFile"
        ",rma::criteria,well_known_file_type[name$eq'EcephysNwb']"
        "[attachable_type$eq'EcephysSession']"
        r"[attachable_id$eq{{session_id}}]"
        ),
        engine=rma_engine.get_rma_tabular, 
        session_id=session_id
    )
    
    return 'http://api.brain-map.org/' + well_known_files['download_link'].iloc[0]

# %%
from pathlib import Path
data_directory = '../data/neuropixel/' # remember to change this to something that exists on your machine
Path(data_directory).mkdir(parents=True, exist_ok=True)


# %% [markdown]
# download the NWB file used in manuscript.

#%%
session_id = 715093703
nwb_path = f'session_{session_id:d}.nwb'
download_links = retrieve_link(session_id)
run(['wget','-q', '-O', f"{data_directory:s}session_{session_id:d}.nwb", download_links],)

# %% [markdown]
# download all available NWB files for the Neuropixels dataset. This will take a while, and you will need a lot of disk space (about 1.5 TB). If you don't have that much space, you can download only a subset of the files by changing the `session_ids` variable below.

# %%
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
sessions = cache.get_session_table()

# %%
download_links = [retrieve_link(session_id) for session_id in sessions.index.values]

# %%
for idx, url in zip(sessions.index.values, download_links):
    run(['wget','-q', '-O', f"{data_directory:s}session_{idx:d}.nwb", url],)
# %% [markdown]
# `download_links` is a list of 58 links that can be used to download the NWB files for all available sessions. Clicking on the links above should start the download automatically.
# 
# If you aren't interested in using the `EcephysProjectCache` object to keep track of what you've downloaded, you can create a `session` object just by passing a path to an NWB file:

# %%
# from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession

# # nwb_path = '/mnt/nvme0/ecephys_cache_dir_10_31/session_721123822/session_721123822.nwb'
# nwb_path = '/mnt/nvme0/ecephys_cache_dir_10_31/session_757216464/session_757216464.nwb'

# session = EcephysSession.from_nwb_path(nwb_path, api_kwargs={
#         "amplitude_cutoff_maximum": np.inf,
#         "presence_ratio_minimum": -np.inf,
#         "isi_violations_maximum": np.inf
#     })

# # %% [markdown]
# # This will load the data for one session, without applying the default unit quality metric filters. Everything will be available except the LFP data, because the `get_lfp()` method can only find the associated LFP files if you're using the `EcephysProjectCache` object.
# # 
# # To obtain similar links for the LFP files, you can use the following code:

# # %%
# def retrieve_lfp_link(probe_id):

#     well_known_files = build_and_execute(
#         (
#             "criteria=model::WellKnownFile"
#             ",rma::criteria,well_known_file_type[name$eq'EcephysLfpNwb']"
#             "[attachable_type$eq'EcephysProbe']"
#             r"[attachable_id$eq{{probe_id}}]"
#         ),
#         engine=rma_engine.get_rma_tabular, 
#         probe_id=probe_id
#     )

#     if well_known_files.shape[0] != 1:
#         return 'file for probe ' + str(probe_id) + ' not found'
        
#     return 'http://api.brain-map.org/' + well_known_files.loc[0, "download_link"]

# probes = cache.get_probes()

# download_links = [retrieve_lfp_link(probe_id) for probe_id in probes.index.values]

# _ = [print(link) for link in download_links]


