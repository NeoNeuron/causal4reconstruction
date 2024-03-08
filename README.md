# Pairwise Causality Measure onto Pulse-output Signals

Original code for paper: "Causal connectivity measures for pulse-output network reconstruction: analysis and applications"

## Requirements

### C/C++ Dependencies

- [`Eigen`](https://eigen.tuxfamily.org): library of vector/matrix operation

  ```bash
  sudo apt-get update
  sudo apt-get install libeigen3-dev
  ```
- [`boost`](http://www.boost.org/users/download/): containing library used for argparsers.

  ```bash
  sudo apt-get install libboost-all-dev
  ```

### Python

numpy, pandas, matplotlib, scipy, sci-kit learn, seaborn, brian2

## Installation

### Compile C/C++ modules

```bash
make all -j
```

### Install Python utilities

```bash
conda create -n causal4 python=3.11 --file requirements.txt -c conda-forge
conda activate causal4
pip install -e .
```

Install additional packages for running large E-I balanced network simulation:
```bash
pip install --upgrade "jax[cpu]"
pip install -U brainpy
pip install brainpylib
pip install -U "ray[default]"
```

## Usage

0. Run simulation of HH10 and HH100 models:
    ```bash
    ./code4paper/run_HH10_scan_S.py
    ./code4paper/run_HH100.py
    ```
    The results will be saved in `./HH/data/EE/N=10/` and `./HH/data/EE/N=100` respectively.

1. Figure 2:
    ```bash
    ./code4paper/pm_scan_kl_HH10.py
    ```
2. Figure 3:
    ```bash
    ./code4paper/pm_scan_kl_HH10.py
    ```
3. Figure 4:
    ```bash
    ./code4paper/HH100_recon_pnas.py
    ```
4. Figure 5:
    To access allen data, you need to download the data using allensdk (or download directly from [Allen Institute](https://portal.brain-map.org/)).
    
    Install allensdk:
    ```bash
    pip install allensdk
    ```

    Then run the following scripts:

    ```bash
    ./code4paper/download_allen_observatory_data.py # download data
    ./code4paper/extract_allen_data_pkl.py          # data preprocessing
    ./code4paper/allen_data_causality_estimation.py
    ./code4paper/test_allen.py
    ```