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

1. Figure 2:
    ```bash
    pm_scan_kl_HH10.py
    ```
2. Figure 3:
    ```bash
    pm_scan_kl_HH10.py
    order_k_th.py
    order_l_th.py
    scan_l.py
    scan_S.py
    ```
3. Figure 4:
    ```bash
    test_HH_recon.py
    test_HH100_recon.py
    ```
4. Figure 5:
    ```bash
    download_allen_observatory_data.py
    extract_allen_data_pkl.py
    gen_TGIC_allen_data.py
    test_allen.py
    ```