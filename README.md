# Causal connectivity measures for pulse-output network reconstruction

Ccode for paper "Causal connectivity measures pulse-output network reconstruction: analysis and applications" (in press), and **it will be available publicly upon publication**.

## Requirements

### C/C++ Dependencies

- [`Eigen`](https://eigen.tuxfamily.org): library of vector/matrix operation
- [`boost`](http://www.boost.org/users/download/): containing library used for argparsers.

### Python

- network simulation: numpy, brian2
- network reconstruction: numpy, pandas matplotlib, scipy, sci-kit learn, seaborn, 

## Installation

### Compile C/C++ modules

```bash
make all -j
```

### Install Python utilities

```bash
pip install -e .
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
