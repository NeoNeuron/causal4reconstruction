# Pairwise Causality Measure onto Pulse-output Signals


## Requirements
#### C/C++ Dependencies:
- [`Eigen`](https://eigen.tuxfamily.org): library of vector/matrix operation
- [`boost`](http://www.boost.org/users/download/): containing library used for argparsers. 
#### Python
- [Anaconda]()
- [Numpy]()
- [Matplotlib]()
- [Scipy]()
- [Sci-kit Learn]()

## Installation
### Causality estimator
```bash
g++ -O2 -w main.cpp -o ./a.out -I ../include/ -lboost_program_options
```

### Network simulators
```bash
g++ -O2 main.cpp -o ./cal_causality.out -I ../include -lboost_program_options
```

### Install python utilities
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