# High-dimensional analysis for Generalized Nonlinear Regression: From Asymptotics to Algorithm
## Intro
This repository provides the code used to run the experiments of the paper "High-dimensional analysis for Generalized Nonlinear Regression: From Asymptotics to Algorithm".
## Environments
- Python 3.7.4
- Pytorch 1.10.0
- NNI 2.5
- CUDA 10.1.168
- cuDnn 7.6.0
- GPU: Nvidia RTX 3090 24G
## Core functions
- auto_kernel_learning.py implements the algorithm to construct random feature regression and RFRed.
- utils.py implements useful tools including load svmlight style dataset and classic datasets used in Pytorch but also various loss functions are introduced.
- parameter_tune.py is used to tune hyperparameters via [NNI](https://nni.readthedocs.io/).
- optimal_parameters.py records optimal parameters for the proposed algorithm.
- exp*_xxx.py are scripts in experiments.
- plot.ipynb reads experiment results and draw images.
## Experiments
1. Download datasets for multi-class classification (https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).
2. Run the script to tune parameters and record them in optimal_parameters.py.
```
nnictl create --config ./config_gpu.yml
```
3. Run the scripts to obtain results in Experiment section
```
python exp*_xxx.py
```
4. Run plot.ipynb to draw figures
