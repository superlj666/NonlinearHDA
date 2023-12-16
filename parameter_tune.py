"""
Tune Hyperparamters for KRR via NNI.
"""

from core_functions.nonlinear_model import RFRed
import core_functions.optimal_parameters as optimal_parameters
import os
import logging
import numpy as np
from nni.utils import merge_parameter
import pickle
import argparse
import math
from core_functions import utils
import torch
import nni

CUDA_LAUNCH_BLOCKING=1
SAVE = False
logger = logging.getLogger('Tune Hyperparamters')

def main(args):
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    res_dict = RFRed(args, True, device, torch.float)
    error = res_dict['test_accuracy']

    nni.report_final_result(error)
    logger.debug('Final result is %.4f', 
    error)
    logger.debug('Send final result done.')

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='KRR Tuner')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument("--dataset", type=str, default='pendigits', help="dataset name")
    parser.add_argument("--p", type=int, default=1000, help="feature dimensionality")
    parser.add_argument("--n", type=int, default=1000, help="sample size")
    parser.add_argument("--m", type=int, default=10, help="subsampling size")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument('--T', type=int, default=30, metavar='N', help='number of epochs to train')
    parser.add_argument('--record_batch', type=int, default=30, metavar='N', help='number of batches to record')
    parser.add_argument("--sigma", type=float, default=0.001, help="kernel hyperparameter")
    parser.add_argument('--lambda_A', type=float, default=0.0, help='regularizer parameter for model complexity')
    parser.add_argument('--lambda_B', type=float, default=0.0, help='regularizer parameter for df2')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--validate', type=bool, default=True, help='If validate')
    args, _ = parser.parse_known_args()
    return args

## set parameters in config_gpu.yml 
## and run nnictl create --config /home/superlj666/Experiment/FedNewton/config_gpu.yml
if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)

    except Exception as exception:
        logger.exception(exception)
        raise