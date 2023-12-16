import pickle
import logging
import nni
import numpy as np
import torch
import core_functions.utils as utils
from core_functions.nonlinear_model import RFFNet

CUDA_LAUNCH_BLOCKING=1
logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    torch.manual_seed(1)
    np.random.seed(1)
    k_par = 0.1 #param['kernel_par'] # 0.1 for kernel
    p = 1000 #param['hidden_size']
    loss_type = 'mse'
    result_dir = './results'
    repeat = 100
    n = 100
    lambda_arr = [0, 1e-6, 1e-4, 1e-2] # [1e-7, 1e-4]
    M_arr = (np.logspace(-2.92, 0, base=4, num=30) * 400).astype(int)

    rf_error = np.empty((len(lambda_arr), len(M_arr), repeat))
    rf_loss = np.empty((len(lambda_arr), len(M_arr), repeat))
    kappa_arr = np.empty((len(lambda_arr), len(M_arr), repeat))
    df1_arr = np.empty((len(lambda_arr), len(M_arr), repeat))
    df2_arr = np.empty((len(lambda_arr), len(M_arr), repeat))
    M_n_arr = M_arr / n
    for r in range(repeat):
        trainloader, _, testloader, d, K = utils.load_data('mnist', n, n)
        X_train, y_train = iter(trainloader).next()    
        X_test, y_test = iter(testloader).next()
        X_train, y_train, X_test, y_test = X_train.to(device), torch.nn.functional.one_hot(y_train, K).to(device), X_test.to(device), y_test.to(device)

        for (i, lambda_reg) in enumerate(lambda_arr):
            for (j, p) in enumerate(M_arr):
                rff = RFFNet(d, p, K, k_par, False, device, torch.float)
                X_train_phi, y_pred = rff.rf_regression(X_train, y_train.float(), X_test, n, lambda_reg, 'identity', device, torch.float)

                rf_loss[i, j, r] = utils.empirical_loss(y_pred, torch.nn.functional.one_hot(y_test, K), loss_type).item()
                rf_error[i, j, r] = 1 - utils.comp_accuracy(y_pred, y_test)[0].item()/100

                C_phi = X_train_phi.T @ X_train_phi
                K_phi = X_train_phi @ X_train_phi.T

                kappa_arr[i, j, r] = 1 / torch.trace(torch.linalg.pinv(K_phi + lambda_reg * torch.eye(n, device=device) * n))

                C_phi_inv = C_phi @ torch.linalg.pinv(C_phi + kappa_arr[i, j, r] * torch.eye(p, device=device) * n)
                df1_arr[i, j, r] = torch.trace(C_phi_inv).item()
                df2_arr[i, j, r] = torch.trace(torch.square(C_phi_inv)).item()
                if df2_arr[i, j, r] > df1_arr[i, j, r]:
                    df2_arr[i, j, r] = df1_arr[i, j, r]
                    
                logger.info("RFF --- MSE: {} Error: {:.2f} df2: {}".format(rf_loss[i, j, r], rf_error[i, j, r], df2_arr[i, j, r]))

    record_ = {
                'lambda_arr': M_arr,
                'M_n_arr': M_n_arr,
                'rf_error': rf_error,
                'rf_loss': rf_loss,
                'kappa_arr': kappa_arr,
                'df1_arr': df1_arr,
                'df2_arr': df2_arr,
                'name': ['$\widehat f$', '$\widehat f_\lambda, \lambda = 10^{-2}$', '$\widehat{f}_\lambda, \lambda = 10^{-4}$', '$\widehat f_\lambda, \lambda = 10^{-6}$', '\widehat f_M']
        }

    result_path = '{}/exp1_{}.pkl'.format(result_dir, 'mnist')
    with open(result_path, "wb") as f:
        pickle.dump(record_, f)

if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        logger.exception(exception)
        raise
    
