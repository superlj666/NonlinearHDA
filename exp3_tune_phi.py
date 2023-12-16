import pickle
import logging
import nni
import numpy as np
import torch
import core_functions.utils as utils
from core_functions.nonlinear_model import RFFNet, RFRed

CUDA_LAUNCH_BLOCKING=1
logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    torch.manual_seed(1)
    np.random.seed(1)
    n = 1000
    loss_type = 'mse'
    result_dir = './results'
    repeat = 10
    dataset = 'mnist'
    k_par = 0.1
    lambda_A = 1e-3
    gamma_arr = [0.5, 1, 2]
    T = 30

    parameter_dic = {
        'dataset' : 'mnist', 
        'sigma' : 0.01,
        'lambda_A' : 1e-1, 
        'lambda_B' : 0, 
        'learning_rate' : 1e-3,
        'n' : n,
        'T' : T,
        'batch_size' : 32,
        'record_batch' : 30,
        'validate' : True
    }

    rf_error = np.empty((len(gamma_arr), T, repeat))
    rf_loss = np.empty((len(gamma_arr), T, repeat))
    kappa_arr = np.empty((len(gamma_arr), T, repeat))
    df2_arr = np.empty((len(gamma_arr), T, repeat))

    rf_loss_sgd = np.empty((len(gamma_arr), T, repeat))
    rf_error_sgd = np.empty((len(gamma_arr), T, repeat))
    kappa_arr_sgd = np.empty((len(gamma_arr), T, repeat))
    df2_arr_sgd = np.empty((len(gamma_arr), T, repeat))

    for r in range(repeat):
        for (i, gamma) in enumerate(gamma_arr):
            p = int(n * gamma)
            trainloader, _, testloader, d, K = utils.load_data(dataset, n, n)
            X_train, y_train = iter(trainloader).next()    
            X_test, y_test = iter(testloader).next()
            X_train, y_train, X_test, y_test = X_train.to(device), torch.nn.functional.one_hot(y_train, K).to(device), X_test.to(device), y_test.to(device)

            rff = RFFNet(d, p, K, k_par, False, device, torch.float)
            X_train_phi, y_pred = rff.rf_regression(X_train, y_train.float(), X_test, n, lambda_A, 'identity', device, torch.float)
                
            rf_loss[i, :, r] = utils.empirical_loss(y_pred, torch.nn.functional.one_hot(y_test, K), loss_type).item()
            rf_error[i, :, r] = 1 - utils.comp_accuracy(y_pred, y_test)[0].item()/100

            C_phi = X_train_phi.T.mm(X_train_phi)
            K_phi = X_train_phi.mm(X_train_phi.T)

            kappa_i_r = 1 / torch.trace(torch.linalg.pinv(K_phi + lambda_A * torch.eye(n, device=device) * n))
            C_phi_inv = C_phi.mm(torch.linalg.pinv(C_phi + kappa_i_r * torch.eye(p, device=device) * n))
            df_i_r = torch.trace(torch.square(C_phi_inv)).item()
            for j in range(T):
                kappa_arr[i, j, r] = kappa_i_r
                df2_arr[i, j, r] = df_i_r
            
            parameter_dic['p'] = p
            res_dict = RFRed(parameter_dic, True, device, torch.float)
            rf_loss_sgd[i, :, r] = res_dict['training_loss_records']
            rf_error_sgd[i, :, r] = 1 - np.array(res_dict['validate_accuracy_records'])/100
            kappa_arr_sgd[i, :, r] = res_dict['training_kappa_records']
            df2_arr_sgd[i, :, r] = res_dict['training_df2_records']

            logger.info("Repeat {}, gamma {}, loss {}, error {}, df2 {}".format(r, gamma, rf_loss[i, -1, r], rf_error[i, -1, r], df2_arr[i, -1, r]))

    record_ = {
                'gamma_arr': gamma_arr,
                'rf_error': rf_error,
                'rf_loss': rf_loss,
                'kappa_arr': kappa_arr,
                'df2_arr': df2_arr,
                'rf_error_sgd': rf_error_sgd,
                'rf_loss_sgd': rf_loss_sgd,
                'kappa_arr_sgd': kappa_arr_sgd,
                'df2_arr_sgd': df2_arr_sgd,
                'name': ['$\widehat f$', '$\widehat f_\lambda, \lambda = 10^{-2}$', '$\widehat{f}_\lambda, \lambda = 10^{-4}$', '$\widehat f_\lambda, \lambda = 10^{-6}$', '\widehat f_M']
        }

    result_path = '{}/exp3_phi_{}.pkl'.format(result_dir, dataset)
    with open(result_path, "wb") as f:
        pickle.dump(record_, f)


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        logger.exception(exception)
        raise
    
