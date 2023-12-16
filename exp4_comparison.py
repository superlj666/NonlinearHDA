import pickle
import logging
import nni
import numpy as np
import torch
import core_functions.utils as utils
from core_functions.nonlinear_model import RFFNet, RFRed
from core_functions.optimal_parameters import get_parameter

CUDA_LAUNCH_BLOCKING=1
logger = logging.getLogger()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def main():
    torch.manual_seed(1)
    np.random.seed(1)
    loss_type = 'mse'
    result_dir = './results'
    repeat = 1
    dataset = 'poker'
    gamma_arr = [0.5, 2]

    parameter_dic = get_parameter(dataset)
    T = parameter_dic['T']
    n = parameter_dic['n']

    rf_error = np.empty((len(gamma_arr), T, repeat))
    rf_loss = np.empty((len(gamma_arr), T, repeat))
    kappa_arr = np.empty((len(gamma_arr), T, repeat))
    df2_arr = np.empty((len(gamma_arr), T, repeat))

    rf_ridgeless_error = np.empty((len(gamma_arr), T, repeat))
    rf_ridgeless_loss = np.empty((len(gamma_arr), T, repeat))
    kappa_ridgeless = np.empty((len(gamma_arr), T, repeat))
    df2_ridgeless = np.empty((len(gamma_arr), T, repeat))

    # Ridge_loss = np.empty((len(gamma_arr), T, repeat))
    # Ridge_error = np.empty((len(gamma_arr), T, repeat))
    # kappa_Ridge = np.empty((len(gamma_arr), T, repeat))
    # df2_Ridge = np.empty((len(gamma_arr), T, repeat))

    RFRed_loss = np.empty((len(gamma_arr), T, repeat))
    RFRed_error = np.empty((len(gamma_arr), T, repeat))
    kappa_RFRed = np.empty((len(gamma_arr), T, repeat))
    df2_RFRed = np.empty((len(gamma_arr), T, repeat))

    for r in range(repeat):
        for (i, gamma) in enumerate(gamma_arr):
            p = int(n * gamma)
            trainloader, _, testloader, d, K = utils.load_data(dataset, n, n)
            X_train, y_train = iter(trainloader).next()    
            X_test, y_test = iter(testloader).next()
            X_train, y_train, X_test, y_test = X_train.to(device), torch.nn.functional.one_hot(y_train, K).to(device), X_test.to(device), y_test.to(device)

            rff = RFFNet(d, p, K, parameter_dic['sigma'], False, device, torch.float)
            X_train_phi, y_pred = rff.rf_regression(X_train, y_train.float(), X_test, n, parameter_dic['lambda_A'], 'identity', device, torch.float)
                
            rf_loss[i, :, r] = utils.empirical_loss(y_pred, torch.nn.functional.one_hot(y_test, K), loss_type).item()
            rf_error[i, :, r] = 1 - utils.comp_accuracy(y_pred, y_test)[0].item()/100

            C_phi = X_train_phi.T.mm(X_train_phi)
            K_phi = X_train_phi.mm(X_train_phi.T)

            kappa_i_r = 1 / torch.trace(torch.linalg.pinv(K_phi + parameter_dic['lambda_A'] * torch.eye(n, device=device) * n))
            C_phi_inv = C_phi.mm(torch.linalg.pinv(C_phi + kappa_i_r * torch.eye(p, device=device) * n))
            df_i_r = torch.trace(torch.square(C_phi_inv)).item()
            for j in range(T):
                kappa_arr[i, j, r] = kappa_i_r
                df2_arr[i, j, r] = df_i_r
            
            parameter_dic['p'] = p
            parameter_dic_ridgeless = parameter_dic
            parameter_dic_ridgeless['lambda_A'] = 0
            parameter_dic_ridgeless['lambda_B'] = 0
            print(parameter_dic_ridgeless)
            res_dict = RFRed(parameter_dic_ridgeless, True, device, torch.float)
            rf_ridgeless_error[i, :, r] = 1 - np.array(res_dict['validate_accuracy_records'])/100
            rf_ridgeless_loss[i, :, r] = res_dict['training_loss_records']
            kappa_ridgeless[i, :, r] = res_dict['training_kappa_records']
            df2_ridgeless[i, :, r] = res_dict['training_df2_records']

            # parameter_dic = get_parameter(dataset)
            # parameter_dic['p'] = p
            # parameter_dic_ridge = parameter_dic
            # parameter_dic_ridge['lambda_B'] = 0
            # print(parameter_dic_ridge)
            # res_dict = RFRed(parameter_dic_ridge, True, device, torch.float)
            # Ridge_error[i, :, r] = 1 - np.array(res_dict['validate_accuracy_records'])/100
            # Ridge_loss[i, :, r] = res_dict['training_loss_records']
            # kappa_Ridge[i, :, r] = res_dict['training_kappa_records']
            # df2_Ridge[i, :, r] = res_dict['training_df2_records']

            parameter_dic = get_parameter(dataset)
            parameter_dic['p'] = p
            parameter_dic_RFRed = parameter_dic
            print(parameter_dic_RFRed)
            res_dict = RFRed(parameter_dic_RFRed, True, device, torch.float)
            RFRed_error[i, :, r] = 1 - np.array(res_dict['validate_accuracy_records'])/100
            RFRed_loss[i, :, r] = res_dict['training_loss_records']
            kappa_RFRed[i, :, r] = res_dict['training_kappa_records']
            df2_RFRed[i, :, r] = res_dict['training_df2_records']

            logger.info("Repeat {}, gamma {}, loss {}, error {}, df2 {}".format(r, gamma, RFRed_loss[i, -1, r], RFRed_error[i, -1, r], df2_RFRed[i, -1, r]))

    record_ = {
                'gamma_arr': gamma_arr,
                'rf_error': rf_error,
                'rf_loss': rf_loss,
                'kappa_arr': kappa_arr,
                'df2_arr': df2_arr,
                'rf_ridgeless_error' : rf_ridgeless_error,
                'rf_ridgeless_loss' : rf_ridgeless_loss,
                'kappa_ridgeless' : kappa_ridgeless,
                'df2_ridgeless' : df2_ridgeless,
                # 'Ridge_loss' : Ridge_loss,
                # 'Ridge_error' : Ridge_error,
                # 'kappa_Ridge' : kappa_Ridge,
                # 'df2_Ridge' : df2_Ridge,
                'RFRed_loss' : RFRed_loss,
                'RFRed_error' : RFRed_error,
                'kappa_RFRed' : kappa_RFRed,
                'df2_RFRed' : df2_RFRed,
                'name': ['$\widehat f$', '$\widehat f_\lambda, \lambda = 10^{-2}$', '$\widehat{f}_\lambda, \lambda = 10^{-4}$', '$\widehat f_\lambda, \lambda = 10^{-6}$', '\widehat f_M']
        }

    result_path = '{}/exp4_{}.pkl'.format(result_dir, dataset)
    with open(result_path, "wb") as f:
        pickle.dump(record_, f)


if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        logger.exception(exception)
        raise
    