# -*- coding: utf-8 -*-
import os
import torch
import torchvision
import math
import matplotlib.pyplot as plt
import time
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
from core_functions.utils import load_data, empirical_loss, matplotlib_imshow
from torch.distributions.uniform import Uniform
from scipy.linalg import hadamard

def subsampling_matrix(m, n, sketch_type, bp = False, device='cuda', dtype=torch.float):
    if sketch_type == 'subset_selection':
        subsampling_idx = np.random.choice(n, m, replace=False)
        S = torch.zeros((m, n), device=device, dtype=dtype, requires_grad=bp)
        for (i, idx) in enumerate(subsampling_idx):
            S[i, idx] = 1
        return S
    elif sketch_type == 'ROS':
        D = np.diag(np.random.choice([1, -1], n))
        H = hadamard(n)
        P = np.eye(n)[np.random.choice(range(n), m, replace=False), :]
        S = np.sqrt(n/m)*P @ D @ H.T
        return torch.tensor(S, device=device, dtype=dtype, requires_grad=bp)
    else:
        return torch.eye(n, device=device, dtype=dtype, requires_grad=bp)

class RFFNet(torch.nn.Module):
    def __init__(self, d, p, K, sigma=0.05, phi_bp=False, device='cuda', dtype=torch.float):
        super(RFFNet, self).__init__()
        self.p = p
        self.W = torch.empty((d, p), device=device, dtype=dtype, requires_grad=phi_bp)
        self.b = torch.empty(p, device=device, dtype=dtype)
        self.theta = torch.randn(p, K, device=device, dtype=dtype, requires_grad=True)
        self.I = torch.eye(p, device=device, dtype=dtype)
        torch.nn.init.normal_(self.W, 0, sigma)
        torch.nn.init.uniform_(self.b, 0, 2*math.pi)

    def forward(self, x):
        x_phi = self.rff(x)
        y = x_phi @ self.theta
        return x_phi, y
    
    def rff(self, x):
        n = x.shape[0]
        return torch.cos(x.mm(self.W) + self.b.repeat(n, 1))*math.sqrt(1/self.p)

    def rf_regression(self, X_train, y_train, X_test, m, lambda_reg = 1e-4, sketch_type='identity', device='cuda', dtype=torch.float):
        n = X_train.shape[0]
        S = subsampling_matrix(m, n, sketch_type, device=device, dtype=dtype)
        S_phi = S @ self.rff(X_train)
        X_test_phi = self.rff(X_test)
        self.theta = torch.linalg.pinv(S_phi.T @ S_phi + lambda_reg * self.I * n) @ (S_phi.T @  (S @ y_train))
        y_pred = X_test_phi.mm(self.theta)
        return S_phi, y_pred

def RFRed(parameter_dic, phi_bp=True, device='cuda', dtype=torch.float):
    dataset = parameter_dic['dataset']
    p = parameter_dic['p']
    n = parameter_dic['n']
    sigma = parameter_dic['sigma']
    batch_size = parameter_dic['batch_size']
    T = parameter_dic['T']
    record_batch = parameter_dic['record_batch']
    lambda_A = parameter_dic['lambda_A']
    lambda_B = parameter_dic['lambda_B']
    learning_rate = parameter_dic['learning_rate']

    # Load data
    trainloader, validateloader, testloader, d, K = load_data(dataset, n, batch_size)

    # Define model and optimizer
    if phi_bp:
        rffNet = RFFNet(d, p, K, sigma, True, device, dtype)
        optimizer = optim.Adam((rffNet.W, rffNet.theta), learning_rate)
    else:        
        rffNet = RFFNet(d, p, K, sigma, False, device, dtype)
        optimizer = optim.Adam((rffNet.W, rffNet.theta), learning_rate)

    # Records variables
    training_loss_records, training_fro_norm_records, training_kappa_records, training_df2_records = [], [], [], []
    validate_loss_records, validate_accuracy_records = [], []

    start = time.time()
    # Training
    for epoch in range(T):
        training_loss, training_fro_norm, training_kappa, training_df2 = [0.0, 0.0, 0.0, 0.0]
        for i_batch, train_batch in enumerate(trainloader, 0):
            optimizer.zero_grad() 
            X_train, y_train = train_batch
            X_train = X_train.to(device).to(dtype)
            y_train = y_train.to(device)
            
            # Forward : predict 
            X_train_phi, y_pred = rffNet(X_train)

            # Forward : calculate objective
            K_phi = X_train_phi @ X_train_phi.T
            kappa = 1 / torch.trace(torch.linalg.pinv(K_phi + lambda_A * torch.eye(K_phi.shape[0], device=device) * K_phi.shape[0]))
            KM_inv =torch.linalg.pinv(K_phi + kappa * K_phi.shape[0] * torch.eye(K_phi.shape[0], device=device))
            df2 = torch.trace(X_train_phi.T @ KM_inv @ K_phi @ KM_inv @ X_train_phi)

            loss = empirical_loss(y_pred,  torch.nn.functional.one_hot(y_train, K).float(), 'mse')
            fro_norm = torch.norm(rffNet.theta, 'fro')
            objective = loss + lambda_A * fro_norm + lambda_B * df2
            
            # Records
            training_loss += loss.item()
            training_fro_norm += fro_norm.item()
            training_kappa += kappa.item()
            training_df2 += df2.item()

            if i_batch % record_batch == record_batch - 1:
                training_loss_records.append(training_loss / record_batch)
                training_fro_norm_records.append(training_fro_norm / record_batch)
                training_kappa_records.append(training_kappa / record_batch)
                training_df2_records.append(training_df2 / record_batch)
                training_loss, training_fro_norm, training_kappa, training_df2 = [0.0, 0.0, 0.0, 0.0]

                if parameter_dic['validate']:
                    validate_loss, validate_accuracy = test_in_batch(testloader, rffNet, device = device)
                    validate_loss_records.append(validate_loss)
                    validate_accuracy_records.append(validate_accuracy)
                    print('[%d, %5d] loss: %.3f, accuracy: %.3f%%' % (epoch + 1, i_batch + 1, validate_loss, validate_accuracy))
                else:
                    print('[%d, %5d] loss: %.3f%%' % (epoch + 1, i_batch + 1, training_loss))

            # Backward
            objective.backward()

            # Update
            optimizer.step()
    end = time.time()
    print('Elapsed time: {} seconds'.format(end - start))

    # Testing
    test_loss, test_accuracy = test_in_batch(testloader, rffNet, device = device)
    print('RFRed test --- loss: %.3f, accuracy: %.3f%%' % (test_loss, test_accuracy))

    # Save results
    result_dict = {
        'parameter_dic' : parameter_dic,

        'training_objective_records' : [training_loss_records[i] + training_fro_norm_records[i] + training_df2_records[i] for i in range(len(training_loss_records))],      
        'training_loss_records' : training_loss_records,
        'training_fro_norm_records' : training_fro_norm_records,
        'training_kappa_records' : training_kappa_records,
        'training_df2_records' : training_df2_records,

        'validate_loss_records' : validate_loss_records, 
        'validate_accuracy_records' : validate_accuracy_records,

        'test_objective' : test_loss,
        'test_loss' : test_loss,
        'test_accuracy' : test_accuracy,
        'training_time' : end - start
    }

    return result_dict

def test_in_batch(testloader, rff, device="cuda:0"):
    total = 0
    loss = 0.0
    correct = 0
    with torch.no_grad():
        for i_batch, test_batch in enumerate(testloader):
            X_test, y_test = test_batch
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            _, y_pred = rff(X_test)
            loss += empirical_loss(y_pred, torch.nn.functional.one_hot(y_test, y_pred.shape[1]), 'mse').item()

            if rff.theta.shape[1] > 1:
                _, predicted = torch.max(y_pred.data, 1)
                correct += (predicted == y_test).sum().item()*100
                total += y_test.size(0)
            else:
                correct += math.sqrt(empirical_loss(y_pred, y_test, 'mse').item())
                total += 1

        return  loss/(i_batch + 1), correct / total