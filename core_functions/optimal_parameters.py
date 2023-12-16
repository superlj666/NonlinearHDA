def get_parameter(dataset):
    if dataset == 'mnist':
        parameter_dic = {
            'dataset' : 'mnist',  # HR6votjL
            'sigma' : 0.01,
            'lambda_A' : 0.0001, 
            'lambda_B' : 0.001,
            'learning_rate' : 1e-3,
            'n' : 1000,
            'T' : 30,
            'batch_size' : 32,
            'record_batch' : 30,
            'validate' : True
        }
    elif dataset == 'usps':
        parameter_dic = {
            'dataset' : 'usps', # EpBm59GH
            'sigma' : 0.1,
            'lambda_A' : 1.00001, 
            'lambda_B' : 0.00001, 
            'learning_rate' : 0.001,
            'n' : 1000,
            'T' : 30,
            'batch_size' : 32,
            'record_batch' : 30,
            'validate' : True
        }
    else:
        parameter_dic = {
            'dataset' : dataset, 
            'sigma' : 0.1,
            'lambda_A' : 1.00001, 
            'lambda_B' : 0.00001, 
            'learning_rate' : 0.001,
            'n' : 1000,
            'T' : 30,
            'batch_size' : 32,
            'record_batch' : 30,
            'validate' : True
        }
    return parameter_dic
