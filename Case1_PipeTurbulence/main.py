
import torch
import numpy as np
from lapy import TriaMesh,Solver
import scipy.io as sio
import time
import pandas as pd
from utils import count_params,LpLoss,GaussianNormalizer
from model import NORM_Net
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main(args):  

    print("\n=============================")
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
    print("=============================\n")
    
    PATH = args.data_dir
 
    ntrain = args.num_train
    ntest = args.num_test
    
    batch_size = args.batch_size
    learning_rate = args.lr
    
    epochs = args.epochs
    
    modes = args.modes
    width = args.width
    
    step_size = 100
    gamma = 0.5
    
    s = args.size_of_nodes
    
    ################################################################
    # reading data and calculating LBO basis
    ################################################################
    
    data = sio.loadmat(PATH)
    
    k = 128
    Points = np.vstack((data['nodes'].T, np.zeros(s).reshape(1,-1)))
    mesh = TriaMesh(Points.T,data['elements'].T-1)
    fem = Solver(mesh)
    evals, LBO_MATRIX = fem.eigs(k=k)
    
    
    x_dataIn = torch.Tensor(data['Input'])    
    y_dataIn = torch.Tensor(data['Output'])
    
    x_data = x_dataIn 
    y_data = y_dataIn 
    
    ################################################################
    # normalization
    ################################################################  
    
    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]
    
    norm_x  = GaussianNormalizer(x_train)
    x_train = norm_x.encode(x_train)
    x_test  = norm_x.encode(x_test)

    
    norm_y  = GaussianNormalizer(y_train)
    y_train = norm_y.encode(y_train)
    y_test  = norm_y.encode(y_test)

    
    x_train = x_train.reshape(ntrain,-1,1)
    x_test = x_test.reshape(ntest,-1,1)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=batch_size, shuffle=False)
    

    
    BASE_MATRIX = LBO_MATRIX[:,:modes]
    BASE_MATRIX = torch.Tensor(BASE_MATRIX).cuda()
    BASE_INVERSE = (BASE_MATRIX.T@BASE_MATRIX).inverse()@BASE_MATRIX.T
      
    model = NORM_Net(modes, width, BASE_MATRIX, BASE_INVERSE).cuda()
    
    ################################################################
    # training and evaluation
    ################################################################
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    myloss = LpLoss(size_average=False)
    
    time_start = time.perf_counter()
    time_step = time.perf_counter()
    
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    ET_list = np.zeros((epochs))
    for ep in range(epochs):
        model.train()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
    
            optimizer.zero_grad()
            out = model(x)
    
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() 
            
            out_real = norm_y.decode(out.view(batch_size, -1).cpu())
            y_real = norm_y.decode(y.view(batch_size, -1).cpu())
            train_l2 += myloss(out_real, y_real).item()   
    
            optimizer.step()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
    
                out = model(x)
                out_real = norm_y.decode(out.view(batch_size, -1).cpu())
                y_real = norm_y.decode(y.view(batch_size, -1).cpu())
                
                test_l2 += myloss(out_real, y_real).item()                
                loss_max_test = (abs(out.view(batch_size, -1)- y.view(batch_size, -1))).max(axis=1).values.mean()
    
        train_l2 /= ntrain
        test_l2 /= ntest
        train_error[ep] = train_l2
        test_error[ep] = test_l2
        
        ET_list[ep] = loss_max_test
        time_step_end = time.perf_counter()
        T = time_step_end - time_step

        print('Step: %d, Train L2: %.5f, Test L2 error: %.5f, Emax_test: %.5f, Time: %.3fs'%(ep, train_l2, test_l2, loss_max_test, T))
        time_step = time.perf_counter()
          
    print("\n=============================")
    print("Training done...")
    print("=============================\n")
    
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=1, shuffle=False)
    pre_test = torch.zeros(y_test.shape)
    y_test   = torch.zeros(y_test.shape)
    x_test   = torch.zeros(x_test.shape[0:2])
    
    index = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            
            out_real = norm_y.decode(out.view(1, -1).cpu())
            y_real   = norm_y.decode(y.view(1, -1).cpu())
            x_real   = norm_x.decode(x.view(1, -1).cpu())
            
            pre_test[index,:] = out_real
            y_test[index,:] = y_real
            x_test[index,:] = x_real
            
            index = index + 1
            
    # ================ Save Data ====================
    current_directory = os.getcwd()
    sava_path = current_directory + "/logs/" + args.CaseName + "/"
    if not os.path.exists(sava_path):
        os.makedirs(sava_path)
    
    dataframe = pd.DataFrame({'Test_loss': [test_l2],
                              'num_paras': [count_params(model)],
                              'train_time':[time_step_end - time_start]})
    dataframe.to_csv(sava_path + 'log.csv', index = False, sep = ',')
    
    loss_dict = {'train_error' :train_error,
                 'test_error'  :test_error}
    
    pred_dict = {'pre_test'    : pre_test.cpu().detach().numpy(),
                    'x_test'   : x_test.cpu().detach().numpy(),
                    'y_test'   : y_test.cpu().detach().numpy(),
                    }
    
    sio.savemat(sava_path +'NORM_loss_' + args.CaseName + '.mat', mdict = loss_dict)                                                     
    sio.savemat(sava_path +'NORM_pre_'  + args.CaseName + '.mat', mdict = pred_dict)
    
    print('\nTesting error: %.3e'%(test_l2))
    print('Training time: %.3f'%(time_step_end - time_start))
    print('Num of paras : %d'%(count_params(model)))


if __name__ == "__main__":
    
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
               
    for i in range(5):
        
        i = i + 1
        for args in [
                {'modes': 128,  
                  'width': 32,
                  'size_of_nodes': 2673,
                  'batch_size': 50, 
                  'epochs': 1000,
                  'data_dir': '../datasets/Turbulence',
                  'num_train': 300, 
                  'num_test': 100,
                  'CaseName': 'Turbulence_'+str(i),
                  'lr' : 0.01},
            ]:
            
            args = objectview(args)
                
        main(args)

    