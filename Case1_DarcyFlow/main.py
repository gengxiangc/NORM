
import torch
import torch.nn.functional as F
import numpy as np
from lapy import TriaMesh,Solver
import scipy.io as sio
import time
import pandas as pd
from utils import count_params,LpLoss,UnitGaussianNormalizer
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
    Points = np.vstack((data['MeshNodes'], np.zeros(s).reshape(1,-1)))
    mesh = TriaMesh(Points.T,data['MeshElements'].T-1)
    fem = Solver(mesh)
    evals, LBO_MATRIX = fem.eigs(k=k)

    y_dataIn = torch.Tensor(data['u_field']) 
    x_dataIn = torch.Tensor(data['c_field'])
    
    x_data = x_dataIn 
    y_data = y_dataIn 
    
    ################################################################
    # normalization
    ################################################################   
    
    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]
    
    print('x_train:', x_train.shape, 'y_train:', y_train.shape)
    print('x_test:', x_test.shape, 'y_test:', y_test.shape)
    
    norm_x  = UnitGaussianNormalizer(x_train)
    x_train = norm_x.encode(x_train)
    x_test  = norm_x.encode(x_test)

    
    norm_y  = UnitGaussianNormalizer(y_train)
    y_train = norm_y.encode(y_train)
    y_test  = norm_y.encode(y_test)

    
    x_train = x_train.reshape(ntrain,-1,1)

    x_test = x_test.reshape(ntest,-1,1)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=batch_size, shuffle=False)
    
    
    BASE_MATRIX = LBO_MATRIX[:,:2*modes]
     
    BASE_MATRIX = torch.Tensor(BASE_MATRIX).cuda()
    BASE_INVERSE = (BASE_MATRIX.T@BASE_MATRIX).inverse()@BASE_MATRIX.T
      
    model = NORM_Net(BASE_MATRIX.shape[1], width, BASE_MATRIX, BASE_INVERSE).cuda()
    
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

    for ep in range(epochs):
        model.train()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
    
            optimizer.zero_grad()
            out = model(x)
    
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() 
            
            out_real = norm_y.decode(out.view(batch_size, -1).cpu())
            y_real = norm_y.decode(y.view(batch_size, -1).cpu())
            train_l2 += myloss(out_real, y_real).item()   

            optimizer.step()
            train_mse += mse.item()
    
        scheduler.step()
        model.eval()
        test_l2 = 0.0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
    
                out = model(x)
                out_real = norm_y.decode(out.view(batch_size, -1).cpu())
                y_real   = norm_y.decode(y.view(batch_size, -1).cpu())
                test_l2 += myloss(out_real, y_real).item()                

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2  /= ntest
        train_error[ep] = train_l2
        test_error [ep] = test_l2
        
        time_step_end = time.perf_counter()
        T = time_step_end - time_step
        
        if ep % 1 == 0:
            print('Step: %d, Train L2: %.5f, Test L2 error: %.5f, Time: %.3fs'%(ep, train_l2, test_l2, T))
        time_step = time.perf_counter()

    print("Training done...")
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=1, shuffle=False)
    pre_train = torch.zeros(y_train.shape)
    y_train   = torch.zeros(y_train.shape)
    x_train   = torch.zeros(x_train.shape[0:2])

    index = 0
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            out = model(x)
            out_real = norm_y.decode(out.view(1, -1).cpu())
            y_real   = norm_y.decode(y.view(1, -1).cpu())
            
            pre_train[index,:] = out_real
            y_train[index,:]   = y_real
            x_train[index]   = norm_x.decode(x.view(1, -1).cpu())
            
            index = index + 1
    
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
            x_test[index] = x_real
            
            index = index + 1
            
    # ================ Save Data ====================
    current_directory = os.getcwd()
    sava_path = current_directory + "/logs/" + args.CaseName + "/"
    if not os.path.exists(sava_path):
        os.makedirs(sava_path)
    
    dataframe = pd.DataFrame({'Test_loss' : [test_l2],
                              'Train_loss': [train_l2],
                              'num_paras' : [count_params(model)],
                              'train_time':[time_step_end - time_start]})
    
    dataframe.to_csv(sava_path + 'log.csv', index = False, sep = ',')
    
    loss_dict = {'train_error' :train_error,
                 'test_error'  :test_error}
    
    pred_dict = {   'pre_test' : pre_test.cpu().detach().numpy(),
                    'pre_train': pre_train.cpu().detach().numpy(),
                    'x_test'   : x_test.cpu().detach().numpy(),
                    'x_train'  : x_train.cpu().detach().numpy(),
                    'y_test'   : y_test.cpu().detach().numpy(),
                    'y_train'  : y_train.cpu().detach().numpy(),
                    }
    
    sio.savemat(sava_path +'NORM_loss.mat', mdict = loss_dict)                                                     
    sio.savemat(sava_path +'NORM_pre.mat', mdict = pred_dict)
    
    test_l2 = (myloss(y_test, pre_test).item())/ntest
    print('\nTesting error: %.3e'%(test_l2))
    print('Training time: %.3f'%(time_step_end - time_start))
    print('Num of paras : %d'%(count_params(model)))


if __name__ == "__main__":
    
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
            
    for i in range(5):
        
        print('====================================')
        print('NO.'+str(i)+' repetition......')
        print('====================================')
        
        for args in [
                        { 'modes' : 128,  
                          'width' : 32,
                          'size_of_nodes' : 2290,
                          'batch_size'    : 100, 
                          'epochs'    : 1000,
                          'data_dir'  : '../datasets/Darcy',
                          'num_train' : 1000, 
                          'num_test'  : 200,
                          'CaseName'  : 'Darcy_node2k_n1200/' + str(i), 
                          'basis'     : 'LBO',
                          'lr'        : 0.001},
                    ]:
            
            args = objectview(args)
                
        main(args)

    