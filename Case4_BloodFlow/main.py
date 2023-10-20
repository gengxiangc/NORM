
import torch
import numpy as np
import scipy.io as sio
import time
import pandas as pd
from utils import count_params,LpLoss, GaussianNormalizer
from model import NORM_net
import os
from Adam import Adam

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(args):  
    
    print("\n=============================")
    print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
    print("=============================\n")
    
    LBO_PATH = args.LBO_dir
    PATH = args.data_dir
    
    ntrain = args.num_train  
    ntest = args.num_test  
    batch_size = args.batch_size 
    learning_rate = args.lr    
    epochs = args.epochs    
    modes = args.modes 
    Fmodes = args.Fmodes 
    width = args.width 
    
    nodes = args.size_of_nodes
    BASIS = args.basis 
    
    step_size = 100
    gamma = 0.1
    
    ################################################################
    # reading data reading LBO basis
    ################################################################   
    
    data = sio.loadmat(PATH) 
    LBOdata = sio.loadmat(LBO_PATH) 
    LBO_MATRIX = LBOdata['Eigenvectors']
    
    x_dataIn = torch.Tensor(data['BC_time'])
    y_dataIn1 = torch.Tensor(data['velocity_x'])
    y_dataIn2 = torch.Tensor(data['velocity_y'])
    y_dataIn3 = torch.Tensor(data['velocity_z'])
    
    x_data = x_dataIn
    y_data = torch.zeros((y_dataIn1.shape[0],y_dataIn1.shape[1],y_dataIn1.shape[2],3))
     
    y_data[:,:,:,0] = y_dataIn1
    y_data[:,:,:,1] = y_dataIn2
    y_data[:,:,:,2] = y_dataIn3
    
    ################################################################
    # normalization
    ################################################################  
    x_train = x_data[:ntrain,:,:]
    y_train = y_data[:ntrain,:,:]
    x_test = x_data[-ntest:,:,:]
    y_test = y_data[-ntest:,:,:]
            

    norm_x1 = GaussianNormalizer(x_train[:,:,0])
    norm_x2 = GaussianNormalizer(x_train[:,:,1:])
    
    x_train[:,:,0] = norm_x1.encode(x_train[:,:,0])
    x_train[:,:,1:] = norm_x2.encode(x_train[:,:,1:])
    x_test[:,:,0] = norm_x1.encode(x_test[:,:,0])
    x_test[:,:,1:] = norm_x2.encode(x_test[:,:,1:])
    
    norm_y  = GaussianNormalizer(y_train)
    y_train = norm_y.encode(y_train)
    y_test  = norm_y.encode(y_test)
       
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                               batch_size=batch_size, shuffle=True, drop_last=True) 
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=batch_size, shuffle=False, drop_last=True) 
    
    if BASIS == 'LBO':
        BASE_MATRIX = LBO_MATRIX[:,:modes] 

    TIME_MATRIX = BASE_MATRIX
    
    TIME_MATRIX = torch.Tensor(TIME_MATRIX).to(device) 
    TIME_INVERSE = (TIME_MATRIX.T@TIME_MATRIX).inverse()@TIME_MATRIX.T 
        
    BASE_MATRIX = torch.Tensor(BASE_MATRIX).to(device) 
    BASE_INVERSE = (BASE_MATRIX.T@BASE_MATRIX).inverse()@BASE_MATRIX.T 
    
    Nt = x_train.shape[1]
    
    model = NORM_net(modes, nodes, Fmodes, width, TIME_MATRIX, TIME_INVERSE, BASE_MATRIX, BASE_INVERSE, Nt).to(device) 
    
    ################################################################
    # training and evaluation
    ################################################################

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) 
    
    myloss = LpLoss(d=3, p=2, size_average  = False)
    
    time_start = time.perf_counter()
    time_step = time.perf_counter()
    
    train_error = np.zeros((epochs))
    test_error = np.zeros((epochs))
    ET_list = np.zeros((epochs))
    
    for ep in range(epochs):
        
        model.train() 
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
    
            optimizer.zero_grad()
            out = model(x)
                                  
            l2 = myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))            
            l2.backward() 
                        
            out_real = norm_y.decode(out.cpu()).contiguous().reshape(batch_size, -1)
            y_real = norm_y.decode(y.cpu()).reshape(batch_size, -1) 
            train_l2 += myloss(out_real, y_real).item()          
                   
            optimizer.step()
            
        scheduler.step()
        model.eval() 
        test_l2 = 0.0
        
        with torch.no_grad(): 
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
    
                out = model(x) 
                out_real = norm_y.decode(out.cpu()).contiguous().reshape(batch_size, -1)
                y_real = norm_y.decode(y.cpu()).reshape(batch_size, -1)
                
                test_l2 += myloss(out_real, y_real).item()                
                loss_max_test= (abs(out_real- y_real)).max(axis=1).values.mean()
    
        train_l2 /= ntrain
        test_l2 /= ntest
        train_error[ep] = train_l2
        test_error[ep] = test_l2
        
        ET_list[ep] = loss_max_test
        time_step_end = time.perf_counter()
        T = time_step_end - time_step

        print('Epoch: %d, Train L2: %.5f, Test L2: %.5f, Emax_te: %.5f, Time: %.3fs'%(ep, train_l2, test_l2, loss_max_test, T))
        time_step = time.perf_counter()
          
    print("\n=============================")
    print("Training done...")
    print("=============================\n")
    
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), 
                                              batch_size=1, shuffle=False)
    pre_test = torch.zeros(y_test.shape)     
    y_test   = torch.zeros(y_test.shape)      
    x_test   = torch.zeros(x_test.shape)      
    
    index = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            
            out_real = norm_y.decode(out.cpu())
            y_real   = norm_y.decode(y.cpu())
            
            x_real   = x
            x_real[:,:,0] = norm_x1.decode(x_real[:,:,0].cpu())
            x_real[:,:,1:] = norm_x2.decode(x_real[:,:,1:].cpu())
            
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
    
    pred_dict = {'pre_test'   : pre_test.cpu().detach().numpy(),
                    'x_test'  : x_test.cpu().detach().numpy(),
                    'y_test'  : y_test.cpu().detach().numpy(),
                    }
    
    sio.savemat(sava_path +'NORM_loss_' + args.CaseName + '.mat', mdict = loss_dict)                                                     
    sio.savemat(sava_path +'NORM_pre_'  + args.CaseName + '.mat', mdict = pred_dict)
    

    print('Training time: %.3f'%(time_step_end - time_start))
    print('Num of paras : %d'%(count_params(model)))
    

if __name__ == "__main__":
    
    class objectview(object):
        def __init__(self, d):
            self.__dict__ = d
            

    for i in range(5):
        
        i = i + 1
        for args in [
            { 'modes': 64,  
              'Fmodes': 16,
              'width': 16,
              'size_of_nodes': 1656,
              'batch_size': 10, 
              'epochs': 500,
              'data_dir': '../datasets/BloodFlow',
              'LBO_dir': '../datasets/BloodFlow_LBO_basis/LBO_basis',
              'num_train': 400, 
              'num_test': 100,
              'CaseName': 'velocity_xyz_'+str(i),
              'basis':'LBO',
              'lr' : 0.001},
        ]:
            args = objectview(args)
    
        main(args)

