
import torch
import numpy as np
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Spatial_Approximation_block(nn.Module):
    def __init__ (self, in_channels, out_channels, modes, Fmodes, LBO_MATRIX, LBO_INVERSE, TIME_MATRIX, TIME_INVERSE):
        super(Spatial_Approximation_block, self).__init__()
        
        '''
        Approximation_block for space-dimension
        '''
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.modes1 = modes 
        self.modes2 = Fmodes 
        self.LBO_MATRIX = LBO_MATRIX 
        self.LBO_INVERSE = LBO_INVERSE 
        
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.modes1, in_channels, out_channels, dtype=torch.float)) 
        
                      
    def forward(self, x):  
        
        x = x.permute(0,3,1,2) 
        x = torch.einsum("txbi,xio->txbo", x, self.weights1) 
        return x
    
  
    
class Temporal_Approximation_block(nn.Module):
    def __init__ (self, in_channels, out_channels, modes, Fmodes, LBO_MATRIX, LBO_INVERSE, TIME_MATRIX, TIME_INVERSE):
        super(Temporal_Approximation_block, self).__init__()
        
        '''
        Approximation_block for time-dimension
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.modes1 = modes              
        self.modes2 = Fmodes 
        self.TIME_MATRIX = TIME_MATRIX
        self.TIME_INVERSE = TIME_INVERSE 

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.modes2, in_channels, out_channels, dtype=torch.cfloat)) 
        
    def compl_mul1d(self, input, weights):
        
        return torch.einsum("xtbi,tio->xtbo", input, weights)
                      
    def forward(self, x): 
    
        out = torch.zeros(self.modes1, x.size(1), x.size(2), self.in_channels, device=x.device, dtype=torch.float)  
        out[:, :self.modes2, :, :] = self.compl_mul1d(x[:, :self.modes2, :, :], self.weights1) 
        
        return out
    
class Spatiotemporal_Parameterization(nn.Module):
    def __init__ (self, nodes1, nodes2, width):
        super(Spatiotemporal_Parameterization, self).__init__()

        '''
        Approximation_block for space&time-dimension
        '''
        self.in_channels = nodes1 
        self.out_channels = nodes2
        self.modes1 =  width 
        
        self.scale = (1 / (self.in_channels*self.out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.modes1, self.in_channels, self.out_channels, dtype=torch.float)) 
        
    def forward(self, x):  
        
        x = x.permute(0,3,1,2) 
        x = torch.einsum("txbi,xio->txbo", x, self.weights1) 
        
        return x   

        
class NORM_net(nn.Module):
    def __init__(self, modes, nodes,Fmodes, width, TIME_MATRIX, TIME_INVERSE, LBO_MATRIX, LBO_INVERSE, Nt):
        super(NORM_net, self).__init__()

        self.modes1 = modes
        self.modes2 = Fmodes
        self.width = width
        self.padding = 2 
        self.fc0 = nn.Linear(6, self.width) 
        
        self.TIME_MATRIX = TIME_MATRIX
        self.TIME_INVERSE = TIME_INVERSE
        self.LBO_MATRIX = LBO_MATRIX 
        self.LBO_INVERSE = LBO_INVERSE 
        self.Nx = LBO_MATRIX.size(0)
        self.Nt = Nt     
        self.nodes = nodes

        self.convt = Spatiotemporal_Parameterization(self.nodes, self.nodes, self.width)
        
        self.conv0 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX,self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)  
        self.conv1 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2,self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)
        self.conv2 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv3 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        
        self.conv4 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv5 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv6 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv7 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )

        self.w0 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x): 
        
        '''
        Input:(n*Nt*6)
        extend channel from 6 to width, obtain n*Nt*width
        '''
        x = self.fc0(x) 
        
        '''
        Prject time-domain to the Frequency-domain, here we can use FFT or 1D LBO
        '''
        x = self.Fmapping_low(x)
        
        '''
        Add a new frequency channel for spatial-domain
        Extend the width of the new channel
        '''
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
        x = self.Extend(x,self.modes1)
        
        '''
        Project the constructed frequency weight back to original domain
        '''
        x1 = x.permute(3, 1, 0, 2)
        x1 = self.iFmapping(x1, self.Nt)
        x1 = self.iLmapping(x1, self.LBO_MATRIX)
        
        '''
        Parameterize on Spatiotemporal domain to increase the expressiveness of the model
        '''
        x = x1.permute(0,1,3,2)
        x = self.convt(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 1, 3)
        

        '''
        layer 1
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
        
        x1 = self.conv0(x1)  
        x1 = self.Fmapping(x1, self.modes2)        
        x1 = self.conv4(x1)           
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3) 
        x2 = self.w0(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 2
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)    
        x1 = self.conv1(x1) 
        x1 = self.Fmapping(x1, self.modes2) 
        x1 = self.conv5(x1) 
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3) 
        x2 = self.w1(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 3
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)      
        x1 = self.conv2(x1) 
        x1 = self.Fmapping(x1, self.modes2)
        x1 = self.conv6(x1) 
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3)
        x2 = self.w2(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 4
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
        x1 = self.conv3(x1)                
        x1 = self.Fmapping(x1, self.modes2)         
        x1 = self.conv7(x1)           
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3)
        x2 = self.w3(x2) 

        x2 = x2.permute(2, 0, 1, 3) 
        x = x1 + x2

        x = x.permute(0, 1, 3, 2)
        x = self.fc1(x) 
        x = torch.relu(x)
        x = self.fc2(x) 
        
        x = x.permute(1, 2, 0, 3)
        
        return x  

    def get_grid(self, shape, device):
        timenodes, batchsize, size_x = shape[2], shape[0], shape[1] 
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([timenodes, batchsize, 1, 1])
        return gridx.to(device)     
    
    def Lmapping(self, x, LBO_MATRIX, LBO_INVERSE):

        x = x = x.permute(0,1,3,2) 
        x = self.LBO_INVERSE @ x  
        x = x.permute(0,1,3,2) 
        return x
        
               
    def Fmapping(self, x, modes2): 

        x = x.permute(1,2,3,0)    
        x_ft = torch.fft.rfft(x)  
        x_ft = x_ft.permute(0,3,1,2) 
        return x_ft
    
    def iFmapping(self, x, Nt):

        x = x.permute(2,3,0,1) 
        
        x_rft = torch.fft.irfft(x, Nt)  
        
        return x_rft
        
    def iLmapping(self, x, LBO_MATRIX): 

        x = x.permute(3,0,1,2) 
        
        x = x @ LBO_MATRIX.T 
        
        return x
    
    def Fmapping_low(self, x): 

        x = x.permute(0,2,1)  
        
        x_ft = torch.fft.rfft(x) 
        
        x_ft = x_ft.permute(0,2,1) 
        
        return x_ft
    
    def Extend(self, x, modes): 

        scale = (1 / (x.shape[2] * modes))
        weights1 = nn.Parameter(scale*torch.rand(x.shape[1], x.shape[2], modes, dtype=torch.float)).cuda()
        x = torch.einsum("txbi,xio->txbo", x, weights1)
        
        return x
        

        
    
