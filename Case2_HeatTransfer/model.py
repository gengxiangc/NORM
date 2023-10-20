
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Approximation_block(nn.Module):
    
    def __init__ (self, in_channels, out_channels, modes):
        super(Approximation_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.float))

    def forward(self, x, LBO_MATRIX, LBO_INVERSE, label = 'True'):
        
        
        ################################################################
        # Encode
        ################################################################
        x = x = x.permute(0, 2, 1)
        x = LBO_INVERSE @ x  
        x = x.permute(0, 2, 1)
            
        ################################################################
        # Approximator
        ################################################################
        x = torch.einsum("bix,iox->box", x[:, :], self.weights1)

        ################################################################
        # Decode
        ################################################################
        x =  x @ LBO_MATRIX.T
        
        return x
    
        
class NORM_net(nn.Module):
    def __init__(self, modes, width, MATRIX_Output, INVERSE_Output, MATRIX_Input, INVERSE_Input):
        super(NORM_net, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) 
        self.fc01 = nn.Linear(self.width, self.width)
        self.LBO_Matri_input = MATRIX_Input
        self.LBO_Inver_input = INVERSE_Input
        self.LBO_Matri_output = MATRIX_Output
        self.LBO_Inver_output = INVERSE_Output
        
        self.conv_encode = Approximation_block(self.width, self.width, self.modes1)
        
        self.conv1 = Approximation_block(self.width, self.width, self.modes1)
        self.conv2 = Approximation_block(self.width, self.width, self.modes1)
        self.conv3 = Approximation_block(self.width, self.width, self.modes1)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        x = self.fc0(x)
        x  = F.gelu(x)
        x = self.fc01(x)
        x = x.permute(0, 2, 1)
        
        x1 = self.conv_encode(x, self.LBO_Matri_input, self.LBO_Inver_input)
        x2 = self.w0(x)
        x  = x1 + x2
        x  = F.gelu(x)

        '''
        from the Input manifold to the Output manifold
        '''
        x = self.conv1(x, self.LBO_Matri_output, self.LBO_Inver_input)
        x = F.gelu(x)
        
        x1 = self.conv2(x, self.LBO_Matri_output, self.LBO_Inver_output)
        x2 = self.w1(x)
        x  = x1  + x2
        
        x1 = self.conv3(x, self.LBO_Matri_output, self.LBO_Inver_output)
        x2 = self.w2(x)
        x  = x1  + x2

        x = x.permute(0, 2, 1)
        
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)     
