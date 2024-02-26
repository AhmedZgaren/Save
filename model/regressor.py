import torch.nn as nn

class RegNet(nn.Module):
    '''
    A regressor network to predict the final count
    multiple linear layers 
    '''
    
    def __init__(self,d_model, dropout = 0.1):
        super(RegNet, self).__init__()

        self.fc1 = nn.Linear(d_model, 1)
        self.dr = nn.Dropout(dropout)
       
        
    def forward(self, x):
        out = self.fc1(x)
        return out