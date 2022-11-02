import numpy as np
import torch
from torch import nn
from nnunet.network_architecture.generic_UNet import Generic_UNet_predict
from nnunet.network_architecture.initialization import InitWeights_He
import torchio as tio


class Classifier(nn.Module):
    def __init__(self, dims, dropout=True):
        super(Classifier, self).__init__()
        layers=[]
        
        for i in range(len(dims)-1):
            if dropout:
                layers.append(nn.Dropout(0.5))
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)   

# train classifier

# main


