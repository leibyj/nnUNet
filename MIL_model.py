import torch
from torch import nn
import torch.nn.functional as F

class ATTN_module(nn.Module):
  def __init__(self, dims = [512, 256], dropout = True):
    """
    Basic 2 layer attention network
    Forward function will return attention tensor (Px1) and original input
    """
    super(ATTN_module, self).__init__()

    self.net = [nn.Linear(dims[0], dims[1])]
    if dropout:
            self.net.append(nn.Dropout(0.25))

    self.net.append(nn.ReLU())
    self.net.append(nn.Linear(dims[1], 1))

    self.net = nn.Sequential(*self.net)


  def forward(self, x):
    return self.net(x), x


class MIL_model(nn.module):
    def __init__(self, dims = [1120, 512, 256], dropout = True):
        super(MIL_model, self).__init__()

    fc = [nn.Linear(dims[0], dims[1]), nn.ReLU()]
    if dropout:
        fc.append(nn.Dropout(0.25))
    fc.extend([nn.Linear(dims[1], dims[1]), nn.ReLU()])
    if dropout:
        fc.append(nn.Dropout(0.25))

    fc.append(ATTN_module([dims[1], dims[2]], dropout = dropout))

    self.attn_net = nn.Sequential(*fc)

    self.classifier = nn.Sequential(
        nn.Linear(dims[1], dims[2]),
        nn.ReLU(),
        nn.Linear(dims[2], 1),
        nn.Sigmoid()
        )

    def forward(self, x):
        A, x = self.attn_net(x) # A.shape == Px1
        A = F.softmax(A, dim=1)

        # A * feature map
        x = torch.mm(A, x) # x.shape == Px512

        # collapse 
        x = x.sum(dim=0) # x.shape == 512
        out = self.classifier(x)

        return out, A



# class Gated_ATTN_module(nn.Module):
#   """
#   Gated attention network designed to take in intermediate VGG convolution
#   """
#   def __init__(self, pool_dim = 7, inter_dim = 1024):
#     super(Gated_ATTN_module, self).__init__()
#     self.attn_base = nn.Sequential(
#         nn.AdaptiveAvgPool2d(pool_dim),
#         nn.Flatten(),
#         nn.Linear(256*pool_dim*pool_dim, inter_dim),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#     )

#     self.attn_t = nn.Sequential(
#         nn.Linear(inter_dim, 256),
#         nn.Tanh()
#     )

#     self.attn_s = nn.Sequential(
#         nn.Linear(inter_dim, 256),
#         nn.Sigmoid()
#     )

#     self.attn_out = nn.Linear(256, 1, bias=False)

#   def forward(self, x):
#     x = self.attn_base(x)
#     b1 = self.attn_t(x)
#     b2 = self.attn_s(x)
#     x = torch.mul(b1, b2)
#     x = self.attn_out(x)
#     return x


# class Classifier(nn.Module):
#     def __init__(self, dims, dropout=True):
#         super(Classifier, self).__init__()
#         layers=[]
        
#         for i in range(len(dims)-1):
#             if dropout:
#                 layers.append(nn.Dropout(0.5))
#             layers.append(nn.Linear(dims[i], dims[i+1]))
#             layers.append(nn.ReLU())
            
#         layers.append(nn.Linear(dims[-1], 1))
#         layers.append(nn.Sigmoid())
        
#         self.net = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.net(x)   
