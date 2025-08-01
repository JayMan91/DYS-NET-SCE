import torch
from torch import nn
import math
from torch.nn import BatchNorm1d as BN
from math import sqrt
# build linear model
class LinearRegression(nn.Module):
    '''
    num_feature: dimension of the feature vector
    num_cost: dimension of the cost vector
    '''
    def __init__(self,num_feature,  num_cost, squeeze= False):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feature,  num_cost )
        # nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.zeros_(self.linear.bias)
        self.squeeze = squeeze

    def forward(self, x):
        out = self.linear(x)
        if self.squeeze:
            out = out.squeeze(-1) 
        return out
class LinearRegressionShared(nn.Module):
    '''
    num_feature: dimension of the feature vector
    num_cost: dimension of the cost vector
    '''
    def __init__(self, num_feature,  num_cost = 1 , squeeze= True):
        super(LinearRegressionShared, self).__init__()
        self.linear = nn.Linear(num_feature,  num_cost )
        # nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.zeros_(self.linear.bias)
        self.squeeze = squeeze

    def forward(self, x):
        out = self.linear(x)
        if self.squeeze:
            out = out.squeeze(-1) 
        return out

### Initialization 
        # nn.init.constant_(self.linear.weight, -1)
        # nn.init.zeros_(self.linear.bias)

class nnsolve(nn.Module):
    def __init__(self , num_sol,  n_layers):
        '''
        num_sol: dimension of the solution and the cost vector
        n_layers: Number of hidden layers
        '''
        super().__init__()

        if n_layers ==0:
            self.linear = nn.Linear(num_sol,  num_sol )
        else:
            n_hidden = 1 #int(math.sqrt(num_sol))
            hiddens= [nn.Linear(num_sol, n_hidden )]
            hiddens.append(nn.ReLU())
            for i in range(n_layers-1):
                hiddens.append( nn.Linear(n_hidden, n_hidden))
                hiddens.append(nn.ReLU())
            hiddens.append(nn.Linear(n_hidden, num_sol ))
            self.linear = nn.Sequential(*hiddens)
        # self.n_layers = n_layers
        # if self.n_layers >0:
        #     n_hidden = int(math.sqrt(num_sol))
        #     hiddens= [nn.Linear(num_sol, n_hidden )]
        #     hiddens.append(nn.ReLU())
        #     for i in range(n_layers):
        #         hiddens.append( nn.Linear(n_hidden, n_hidden))
        #         hiddens.append(nn.ReLU())
   
        #     hiddens.append(nn.Linear(n_hidden, num_sol ))
        #     self.hidden = nn.Sequential(*hiddens)

    def forward(self, x):
        out = self.linear(x)

        out = nn.Sigmoid()(out)
        # out = out/(torch.norm(out,float('inf'), dim=1)[:, None])
        # out = nn.functional.normalize(out, p=1.0, dim=1)
        # print (out) 
        return out  #nn.Softmax(dim=1)(out)


class convsolve(nn.Module):
    def __init__(self , num_sol,  n_layers):
        '''
        num_sol: dimension of the solution and the cost vector
        n_layers: Number of hidden layers
        '''
        super().__init__()
        self.conv = nn.Conv1d(1,1, 1)
        self.linear = nn.Linear(num_sol-1+1,  num_sol )
        

   
     


        # self.n_layers = n_layers
        # if self.n_layers >0:
        #     n_hidden = int(math.sqrt(num_sol))
        #     hiddens= [nn.Linear(num_sol, n_hidden )]
        #     hiddens.append(nn.ReLU())
        #     for i in range(n_layers):
        #         hiddens.append( nn.Linear(n_hidden, n_hidden))
        #         hiddens.append(nn.ReLU())
   
        #     hiddens.append(nn.Linear(n_hidden, num_sol ))
        #     self.hidden = nn.Sequential(*hiddens)

    def forward(self, x):
        out = self.conv(x.unsqueeze(1) )
        out = self.linear(out.squeeze(1) )

        out = nn.Sigmoid()(out)
        # out = out/(torch.norm(out,float('inf'), dim=1)[:, None])
        # out = nn.functional.normalize(out, p=1.0, dim=1)
        # print (out) 
        return out  #nn.Softmax(dim=1)(out)


## sigmoid followed by divide by norm gives 14.5; so is only divide by norm
### Howver, restoration is better if sigmoid is used
### Only divide by norm: 14.5
### Only Sigmois: 13.5
### Both Sigmoid followee by  divide by norm: 14.5
### None:  14.5
# class nnsolve(nn.Module):
#     def __init__(self, num_sol):
#         super().__init__()
        
#         n_hidden =  5 #int(math.sqrt(num_sol))
#         # self.linear1 =  nn.Sequential ( nn.Linear(num_sol,n_hidden ), nn.ReLU() )
#         # self.linear2 = nn.Sequential ( nn.Linear( n_hidden , num_sol)  )

#         self.linear = nn.Sequential ( nn.Linear(num_sol, n_hidden ), nn.ReLU(), BN(n_hidden),
#                                      nn.Linear(n_hidden, n_hidden ), nn.ReLU(), BN(n_hidden),
#                                      nn.Linear(n_hidden, 1 ) )


#         # self.linear =  nn.Sequential ( nn.Linear(num_sol, num_sol ), nn.ReLU(), BN(num_sol) )
#     def forward(self, x):
#         # out = self.linear1(x)
#         # out = self.linear2(out)

#         # # out = self.linear(x)

#         # # out = self.linear (x.unsqueeze(2) )
#         # out = nn.Sigmoid()(out)
    
#         # # out = nn.functional.normalize(out, p=1.0, dim=1)
        
#         # # print (out) 
#         out = self.linear (x)
#         return out  #nn.Softmax(dim=1)(out)
        
#         # return out  #nn.Softmax(dim=1)(out)

from torchvision.models import resnet18
class partialResNet(nn.Module):

    def __init__(self, k):
        super(partialResNet, self).__init__()
        # init resnet 18
        resnet = resnet18(pretrained=False)
        # first five layers of ResNet18
        self.conv1 = resnet.conv1
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool1 = resnet.maxpool
        self.block = resnet.layer1
        # conv to 1 channel
        self.conv2  = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
        # max pooling
        self.maxpool2 = nn.AdaptiveMaxPool2d((k,k))

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn(h)
        h = self.relu(h)
        h = self.maxpool1(h)
        h = self.block(h)
        h = self.conv2(h)
        out = self.maxpool2(h)
        # reshape for optmodel
        out = torch.squeeze(out, 1)
        out = out.reshape(out.shape[0], -1)
        return nn.ReLU()(out)


class CombRenset18(nn.Module):

    def __init__(self, out_features, in_channels):
        super().__init__()
        self.resnet_model = resnet18(pretrained=False, num_classes=out_features)
        del self.resnet_model.conv1
        self.resnet_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        output_shape = (int(sqrt(out_features)), int(sqrt(out_features)))
        self.pool = nn.AdaptiveMaxPool2d(output_shape)
        #self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)


    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        x = self.resnet_model.layer1(x)
        #x = self.resnet_model.layer2(x)
        #x = self.resnet_model.layer3(x)
        #x = self.last_conv(x)
        x = self.pool(x)
        x = x.mean(dim=1)
        # print ("Shape", x.shape)
        x = x.view(x.size(0), -1)
        return x


# class DYSCombRenset18(nn.Module):

#     def __init__(self, out_features, in_channels, num_totaledges):
#         super().__init__()
#         self.num_totaledges = num_totaledges
#         self.resnet_model = resnet18(pretrained=False, num_classes=out_features)
#         del self.resnet_model.conv1
#         self.resnet_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         output_shape = (int(sqrt(out_features)), int(sqrt(out_features)))
#         self.pool = nn.AdaptiveMaxPool2d(output_shape)
#         #self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)


#     def forward(self, x):
#         x = self.resnet_model.conv1(x)
#         x = self.resnet_model.bn1(x)
#         x = self.resnet_model.relu(x)
#         x = self.resnet_model.maxpool(x)
#         x = self.resnet_model.layer1(x)
#         #x = self.resnet_model.layer2(x)
#         #x = self.resnet_model.layer3(x)
#         #x = self.last_conv(x)
#         x = self.pool(x)
#         x = x.mean(dim=1)
#         # print ("Shape", x.shape)
#         x = x.view(x.size(0), -1)

        return x

