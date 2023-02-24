import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class CNN_VAE(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1,fc2):
        super(CNN_VAE, self).__init__()
 
        self.nc = nc
        self.hidden5=hidden5
        self.fc1=fc1
        self.fc2=fc2
 
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, hidden1, kernel, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden1, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden2, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden3, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden4, hidden5, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden5),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder=nn.DataParallel(self.encoder)
 
        self.fcE1 = nn.DataParallel(nn.Linear(fc1, fc2))
        self.fcE2 = nn.DataParallel(nn.Linear(fc1,fc2))
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden5, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden4, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden3, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden2, hidden1, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden1, nc, kernel, stride, padding, bias=False),
            nn.Sigmoid(),
        )
        self.decoder=nn.DataParallel(self.decoder)
 
        self.fcD1 = nn.Sequential(
            nn.Linear(fc2, fc1),
            nn.ReLU(inplace=True),
            )
        self.fcD1=nn.DataParallel(self.fcD1)
#         self.bn_mean = nn.BatchNorm1d(fc2)
 
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(-1, h.size()[1]*h.size()[2]*h.size()[3])
        return self.fcE1(h), self.fcE2(h)
 
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, z):
        h = self.fcD1(z)
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1/self.hidden5)), int(np.sqrt(self.fc1/self.hidden5)))
        return self.decoder(h)
 
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar
    
class CNN_VAE_hook(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1,fc2):
        super(CNN_VAE_hook, self).__init__()
 
        self.nc = nc
        self.hidden5=hidden5
        self.fc1=fc1
        self.fc2=fc2
 
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, hidden1, kernel, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden1, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden2, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden3, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden4, hidden5, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden5),
            nn.LeakyReLU(0.2, inplace=True),
        )
 
        self.fcE1 = nn.Linear(fc1, fc2)
        self.fcE2 = nn.Linear(fc1,fc2)
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden5, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden4, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden3, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden2, hidden1, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden1, nc, kernel, stride, padding, bias=False),
            nn.Sigmoid(),
        )
 
        self.fcD1 = nn.Sequential(
            nn.Linear(fc2, fc1),
            nn.ReLU(inplace=True),
            )
#         self.bn_mean = nn.BatchNorm1d(fc2)
        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def encode(self, x):
        h = self.encoder(x)
        hook = h.register_hook(self.activations_hook)
#         print(h.size())
        if torch.isnan(torch.sum(h)):
            print('convolution exploded')
        h = h.view(-1, h.size()[1]*h.size()[2]*h.size()[3])
        return self.fcE1(h), self.fcE2(h)
 
    def reparameterize(self, mu, logvar):
#         return mu
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, z):
        h = self.fcD1(z)
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1/self.hidden5)), int(np.sqrt(self.fc1/self.hidden5)))
        return self.decoder(h)
 
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar
    
    def get_activations_gradient(self):
        return self.gradients
    def get_activations(self, x):
        return self.encoder(x)
    
class CNN_VAE_sharded(nn.Module):
    def __init__(self, kernel, stride, padding, nc, hidden1, hidden2, hidden3, hidden4, hidden5, fc1,fc2):
        super(CNN_VAE_sharded, self).__init__()
 
        self.nc = nc
        self.hidden5=hidden5
        self.fc1=fc1
        self.fc2=fc2
 
        self.encoder = nn.Sequential(
            # input is nc x imsize x imsize
            nn.Conv2d(nc, hidden1, kernel, stride, padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden1) x imsize/stride^2
            nn.Conv2d(hidden1, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden2, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden3, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden4, hidden5, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden5),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder.cuda(0)
 
        self.fcE1 = nn.Linear(fc1, fc2)
        self.fcE2 = nn.Linear(fc1,fc2)
        self.fcE1.cuda(3)
        self.fcE2.cuda(3)
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden5, hidden4, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden4, hidden3, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden3, hidden2, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden2, hidden1, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(hidden1, nc, kernel, stride, padding, bias=False),
            nn.Sigmoid(),
        )
        self.decoder.cuda(0)
 
        self.fcD1 = nn.Sequential(
            nn.Linear(fc2, fc1),
            nn.ReLU(inplace=True),
            )
        self.fcD1.cuda(0)
#         self.bn_mean = nn.BatchNorm1d(fc2)
 
    def encode(self, x):
        h = self.encoder(x.cuda(0).float())
#         print(h.size())
        if torch.isnan(torch.sum(h)):
            print('convolution exploded')
        h = h.view(-1, h.size()[1]*h.size()[2]*h.size()[3])
        h=h.cuda(3)
        return self.fcE1(h), self.fcE2(h)
 
    def reparameterize(self, mu, logvar):
#         return mu
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
 
    def decode(self, z):
        h = self.fcD1(z.cuda(0))
        h = h.view(-1, self.hidden5, int(np.sqrt(self.fc1/self.hidden5)), int(np.sqrt(self.fc1/self.hidden5)))
        return self.decoder(h)
 
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar

    
class FC_l3(nn.Module):
    def __init__(self,inputdim,fcdim1,fcdim2,fcdim3, num_classes,dropout=0.5,regrs=True):
        super(FC_l3, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(inputdim, fcdim1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim1, fcdim2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim2, fcdim3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fcdim3, num_classes),            
        )
        self.regrs=regrs
        if regrs:
            self.regrsAct=nn.ReLU()

    def forward(self, x):
        x = self.classifier(x)
        if self.regrs:
            x=self.regrsAct(x)
        return x   

class FC_l1(nn.Module):
    def __init__(self,inputdim,fcdim1, num_classes,regrs=True):
        super(FC_l1, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(inputdim, fcdim1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fcdim1, num_classes),            
        )
        self.regrs=regrs
        if regrs:
            self.regrsAct=nn.ReLU()

    def forward(self, x):
        x = self.classifier(x)
        if self.regrs:
            x=self.regrsAct(x)
        return x   
    
class FC_l0(nn.Module):
    def __init__(self,inputdim, num_classes,regrs=True):
        super(FC_l0, self).__init__()
        
        self.classifier = nn.Linear(inputdim, num_classes)         
        self.regrs=regrs
        if regrs:
            self.regrsAct=nn.ReLU()

    def forward(self, x):
        x = self.classifier(x)
        if self.regrs:
            x=self.regrsAct(x)
        return x 
    
class FC_l5(nn.Module):
    def __init__(self,inputdim,fcdim1,fcdim2,fcdim3,fcdim4,fcdim5, num_classes,dropout=0.5,regrs=True):
        super(FC_l5, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(inputdim, fcdim1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim1, fcdim2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim2, fcdim3),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim3, fcdim4),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fcdim4, fcdim5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fcdim5, num_classes),            
        )
        self.regrs=regrs
        if regrs:
            self.regrsAct=nn.ReLU()

    def forward(self, x):
        x = self.classifier(x)
        if self.regrs:
            x=self.regrsAct(x)
        return x   