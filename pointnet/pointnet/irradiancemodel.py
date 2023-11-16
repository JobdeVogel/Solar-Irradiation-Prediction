from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys

class STN3d(nn.Module):
    # Feature transformer
    def __init__(self, device='cuda:0'):
        super(STN3d, self).__init__()
        self.device = device
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Identity matrix
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.to(self.device)
        x = x + iden
        
        # Matrix 3x3
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    # Second feature transformation
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        
        # Output is 64x64
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    # Combines everything from input to global features
    
    def __init__(self, device='cuda:0', global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(device=device)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x, meta=None):      
        n_pts = x.size()[2]
        trans = self.stn(x)
        
        x = x.transpose(2, 1)
        
        # Multiplication of input and first feature transformation
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        # Multiplication with second feature transformation
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        '''
        # ! Here the global features are concatenated with the output from the feature transform
        
        # That would mean that the normal data should be added HERE, just after getting the pointfeatures
            * pointfeat shape:  [32, 64, 2500]
            * normal shape: [32, 3, 2500]
            
            Should be joined to [32, 67, 2500]
            
            x = torch.cat((pointfeat, normals), dim=1)
            
        This is then combined with x of shape [32, 1024] (global features)
            
            * x shape: [32, 67, 2500]
            * global_features shape [32, 1024]
            
            To:
            
            * [32, 1091, 1024]   
        '''
        
        # ! New implementation with meta:
        if meta != None:
            pointfeat = torch.cat((pointfeat, meta), dim=1)
        
        ''' AS USUAL FROM HERE '''
        
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetDenseCls(nn.Module):
    # Add the segmentation part
    
    def __init__(self, k = 2500, m=0, feature_transform=False, config=None, device='cuda:0'):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.m = m
        self.device = device
        self.feature_transform = feature_transform
        self.config = config
        self.feat = PointNetfeat(global_feat=False, device=device, feature_transform=feature_transform)
        
        # ! Here we should change 1088 to 1091 since normal data is added
        self.conv1 = torch.nn.Conv1d(1088 + self.m, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        
        if self.config == None:
            self.conv4 = torch.nn.Conv1d(128, 4, 1)
            
            self.fc1 = nn.Linear(4 * self.k, 1024)  # Add an FC layer to reduce dimensionality
            self.fc2 = nn.Linear(1024, 512)  # Add another FC layer
            self.fc3 = nn.Linear(512, self.k)  # The final FC layer with 'k' output neurons
        else:           
            fc1_kernels = self.config.fc1
            fc2_kernels = self.config.fc2
            fc3_kernels = self.config.fc3
            
            self.conv4 = torch.nn.Conv1d(128, fc1_kernels, 1)
            
            self.fc1 = nn.Linear(fc1_kernels * self.k, self.k)
            self.fc2 = nn.Linear(fc2_kernels, fc3_kernels)
            self.fc3 = nn.Linear(fc3_kernels, self.k)
            
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x, meta=None):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        
        x, trans, trans_feat = self.feat(x, meta)       
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        
        # Flatten the output of the last convolutional layer
        x = x.view(batchsize, -1)
        
        # Pass it through FC layers
        x = self.fc1(x)    

        # x = self.fc2(x)    
        # x = self.fc3(x)  # No activation function for the final output
        
        return x, trans, trans_feat

class dummy(nn.Module):
    # Add the segmentation part
    
    def __init__(self, k = 2500, m=0, feature_transform=False, single_output=False, config=None, device='cuda:0'):
        super(dummy, self).__init__()
        self.k = k
        self.fc1 = nn.Linear(3 * self.k, self.k)

    def forward(self, x, meta=None):
        batchsize = x.size()[0]
    
        x = x.reshape(batchsize, -1)
        
        x = self.fc1(x)
        
        # x = x.view(-1)
        
        return x, None, None


def init_weights(m, config=None):
    if config != None:
        if config.initialization == 'kaiming':
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif config.initialization == 'xavier':
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif config.initialization == None:
            pass
        else:
            print('WARNING: weight initialization type does not exist!')
            sys.exit()

def feature_transform_regularizer(trans, device='cuda:0'):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.to(device)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    normal_data = Variable(torch.rand(32,3,2500))


    sim_data = sim_data[:, 0, :]
    seg = dummy(k=2500, m=3)
    
    out, _, _ = seg(sim_data, meta=normal_data)
    
    print(out)