import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as torch_func

from torch.autograd import Variable


class InputTransNet(nn.Module):
    def __init__(self, channel):
        super(InputTransNet, self).__init__()
        # defining conv layers
        self.conv1 = nn.Conv1d(9, 64, 1) # Kernel Size=1
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # defining fc layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.relu = nn.ReLU
        # self.leaky_relu = nn.LeakyReLU

        # defining batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, inp):
        batch_size = inp.size()[0]
        print("InputTransNet --> inp.size(): ", inp.size()) # 32 x 9 x 128
        inp = torch_func.relu(self.bn1(self.conv1(inp)))
        inp = torch_func.relu(self.bn2(self.conv2(inp)))
        inp = torch_func.relu(self.bn3(self.conv3(inp)))
        inp = torch.max(inp, 2, keepdim=True)[0]
        inp = inp.view(-1, 1024)

        inp = torch_func.relu(self.bn4(self.fc1(inp)))
        inp = torch_func.relu(self.bn5(self.fc2(inp)))
        inp = self.fc3(inp)

        identity_mat = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batch_size, 1)
        if inp.is_cuda:
            identity_mat = identity_mat.cuda()
        inp = inp + identity_mat
        inp = inp.view(-1, 3, 3)
        print("InputTransNet --> inp.size() after op: ", inp.size()) # 32 x 3 x 3

        return inp


"""
    Class for Semantic Segmentation with Feature Transform 
"""


class FeatureTransNet(nn.Module):
    def __init__(self, k=64):
        super(FeatureTransNet, self).__init__()
        # defining convolution layers
        self.conv1 = nn.Conv1d(k, 64, 1)  # Kernel Size=1
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # defining fc layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()
        # self.leaky_relu = nn.LeakyReLU

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.dim = k

    def forward(self, inp):
        batch_size = inp.size()[0]
        inp = torch_func.relu(self.bn1(self.conv1(inp)))
        inp = torch_func.relu(self.bn2(self.conv2(inp)))
        inp = torch_func.relu(self.bn3(self.conv3(inp)))
        inp = torch.max(inp, 2, keepdim=True)[0]
        inp = inp.view(-1, 1024)

        inp = torch_func.relu(self.bn4(self.fc1(inp)))
        inp = torch_func.relu(self.bn5(self.fc2(inp)))
        inp = self.fc3(inp)

        identity_mat = Variable(torch.from_numpy(np.eye(self.dim).flatten().astype(np.float32))).view(1, self.dim * self.dim).repeat(batch_size, 1)
        if inp.is_cuda:
            identity_mat = identity_mat.cuda()
        inp = inp + identity_mat
        inp = inp.view(-1, self.dim, self.dim)
        return inp


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss


class PointNetBaseArchitecture(nn.Module):
    def __init__(self, feat_trans=False, global_feature=True, channel=3):
        super(PointNetBaseArchitecture, self).__init__()
        self.inp_transformation = InputTransNet(channel)
        self.feature_transform = feat_trans
        if self.feature_transform:
            self.seg_feat_transformation = FeatureTransNet(k=64)

        # Define convolution operation
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # Define Batch Normalization operation
        # This is used for shared MLP with output size (64, 128, 1024)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.global_feature = global_feature

    def forward(self, inp_data):
        batch, dim, N = inp_data.size() # default: 32 x 9 x 4096
        print("In PointNetBaseArchitecture --> batch: ", batch, " dimension: ", dim, " N: ", N)
        trans = self.inp_transformation(inp_data)
        print("trans.size()  : ", trans.size())
        inp_data = inp_data.transpose(2, 1)
        print("ist inp_data  : ", inp_data.size())
        if dim > 3:
            inp_data, feature, _ = inp_data.split(3, dim=2)
        print("feature size: ", feature.size())
        inp_data = torch.bmm(inp_data, trans)
        print("bmm er pore : ", inp_data.size())
        if dim > 3:
            inp_data = torch.cat([inp_data, feature], dim=2)
        inp_data = inp_data.transpose(2, 1)
        #print("Bari khawar age : ", inp_data.size())
        inp_data = torch_func.relu(self.bn1(self.conv1(inp_data))) # 32 x 64 x 128

        if self.feature_transform:
            trans_feat = self.seg_feat_transformation(inp_data)
            print("trans_feat.size(): ", trans_feat.size()) # 32 x 64 x 64
            inp_data = inp_data.transpose(2, 1)
            print("feat-trans --> inp_data: ", inp_data.size())
            inp_data = torch.bmm(inp_data, trans_feat)
            inp_data = inp_data.transpose(2, 1)
            print("feature transform --> inp:", inp_data.size()) # 32 x 64 x 128
        else:
            trans_feat = None

        pointfeat = inp_data
        inp_data = torch_func.relu(self.bn2(self.conv2(inp_data)))
        inp_data = self.bn3(self.conv3(inp_data))
        inp_data = torch.max(inp_data, 2, keepdim=True)[0]
        inp_data = inp_data.view(-1, 1024)
        if self.global_feature:
            return inp_data, trans, trans_feat
        else:
            inp_data = inp_data.view(-1, 1024, 1).repeat(1, 1, N)
            #print("final --> inp_data", inp_data.size())
            #print("pointfeat.size(): ", pointfeat.size())
            #c = torch.cat([inp_data, pointfeat], 1)
            #print("c shape: ", c.size())
            return torch.cat([inp_data, pointfeat], 1), trans, trans_feat






