import torch
import torch.nn as nn
import torch.cuda
import torch.nn.functional as torch_func

from pointnet.base_architecture import PointNetBaseArchitecture, feature_transform_regularizer

"""
    Segmentation Network model for PointNet (CVPR 2017).
    ModelCreation() class contains a MLP(512, 256, 128) that reduced to point features (n x 128) which is
    feed to another MLP(128 x m) to get the output scores (n x m) where, m = no. of obj classes. 

"""


class ModelCreation(nn.Module):
    def __init__(self, num_class, is_rgb=True):
        super(ModelCreation, self).__init__()
        if is_rgb:
            channel = 6
        else:
            channel = 3
        # Creating Skeleton of the Network
        print("Call in ModelCreation ..........")
        self.k = num_class
        self.feat = PointNetBaseArchitecture(feat_trans=True, global_feature=False, channel=channel)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, inp):
        print("In ModelCreation --> Batch Size: ", inp.size())
        # print("In ModelCreation --> self.chn: ", self.chn)
        batch_size = inp.size()[0]
        n_pts = inp.size()[2]
        inp, trans, trans_feat = self.feat(inp)
        inp = torch_func.relu(self.bn1(self.conv1(inp)))
        inp = torch_func.relu(self.bn2(self.conv2(inp)))
        inp = torch_func.relu(self.bn3(self.conv3(inp)))
        inp = self.conv4(inp)
        print("Input dim befor transpose op: ", inp.size())
        inp = inp.transpose(2, 1).contiguous()
        inp = torch_func.log_softmax(inp.view(-1, self.k), dim=-1)
        inp = inp.view(batch_size, n_pts, self.k)
        print("Input dim after transformation op: ", inp.size())
        print("just before return --> trans_feat.size()", trans_feat.size())
        return inp, trans_feat


class GetLoss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(GetLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        print("pred.shape: ", pred.size())
        print("target.shape: ", target.size())
        print("trans_feat.shape: ", trans_feat.size())
        print("weight.shape: ", weight.size())
        loss = torch_func.nll_loss(pred, target, weight=weight)
        # mat_diff_loss = feature_transform_regularizer(trans_feat)
        # total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        # return total_loss
        return loss


if __name__ == '__main__':
    model = ModelCreation(13, is_rgb=True)
    xyz = torch.rand(12, 9, 2048)
    print(model(xyz))
