import torch
import torch.nn as nn
import torch.cuda
import torch.nn.functional as torch_func
import deep_convolution.utility_dpconv as pcutil
import deep_convolution.utility_pointconv as convutil


class DensityNet(nn.Module):
    def __init__(self, hidden_unit=[8, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.mlp_convs.append(nn.Conv1d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv1d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv1d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm1d(1))

    def forward(self, xyz_density):
        B, N = xyz_density.shape
        density_scale = xyz_density.unsqueeze(1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale = bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = torch_func.sigmoid(density_scale) + 0.5
            else:
                density_scale = torch_func.relu(density_scale)

        return density_scale

class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights = torch_func.relu(bn(conv(weights)))

        return weights

"""
    Hierarchical class for feature aggregation
"""

class AbsFeatureEstimates(nn.Module):
    def __init__(self, num_points, k, in_channel, mlp_layers, bandwidth, group_all):
        super(AbsFeatureEstimates, self).__init__()
        self.num_points = num_points
        self.num_samples = k
        self.mlp_conv_iter = nn.ModuleList()
        self.mlp_batchnorm_iter = nn.ModuleList()
        current_channel = in_channel
        for output_channel in mlp_layers:
            self.mlp_conv_iter.append(nn.Conv2d(current_channel, output_channel, 1))
            self.mlp_batchnorm_iter.append(nn.BatchNorm2d(output_channel))
            current_channel = output_channel

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp_layers[-1], mlp_layers[-1])
        self.bn_linear = nn.BatchNorm1d(mlp_layers[-1])
        self.densitynet = DensityNet()
        self.group_all = group_all
        self.bandwidth = bandwidth


    def forward(self, xyz, points):
        print("In AbsFeatureEstimate class ... ")
        print("xyz.shape: ", xyz.shape)
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        xyz_density = pcutil.compute_density(xyz, self.bandwidth)
        # import ipdb; ipdb.set_trace()
        density_scale = self.densitynet(xyz_density)

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm, grouped_density = \
                pcutil.sample_and_group_all(xyz, points, density_scale.view(B, N, 1))
        else:
            new_xyz, new_points, grouped_xyz_norm, _, grouped_density = \
                pcutil.sample_and_group(self.num_points, self.num_samples, xyz, points, density_scale.view(B, N, 1))

            # new_xyz: sampled points position data, [B, npoint, C]
            # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_conv_iter):
            bn = self.mlp_batchnorm_iter[i]
            new_points = torch_func.relu(bn(conv(new_points)))

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = new_points * grouped_density.permute(0, 3, 2, 1)
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2),
                                  other=weights.permute(0, 3, 2, 1)).view(B, self.num_points, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = torch_func.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


# class for feature encoding according to PointConv CVPR 2019

class FeatureEncoding(nn.Module):
    def __init__(self, npoints, radius, sigma, k, mlp):
        super(FeatureEncoding, self).__init__()
        self.npoints = npoints
        self.radius = radius
        self.sigma = sigma
        self.k = k


    def forward(self, xyz, features):
        B = xyz.shape[0]
        N = xyz.shape[2]
        num_points = xyz.shape[1]

        if num_points == self.npoints:
            new_xyz = xyz
        else:
            new_xyz = convutil.sampling(xyz, self.npoints)



# Deep Convolution For testing purpose

class ModelCreation(nn.Module):
    def __init__(self, num_classes=13):
        super(ModelCreation, self).__init__()
        self.abf1 = AbsFeatureEstimates(num_points=512, k=32, in_channel=3, mlp_layers=[64, 64, 128],
                                                  bandwidth=0.1, group_all=False)
        self.abf2 = AbsFeatureEstimates(num_points=128, k=64, in_channel=128 + 3, mlp_layers=[128, 128, 256],
                                                  bandwidth=0.2, group_all=False)
        self.abf3 = AbsFeatureEstimates(num_points=1, k=None, in_channel=256 + 3, mlp_layers=[256, 512, 1024],
                                                  bandwidth=0.4, group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)


    def forward(self, inp):
        print("[log:DeepConv]inp shape: ", inp.shape)
        inp_data, feature, _ = inp.split(3, dim=1)
        print("inp_data: ", inp_data.shape)
        B, _, _ = inp.shape
        l1_xyz, l1_points = self.abf1(inp_data, None)
        l2_xyz, l2_points = self.abf2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.abf3(l2_xyz, l2_points)
        print("l3_xyz shape: ", l3_xyz.shape)
        x = l3_points.view(B, 1024)
        print("x (after view): ", x.shape)
        x = self.drop1(torch_func.relu(self.bn1(self.fc1(x))))
        x = self.drop2(torch_func.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = torch_func.log_softmax(x, -1)
        return x

class GetLoss(nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(GetLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        print("pred.shape: ", pred.size())
        print("target.shape: ", target.size())
        print("trans_feat.shape: ", trans_feat.size())
        print("weight.shape: ", weight.size())
        loss = torch_func.nll_loss(pred, target, weight = weight)
        #mat_diff_loss = feature_transform_regularizer(trans_feat)
        #total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        #return total_loss
        return loss

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((32, 9 ,2048))
    #label = torch.randn(8,16)
    model = ModelCreation(num_classes=13)
    output= model(input)
    print(output.size())