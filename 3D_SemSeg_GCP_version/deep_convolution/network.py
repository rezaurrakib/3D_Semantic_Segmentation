import torch
import torch.nn as nn
import torch.nn.functional as F
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
                density_scale = F.sigmoid(density_scale) + 0.5
            else:
                density_scale = F.relu(density_scale)

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
            weights = F.relu(bn(conv(weights)))

        return weights

# class for feature encoding according to PointConv CVPR 2019


class FeatureEncodingLayer(nn.Module):
    def __init__(self, npoints, radius, sigma, k, mlp):
        super(FeatureEncodingLayer, self).__init__()
        self.npoints = npoints # No. of points selected in a specific layer
        self.radius = radius
        self.sigma = sigma
        self.k = k
        self.mlp = mlp
        self.density_net = DensityNet()
        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.mlp_conv = nn.ModuleList()
        self.mlp_bn = nn.ModuleList()

    def forward(self, xyz, features):
        batch = xyz.shape[0]
        data_points = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1) # dim(xyz): B x N x C
        features = features.permute(0, 2, 1)
        num_points = xyz.shape[2] # Number of points in the selected point set
        if num_points == self.npoints:
            xyz_fps = xyz
        else:
            xyz_fps = convutil.sampling(xyz, self.npoints) # Sampled points from point set using fps method
        print("Encoding Layer (before grouping): xyz.shape: ", xyz.shape, " xyz_fps.shape: ", xyz_fps.shape)
        grouped_xyz, grouped_features, idx = convutil.grouping(features, self.k, xyz, xyz_fps)
        print("[Encoding Layer]: grouped_xyz.shape: ", grouped_xyz.shape)
        print("[Encoding Layer]: grouped_features.shape: ", grouped_features.shape)
        print("[Encoding Layer]: idx.shape: ", idx.shape)
        density = convutil.compute_density(xyz, self.sigma)
        # print("density: ", density.shape)
        density_scale = self.density_net(density)
        print("[Encoding Layer]: density_scale.shape: ", density_scale.shape)
        density_scale = density_scale.view(batch, data_points, 1) # ([B, npoints, 1])
        print("[Encoding Layer]: density_scale.shape(after): ", density_scale.shape)
        grouped_density = convutil.index_points(density_scale, idx)

        # Processing grouped_features with a series of convolution ops
        current_channel = grouped_features.shape[3] # C+D
        for output_channel in self.mlp:
            self.mlp_conv.append(nn.Conv2d(current_channel, output_channel, 1))
            self.mlp_bn.append(nn.BatchNorm2d(output_channel))
            current_channel = output_channel

        grouped_features = grouped_features.permute(0, 3, 2, 1)  # [B x C+D x K x npoints]
        for i, conv in enumerate(self.mlp_conv):
            bn = self.mlp_bn[i]
            grouped_features = F.relu(bn(conv(grouped_features)))

        # print("grouped_features (after mlp ops): ", grouped_features.shape) # [B x mlp[-1] x K x npoints]
        grouped_xyz = grouped_xyz.permute(0, 3, 2, 1) # [B x C x K x npoints]
        weightnet = WeightNet(3, 16)
        weights = weightnet(grouped_xyz)
        grouped_density = grouped_density.permute(0, 3, 2, 1) # [B x 1 x K x npoints]
        print("[Encoding layer]: grouped_density.shape: ", grouped_density.shape)
        new_points = grouped_features * grouped_density
        new_points = new_points.permute(0, 3, 1, 2) # [B x npoints x mlp[-1] x k]
        weights = weights.permute(0, 3, 2, 1) # [B x npoints x K x 16]
        new_points = torch.matmul(new_points, weights) # [B x npoints x mlp[-1] x 16]
        new_points = new_points.view(batch, self.npoints, -1) # B x npoints x feat(= mlp[-1] * 16)]
        new_points = self.linear(new_points) # [B x npoints x mlp[-1]]
        # print("new_points (linear): ", new_points.shape) # [B x npoints x mlp[-1]]

        # Batch norm in last layer
        new_points = new_points.permute(0, 2, 1) # [B x mlp[-1] x npoints]
        new_points = F.relu(self.bn_linear(new_points))
        xyz_fps = xyz_fps.permute(0, 2, 1)
        print("new_points: ", new_points.shape)
        print("xyz_fps: ", xyz_fps.shape)
        return xyz_fps, new_points

# Feature Decoding Layer


class FeatureDecodingLayer(nn.Module):
    """
    @Author: Reza, TU Munich
    Class for decoding the feature embedding of point convolution ops.
    """
    def __init__(self, sigma, k, mlp):
        super(FeatureDecodingLayer, self).__init__()
        self.k = k
        self.sigma = sigma
        self.linear = nn.Linear(16 * (mlp[-1]+3), mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.density_net = DensityNet()
        self.weightnet = WeightNet(3, 16)
        self.mlp_conv = nn.ModuleList()
        self.mlp_bn = nn.ModuleList()
        self.mlp = mlp

    def forward(self, xyz1, xyz2, feat_points1, feat_points2):
        batch = xyz1.shape[0]
        npoints = xyz1.shape[2]
        dist, idx = convutil.three_nn_cpu_version(xyz1, xyz2)
        dist = torch.clamp(dist, min=1e-10)
        print(dist.shape)
        norm = torch.sum((1/dist), 2, keepdim=True)
        print(norm.shape)
        norm = norm.repeat(1, 1, 3)
        print("After repeating norm.shape: ", norm.shape)
        weights = (1.0 / dist) / norm # dim: [B x npoints x 3]
        feat_points2 = feat_points2.permute(0, 2, 1) # dim: [B x npoints x channels]
        interpolated_points = convutil.three_interpolation_efc(feat_points2, idx, weights) # dim: [B x npoints x channels]

        # setup for deConv
        xyz1 = xyz1.permute(0, 2, 1) # dim: [B x npoints x 3]
        # xyz2 = xyz2.permute(0, 2, 1)
        print("[Decoding Layer (before grouping)]: xyz1.shape: ", xyz1.shape, " xyz1.shape: ", xyz1.shape)
        grouped_xyz, grouped_features, idx = convutil.grouping(interpolated_points, self.k, xyz1, xyz1)
        # grouped_xyz.shape: [B x npoints x k x 3]
        # grouped_features.shape: [B x npoints x k x (channels + D) (D=3 is (x, y, z) point coordinates)]
        # idx.shape: [B x npoints x k]

        density = convutil.compute_density(xyz1, self.sigma)
        density_scale = self.density_net(density) # dim: [B x 1 x npoints]
        density_scale = density_scale.permute(0, 2, 1) # dim: [B x npoints x 1]
        grouped_density = convutil.index_points(density_scale, idx) # dim: [B x npoints x k x 1]
        grouped_xyz = grouped_xyz.permute(0, 3, 2, 1)  # dim: [B x 3 x k x npoints]
        weightnet = WeightNet(3, 16)
        weights = weightnet(grouped_xyz) # dim: [B x wgt_channels x k x npoints]
        # print("[Decoding Layer]: weights.shape: ", weights.shape)
        grouped_density = grouped_density.permute(0, 3, 2, 1)  # [B x 1 x k x npoints]
        grouped_features = grouped_features.permute(0, 3, 2, 1)  # [B x (channels + D) x k x npoints]
        new_points = grouped_features * grouped_density # [B x (channels + D) x k x npoints]
        new_points = new_points.permute(0, 3, 1, 2)  # [B x npoints x (channels + D) x k]
        weights = weights.permute(0, 3, 2, 1) # [B x npoints x k x wgt_channels]
        new_points = torch.matmul(new_points, weights)  # [B x npoints x (channels + D) x k]
        new_points = new_points.permute(0, 2, 3, 1) # [B x (channels + D) x k x npoints]
        print("new_points.shape: (before conv2d)", new_points.shape)

        # conv2d ops on new_points
        inp_ch = new_points.shape[1]  # (channels + D)
        conv = nn.Conv2d(inp_ch, self.mlp[0], 1)
        bn = nn.BatchNorm2d(self.mlp[0])
        new_points = F.relu(bn(conv(new_points))) # [B x mlp[0] x k x npoints]

        print("new_points.shape (after conv ops): ", new_points.shape)
        new_points = new_points.permute(0, 3, 2, 1) # [B x npoints x k x channels/mlp[0]]
        new_points = torch.reshape(new_points, (batch, npoints, 1, -1)) # dim: [B x npoints x 1 x mlp[0]*k]
        # print("new_points.shape: after reshape ", new_points.shape)
        feat_points1 = feat_points1.permute(0, 2, 1)  # dim: [B x npoints x channels]
        feat_points1 = feat_points1.unsqueeze(2) # dim: [B x npoints x 1 x channels]
        print("feat_points1.shape: (after expand_dim) ", feat_points1.shape)

        if feat_points1 is not None:
            new_points1 = torch.cat([new_points, feat_points1], dim=3) # dim: [B x npoints x 1 x (chn1 + chn2)]
        else:
            new_points1 = new_points

        print("new_points1.shape: ", new_points1.shape)
        new_points1 = new_points1.permute(0, 3, 2, 1) # dim: [B x (chn1 + chn2) x 1 x npoints]

        # Processing new_points1 with a series of convolution ops
        current_channel = new_points1.shape[1]  # chn1 + chn2
        print("current channel: ", current_channel)
        for output_channel in self.mlp:
            self.mlp_conv.append(nn.Conv2d(current_channel, output_channel, 1))
            self.mlp_bn.append(nn.BatchNorm2d(output_channel))
            current_channel = output_channel

        for i, conv in enumerate(self.mlp_conv):
            bn = self.mlp_bn[i]
            new_points1 = F.relu(bn(conv(new_points1)))

        print("new_points1.shape (after BN): ", new_points1.shape)
        new_points1.squeeze_(2)
        new_points1 = new_points1.permute(0, 2, 1) # B,ndataset1,mlp[-1]
        print("new_points.shape: ", new_points1.shape)
        return new_points1


class DeepConv(nn.Module):
    def __init__(self, num_classes):
        super(DeepConv, self).__init__()
        self.encode_layer_1 = FeatureEncodingLayer(npoints=512, radius=0.1, sigma=0.05, k=32, mlp=[16, 16, 32])
        self.encode_layer_2 = FeatureEncodingLayer(npoints=256, radius=0.2, sigma=0.05 * 2, k=32, mlp=[32, 32, 64])
        self.encode_layer_3 = FeatureEncodingLayer(npoints=64, radius=0.4, sigma=0.05 * 4, k=32, mlp=[64, 64, 128])
        self.encode_layer_4 = FeatureEncodingLayer(npoints=36, radius=0.8, sigma=0.05 * 8, k=32, mlp=[128, 128, 256])

        self.decode_layer_1 = FeatureDecodingLayer(sigma=0.05 * 8, k=16, mlp=[256, 256])
        self.decode_layer_2 = FeatureDecodingLayer(sigma=0.05 * 4, k=16, mlp=[128, 128])
        self.decode_layer_3 = FeatureDecodingLayer(sigma=0.05 * 2, k=16, mlp=[128, 64])
        self.decode_layer_4 = FeatureDecodingLayer(sigma=0.05, k=16, mlp=[64, 64, 64])


        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, point_cloud):
        batch, feature, xyz = point_cloud.shape
        xyz, rgb, norm_xyz = point_cloud.split(3, dim=1) # point_cloud is the initial input point set
        feature = torch.cat([rgb, norm_xyz], dim=1)
        # print("feature.shape: ", feature.shape)
        layer_0_xyz = xyz
        layer_0_features = feature
        layer_1_xyz, layer_1_features = self.encode_layer_1(layer_0_xyz, layer_0_features)
        layer_2_xyz, layer_2_features = self.encode_layer_2(layer_1_xyz, layer_1_features)
        layer_3_xyz, layer_3_features = self.encode_layer_3(layer_2_xyz, layer_2_features)
        layer_4_xyz, layer_4_features = self.encode_layer_4(layer_3_xyz, layer_3_features)
        layer_3_points = self.decode_layer_1(layer_3_xyz, layer_4_xyz, layer_3_features, layer_4_features)
        layer_2_points = self.decode_layer_2(layer_2_xyz, layer_3_xyz, layer_2_features, layer_3_points)
        layer_1_points = self.decode_layer_3(layer_1_xyz, layer_2_xyz, layer_1_features, layer_2_points)
        layer_0_points = self.decode_layer_4(layer_0_xyz, layer_1_xyz, layer_0_features, layer_1_points)



if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((32, 9, 1024))
    #label = torch.randn(8,16)
    model = DeepConv(num_classes=13)
    # output= model(input)
    model(input)
    # print(output.size())