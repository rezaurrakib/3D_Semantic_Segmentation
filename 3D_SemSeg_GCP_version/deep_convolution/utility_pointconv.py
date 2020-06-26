import torch


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    #print("Batch Ind: ", batch_indices)
    #print("farthest: ", farthest)
    for i in range(npoint):
        centroids[:, i] = farthest
        #print("centroids: ", centroids.shape)
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def sampling(xyz, npoints):
    B, N, C = xyz.shape
    S = npoints
    fps_idx = farthest_point_sample(xyz, npoints)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    return new_xyz

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    # print("sqrdists.shape: ", sqrdists.shape)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    # print("group_idx: ", group_idx.shape)
    return group_idx

"""
def grouping(feature, K, src_xyz, q_xyz, use_xyz = True):
    '''
    K: neighbor size
    src_xyz: original point xyz (batch_size, ndataset, 3)
    q_xyz: query point xyz (batch_size, npoint, 3)
    '''

    batch_size = src_xyz.get_shape()[0]
    npoint = q_xyz.get_shape()[1]

    point_indices = tf.py_func(knn_kdtree, [K, src_xyz, q_xyz], tf.int32)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, npoint, K, 1))
    idx = tf.concat([batch_indices, tf.expand_dims(point_indices, axis = 3)], axis = 3)
    idx.set_shape([batch_size, npoint, K, 2])

    grouped_xyz = tf.gather_nd(src_xyz, idx)
    grouped_xyz -= tf.tile(tf.expand_dims(q_xyz, 2), [1,1,K,1]) # translation normalization

    grouped_feature = tf.gather_nd(feature, idx)
    if use_xyz:
        new_points = tf.concat([grouped_xyz, grouped_feature], axis = -1)
    else:
        new_points = grouped_feature
    
    return grouped_xyz, new_points, idx

"""

def compute_density(xyz, bandwidth):
    '''
    xyz: input points position data, [B, N, C]
    '''
    # import ipdb; ipdb.set_trace()
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim=-1)

    return xyz_density


def grouping(features, k, source_xyz, query_xyz):
    '''
    k: Neighbor size
    source_xyz: original point xyz (batch_size, ndataset, 3) [B x N x 3]
    query_xyz: query point xyz (batch_size, npoint, 3)
    return: grouped_xyz, new_points, idx
    '''

    batch_size = source_xyz.shape[0]
    npoints = query_xyz.shape[1]
    B, N, C = source_xyz.shape
    point_idx = knn_point(k, source_xyz, query_xyz) # point_idx shape: B x npoints x k
    grouped_xyz = index_points(source_xyz, point_idx) # [B x npoints x K x C]
    grouped_xyz_norm = grouped_xyz - query_xyz.view(B, npoints, 1, C) # [B x npoints x K x C]
    grouped_features = index_points(features, point_idx) # [B x npoints x K x D]
    new_points = torch.cat([grouped_xyz_norm, grouped_features], dim=-1) # [B x npoints x K x C+D]
    return grouped_xyz_norm, new_points, point_idx


def three_nn_cpu_version(xyz1, xyz2):
    """
    Find three nearest neigbors with square distance. Reimplementation of the C++ version in python 3
    :param xyz1: dimension: [B, N, 3]
    :param xyz2: dimensino: [B, M, 3], TF tensor, sparser than xyz1
    :return: dist [B, N, 3], idx [B, N, 3]
    """
    xyz1 = xyz1.permute(0, 2, 1)
    xyz2 = xyz2.permute(0, 2, 1)
    print("xyz1.shape: ", xyz1.shape)
    print("xyz2.shape: ", xyz2.shape)
    sqrdists = square_distance(xyz1, xyz2)
    dist, idx = torch.topk(sqrdists, 3, dim=-1, largest=False, sorted=False)
    print("dist.shape: ", dist.shape)
    print("idx.shape: ", idx.shape)
    #print(dist)
    return dist, idx


def three_interpolate(points, idx, weights):
    """

    :param points: [B, M, C]
    :param idx: [B, N, 3]
    :param weights: [B, N, 3]
    :return: [B, N, C]
    """
    B, M, C = points.shape
    print("M: ", M, "C: ", C)
    _, N, _ = idx.shape
    interpolated_points = torch.randn(B * N * C)
    w1, w2, w3 = weights[0, 0, :]
    print("w1: ", w1, " w2: ", w2, " w3: ", w3)
    for b in range(B):
        for j in range(N):
            w1, w2, w3 = weights[b, j, :]
            i1, i2, i3 = idx[b, j, :]
            for l in range(C):
                interpolated_points[j*C+l] = points[b, i1, l] * w1 + points[b, i2, l] * w2 + points[b, i3, l] * w3

    interpolated_points = interpolated_points.view(B, N, -1)
    print("interpolated_points.shape: ", interpolated_points.shape)
    return interpolated_points


def three_interpolation_efc(points, idx, weights):
    """

    :param points: [B, M, C]
    :param idx: [B, N, 3]
    :param weights: [B, N, 3]
    :return: [B, N, C]
    """
    B, M, C = points.shape
    _, N, _ = idx.shape
    print("B: ", B, "M: ", M, "C: ", C, " N: ", N)

    flatten_points = torch.flatten(points)
    batch_idx_list = torch.flatten(idx)
    batch_weights_list = torch.flatten(weights)
    #print("flatten_points.shape: ", flatten_points.shape)
    #print("idx.shape: ", batch_idx_list.shape)
    #print("weights.shape: ", batch_weights_list.shape)
    interpolated_points = torch.randn(B * N * C)

    for b in range(B):
        b_wgt = batch_weights_list[(b * N * 3): (b * N * 3) + (N * 3)]
        b_id = batch_idx_list[(b * N * 3): (b * N * 3) + (N * 3)]
        b_points = flatten_points[(b * M * C): (b * M * C) + (M * C)]

        print("b_id.shape", b_id.shape)
        print("b_wgt.shape", b_wgt.shape)
        # print("b_points.shape", b_points.shape)
        print("current-->  wgt.shape.begin: ", (b * N * 3) + 1, " wgt.shape.end: ", (b * N * 3) + (N * 3))
        # print("current-->  b_points.shape.begin: ", (b * M * C) + 1, " b_points.shape.end: ", (b * M * C) + (M * C))

        for j in range(N):
            w1 = b_wgt[j * 3].float()
            w2 = b_wgt[j * 3 + 1].float()
            w3 = b_wgt[j * 3 + 2].float()
            i1 = b_id[j * 3].long()
            i2 = b_id[j * 3 + 1].long()
            i3 = b_id[j * 3 + 2].long()

            for l in range(C):
                interpolated_points[(b * N * C) + (j * C + l)] = flatten_points[i1 * C + l] * w1 + \
                                                                 flatten_points[i2 * C + l] * w2 + \
                                                                 flatten_points[i3 * C + l] * w3

    interpolated_points = interpolated_points.view(B, N, -1)
    print("interpolated_points.shape: ", interpolated_points.shape)
    return interpolated_points

if __name__ == '__main__':
    import torch
    points = torch.randn((32, 128, 256))
    idx = torch.randn((32, 512, 3))
    weights = torch.randn((32, 512, 3))

    out = three_interpolation_efc(points, idx, weights)