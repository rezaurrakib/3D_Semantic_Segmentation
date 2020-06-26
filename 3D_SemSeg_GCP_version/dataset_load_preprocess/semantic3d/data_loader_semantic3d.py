import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset
from .ply_writer import read_ply


class Semantic3dDataset(Dataset):
    def __init__(self, data_root, split='train', num_point=4096, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.global_counter = 0
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        if split == 'train':
            data_root = os.path.join(data_root, 'training_split')
        elif split == 'test':
            data_root = os.path.join(data_root, 'testing_split')

        area_list = sorted(os.listdir(data_root))
        areas = [area for area in area_list]

        self.area_points, self.area_labels = [], []
        self.area_coord_min, self.area_coord_max = [], []
        num_point_all = []
        label_weights = np.zeros(9)
        print("Initial label_weights: ", label_weights)

        for area_name in areas:
            print("Current area name : ", area_name)
            area_path = os.path.join(data_root, area_name)
            area_pcd = read_ply(area_path)
            points = np.vstack((area_pcd['x'], area_pcd['y'], area_pcd['z'], area_pcd['red'],
                                area_pcd['green'], area_pcd['blue'])).T
            labels = area_pcd['class']
            print("points shape: ", points.shape)
            print("labels shape: ", labels.shape)
            # area_pcd = np.load(load_pcd) # xyzrgbl, N*7
            #points, labels = area_pcd[:, 0:6], area_pcd[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(10))
            label_weights += tmp
            print("tmp: ", tmp)
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.area_points.append(points)
            self.area_labels.append(labels)
            self.area_coord_min.append(coord_min)
            self.area_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        # print("label_weights (before float): ", label_weights)
        label_weights = label_weights.astype(np.float32)
        label_weights = label_weights / np.sum(label_weights)
        print("label_weights: ", label_weights)
        self.labelweights = np.power(np.amax(label_weights) / label_weights, 1 / 3.0)
        print("self.label_weights: ", self.labelweights)
        print("points in all areas : ", num_point_all)
        sample_prob = num_point_all / np.sum(num_point_all)
        print("Sample Probability: ", sample_prob)
        print("Total No. of points in training set: ", np.sum(num_point_all))
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        print("num_iter: ", num_iter)
        block_idxs = []
        print("Total training Area : ", len(areas))
        for index in range(len(areas)):
            block_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.block_idxs = np.array(block_idxs)
        #print("block_idxs size : ", block_idxs)
        print("Totally {} block samples in {} set.".format(len(self.block_idxs), split))

    def __getitem__(self, idx):
        #print("Current idx: ", idx)
        #print("Self.room_idxs: ", len(self.room_idxs))
        block_idx = self.block_idxs[idx]
        #print("room idx to aways zero hoibek: ", room_idx)
        points = self.area_points[block_idx]   # N * 6
        labels = self.area_labels[block_idx]   # N
        n_points = points.shape[0]

        while (True):
            center = points[np.random.choice(n_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) &
                                  (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            #print("point_idxs : ", point_idxs.shape)
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        #print("current point shape: ", current_points.shape)
        current_points[:, 6] = selected_points[:, 0] / self.area_coord_max[block_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.area_coord_max[block_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.area_coord_max[block_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.block_idxs)

if __name__ == '__main__':
    dataset_root = 'F:/thesis_stuffs/dataset_3dVision/dummy_semantic3d_outdoor/'
    obj = Semantic3dDataset(dataset_root)
