import os
import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


# Add cross-validation split up
# 25.06.2020

class S3DISDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=[5], block_size=1.0,
                 sample_rate=1.0, transform=None):
        super().__init__()
        self.global_counter = 0
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        all_rooms = sorted(os.listdir(data_root))
        all_rooms = [room for room in all_rooms if 'Area_' in room]

        '''
        if split == 'train':
            rooms_split = [room for room in all_rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in all_rooms if 'Area_{}'.format(test_area) in room]
        '''

        if split == 'train':
            rooms_split = [room for area in test_area for room in all_rooms if not 'Area_{}'.format(area) in room]
        else:
            rooms_split = [room for area in test_area for room in all_rooms if 'Area_{}'.format(area) in room]


        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)
        print("initial labelweights: ", labelweights)
        for room_name in rooms_split:
            print("Current room name : ", room_name)
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)  # xyzrgbl, N*7
            points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
            tmp, _ = np.histogram(labels, range(14))
            labelweights += tmp
            #print("tmp: ", tmp)
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        print("labelweights (before float): ", labelweights)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        print("labelweights: ", labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print("self.labelweights: ", self.labelweights)
        print("points in all room : ", num_point_all)
        sample_prob = num_point_all / np.sum(num_point_all)
        print("Sample Probability: ", sample_prob)
        print("Total No. of points in training set: ", np.sum(num_point_all))
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        print("num_iter: ", num_iter)
        room_idxs = []
        print("Total training room : ", len(rooms_split))
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        #print("room_idxs size : ", room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        #print("Current idx: ", idx)
        #print("Self.room_idxs: ", len(self.room_idxs))
        room_idx = self.room_idxs[idx]
        #print("room idx to aways zero hoibek: ", room_idx)
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
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
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)


if __name__ == '__main__':
    s3dis_dataset_path = '/home/reza/Desktop/thesis_tum/dataset/stanford_indoor3d/'
    train_dataset = S3DISDataset(split='train', data_root=s3dis_dataset_path, num_point=128,
                                     test_area=[1, 3, 6], block_size=1.0, sample_rate=1.0, transform=None)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True,
                                               num_workers=4, pin_memory=True, drop_last=True,
                                               worker_init_fn=lambda x: np.random.seed(x + int(time.time())))

    print("Train Loader: ", train_loader.dataset.__len__())
