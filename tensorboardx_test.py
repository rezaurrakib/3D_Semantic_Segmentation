import tensorboardX
import os
import re
import pptk
import numpy as np
from tensorboardX import SummaryWriter
from dataset_preprocessing.semantic3d.ply_writer import read_ply


log_dir = "F:/thesis_stuffs"

writer1 = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboardX_run/train'))
writer2 = SummaryWriter(log_dir=os.path.join(log_dir, 'tensorboardX_run/validation'))

for i in range(100):
    writer1.add_scalar('Loss', np.random.random(), i)
    writer2.add_scalar('Loss', np.random.random(), i)
    writer1.add_scalar('Accuracy', np.random.random(), i)
    writer2.add_scalar('Accuracy', np.random.random(), i)

'''
for i in range(100):

    writer.add_scalar('Loss/train', np.random.random(), i)
    writer.add_scalar('Loss/test', np.random.random(), i)
    writer.add_scalar('Accuracy/train', np.random.random(), i)
    writer.add_scalar('Accuracy/test', np.random.random(), i)

writer.close()
'''

idx = [1, 3, 5]
batch_data = np.random.rand(1, 10)[0]
print(batch_data)
train_data = batch_data[idx]
print(train_data)

# test pptk visualizer


def visualize_plyfile(filepath):
    points = None
    labels = None
    file_ext = re.split(r'(/+|\\)', filepath)[-1][-4:]
    print("file extension: ", file_ext)
    if file_ext == ".ply":
        area_pcd = read_ply(filepath)
        points = np.vstack((area_pcd['x'], area_pcd['y'], area_pcd['z'], area_pcd['red'],
                            area_pcd['green'], area_pcd['blue'])).T
        labels = area_pcd['class']

    elif file_ext == '.npy':
        room_data = np.load(filepath)
        points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N

    print("points shape: ", points.shape)
    print("labels shape: ", labels.shape)
    v = pptk.viewer(points[:, 0:3])
    v.attributes(points[:, 3:6]/255., labels)
    v.set(point_size=0.050)

# visualize_plyfile('F:/thesis_stuffs/dataset_3dVision/stanford_indoor3d/Area_1_conferenceRoom_1.npy')
visualize_plyfile('F:/thesis_stuffs/dataset_3dVision/semantic3d_ply/training_split/bildstein_station3_xyz_rgb.ply')