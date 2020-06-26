import os
import glob
import pandas
import numpy as np
import dataset_load_preprocess.semantic3d.ply_writer as ply_util



target_dir = '/home/reza/Desktop/dummy/new/'
dataset_dir = '/home/reza/Desktop/dummy'


class DownsampleSemantic3D():
    def __init__(self):
        self.remove_unlabeled_data()

    def remove_unlabeled_data(self):
        for filename in glob.glob(os.path.join(dataset_dir, '*.ply')):
            ply_filename = filename.split('/')[-1]
            save_path = target_dir + ply_filename
            pcd = ply_util.read_ply(filename)
            pc_red = pcd['red']
            pc_green = pcd['green']
            pc_blue = pcd['blue']
            pc_x = pcd['x']
            pc_y = pcd['y']
            pc_z = pcd['z']
            pc_labels = pcd['class']
            actual_lbl_idx = pc_labels.nonzero()[0]
            actual_labels = pc_labels[pc_labels!=0]
            actual_red = np.array(pc_red)[actual_lbl_idx]
            actual_green = np.array(pc_green)[actual_lbl_idx]
            actual_blue = np.array(pc_blue)[actual_lbl_idx]

            actual_x = np.array(pc_x)[actual_lbl_idx]
            actual_y = np.array(pc_y)[actual_lbl_idx]
            actual_z = np.array(pc_z)[actual_lbl_idx]
            actual_xyz = np.column_stack((actual_x, actual_y, actual_z))
            xyz = actual_xyz[:, :3].astype(np.float32)
            actual_rgb = np.column_stack((actual_red, actual_green, actual_blue))
            rgb = actual_rgb[:, :3].astype(np.uint8)

            ply_util.write_ply(save_path, (xyz, rgb, actual_labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            print("labels length: ", len(pc_labels))
            print("x.length: ", len(actual_x))
            print("y.length: ", len(actual_y))
            print("z.length: ", len(actual_z))
            print("actual_red.length: ", len(actual_red))
            print("actual_green.length: ", len(actual_green))
            print("actual_blue.length: ", len(actual_blue))
            print("xyz: ", xyz.shape)



class Semantic3DPlyExtConverter():
    def __init__(self):
        self.create_ply_ext()

    def create_ply_ext(self):
        for filename in glob.glob(os.path.join(dataset_dir, '*.txt')):
            print("Current processing file: ", filename)
            print("Reading xyz coordinates ....")
            pc_xyz = pandas.read_csv(filename, usecols=[0, 1, 2], delim_whitespace=True, dtype=np.float32).values
            print("pc_xyz.shape: ", pc_xyz.shape)
            print("Reading rgb ....")
            pc_rgb = pandas.read_csv(filename, usecols=[4, 5, 6], delim_whitespace=True, dtype=np.uint8).values
            #point_cloud_data = pc_rgb.values

            label_path = filename[:-4] + '.labels'
            if os.path.exists(label_path):
                print("path exists ......")
                #label_pd = pandas.read_csv(label_path, delim_whitespace=True, dtype=np.uint8)
                pc_labels = pandas.read_csv(label_path, delim_whitespace=True, dtype=np.uint8).values
                print("Label read completes .....")
                #pc_labels = label_pd.values
                ply_filename = filename[:-4] + '.ply'
                save_path = os.path.join(target_dir, ply_filename)
                #xyz = point_cloud_data[:, :3].astype(np.float32)
                #rgb = point_cloud_data[:, 4:7].astype(np.uint8)
                xyz = pc_xyz[:, :3].astype(np.float32)
                rgb = pc_rgb[:, :3].astype(np.uint8)
                ply_util.write_ply(save_path, (xyz, rgb, pc_labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            else:
                print("In else ........... ")
                pc_labels = np.zeros(pc_xyz.shape[0], dtype=np.uint8)
                ply_filename = filename[:-4] + '.ply'
                save_path = os.path.join(target_dir, ply_filename)
                xyz = pc_xyz[:, :3].astype(np.float32)
                rgb = pc_rgb[:, :3].astype(np.uint8)
                ply_util.write_ply(save_path, (xyz, rgb, pc_labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])






if __name__ == "__main__":
    # Semantic3DPlyExtConverter()
    # file = pcd = o3d.io.read_point_cloud("/home/reza/Desktop/dummy/marketplacefeldkirch_station4_intensity_rgb.ply")
    DownsampleSemantic3D()
