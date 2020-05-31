import os
import glob
import pandas
import numpy as np
import semantic3d_data_processing.ply_writer as ply_util

train_file_prefixes = [
    "bildstein_station1_xyz_intensity_rgb",
    "bildstein_station3_xyz_intensity_rgb",
    "bildstein_station5_xyz_intensity_rgb",
    "domfountain_station1_xyz_intensity_rgb",
    "domfountain_station2_xyz_intensity_rgb",
    "domfountain_station3_xyz_intensity_rgb",
    "neugasse_station1_xyz_intensity_rgb",
    "sg27_station1_intensity_rgb",
    "sg27_station2_intensity_rgb",
    "sg27_station4_intensity_rgb",
    "sg27_station5_intensity_rgb",
    "sg27_station9_intensity_rgb",
    "sg28_station4_intensity_rgb",
    "untermaederbrunnen_station1_xyz_intensity_rgb",
    "untermaederbrunnen_station3_xyz_intensity_rgb",
]

base_dir = '/home/reza/Desktop/thesis_tum/dataset/semantic3d_ply/'
dataset_dir = "/home/reza/Desktop/thesis_tum/dataset/semantic3d_ply/"

class Semantic3DPlyExtConverter():
    def __init__(self):
        self.create_ply_ext()

    def create_ply_ext(self):
        for filename in glob.glob(os.path.join(dataset_dir, '*.txt')):
            print("Current processing file: ", filename)
            point_cloud_pd = pandas.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float64)
            point_cloud_data = point_cloud_pd.values

            label_path = filename[:-4] + '.labels'
            if os.path.exists(label_path):
                print("path exists ......")
                label_pd = pandas.read_csv(label_path, header=None, delim_whitespace=True, dtype=np.uint8)
                print("Label read completes .....")
                pc_labels = label_pd.values
                save_path = os.path.join(base_dir, filename[:-4] + '.ply')
                xyz = point_cloud_data[:, :3].astype(np.float32)
                rgb = point_cloud_data[:, 4:7].astype(np.uint8)
                ply_util.write_ply(save_path, (xyz, rgb, pc_labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])


if __name__ == "__main__":
    Semantic3DPlyExtConverter()