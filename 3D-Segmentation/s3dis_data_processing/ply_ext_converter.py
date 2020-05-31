import os
import numpy as np

from plyfile import PlyData, PlyElement

_authors_ = "Md Rezaur Rahman"
_copyright_ = "Copyright 2020"
_license_ = "MIT"
_version_ = "1.0.1"
_maintainer_ = "Reza"
_status_ = "Dev"

"""
    A converter class for point cloud dataset creation.
    Changes the *.off format files to *.ply format
    Test Dataset: ModelNet40 (Princeton University)
"""


class PLYConverter():
    def __init__(self):
        self.DATASET_ROOT = "C:/modelnet40/"
        self.PLY_WRITE_LOCATION = "F:/dataset_3dVision/modelnet40_plyFormat/"
        self.obj_categories = []

    def prepare_train_test_samples(self):
        # List all the object categories in the root Dataset directory
        self.obj_categories = os.listdir(self.DATASET_ROOT)
        print("Object Categories: ", self.obj_categories)

        for obj in self.obj_categories:
            obj_path = os.path.join(self.DATASET_ROOT, obj)
            #print(obj_path)
            if os.path.isdir(obj_path):
                folder_names = os.listdir(obj_path)
                for fld in folder_names:
                    samples_path = os.path.join(obj_path, fld)
                    filelist = os.listdir(samples_path)
                    for f in filelist:
                        filepath = os.path.join(samples_path, f)
                        #print(file)
                        # set filepath from obj category + train/test
                        new_obj_path = os.path.join(obj, fld)
                        print(new_obj_path)
                        self.off_to_ply_conversion(new_obj_path, filepath, f)

    def off_to_ply_conversion(self, new_filepath, off_file_path, filename, text=True):
        write_ply_filename = os.path.join(self.PLY_WRITE_LOCATION, new_filepath)
        if not os.path.exists(write_ply_filename):
            os.makedirs(write_ply_filename)
        filename = filename.replace('off', 'ply')
        print("Current Filename: ", filename)
        print(write_ply_filename)
        num_faces = 0
        num_vertices = 0

        with open(off_file_path, 'r') as file:
            datastreams = file.readlines()  # Reading OFF file

        if datastreams[0].strip().lower() != 'off':
            num_vertices = int(datastreams[0].strip()[3:].split(' ')[0])
            num_faces = int(datastreams[0].strip().split(' ')[1])
            start = 1
        else:
            num_vertices, num_faces, n_smth = tuple(int(s) for s in datastreams[1].strip().split(' '))
            start = 2

        print("Num Vertices: ", num_vertices, " Num Faces: ", num_faces)
        points = [tuple(float(s) for s in datastreams[i].strip().split(' ')) for i in range(start, start + num_vertices)]

        start = start + num_vertices
        faces = []
        for i in range(start, start + num_faces):
            data = [int(s) for s in datastreams[i].strip().split(' ')]
            vertex = [data[i] for i in range(1, 4)]
            faces.append(tuple([vertex]))

        print("points length: ", len(points))
        print("faces length: ", len(faces))
        print(points[0])
        print(faces[0])

        # input: Nx3, write points to filename as PLY format.
        vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        faces = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])
        el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
        el2 = PlyElement.describe(faces, 'faces', comments=['faces'])
        PlyData([el, el2], text=text).write(write_ply_filename + "/" + filename)


if __name__ == '__main__':
    pobj = PLYConverter()
    pobj.prepare_train_test_samples()
