import os
import sys
import torch
import logging
import argparse
import importlib
import numpy as np

from tqdm import tqdm
from pathlib import Path
from dataset_preprocessing.wholescene_dataprocessing import ScannetDatasetWholeScene


class TestSemSeg():
    def __init__(self, parser):
        super(TestSemSeg, self).__init__()
        self.dataset = parser.dataset
        self.model = parser.model
        self.test_area = parser.test_area
        self.num_points = parser.num_point
        self.batch_size = parser.batchsize
        self.visual = parser.visual
        self.num_votes = parser.num_votes
        self.classes_annotation = None
        self.semantic3d_annotation = None
        self.class_label = {}
        self.seg_label_to_category = {}
        self.num_class = None
        self.dataset_path = None
        self.experiment_dir = None
        self.visual_dir = None
        os.environ["CUDA_VISIBLE_DEVICES"] = parser.gpu

        self.dataset_preprocessing()
        self.set_visual_directory()
        self.evaluate_scene()

    def print_log(self, str):
        self.logger.info(str)
        print(str)

    def dataset_preprocessing(self):
        if self.dataset == 's3dis':
            self.num_class = 13
            #self.dataset_path = '/home/rakib08cse/dataset/stanford_indoor3d/'
            self.dataset_path = '/home/reza/Desktop/thesis_tum/dataset/dummy_stanford_indoor3d/'
            self.s3dis_annotation()
        elif self.dataset == 'semantic3d':
            self.num_class = 9
            self.dataset_path = '/home/rakib08cse/dataset/semantic3d/'
            self.semantic3d_annotation()

    def s3dis_annotation(self):
        self.classes_annotation = ['beam', 'board', 'bookcase' ,'ceiling', 'chair', 'clutter', 'column', 'door',
                                   'floor', 'sofa', 'table', 'wall', 'window']
        for idx, cls in enumerate(self.classes_annotation):
            self.class_label[cls] = idx
        for idx, cat in enumerate(self.class_label.keys()):
            self.seg_label_to_category[idx] = cat

        self.s3dis_class_coloring()

    def s3dis_class_coloring(self):
        g_class2color = {'ceiling': [0, 255, 0],
                         'floor': [0, 0, 255],
                         'wall': [0, 255, 255],
                         'beam': [255, 255, 0],
                         'column': [255, 0, 255],
                         'window': [100, 100, 255],
                         'door': [200, 200, 100],
                         'table': [170, 120, 200],
                         'chair': [255, 0, 0],
                         'sofa': [200, 100, 100],
                         'bookcase': [10, 200, 100],
                         'board': [200, 200, 200],
                         'clutter': [50, 50, 50]}
        self.g_label2color = {self.classes_annotation.index(cls): g_class2color[cls] for cls in self.classes_annotation}

    def semantic3d_annotation(self):
        self.classes_annotation = ['unlabeled', 'man-made terrain', 'natural terrain', 'high vegetation',
                                   'low vegetation', 'buildings', 'hard scape', 'scanning artefacts', 'cars']
        for idx, cls in enumerate(self.classes_annotation):
            self.class_label[cls] = idx
        for idx, cat in enumerate(self.class_label.keys()):
            self.seg_label_to_category[idx] = cat

    def set_visual_directory(self):
        cmn_pth = '/home/reza/Desktop/thesis_tum/gcp_training_result/saved_models/sem_seg_3d/'
        self.experiment_dir = cmn_pth + '/' + self.model + '/' + self.dataset
        self.visual_dir = self.experiment_dir + '/visual/'
        self.visual_dir = Path(self.visual_dir)
        self.visual_dir.mkdir(exist_ok=True)

    def add_vote(self, vote_label_pool, point_idx, pred_label, weight):
        B = pred_label.shape[0]
        N = pred_label.shape[1]
        for b in range(B):
            for n in range(N):
                if weight[b, n]:
                    vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
        return vote_label_pool

    """
    def log_directory(self):
        '''LOG'''
        args = parse_args()
        logger = logging.getLogger("Model")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        log_string('PARAMETER ...')
        log_string(args)
    """

    def load_model(self):
        if self.model == 'pointnet':
            model = importlib.import_module('.pointnet_plus', package=self.model)
        classifier = model.ModelCreation(self.num_class).cuda()
        checkpoint = torch.load(str(self.experiment_dir) + '/e25_p4096/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
        return classifier

    def evaluate_scene(self):

        classifier = self.load_model()
        test_scene = ScannetDatasetWholeScene(self.dataset_path, split='test', test_area=self.test_area,
                                                            block_points=self.num_points)
        print("The number of test data is: ", len(test_scene))

        with torch.no_grad():
            scene_id = test_scene.file_list
            scene_id = [x[:-4] for x in scene_id]
            num_batches = len(test_scene)

            total_seen_class = [0 for _ in range(self.num_class)]
            total_correct_class = [0 for _ in range(self.num_class)]
            total_iou_deno_class = [0 for _ in range(self.num_class)]

            print('---- Evaluating the whole Scene ----')
            pred_scene_output = None
            scene_ground_truth = None

            for batch_idx in range(num_batches):
                print("visualize [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
                total_seen_class_tmp = [0 for _ in range(self.num_class)]
                total_correct_class_tmp = [0 for _ in range(self.num_class)]
                total_iou_deno_class_tmp = [0 for _ in range(self.num_class)]
                if self.visual:
                    pred_scene_output = open(os.path.join(self.visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                    scene_ground_truth = open(os.path.join(self.visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

                whole_scene_data = test_scene.scene_points_list[batch_idx]
                whole_scene_label = test_scene.semantic_labels_list[batch_idx]
                vote_label_pool = np.zeros((whole_scene_label.shape[0], self.num_class))
                for _ in tqdm(range(self.num_votes), total=self.num_votes):
                    print("In range loop .... 1")
                    scene_data, scene_label, scene_smpw, scene_point_index = test_scene[batch_idx]
                    num_blocks = scene_data.shape[0]
                    s_batch_num = (num_blocks + self.batch_size - 1) // self.batch_size
                    batch_data = np.zeros((self.batch_size, self.num_points, 9))

                    batch_label = np.zeros((self.batch_size, self.num_points))
                    batch_point_index = np.zeros((self.batch_size, self.num_points))
                    batch_smpw = np.zeros((self.batch_size, self.num_points))
                    print("s_batch_num: ", s_batch_num)
                    print("num_blocks: ", num_blocks)

                    for sbatch in range(s_batch_num):
                        print("sbatch no .... ", sbatch)
                        start_idx = sbatch * self.batch_size
                        end_idx = min((sbatch + 1) * self.batch_size, num_blocks)
                        real_batch_size = end_idx - start_idx
                        batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                        batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                        batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                        batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                        batch_data[:, :, 3:6] /= 1.0

                        torch_data = torch.Tensor(batch_data)
                        torch_data = torch_data.float().cuda()
                        torch_data = torch_data.transpose(2, 1)
                        seg_pred, _ = classifier(torch_data)
                        batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                        vote_label_pool = self.add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                                   batch_pred_label[0:real_batch_size, ...],
                                                   batch_smpw[0:real_batch_size, ...])

                pred_label = np.argmax(vote_label_pool, 1)

                for l in range(self.num_class):
                    print("In range num_class : ", l)
                    total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                    total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                    total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                    total_seen_class[l] += total_seen_class_tmp[l]
                    total_correct_class[l] += total_correct_class_tmp[l]
                    total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

                iou_map = np.array(total_correct_class_tmp) / (
                            np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
                print(iou_map)
                arr = np.array(total_seen_class_tmp)
                tmp_iou = np.mean(iou_map[arr != 0])
                # log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
                print('Mean IoU of ', (scene_id[batch_idx], ' is : ',  tmp_iou))
                print('----------------------------')

                filename = os.path.join(self.visual_dir, scene_id[batch_idx] + '.txt')
                with open(filename, 'w') as pl_save:
                    for i in pred_label:
                        pl_save.write(str(int(i)) + '\n')
                    pl_save.close()
                for i in range(whole_scene_label.shape[0]):
                    color = self.g_label2color[pred_label[i]]
                    color_gt = self.g_label2color[whole_scene_label[i]]
                    if self.visual:
                        pred_scene_output.write('v %f %f %f %d %d %d\n' % (
                            whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                            color[2]))
                        pred_scene_output.write(
                            'v %f %f %f %d %d %d\n' % (
                                whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                                color_gt[1], color_gt[2]))
                if self.visual:
                    pred_scene_output.close()
                    pred_scene_output.close()

            IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
            iou_per_class_str = '------- IoU --------\n'
            for l in range(self.num_class):
                iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                    self.seg_label_to_category[l] + ' ' * (14 - len(self.seg_label_to_category[l])),
                    total_correct_class[l] / float(total_iou_deno_class[l]))
            # log_string(iou_per_class_str)
            print("IOU_per-class_str: ", iou_per_class_str)
            # log_string('eval point avg class IoU: %f' % np.mean(IoU))
            print(('eval point avg class IoU: ', np.mean(IoU)))
            print('eval whole scene point avg class acc: ',(np.mean(np.array(total_correct_class)
                                                            / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

            print('eval whole scene point accuracy: ', (np.sum(total_correct_class)
                                                                / float(np.sum(total_seen_class) + 1e-6)))

            # log_string('eval whole scene point avg class acc: %f' % (
            #    np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
            # log_string('eval whole scene point accuracy: %f' % (np.sum(total_correct_class)
            #                                                    / float(np.sum(total_seen_class) + 1e-6)))

            print("Done!")

def parse_args():
    parser = argparse.ArgumentParser('Testing_Semantic_Segmentation_Model')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--model', type=str, default='pointnet', help='specify pointnet_plus to test [default: pointnet]')
    parser.add_argument('--dataset', type=str, default='s3dis', help='specify on which dataset model is trained')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=128, help='Point Number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='2020-06-21_test', help='Experiment root')
    parser.add_argument('--visual', action='store_true', default=True, help='Whether visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=5, help='Aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    obj = TestSemSeg(parser)
