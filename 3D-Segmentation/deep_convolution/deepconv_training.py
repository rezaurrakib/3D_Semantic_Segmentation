import os
import time
import datetime
import random
import argparse
import importlib
import provider
import logging
import numpy as np
import torch
import torch.utils.data

from pathlib import Path
from tqdm import tqdm
from utils import helper
from s3dis_data_processing.s3dis_preprocess import S3DISDataset
from s3dis_data_processing.shapenet_preprocess import ShapeNetDataProcess


_authors_ = "Md Rezaur Rahman"
_license_ = "MIT"
_version_ = "1.0"
_maintainer_ = "Reza"
_status_ = "Dev"

def _arg_parsing():
    parser = argparse.ArgumentParser("SegmentationModel")
    parser.add_argument('--model', type=str, default='deep_convolution', help='Load model for training [default: deep_convolution]')
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size definition')
    parser.add_argument('--epoch', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--output', type=str, default='', help='Output folder to save model')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--npoint', type=int, default=128, help='Point Number [default: 256]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--test_area', type=int, default=5, help='S3DIS area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

    return parser.parse_args()


class SemanticSegTraining():
    def __init__(self):
        self.parser_obj = _arg_parsing()
        #self.cmd_option = self.parser_obj.parse_args()
        #self.shapenet_dataset = ShapeNetDataProcess()
        #self.s3dis_dataset_obj = S3DISDataset()
        self.root = None
        self.log_dir = None
        self.experiment_dir = None

        # Variables defined for S3DIS datasets obj annotation
        self.classes_annotation = None
        self.class_label = {}
        self.seg_label_to_categry = {}
        self.num_class = 13
        self.s3dis_dataset_path = '/home/reza/Desktop/thesis_tum/dataset/dummy_stanford_indoor3d/'
        self.num_points = self.parser_obj.npoint
        self.batch_size = self.parser_obj.batchsize
        self.test_area = self.parser_obj.test_area

        # Hyper Parameter creation
        os.environ['CUDA_VISIBLE_DEVICES'] = self.parser_obj.gpu


        # Log folder creations
        self.set_log_directory()
        self.set_logging_info()
        self.s3dis_dataset_seg_processing()

    def print_log(self, str):
        self.logger.info(str)
        print(str)

    def set_log_directory(self):
        time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        self.experiment_dir = Path('/home/reza/Desktop/thesis_tum/nn_training_stuffs/log_directory')
        print("Experimental Dir: ", self.experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.experiment_dir = self.experiment_dir.joinpath('deep_convolution')
        self.experiment_dir.mkdir(exist_ok=True)
        if self.parser_obj.log_dir is None:
            self.experiment_dir = self.experiment_dir.joinpath(time_str)
        else:
            self.experiment_dir = self.experiment_dir.joinpath(self.parser_obj.log_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.experiment_dir.joinpath('checkpoints/')
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.log_dir = self.experiment_dir.joinpath('logs/')
        self.log_dir.mkdir(exist_ok=True)
        print("Final Log Directory: ", self.log_dir)

    def set_logging_info(self):
        self.logger = logging.getLogger("SegmentationModel")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/%s.txt' % (self.log_dir, self.parser_obj.model))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.print_log('Parameters ...')
        self.print_log(self.parser_obj)

    def s3dis_dataset_seg_processing(self):
        self.classes_annotation = ['beam', 'board', 'bookcase' ,'ceiling', 'chair', 'clutter', 'column', 'door',
                                   'floor', 'sofa', 'table', 'wall', 'window']
        for idx, cls in enumerate(self.classes_annotation):
            self.class_label[cls] = idx
        for idx, cat in enumerate(self.class_label.keys()):
            self.seg_label_to_categry[idx] = cat

    def load_model_for_training(self):
        model = importlib.import_module('.deep_conv_network', package=self.parser_obj.model)
        classifier = model.DeepConv(self.num_class).cuda()
        #print("in load_model --> classifier.channel: ", classifier.chn)
        classifier_loss = model.GetLoss().cuda()
        return classifier, classifier_loss

    def weight_initialization(self, m):
        classname = m.__class__.__name__
        print("m: ", m)
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def start_training(self):
        self.root = self.s3dis_dataset_path
        print('Loading S3DIS Training Data for Deep Convolution Training .................. ')

        # Load train data ...
        train_dataset = S3DISDataset(split='train', data_root=self.root, num_point=self.num_points,
                                     test_area=self.test_area, block_size=1.0, sample_rate=1.0, transform=None)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=4, pin_memory=True, drop_last=True,
                                                   worker_init_fn = lambda x: np.random.seed(x+int(time.time())))

        # Load data for testing ...
        test_dataset = S3DISDataset(split='test', data_root=self.root, num_point=self.num_points,
                                     test_area=self.test_area, block_size=1.0, sample_rate=1.0, transform=None)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=4, pin_memory=True, drop_last=True)

        weights = torch.Tensor(train_dataset.labelweights).cuda()
        classifier, classifier_loss = self.load_model_for_training()


        self.print_log("Number of training data from S3DIS Dataset: %d" % len(train_dataset))
        #print("Number of training data from S3DIS Dataset: ", len(train_dataset))

        # Check for Pretrained Model, else training from scratch
        try:
            checkpoint = torch.load(str(self.experiment_dir) + '/checkpoints/best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            self.print_log("Using Pretrained Model")
        except:
            self.print_log('No existing model, starting training from scratch...')
            start_epoch = 0
            classifier = classifier.apply(self.weight_initialization)


        if self.parser_obj.optimizer == 'Adam':
            optimizer = torch.optim.Adam(classifier.parameters(), lr=self.parser_obj.learning_rate, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=self.parser_obj.decay_rate)
        else:
            optimizer = torch.optim.SGD(classifier.parameters(), lr=self.parser_obj.learning_rate, momentum=0.9)

        def batch_norm_momentum_adjust(m, momentum):
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                m.momentum = momentum

        learning_rate_clip = 1e-5
        momentum_original = 0.1
        momentum_deccay = 0.5
        momentum_deccay_step = self.parser_obj.step_size

        global_epoch = 0
        best_iou = 0

        for epoch in range(start_epoch, self.parser_obj.epoch):
            lr = max(self.parser_obj.learning_rate * (self.parser_obj.lr_decay ** (epoch // self.parser_obj.step_size)),
                     learning_rate_clip)
            self.print_log('Learning rate: %f' % lr)
            for param_grp in optimizer.param_groups:
                param_grp['lr'] = lr
            momentum = momentum_original * (momentum_deccay ** (epoch // momentum_deccay_step))
            if momentum < 0.01:
                momentum = 0.01
            self.print_log('BN momentum updated to: %f' % momentum)
            classifier = classifier.apply(lambda x: batch_norm_momentum_adjust(x, momentum))
            num_batches = len(train_loader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            print("num_batches: ", num_batches)

            for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
                coord_points, labels = data
                print("coord points hihihi: ", coord_points.size())
                print("labels size: ", labels.size())
                coord_points = coord_points.data.numpy()
                coord_points[:, :, :3] = helper.rotate_point_cloud_z(coord_points[:, :, :3])
                coord_points = torch.Tensor(coord_points)
                print("coord points shape: ", coord_points.size())
                coord_points = coord_points.float().cuda()
                labels = labels.long().cuda()

                coord_points = coord_points.transpose(2, 1)
                optimizer.zero_grad()
                classifier = classifier.train()
                #print("in start_training() --> classifier.chn: ", classifier.chn)
                print("[Seg Training] Point Coord Shape: ", coord_points.shape)
                seg_pred, trans_feat = classifier(coord_points)
                print("seg_pred.shape(): ", seg_pred.size())
                print("trans_feat.shape(): ", trans_feat.size())
                seg_pred = seg_pred.contiguous().view(-1, self.num_class)
                print("seg_pred.shape() bal chal change: ", seg_pred.size())
                batch_label = labels.view(-1, 1)[:, 0].cpu().data.numpy()
                print("batch_label: ", len(batch_label))
                target = labels.view(-1, 1)[:, 0]
                loss = classifier_loss(seg_pred, target, trans_feat, weights)
                loss.backward()
                optimizer.step()
                pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (self.batch_size * self.num_points)
                loss_sum += loss
            self.print_log('Training mean loss: %f' % (loss_sum / num_batches))
            self.print_log('Training accuracy: %f' % (total_correct / float(total_seen)))
            print("Training mean loss: ", loss_sum / num_batches)
            print("Training accuracy:", total_correct / float(total_seen))

            if epoch % 5 == 0:
                self.logger.info('Save model...')
                savepath = str(self.checkpoints_dir) + '/model.pth'
                self.print_log('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                self.print_log('Saving model....')


            ###  "Evaluate on chopped scenes" ###
            with torch.no_grad():
                num_batches = len(test_loader)
                total_correct = 0
                total_seen = 0
                loss_sum = 0
                labelweights = np.zeros(self.num_class)
                total_seen_class = [0 for _ in range(self.num_class)]
                total_correct_class = [0 for _ in range(self.num_class)]
                total_iou_deno_class = [0 for _ in range(self.num_class)]
                self.print_log('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))

                for i, data in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
                    points, target = data
                    points = points.data.numpy()
                    points = torch.Tensor(points)
                    points, target = points.float().cuda(), target.long().cuda()
                    points = points.transpose(2, 1)
                    classifier = classifier.eval()
                    seg_pred, trans_feat = classifier(points)
                    pred_val = seg_pred.contiguous().cpu().data.numpy()
                    seg_pred = seg_pred.contiguous().view(-1, self.num_class)
                    batch_label = target.cpu().data.numpy()
                    target = target.view(-1, 1)[:, 0]
                    loss = classifier_loss(seg_pred, target, trans_feat, weights)
                    loss_sum += loss
                    pred_val = np.argmax(pred_val, 2)
                    correct = np.sum((pred_val == batch_label))
                    total_correct += correct
                    total_seen += (self.batch_size * self.num_points)
                    tmp, _ = np.histogram(batch_label, range(self.num_class + 1))
                    labelweights += tmp
                    for l in range(self.num_class):
                        total_seen_class[l] += np.sum((batch_label == l))
                        total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                        total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
                    labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
                    mIoU = np.mean(
                        np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
                    self.print_log('eval mean loss: %f' % (loss_sum / float(num_batches)))
                    self.print_log('eval point avg class IoU: %f' % (mIoU))
                    self.print_log('eval point accuracy: %f' % (total_correct / float(total_seen)))
                    self.print_log('eval point avg class acc: %f' % (
                        np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
                    iou_per_class_str = '------- IoU --------\n'
                    for l in range(self.num_class):
                        iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                            self.seg_label_to_categry[l] + ' ' * (14 - len(self.seg_label_to_categry[l])), labelweights[l - 1],
                            total_correct_class[l] / float(total_iou_deno_class[l]))

                    self.print_log(iou_per_class_str)
                    self.print_log('Eval mean loss: %f' % (loss_sum / num_batches))
                    self.print_log('Eval accuracy: %f' % (total_correct / float(total_seen)))
                    if mIoU >= best_iou:
                        best_iou = mIoU
                        self.logger.info('Save model...')
                        savepath = str(self.checkpoints_dir) + '/best_model.pth'
                        self.print_log('Saving at %s' % savepath)
                        state = {
                            'epoch': epoch,
                            'class_avg_iou': mIoU,
                            'model_state_dict': classifier.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }
                        torch.save(state, savepath)
                        self.print_log('Saving model....')
                    self.print_log('Best mIoU: %f' % best_iou)
                global_epoch += 1


if __name__ == '__main__':
    seg_train_obj = SemanticSegTraining()
    seg_train_obj.start_training()

