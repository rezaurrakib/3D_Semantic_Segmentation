import os
import time
import logging
import datetime
import importlib
import numpy as np
import torch
import torch.utils.data

from tqdm import tqdm
from pathlib import Path
from pointnet import helper_tool
from tensorboardX import summary
from dataset_load_preprocess.s3dis.data_loader import S3DISDataset
from dataset_load_preprocess.semantic3d.data_loader_semantic3d import Semantic3dDataset


class SemanticSegTraining():
    def __init__(self, parser):
        self.model = parser.model
        self.gpu = parser.gpu
        self.epoch = parser.epoch
        self.output = parser.output
        self.num_points = parser.npoint
        self.batch_size = parser.batchsize
        self.dataset = parser.dataset
        self.learning_rate = parser.learning_rate
        self.decay_rate = parser.decay_rate # weight decay
        self.optimizer = parser.optimizer
        self.test_area = parser.test_area
        self.step_size = parser.step_size
        self.lr_decay = parser.lr_decay # learning rate decay
        self.log_dir = None
        self.experiment_dir = None
        self.num_class = None
        self.dataset_path = None

        # Variables defined for datasets obj annotation
        self.classes_annotation = None
        self.class_label = {}
        self.seg_label_to_category = {}

        # Hyper Parameter creation
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu
        self.dataset_preprocessing()

        # Log folder creations
        self.set_log_directory()
        self.set_logging_info()
        self.start_training()

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

    def semantic3d_annotation(self):
        self.classes_annotation = ['unlabeled', 'man-made terrain', 'natural terrain', 'high vegetation',
                                   'low vegetation', 'buildings', 'hard scape', 'scanning artefacts', 'cars']
        for idx, cls in enumerate(self.classes_annotation):
            self.class_label[cls] = idx
        for idx, cat in enumerate(self.class_label.keys()):
            self.seg_label_to_category[idx] = cat

    def print_log(self, str):
        self.logger.info(str)
        print(str)

    def set_log_directory(self):
        #time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        self.experiment_dir = Path('/home/reza/Desktop/thesis_tum/nn_training_stuffs/log_directory/')
        print("Experimental Dir: ", self.experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.experiment_dir = self.experiment_dir.joinpath('3d_semantic_segmentation')
        self.experiment_dir.mkdir(exist_ok=True)
        self.experiment_dir = self.experiment_dir.joinpath(self.model)
        self.experiment_dir.mkdir(exist_ok=True)
        self.experiment_dir = self.experiment_dir.joinpath(self.dataset)
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
        file_handler = logging.FileHandler('%s/%s.txt' % (self.log_dir, self.model))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.print_log('Parameters ...')
        #self.print_log(self.parser_obj)

    def load_model_for_training(self):
        if self.model == 'pointnet':
            model = importlib.import_module('.network', package=self.model)
        elif self.model == 'deep_convolution':
            model = importlib.import_module('.network_dpconv', package=self.model)

        classifier = model.ModelCreation(self.num_class).cuda()
        #print("in load_model --> classifier.channel: ", classifier.chn)
        classifier_loss = model.GetLoss().cuda()
        return classifier, classifier_loss

    # See weight init details from below stack-overflow link:
    # https://stackoverflow.com/a/49433937

    def weight_initialization(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def start_training(self):
        if self.dataset == 'semantic3d':
            print('Loading Semantic3D Training Data for Training .................. ')
            # Add the code for train/test loader
            train_dataset = Semantic3dDataset(self.dataset_path, split='train', num_point=self.num_points,
                                              block_size=1.0, sample_rate=1.0, transform=None)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                       num_workers=0, pin_memory=True, drop_last=True,
                                                       worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
            self.print_log("Number of training data from Semantic3D Dataset: %d" % len(train_dataset))

            test_dataset = Semantic3dDataset(self.dataset_path, split='test', num_point=self.num_points,
                                             block_size=1.0, sample_rate=1.0, transform=None)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,
                                                      num_workers=0, pin_memory=True, drop_last=True,
                                                      worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
            self.print_log("Number of testing data from Semantic3D Dataset: %d" % len(test_dataset))

        elif self.dataset == 's3dis':
            print('Loading S3DIS Training Data .................. ')
            # Load train data ...
            train_dataset = S3DISDataset(split='train', data_root=self.dataset_path, num_point=self.num_points,
                                         test_area=self.test_area, block_size=1.0, sample_rate=1.0, transform=None)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                       num_workers=0, pin_memory=True, drop_last=True,
                                                       worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
            self.print_log("Number of training data from S3DIS Dataset: %d" % len(train_dataset))

            # Load data for testing ...
            test_dataset = S3DISDataset(split='test', data_root=self.dataset_path, num_point=self.num_points,
                                        test_area=self.test_area, block_size=1.0, sample_rate=1.0, transform=None)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True,
                                                      num_workers=0, pin_memory=True, drop_last=True)

        weights = torch.Tensor(train_dataset.labelweights).cuda()
        classifier, classifier_loss = self.load_model_for_training()

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

        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(classifier.parameters(), lr=self.learning_rate, betas=(0.9, 0.999),
                                         eps=1e-08, weight_decay=self.decay_rate)
        else:
            optimizer = torch.optim.SGD(classifier.parameters(), lr=self.learning_rate, momentum=0.9)

        def batch_norm_momentum_adjust(m, momentum):
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                m.momentum = momentum

        learning_rate_clip = 1e-5
        momentum_original = 0.1
        momentum_deccay = 0.5
        momentum_deccay_step = self.step_size

        global_epoch = 0
        best_iou = 0

        for epoch in range(start_epoch, self.epoch):
            lr = max(self.learning_rate * (self.lr_decay ** (epoch // self.step_size)), learning_rate_clip)
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
                #print("coord points Shape: ", coord_points.size())
                #print("labels shape: ", labels.size())
                coord_points = coord_points.data.numpy()
                coord_points[:, :, :3] = helper_tool.rotate_point_cloud_z(coord_points[:, :, :3])
                coord_points = torch.Tensor(coord_points)
                #print("coord points shape: ", coord_points.size())
                coord_points = coord_points.float().cuda()
                labels = labels.long().cuda()
                coord_points = coord_points.transpose(2, 1)
                optimizer.zero_grad()
                classifier = classifier.train()
                # print("in start_training() --> classifier.chn: ", classifier.chn)
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
                            self.seg_label_to_category[l] + ' ' * (14 - len(self.seg_label_to_category[l])),
                            labelweights[l - 1],
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
