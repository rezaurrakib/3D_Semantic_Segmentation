import argparse

from semseg_training import SemanticSegTraining

_authors_ = "Md Rezaur Rahman"
_license_ = "MIT"
_version_ = "1.0"
_maintainer_ = "Reza"
_status_ = "Dev"

"""
    Starter file for 3D Sem. Segmentation training in Google Cloud Platform
    Training Networks: [pointnet] [pointnet_plus] [Deep Convolution]
    Training Dataset : [S3DIS] [Semantic3D]
"""


def _arg_parsing():
    parser = argparse.ArgumentParser("3D-Semantic-Segmentation-GCP-Instance")
    parser.add_argument('--model', type=str, default='pointnet_plus', help='Load model for training [default: deep_convolution]')
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size definition')
    parser.add_argument('--dataset', type=str, default='s3dis', help='Load dataset for training [default: s3dis]')
    parser.add_argument('--epoch', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--output', type=str, default='', help='Output folder to save model')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--npoint', type=int, default=128, help='Point Number [default: 256]')
    parser.add_argument('--test_area', type=int, default=5, help='S3DIS area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

    return parser.parse_args()


if __name__ == '__main__':
    parser = _arg_parsing()
    # Invoking training file with arguments ...
    train_obj = SemanticSegTraining(parser)
