import torch
import argparse
import importlib

# Semantic Segmentation Prediction for random Indoor 3D scene

_authors_ = "Reza, TU Munich"


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--network', type=str, default='pointnet', help='Load Network for testing [default: pointnet]')
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size definition [default: 32]')

    return parser.parse_args()


class SemSegPred():
    def __init__(self):
        self.args = parse_args()
        self.cls2lbl_dict, self.lbl2cls_dict = self.set_obj_classes()
        # print(self.cls2lbl_dict)
        # print(self.lbl2cls_dict)
        self.batch_sz = self.args.batchsize
        self.classifier, self.checkpoints = self.load_model()
        self.classifier.load_state_dict(self.checkpoints['model_state_dict'])
        print("Done!")


    def set_obj_classes(self):
        """
        For the time being, model is trained on S3DIS dataset. So, according to this dataset,
        13 classes is used.

        :return: dict of object classes and corresponding category.
        """
        classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
                   'board', 'clutter']
        seg_classes = {cls: i for i, cls in enumerate(classes)} # seg_classes['ceiling'] = 0, ... etc
        seg_label_to_cat = {}
        for i, cat in enumerate(seg_classes.keys()):
            seg_label_to_cat[i] = cat
        return seg_classes, seg_label_to_cat

    def load_model(self):
        """
        Load the model for prediction
        :return: model object
        """
        classifier = None
        if self.args.network == 'pointnet':
            num_class = 13
            model_name = importlib.import_module('.semantic_seg_network', package='pointnet')
            classifier = model_name.ModelCreation(num_class).cuda()

        base_dir = '/home/reza/Desktop/thesis_tum/nn_training_stuffs/'
        path = ''.join([base_dir, 'log_directory/semantic_segmentation/s3dis_log/checkpoints/best_model.pth'])
        print("Path: ", path)
        checkpoint = torch.load(path)
        return classifier, checkpoint

    def evaluate_scene(self):

        pass


if __name__ == '__main__':
    obj = SemSegPred()
