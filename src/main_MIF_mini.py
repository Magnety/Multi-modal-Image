import argparse
import os
from solver_MIF import Solver
# from Code.data_loader import get_loader
from torch.backends import cudnn
from data_loader_mm_mini import get_loader
import random
import time


def main(config):
    print("main")

    cudnn.benchmark = True

    cur_time = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
    # Create directories if not exist
    config.model_path = os.path.join(config.output_path, config.model_type + '_' + config.lr_decay + '_' + config.opt,
                                     str(config.modal), cur_time, 'model')
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    config.result_path = os.path.join(config.output_path, config.model_type + '_' + config.lr_decay + '_' + config.opt,
                                      str(config.modal), cur_time, 'result')
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.outimg_path = os.path.join(config.output_path, config.model_type + '_' + config.lr_decay + '_' + config.opt,
                                      str(config.modal), cur_time, 'img')
    if not os.path.exists(config.outimg_path):
        os.makedirs(config.outimg_path)
    # augmentation_prob= random.random()*0.7
    decay_ratio = 0.95
    decay_epoch = int(config.num_epochs * decay_ratio)

    # config.augmentation_prob = augmentation_prob
    config.num_epochs_decay = decay_epoch

    print(config)

    train_loader = get_loader(
        data_path=config.data_path,
        name_path=config.train_path,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mode='train',
        augmentation_prob=config.augmentation_prob,
    )
    valid_loader = get_loader(
        name_path=config.valid_path,
        data_path=config.data_path,

        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,

        mode='valid',
        augmentation_prob=0.)
    test_loader = get_loader(
        name_path=config.test_path,
        data_path=config.data_path,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,

        mode='test',
        augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    for i in range(0,5):

        parser = argparse.ArgumentParser()

        # model hyper-parameters
        parser.add_argument('--image_size', type=tuple, default=(256, 256))
        # training hyper-parameters
        parser.add_argument('--img_ch', type=int, default=6)
        parser.add_argument('--output_ch', type=int, default=2)
        parser.add_argument('--num_epochs', type=int, default=8000)
        parser.add_argument('--num_epochs_decay', type=int, default=5)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--lr', type=float, default=0.0005)
        parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
        parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
        parser.add_argument('--augmentation_prob', type=float, default=0.8)
        parser.add_argument('--lr_decay', default='StepLR', help='Cosine/Lambda')

        parser.add_argument('--log_step', type=int, default=2)
        parser.add_argument('--val_step', type=int, default=2)
        # misc
        parser.add_argument('--opt', type=str, default='SGD', help='SGD/Adam')
        parser.add_argument('--modal', type=int, default=2)  # 0=2d 1=els 2=2d c els(last) 3=2d c els(first)  4=ours

        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--model_type', type=str, default='resnet_last_cat',
                            help='')
        parser.add_argument('--output_path', type=str,
                            default='/breast_multimodal_project/mda_output/fix_testset/fold_%d'%i)
        parser.add_argument('--data_path', type=str,
                            default='/breast_multimodal_project/dataset/data')
        parser.add_argument('--name_path', type=str,
                            default='/breast_multimodal_project/dataset/fix_testset/ac_k_fold_%d'%i)
        parser.add_argument('--checkpoint', type=str,
                            default='')

        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--result_path', type=str, default='')
        parser.add_argument('--train_path', type=str, default='')
        parser.add_argument('--valid_path', type=str, default='')
        parser.add_argument('--test_path', type=str, default='')
        parser.add_argument('--outimg_path', type=str, default='')
        parser.add_argument('--cuda_idx', type=int, default=1)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        config = parser.parse_args()
        print("into main")
        config.train_path = config.name_path + "/train"
        config.valid_path = config.name_path + "/valid"
        config.test_path = config.name_path + "/test"

        main(config)

