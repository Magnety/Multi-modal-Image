import os
import numpy as np
import time
import datetime
import torchvision
from pathlib import Path
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.backends import cudnn

from arch.ours import ours
from torchcam.utils import overlay_mask
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt
from torchvision.transforms.functional import normalize, resize, to_pil_image
from arch.BigGAN.utils import tensor2var, denorm
from torchvision.utils import save_image

from tqdm import tqdm
from dqn import DQNAgent
from collections import OrderedDict
import csv
from PIL import Image
import torch
import scipy.misc
import cv2
import torch.nn as nn
from torchcam.methods import SmoothGradCAMpp, XGradCAM
import torchvision.transforms as transforms
from focal_loss import FocalLoss
from show_grid import grid_map, grid_map_1


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        # Models
        self.unet = None
        self.optimizer = None
        self.modal = config.modal
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCELoss()
        # self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.alpha = Variable(torch.ones(2, 1))
        self.alpha[0, 0] = 0.5
        self.alpha[1, 0] = 0.5
        self.action1 =[-0.002,-0.001,0,0.001,0.002]# [-0.02, -0.005, 0, 0.005, 0.1]  # [-0.005,-0.002,0,0.002,0.005]

        self.cls_criterion = FocalLoss(class_num=2, alpha=self.alpha)
        self.augmentation_prob = config.augmentation_prob
        self.img_size = config.image_size
        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lr_decay = config.lr_decay
        self.opt = config.opt
        self.warmup = config.warmup
        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step
        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.result_dir = config.result_path
        self.dqn = DQNAgent()
        self.mode = config.mode
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.result_path = self.result_path + '/%d-%.6f-%d-%.4f.txt' % (
            self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob)
        self.checkpoint = config.checkpoint
        with open(Path(self.result_path), "a") as f:
            f.write('%s\n' % str(config))

        self.t = config.t
        self.build_model()

    def build_model(self):
        torch.cuda.empty_cache()
        np.random.seed(19970916)
        torch.manual_seed(19970916)
        torch.cuda.manual_seed_all(19970916)
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        """Build generator and discriminator."""
        if self.model_type == 'resnet18':
            self.unet = resnet18()

        elif self.model_type == 'ours':
            self.unet = ours()

        if self.opt == "Adam":
            self.optimizer = optim.Adam([{'params': [param for name, param in self.unet.named_parameters()]}],
                                        lr=self.lr, betas=[self.beta1, self.beta2], weight_decay=0.05)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,'min',factor=0.98,patience=5)
        elif self.opt == 'SGD':
            self.optimizer = optim.SGD([{'params': [param for name, param in self.unet.named_parameters()]}],
                                       lr=self.lr, momentum=0.9, weight_decay=0.05)

        if self.lr_decay == 'Cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)
        elif self.lr_decay == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.985)
        elif self.lr_decay == 'Lambda':
            lambda1 = lambda epoch: np.sin(epoch) / epoch
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        #self.unet = nn.DataParallel(self.unet)

        self.unet.cuda()
        # self.unet.to(self.device)

    # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)
        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def test(self):
        self.unet.load_state_dict(torch.load(self.checkpoint))
        self.unet.train(False)
        self.unet.eval()

        total = 0
        true = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        label_list = []
        probility_list = []
        predict_list = []
        name_list = []
        for i, (images, label, name) in enumerate(self.test_loader):
            # print(name)
            images = images.to(self.device)
            # GT = GT.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                if self.modal == 0:
                    cls = self.unet(images[:, 0:3, :, :])
                elif self.modal == 1:
                    cls = self.unet(images[:, 3:6, :, :])
                elif self.modal == 2:
                    cls = self.unet(images[:, 0:6, :, :])

            batch_size = label.size(0)

            cls_cpu = cls.data.cpu().numpy()
            label_cpu = label.data.cpu().numpy()

            # cls_out = cls_cpu.argmax()
            for b in range(batch_size):
                total += 1
                name_list.append(name[b])
                probility_list.append(cls_cpu[b, 1])
                cls_out = cls_cpu[b].argmax()
                """if np.exp(cls_cpu[b, 1]) > 0.5:
                    cls_out = 1
                else:
                    cls_out = 0"""
                label_out = label_cpu[b]
                label_list.append(label_out)
                predict_list.append(cls_out)
                if cls_out == label_out:
                    true += 1
                    if cls_out == 1:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if cls_out == 1:
                        fp += 1
                    else:
                        fn += 1

        cls_acc = true / total
        cls_pre = tp / (tp + fp + 1e-8)
        cls_rec = tp / (tp + fn + 1e-8)
        cls_spe = tn / (tn + fp + 1e-8)
        cls_f1 = (2 * cls_pre * cls_rec) / (cls_pre + cls_rec + 1e-8)
        unet_score = cls_f1 + cls_acc

        print('[      Test_cls] Acc: %.4f, PRE: %.4f, REC: %.4f, SPE: %.4f, F1: %.4f' % (
            cls_acc, cls_pre, cls_rec, cls_spe, cls_f1))
        """if not os.path.exists(Path(self.result_path+"/log.txt")):
            with open(Path(self.result_path+"/log.txt"), "w") as f:
                print(f)"""
        with open(Path(self.result_path), "a") as f:
            f.write(('[Test_cls] Acc: %.4f, PRE: %.4f, REC: %.4f, SPE: %.4f, F1: %.4f\n' % (
                cls_acc, cls_pre, cls_rec, cls_spe, cls_f1)))
            f.write(('[Name] %s\n' % str(name_list)))
            f.write(('[Probility] %s\n' % str(probility_list)))
            f.write(('[Predict] %s\n' % str(predict_list)))
            f.write(('[Label  ] %s\n' % str(label_list)))
            f.write(('\n'))

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#
        unet_path = os.path.join(self.model_path, '%d-%.6f-%d-%.4f' % (
            self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))
        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:

            # Train for Encoder
            lr = self.lr
            best_unet_score = 0.
            best_acc = 0.
            best_f1 = 0.
            # print(self.unet)

            for epoch in range(self.num_epochs):
                self.unet.train(True)
                epoch_loss = 0
                total = 0
                true = 0
                tp = 0
                fp = 0
                tn = 0
                fn = 0
                label_list = []
                probility_list = []
                predict_list = []
                name_list = []
                next_images = None
                next_label = None
                next_name = None
                index = 0
                data_iter = iter(self.train_loader)
                # print('len(data_loader):',len(self.train_loader))
                for _images, _label, _name in tqdm(self.train_loader):
                    # print(index)
                    if next_images is not None:
                        cur_images = next_images
                        cur_label = next_label
                        cur_name = next_name
                        if index == (len(self.train_loader) - 1):
                            next_images, next_label, next_name = next(data_iter)
                        else:
                            next_images = _images
                            next_label = _label
                            next_name = _name
                    else:
                        cur_images = _images
                        cur_label = _label
                        cur_name = _name
                        next_images, next_label, next_name = next(data_iter)
                    images = cur_images.to(self.device)
                    label = cur_label.to(self.device)
                    name = cur_name
                    index += 1
                    cam_extractor = XGradCAM(self.unet, ['fire9_1', 'fire9_2'])
                    # cam_extractor_e = XGradCAM(self.unet, 'module.e_conv5_x')
                    cam_extractor._hooks_enabled = True
                    # cam_extractor_e._hooks_enabled = True
                    activation_map_g = torch.ones((images.shape[0], 32, 32)).to(self.device)
                    activation_map_e = torch.ones((images.shape[0], 32, 32)).to(self.device)

                    if self.modal == 0:
                        cls = self.unet(images[:, 0:3, :, :])
                    elif self.modal == 1:
                        cls = self.unet(images[:, 3:6, :, :])
                    elif self.modal == 2:
                        cls = self.unet(images[:, 0:6, :, :])
                    elif self.modal == 4:
                        if epoch < self.warmup:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map_g,
                                activation_map_e)
                        else:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map_g,
                                activation_map_e)
                        # print('*******finish')

                    activation_map = cam_extractor(cls.argmax(axis=1).squeeze(0).tolist(), cls)
                    # activation_map_e = cam_extractor_e(cls.argmax(axis=1).squeeze(0).tolist(), cls)
                    # plt.imshow(activation_map1[0][0].squeeze(0).cpu().numpy())
                    # plt.axis('off')
                    # plt.tight_layout()
                    # plt.show()

                    cam_extractor.remove_hooks()
                    cam_extractor._hooks_enabled = False
                    # cam_extractor_e.remove_hooks()
                    # cam_extractor_e._hooks_enabled = False

                    # print(np.array(activation_map).shape)
                    if self.modal == 4:
                        if epoch < self.warmup:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map[0],
                                activation_map[1])
                        else:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                torch.ones((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map[0],
                                activation_map[1])
                    cls_cpu = cls.data.cpu().numpy()
                    label_cpu = label.data.cpu().numpy()
                    # print('cls_cpu[:,1]:',cls_cpu[:,1].shape)
                    # print('np.sum(label_cpu==0):',np.array(np.sum(label_cpu==0)).unsqueeze(0).shape)
                    state = np.append(cls_cpu[:, 1],
                                      [np.array(np.sum(label_cpu == 0)), np.array(np.sum(label_cpu == 1))])
                    # print('label_1:',label_cpu)

                    _action1, _action2 = self.dqn.get_action(state)
                    print('action prob:',_action1,_action2)
                    if self.alpha[0, 0] <= 0.1:
                        if _action1 < 2:
                            _action1 = 2
                    if self.alpha[0, 0] >= 0.9:
                        if _action1 > 2:
                            _action1 = 2
                    if self.cls_criterion.gamma <= 0.1:
                        if _action2 < 2:
                            _action2 = 2
                    if self.cls_criterion.gamma >= 3.9:
                        if _action2 > 2:
                            _action2 = 2

                    if epoch > self.warmup:
                        self.alpha[0, 0] += self.action1[_action1]
                        self.alpha[1, 0] -= self.action1[_action1]
                        print('action1:', self.action1[_action1], 'action1:', -self.action1[_action1], 'action2:',
                              self.action1[_action2])

                        self.cls_criterion.alpha = self.alpha
                        self.cls_criterion.gamma = self.cls_criterion.gamma + self.action1[_action2]
                    cls = cls.view(-1, cls.size(-1))
                    label = label.view(-1)
                    cls_loss = self.cls_criterion(cls, label.long())
                    loss = cls_loss
                    epoch_loss += loss.item()
                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                    batch_size = label.size(0)

                    total_b = 0
                    true_b = 0
                    tp_b = 0
                    fp_b = 0
                    tn_b = 0
                    fn_b = 0
                    for b in range(batch_size):
                        total_b += 1
                        cls_out = cls_cpu[b].argmax()
                        label_out = label_cpu[b]
                        if cls_out == label_out:
                            true_b += 1
                            if cls_out == 1:
                                tp_b += 1
                            else:
                                tn_b += 1
                        else:
                            if cls_out == 1:
                                fp_b += 1
                            else:
                                fn_b += 1

                    cls_acc_b = true_b / total_b
                    cls_pre_b = tp_b / (tp_b + fp_b + 1e-8)
                    cls_rec_b = tp_b / (tp_b + fn_b + 1e-8)
                    cls_spe_b = tn / (tn_b + fp_b + 1e-8)
                    cls_f1_b = (2 * cls_pre_b * cls_rec_b) / (cls_pre_b + cls_rec_b + 1e-8)
                    youden = cls_rec_b + cls_spe_b -1.0
                    images = next_images.to(self.device)
                    label = next_label.to(self.device)
                    # print(images.shape)
                    # print(label.shape)
                    activation_map_g = torch.ones((images.shape[0], 32, 32)).to(self.device)
                    activation_map_e = torch.ones((images.shape[0], 32, 32)).to(self.device)
                    if self.modal == 4:
                        if epoch < self.warmup:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map_g,
                                activation_map_e)
                        else:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map_g,
                                activation_map_e)
                    # print(cls.shape)
                    activation_map = cam_extractor(cls.argmax(axis=1).squeeze(0).tolist(), cls)
                    # activation_map_e = cam_extractor_e(cls.argmax(axis=1).squeeze(0).tolist(), cls)
                    # plt.imshow(activation_map1[0][0].squeeze(0).cpu().numpy())
                    # plt.axis('off')
                    # plt.tight_layout()
                    # plt.show()

                    cam_extractor.remove_hooks()
                    cam_extractor._hooks_enabled = False

                    if self.modal == 4:
                        if epoch < self.warmup:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map[0],
                                activation_map[1])
                        else:
                            cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                torch.ones((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                activation_map[0],
                                activation_map[1])

                    cls_cpu = cls.data.cpu().numpy()
                    label_cpu = label.data.cpu().numpy()
                    # print('cls_cpu[:,1]:',cls_cpu[:,1].shape)
                    # print('np.sum(label_cpu==0):',np.sum(label_cpu==0).shape)
                    # cls_f1_b*cls_f1_b*cls_f1_b+(cls_rec_b+cls_spe_b-1)*(cls_rec_b+cls_spe_b-1)
                    next_state = np.append(cls_cpu[:, 1],
                                           [np.array(np.sum(label_cpu == 0)), np.array(np.sum(label_cpu == 1))])
                    self.dqn.replay_buffer.push(state, [_action1, _action2], (0.5*cls_f1_b+0.5*youden), next_state, 1)
                    if len(self.dqn.replay_buffer) > 8:
                        self.dqn.update(8)
                    # cls_out = cls_cpu.argmax()
                    for b in range(batch_size):
                        total += 1

                        name_list.append(name[b])
                        probility_list.append(cls_cpu[b, 1])
                        cls_out = cls_cpu[b].argmax()
                        """if np.exp(cls_cpu[b, 1]) > 0.5:
                            cls_out = 1
                        else:
                            cls_out = 0"""
                        label_out = label_cpu[b]
                        label_list.append(label_out)
                        predict_list.append(cls_out)
                        if cls_out == label_out:
                            true += 1
                            if cls_out == 1:
                                tp += 1
                            else:
                                tn += 1
                        else:
                            if cls_out == 1:
                                fp += 1
                            else:
                                fn += 1

                # self.scheduler.step(epoch_loss)
                self.scheduler.step()
                cls_acc = true / total
                cls_pre = tp / (tp + fp + 1e-8)
                cls_rec = tp / (tp + fn + 1e-8)
                cls_spe = tn / (tn + fp + 1e-8)
                cls_f1 = (2 * cls_pre * cls_rec) / (cls_pre + cls_rec + 1e-8)
                unet_score = cls_f1 + cls_acc
                print('Epoch [%d/%d], Loss: %.4f , LR: %.8f' % (
                    epoch + 1, self.num_epochs, epoch_loss, self.optimizer.state_dict()['param_groups'][0]['lr']))

                print('[     Train_cls] Acc: %.4f, PRE: %.4f, REC: %.4f, SPE: %.4f, F1: %.4f' % (
                    cls_acc, cls_pre, cls_rec, cls_spe, cls_f1))
                print('[     DQN] Alpha1: %.4f, Alpha2: %.4f, Gamma: %.4f' % (
                    self.cls_criterion.alpha[0, 0], self.cls_criterion.alpha[1, 0], self.cls_criterion.gamma))

                """ if self.modal == 0:
                    save_image(denorm(images[:, 0:3, :, :].data),
                               os.path.join(self.result_dir, '{}_g_train.png'.format(epoch + 1)))
                elif self.modal == 1:
                    save_image(denorm(images[:, 3:6, :, :].data),
                               os.path.join(self.result_dir, '{}_e_train.png'.format(epoch + 1)))
                else:
                    save_image(denorm(images[:, 3:6, :, :].data),
                               os.path.join(self.result_dir, '{}_e_train.png'.format(epoch + 1)))
                    save_image(denorm(images[:, 0:3, :, :].data),
                               os.path.join(self.result_dir, '{}_g_train.png'.format(epoch + 1)))"""
                # ===================================== Validation ====================================#
                if epoch % self.val_step == 0:
                    self.unet.train(False)
                    self.unet.eval()
                    total = 0
                    true = 0
                    tp = 0
                    fp = 0
                    tn = 0
                    fn = 0

                    label_list = []
                    probility_list = []
                    predict_list = []
                    name_list = []
                    for i, (images, label, name) in enumerate(self.valid_loader):
                        # print(name)
                        epoch_loss = 0
                        images = images.to(self.device)

                        label = label.to(self.device)
                        cam_extractor = XGradCAM(self.unet, ['fire9_1', 'fire9_2'])
                        # cam_extractor_e = XGradCAM(self.unet, 'module.e_conv5_x')

                        cam_extractor._hooks_enabled = True
                        # cam_extractor_e._hooks_enabled = True
                        activation_map_g = torch.ones((images.shape[0], 32, 32)).to(self.device)
                        activation_map_e = torch.ones((images.shape[0], 32, 32)).to(self.device)

                        if self.modal == 0:
                            cls = self.unet(images[:, 0:3, :, :])
                        elif self.modal == 1:
                            cls = self.unet(images[:, 3:6, :, :])
                        elif self.modal == 2:
                            cls = self.unet(images[:, 0:6, :, :])
                        elif self.modal == 4:
                            if epoch < self.warmup:
                                cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                    torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                    activation_map_g,
                                    activation_map_e)
                            else:
                                cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                    torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                    activation_map_g,
                                    activation_map_e)
                            # print('*******finish')
                        # print(cls.shape)
                        # print(cls.argmax(axis=1).squeeze(0).tolist())
                        activation_map = cam_extractor(cls.argmax(axis=1).squeeze(0).tolist(), cls)
                        # activation_map_e = cam_extractor_e(cls.argmax(axis=1).squeeze(0).tolist(), cls)
                        # plt.imshow(activation_map1[0][0].squeeze(0).cpu().numpy())
                        # plt.axis('off')
                        # plt.tight_layout()
                        # plt.show()

                        if self.modal == 4:
                            if epoch < self.warmup:
                                cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                    torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :], activation_map[0],
                                    activation_map[1])
                            else:
                                cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                    torch.ones((1)), images[:, 0:3, :, :], images[:, 3:6, :, :], activation_map[0],
                                    activation_map[1])

                        cam_extractor.remove_hooks()
                        cam_extractor._hooks_enabled = False
                        batch_size = label.size(0)
                        cls_cpu = cls.data.cpu().numpy()
                        label_cpu = label.data.cpu().numpy()
                        # cls_out = cls_cpu.argmax()
                        for b in range(batch_size):
                            total += 1
                            name_list.append(name[b])
                            probility_list.append(cls_cpu[b, 1])
                            cls_out = cls_cpu[b].argmax()
                            """if np.exp(cls_cpu[b, 1]) > 0.5:
                                cls_out = 1
                            else:
                                cls_out = 0"""
                            label_out = label_cpu[b]
                            label_list.append(label_out)
                            predict_list.append(cls_out)
                            if cls_out == label_out:
                                true += 1
                                if cls_out == 1:
                                    tp += 1
                                else:
                                    tn += 1
                            else:
                                if cls_out == 1:
                                    fp += 1
                                else:
                                    fn += 1
                        if self.modal == 0:
                            save_image(denorm(images[:, 0:3, :, :].data),
                                       os.path.join(self.result_dir, '{}_{}_g_val.png'.format(epoch + 1, i)))
                        elif self.modal == 1:
                            save_image(denorm(images[:, 3:6, :, :].data),
                                       os.path.join(self.result_dir, '{}_{}_e_val.png'.format(epoch + 1, i)))
                        else:
                            if epoch == 0:
                                save_image(denorm(images[:, 3:6, :, :].data),
                                           os.path.join(self.result_dir, '{}_{}_e_val.png'.format(epoch + 1, i)))
                                save_image(denorm(images[:, 0:3, :, :].data),
                                           os.path.join(self.result_dir, '{}_{}_g_val.png'.format(epoch + 1, i)))
                        # Save Best U-Net model


                        img_map0 = []
                        att_f_g = []
                        affin_grid_g = []
                        affin_grid_e = []
                        att_f_e = []
                        g_att_map = []
                        e_att_map = []
                        att_f_g_before = []

                        att_f_e_before = []

                        edge_1 = []
                        color_1 = []
                        am0 = activation_map[0][0]
                        transf = transforms.ToTensor()
                        for j in range(0, activation_map[0].shape[0]):
                            # att_f_g.append(denorm(g_att_output[j,10:11,:,:]).data)
                            # att_f_e.append(denorm(e_att_output[j,10:11,:,:]).data)
                            result = overlay_mask(to_pil_image(denorm(images[j, 0:3, :, :].data)),
                                                  to_pil_image(activation_map[0][j], mode='F'), alpha=0.5)
                            input_tensor = transf(result)
                            img_map0.append(input_tensor)
                        # for j in range(0,affin_grid_1.shape[0]):
                        grid_map_1(affin_grid_1[0, :, :, :].data.cpu(),
                                   os.path.join(self.result_dir, '{}_{}_g_affin.png'.format(epoch + 1, i)))
                        grid_map_1(affin_grid_2[0, :, :, :].data.cpu(),
                                   os.path.join(self.result_dir, '{}_{}_e_affin.png'.format(epoch + 1, i)))

                        for j in range(0, g_att.shape[0]):
                            g_att_map_ = overlay_mask(to_pil_image(denorm(images[j, 0:3, 0:128, 0:128].data)),
                                                      to_pil_image(denorm(g_att[j, :, :].data), mode='F'), alpha=0.01)
                            g_att_map_ = transf(g_att_map_)
                            g_att_map.append(g_att_map_)

                            e_att_map_ = overlay_mask(to_pil_image(denorm(images[j, 3:6, 0:128, 0:128].data)),
                                                      to_pil_image(denorm(e_att[j, :, :].data), mode='F'), alpha=0.01)
                            e_att_map_ = transf(e_att_map_)
                            e_att_map.append(e_att_map_)

                        for j in range(0, x1_1.shape[1]):
                            att_f_g_ = overlay_mask(to_pil_image(denorm(images[0, 0:3, 0:128, 0:128].data)),
                                                    to_pil_image(denorm(x1_1[0, j:j + 1, :, :].data), mode='F'),
                                                    alpha=0.01)

                            att_f_g_ = transf(att_f_g_)
                            att_f_g.append(att_f_g_)

                            att_f_g_before_ = overlay_mask(to_pil_image(denorm(images[0, 0:3, 0:128, 0:128].data)),
                                                           to_pil_image(denorm(x1_1_before[0, j:j + 1, :, :].data),
                                                                        mode='F'), alpha=0.01)

                            att_f_g_before_ = transf(att_f_g_before_)
                            att_f_g_before.append(att_f_g_before_)

                            att_f_e_ = overlay_mask(to_pil_image(denorm(images[0, 3:6, 0:128, 0:128].data)),
                                                    to_pil_image(denorm(x2_1[0, j:j + 1, :, :].data), mode='F'),
                                                    alpha=0.01)
                            att_f_e_ = transf(att_f_e_)

                            att_f_e.append(att_f_e_)
                            att_f_e_before_ = overlay_mask(to_pil_image(denorm(images[0, 3:6, 0:128, 0:128].data)),
                                                           to_pil_image(denorm(x2_1_before[0, j:j + 1, :, :].data),
                                                                        mode='F'), alpha=0.01)
                            att_f_e_before_ = transf(att_f_e_before_)

                            att_f_e_before.append(att_f_e_before_)
                        for j in range(0, edge.shape[0]):
                            edge_ = overlay_mask(to_pil_image(denorm(images[j, 0:3, :, :].data)),
                                                 to_pil_image(denorm(edge[j, 0, :, :].data), mode='F'), alpha=0.01)
                            edge_ = transf(edge_)
                            edge_1.append(edge_)
                        for j in range(0, color.shape[1]):
                            color_1.append(denorm(color[0, j:j + 1, :, :]).data)
                        # print("**************shape********************")
                        save_image(edge_1,
                                   os.path.join(self.result_dir, '{}_{}_edge.png'.format(epoch + 1, i)))
                        save_image(color_1,
                                   os.path.join(self.result_dir, '{}_{}_color.png'.format(epoch + 1, i)))
                        # print("affin:",affin_grid_2.shape)
                        # print("edge:",edge.shape)

                        save_image(img_map0,
                                   os.path.join(self.result_dir, '{}_{}_g_cam.png'.format(epoch + 1, i)))
                        save_image(att_f_g,
                                   os.path.join(self.result_dir, '{}_{}_g_f.png'.format(epoch + 1, i)))
                        save_image(att_f_e,
                                   os.path.join(self.result_dir, '{}_{}_e_f.png'.format(epoch + 1, i)))
                        save_image(att_f_g_before,
                                   os.path.join(self.result_dir, '{}_{}_g_f_before.png'.format(epoch + 1, i)))
                        save_image(att_f_e_before,
                                   os.path.join(self.result_dir, '{}_{}_e_f_before.png'.format(epoch + 1, i)))
                        """save_image(affin_grid_g,
                                   os.path.join(self.result_dir, '{}_{}_g_affin.png'.format(epoch + 1, i)))
                        save_image(affin_grid_e,
                                   os.path.join(self.result_dir, '{}_{}_e_affin.png'.format(epoch + 1, i)))"""

                        save_image(g_att_map,
                                   os.path.join(self.result_dir, '{}_{}_g_att.png'.format(epoch + 1, i)))
                        save_image(e_att_map,
                                   os.path.join(self.result_dir, '{}_{}_e_att.png'.format(epoch + 1, i)))
                        img_map1 = []
                        am1 = activation_map[1][0]
                        for j in range(0, activation_map[1].shape[0]):
                            result = overlay_mask(to_pil_image(denorm(images[j, 3:6, :, :].data)),
                                                  to_pil_image(activation_map[1][j], mode='F'), alpha=0.5)
                            input_tensor = transf(result)
                            img_map1.append(input_tensor)
                        save_image(img_map1,
                                   os.path.join(self.result_dir, '{}_{}_e_cam.png'.format(epoch + 1, i)))

                    cls_acc = true / total
                    cls_pre = tp / (tp + fp + 1e-8)
                    cls_rec = tp / (tp + fn + 1e-8)
                    cls_spe = tn / (tn + fp + 1e-8)
                    cls_f1 = (2 * cls_pre * cls_rec) / (cls_pre + cls_rec + 1e-8)
                    unet_score = cls_f1 + cls_acc
                    unet_acc = cls_acc
                    unet_f1 = cls_f1
                    print('[Validation_cls] Acc: %.4f, PRE: %.4f, REC: %.4f, SPE: %.4f, F1: %.4f' % (
                        cls_acc, cls_pre, cls_rec, cls_spe, cls_f1))
                    """if not os.path.exists(Path(self.result_path+"/log.txt")):
                        with open(Path(self.result_path+"/log.txt"), "w") as f:
                            print(f)"""
                    with open(Path(self.result_path), "a") as f:

                        f.write(('[Validation_cls] Acc: %.4f, PRE: %.4f, REC: %.4f, SPE: %.4f, F1: %.4f\n' % (
                            cls_acc, cls_pre, cls_rec, cls_spe, cls_f1)))
                        f.write(('[Name] %s\n' % str(name_list)))

                        f.write(('[Probility] %s\n' % str(probility_list)))
                        f.write(('[Predict] %s\n' % str(predict_list)))

                        f.write(('[Label  ] %s\n' % str(label_list)))
                    # print(images.data)

                    flag = 0
                    if unet_score >= best_unet_score:
                        flag += 1
                        best_unet_score = unet_score
                        best_epoch = epoch
                        best_unet = self.unet.state_dict()
                        with open(Path(self.result_path), "a") as f:
                            f.write('Best %s model score : %.4f\n' % (self.model_type, best_unet_score))
                        print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
                        torch.save(best_unet, unet_path + '_best.pkl')
                    if unet_f1 >= best_f1:
                        flag += 1
                        best_f1 = unet_f1
                        best_epoch = epoch
                        best_unet = self.unet.state_dict()
                        with open(Path(self.result_path), "a") as f:
                            f.write('Best %s f1 score : %.4f\n' % (self.model_type, best_f1))
                        print('Best %s f1 score : %.4f' % (self.model_type, best_f1))
                        torch.save(best_unet, unet_path + '_best_f1.pkl')
                    if unet_acc >= best_acc:
                        flag += 1
                        best_acc = unet_acc
                        best_epoch = epoch
                        best_unet = self.unet.state_dict()
                        with open(Path(self.result_path), "a") as f:
                            f.write('Best %s acc score : %.4f\n' % (self.model_type, best_acc))
                        print('Best %s acc score : %.4f' % (self.model_type, best_acc))
                        torch.save(best_unet, unet_path + '_best_acc.pkl')
                    if flag > 0:
                        self.unet.train(False)
                        self.unet.eval()
                        total = 0
                        true = 0
                        tp = 0
                        fp = 0
                        tn = 0
                        fn = 0

                        label_list = []
                        probility_list = []
                        predict_list = []
                        name_list = []
                        for i, (images, label, name) in enumerate(self.test_loader):
                            # print(name)
                            epoch_loss = 0
                            images = images.to(self.device)

                            label = label.to(self.device)
                            cam_extractor = XGradCAM(self.unet, ['fire9_1', 'fire9_2'])
                            # cam_extractor_e = XGradCAM(self.unet, 'module.e_conv5_x')

                            cam_extractor._hooks_enabled = True
                            # cam_extractor_e._hooks_enabled = True
                            activation_map_g = torch.ones((images.shape[0], 32, 32)).to(self.device)
                            activation_map_e = torch.ones((images.shape[0], 32, 32)).to(self.device)

                            if self.modal == 0:
                                cls = self.unet(images[:, 0:3, :, :])
                            elif self.modal == 1:
                                cls = self.unet(images[:, 3:6, :, :])
                            elif self.modal == 2:
                                cls = self.unet(images[:, 0:6, :, :])
                            elif self.modal == 4:
                                if epoch < self.warmup:
                                    cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                        torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                        activation_map_g,
                                        activation_map_e)
                                else:
                                    cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                        torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                        activation_map_g,
                                        activation_map_e)
                                # print('*******finish')

                            activation_map = cam_extractor(cls.argmax(axis=1).squeeze(0).tolist(), cls)
                            # activation_map_e = cam_extractor_e(cls.argmax(axis=1).squeeze(0).tolist(), cls)
                            # plt.imshow(activation_map1[0][0].squeeze(0).cpu().numpy())
                            # plt.axis('off')
                            # plt.tight_layout()
                            # plt.show()

                            cam_extractor.remove_hooks()
                            cam_extractor._hooks_enabled = False
                            if self.modal == 4:
                                if epoch < self.warmup:
                                    cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                        torch.zeros((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                        activation_map[0],
                                        activation_map[1])
                                else:
                                    cls, g_att_output, e_att_output, x1_1, x2_1, affin_grid_1, affin_grid_2, edge, color, g_att, e_att, x1_1_before, x2_1_before = self.unet(
                                        torch.ones((1)), images[:, 0:3, :, :], images[:, 3:6, :, :],
                                        activation_map[0],
                                        activation_map[1])

                            """plt.imshow(activation_map[0].squeeze(0).cpu().numpy())
                            plt.axis('off')
                            plt.tight_layout()
                            plt.show()"""
                            batch_size = label.size(0)

                            cls_cpu = cls.data.cpu().numpy()
                            label_cpu = label.data.cpu().numpy()

                            # cls_out = cls_cpu.argmax()
                            for b in range(batch_size):
                                total += 1
                                name_list.append(name[b])
                                probility_list.append(cls_cpu[b, 1])
                                cls_out = cls_cpu[b].argmax()
                                """if np.exp(cls_cpu[b, 1]) > 0.5:
                                    cls_out = 1
                                else:
                                    cls_out = 0"""
                                label_out = label_cpu[b]
                                label_list.append(label_out)
                                predict_list.append(cls_out)
                                if cls_out == label_out:
                                    true += 1
                                    if cls_out == 1:
                                        tp += 1
                                    else:
                                        tn += 1
                                else:
                                    if cls_out == 1:
                                        fp += 1
                                    else:
                                        fn += 1

                        cls_acc = true / total
                        cls_pre = tp / (tp + fp + 1e-8)
                        cls_rec = tp / (tp + fn + 1e-8)
                        cls_spe = tn / (tn + fp + 1e-8)
                        cls_f1 = (2 * cls_pre * cls_rec) / (cls_pre + cls_rec + 1e-8)
                        unet_score = cls_f1 + cls_acc
                        unet_acc = cls_acc
                        unet_f1 = cls_f1
                        print('[      Test_cls] Acc: %.4f, PRE: %.4f, REC: %.4f, SPE: %.4f, F1: %.4f' % (
                            cls_acc, cls_pre, cls_rec, cls_spe, cls_f1))
                        """if not os.path.exists(Path(self.result_path+"/log.txt")):
                            with open(Path(self.result_path+"/log.txt"), "w") as f:
                                print(f)"""
                        with open(Path(self.result_path), "a") as f:

                            f.write(('[      Test_cls] Acc: %.4f, PRE: %.4f, REC: %.4f, SPE: %.4f, F1: %.4f\n' % (
                                cls_acc, cls_pre, cls_rec, cls_spe, cls_f1)))
                            f.write(('[Name] %s\n' % str(name_list)))

                            f.write(('[Probility] %s\n' % str(probility_list)))
                            f.write(('[Predict] %s\n' % str(predict_list)))

                            f.write(('[Label  ] %s\n' % str(label_list)))
                        if self.modal == 0:
                            save_image(denorm(images[:, 0:3, :, :].data),
                                       os.path.join(self.result_dir, '{}_g_test.png'.format(epoch + 1)))
                        elif self.modal == 1:
                            save_image(denorm(images[:, 3:6, :, :].data),
                                       os.path.join(self.result_dir, '{}_e_test.png'.format(epoch + 1)))
                        elif self.modal == 2:
                            save_image(denorm(images[:, 3:6, :, :].data),
                                       os.path.join(self.result_dir, '{}_e_test.png'.format(epoch + 1)))
                            save_image(denorm(images[:, 0:3, :, :].data),
                                       os.path.join(self.result_dir, '{}_g_test.png'.format(epoch + 1)))

                        """save_image((activation_map0[0].unsqueeze(1).data),
                                   os.path.join(self.result_dir, '{}_cam_val.png'.format(epoch + 1)))"""

                        # plt.imsave(os.path.join(self.result_dir, '{}_cam_test.png'.format(epoch + 1)),activation_map[0][0].squeeze(0).cpu().numpy())
            # ===================================== Test ====================================#

