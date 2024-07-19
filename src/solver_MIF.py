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
from arch.TransMed import MultiModalNet as TransMed
from ptflops import get_model_complexity_info

from torchvision.utils import save_image

from arch.convnext_first_cat import convnext_base as convnext_first_cat
from arch.ours_backbone import ours as ours_backbone

from arch.mobilenetv2_last_cat import mobilenetv2 as mobilenetv2_last_cat
from arch.googlenet_last_cat import googlenet as googlenet_last_cat
from arch.nasnet_last_cat import nasnet as nasnet_last_cat
from arch.densenet_last_cat import densenet121 as densenet_last_cat
from arch.senet_last_cat import seresnet18 as senet_last_cat
from arch.shufflenetv2_last_cat import shufflenetv2 as shufflenet_last_cat
from arch.convnext_last_cat import convnext_base as convnext_last_cat
from arch.vgg_last_cat import vgg11_bn as vgg_last_cat
from arch.resnet_last_cat import resnet18 as resnet18_last_cat
from arch.resnext_last_cat import resnext50 as resnext_last_cat
from arch.xception_last_cat import xception as xception_last_cat
from arch.VIT_last_cat import ViT as ViT_last_cat
from arch.swin_transformer_last_cat import SwinTransformer as swin_transformer_last_cat
from tqdm import tqdm
from collections import OrderedDict
import csv
from PIL import Image
import torch
import scipy.misc
import cv2
import torch.nn as nn

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[len('module.'):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict
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
        self.cls_criterion = torch.nn.CrossEntropyLoss()
        self.augmentation_prob = config.augmentation_prob
        self.img_size = config.image_size
        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lr_decay = config.lr_decay
        self.opt = config.opt
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
        if self.model_type == 'ours':
            self.unet = ours_backbone()
        elif self.model_type == 'resnet18_last_cat':
            self.unet = resnet18_last_cat()
        elif self.model_type == 'resnet18_first_cat':
            self.unet = resnet18_first_cat()
        elif self.model_type == 'mobilenetv2_first_cat':
            self.unet = mobilenetv2_first_cat()
        elif self.model_type == 'googlenet_first_cat':
            self.unet = googlenet_first_cat()
        elif self.model_type == 'nasnet_first_cat':
            self.unet = nasnet_first_cat()
        elif self.model_type == 'senet_first_cat':
            self.unet = senet_first_cat()
        elif self.model_type == 'shufflenetv2_first_cat':
            self.unet = shufflenetv2_first_cat()
        elif self.model_type == 'resnext50_first_cat':
            self.unet = resnext50_first_cat()
        elif self.model_type == 'vgg_first_cat':
            self.unet = vgg_first_cat()
        elif self.model_type == 'convnext_first_cat':
            self.unet = convnext_first_cat()
        elif self.model_type == 'mobilenetv2_last_cat':
            self.unet = mobilenetv2_last_cat()
        elif self.model_type == 'googlenet_last_cat':
            self.unet = googlenet_last_cat()
        elif self.model_type == 'nasnet_last_cat':
            self.unet = nasnet_last_cat()
        elif self.model_type == 'densenet_last_cat':
            self.unet = densenet_last_cat()
        elif self.model_type == 'senet_last_cat':
            self.unet = senet_last_cat()
        elif self.model_type == 'shufflenet_last_cat':
            self.unet = shufflenet_last_cat()
        elif self.model_type == 'convnext_last_cat':
            self.unet = convnext_last_cat()
        elif self.model_type == 'vgg_last_cat':
            self.unet = vgg_last_cat()
        elif self.model_type == 'resnext_last_cat':
            self.unet = resnext_last_cat()
        elif self.model_type == 'xception_last_cat':
            self.unet = xception_last_cat()
        elif self.model_type == 'ViT_last_cat':
            self.unet = ViT_last_cat(image_size=256, patch_size=32, num_classes=2, dim=1024, depth=6, heads=16, mlp_dim=2048,
                        dropout=0.1, emb_dropout=0.1)
        elif self.model_type == 'swin_transformer_last_cat':
            self.unet = swin_transformer_last_cat()
        elif self.model_type == 'TransMed':
            self.unet = TransMed()

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
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.98)
        elif self.lr_decay == 'Lambda':
            lambda1 = lambda epoch: np.sin(epoch) / epoch
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        #self.unet = nn.DataParallel(self.unet)
        self.unet.cuda()
        # self.unet.to(self.device)
        flops, params = get_model_complexity_info(self.unet, (3, 256, 256), as_strings=True,
                                                  print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
        print('-------------------------------------------')
        print('Flops:  ' + flops)
        print('Params: ' + params)
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
        #self.unet.load_state_dict(torch.load(self.checkpoint))

        checkpoint = torch.load(self.checkpoint)
        state_dict = remove_module_prefix(checkpoint)
        self.unet.load_state_dict(state_dict)

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
        feature1_1_list = []
        feature1_2_list = []
        feature2_1_list = []
        feature2_2_list = []
        feature3_1_list = []
        feature3_2_list = []
        feature4_1_list = []
        feature4_2_list = []
        feature5_1_list = []
        feature5_2_list = []
        for i, (images, label, name) in enumerate(self.test_loader):
            print(name)
            images = images.to(self.device)
            # GT = GT.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                if self.modal == 2:
                    cls,f1_1,f1_2,f2_1,f2_2,f3_1,f3_2,f4_1,f4_2,f5_1,f5_2 = self.unet(images[:, 0:3, :, :], images[:, 3:6, :, :])
                elif self.modal == 3:
                    cls = self.unet(images[:, :, :, :])
            batch_size = label.size(0)

            cls_cpu = cls.data.cpu().numpy()
            label_cpu = label.data.cpu().numpy()
            f1_1_cpu = f1_1.data.cpu().numpy()
            f1_2_cpu = f1_2.data.cpu().numpy()
            f2_1_cpu = f2_1.data.cpu().numpy()
            f2_2_cpu = f2_2.data.cpu().numpy()
            f3_1_cpu = f3_1.data.cpu().numpy()
            f3_2_cpu = f3_2.data.cpu().numpy()
            f4_1_cpu = f4_1.data.cpu().numpy()
            f4_2_cpu = f4_2.data.cpu().numpy()
            f5_1_cpu = f5_1.data.cpu().numpy()
            f5_2_cpu = f5_2.data.cpu().numpy()
            # cls_out = cls_cpu.argmax()
            for b in range(batch_size):
                total += 1
                name_list.append(name[b])
                probility_list.append(np.exp(cls_cpu[b, 1]))
                cls_out = cls_cpu[b].argmax()
                """if np.exp(cls_cpu[b, 1]) > 0.5:
                    cls_out = 1
                else:
                    cls_out = 0"""
                label_out = label_cpu[b]
                label_list.append(label_out)
                predict_list.append(cls_out)

                f1_1_out = f1_1_cpu[b]
                feature1_1_list.append(f1_1_out.tolist())
                f1_2_out = f1_2_cpu[b]
                feature1_2_list.append(f1_2_out.tolist())
                f2_1_out = f2_1_cpu[b]
                feature2_1_list.append(f2_1_out.tolist())
                f2_2_out = f2_2_cpu[b]
                feature2_2_list.append(f2_2_out.tolist())
                f3_1_out = f3_1_cpu[b]
                feature3_1_list.append(f3_1_out.tolist())
                f3_2_out = f3_2_cpu[b]
                feature3_2_list.append(f3_2_out.tolist())
                f4_1_out = f4_1_cpu[b]
                feature4_1_list.append(f4_1_out.tolist())
                f4_2_out = f4_2_cpu[b]
                feature4_2_list.append(f4_2_out.tolist())
                f5_1_out = f5_1_cpu[b]
                feature5_1_list.append(f5_1_out.tolist())
                f5_2_out = f5_2_cpu[b]
                feature5_2_list.append(f5_2_out.tolist())


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

        f1_1_np = np.array(feature1_1_list)
        np.save(self.result_dir + '/feature1_1_np', f1_1_np)
        f1_2_np = np.array(feature1_2_list)
        np.save(self.result_dir + '/feature1_2_np', f1_2_np)
        f2_1_np = np.array(feature2_1_list)
        np.save(self.result_dir + '/feature2_1_np', f2_1_np)
        f2_2_np = np.array(feature2_2_list)
        np.save(self.result_dir + '/feature2_2_np', f2_2_np)
        f3_1_np = np.array(feature3_1_list)
        np.save(self.result_dir + '/feature3_1_np', f3_1_np)
        f3_2_np = np.array(feature3_2_list)
        np.save(self.result_dir + '/feature3_2_np', f3_2_np)
        f4_1_np = np.array(feature4_1_list)
        np.save(self.result_dir + '/feature4_1_np', f4_1_np)
        f4_2_np = np.array(feature4_2_list)
        np.save(self.result_dir + '/feature4_2_np', f4_2_np)
        f5_1_np = np.array(feature5_1_list)
        np.save(self.result_dir + '/feature5_1_np', f5_1_np)
        f5_2_np = np.array(feature5_2_list)
        np.save(self.result_dir + '/feature5_2_np', f5_2_np)





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
                for images, label, name in tqdm(self.train_loader):

                    images = images.to(self.device)

                    label = label.to(self.device)
                    if self.modal == 2:
                        cls = self.unet(images[:, 0:3, :, :], images[:, 3:6, :, :])
                    elif self.modal == 3:
                        cls = self.unet(images[:, :, :, :])
                    cls_loss = self.cls_criterion(cls, label.long())
                    loss = cls_loss
                    epoch_loss += loss.item()

                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                    batch_size = label.size(0)

                    cls_cpu = cls.data.cpu().numpy()
                    label_cpu = label.data.cpu().numpy()

                    # cls_out = cls_cpu.argmax()
                    for b in range(batch_size):
                        total += 1

                        name_list.append(name[b])
                        probility_list.append(np.exp(cls_cpu[b, 1]))
                        # cls_out = cls_cpu[b].argmax()
                        if np.exp(cls_cpu[b, 1]) > 0.5:
                            cls_out = 1
                        else:
                            cls_out = 0
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
                        # GT = GT.to(self.device)
                        label = label.to(self.device)
                        with torch.no_grad():
                            if self.modal == 2:
                                cls = self.unet(images[:, 0:3, :, :], images[:, 3:6, :, :])
                            elif self.modal == 3:
                                cls = self.unet(images[:, :, :, :])
                        batch_size = label.size(0)
                        cls_cpu = cls.data.cpu().numpy()
                        label_cpu = label.data.cpu().numpy()
                        # cls_out = cls_cpu.argmax()
                        for b in range(batch_size):
                            total += 1
                            name_list.append(name[b])
                            probility_list.append(np.exp(cls_cpu[b, 1]))
                            # cls_out = cls_cpu[b].argmax()
                            if np.exp(cls_cpu[b, 1]) > 0.5:
                                cls_out = 1
                            else:
                                cls_out = 0
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
                    save_image(denorm(images[:, 0:3, :, :].data),
                               os.path.join(self.result_dir, '{}_g_val.png'.format(epoch + 1)))
                    save_image(denorm(images[:, 3:6, :, :].data),
                               os.path.join(self.result_dir, '{}_e_val.png'.format(epoch + 1)))
                    # Save Best U-Net model
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
                            # GT = GT.to(self.device)
                            label = label.to(self.device)

                            with torch.no_grad():
                                if self.modal == 2:
                                    cls = self.unet(images[:, 0:3, :, :], images[:, 3:6, :, :])
                                elif self.modal == 3:
                                    cls = self.unet(images[:, :, :, :])
                            batch_size = label.size(0)

                            cls_cpu = cls.data.cpu().numpy()
                            label_cpu = label.data.cpu().numpy()

                            # cls_out = cls_cpu.argmax()
                            for b in range(batch_size):
                                total += 1
                                name_list.append(name[b])
                                probility_list.append(np.exp(cls_cpu[b, 1]))
                                # cls_out = cls_cpu[b].argmax()
                                if np.exp(cls_cpu[b, 1]) > 0.5:
                                    cls_out = 1
                                else:
                                    cls_out = 0
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
                        save_image(denorm(images[:, 0:3, :, :].data),
                                   os.path.join(self.result_dir, '{}_g_test.png'.format(epoch + 1)))
                        save_image(denorm(images[:, 3:6, :, :].data),
                                   os.path.join(self.result_dir, '{}_e_test.png'.format(epoch + 1)))
            # ===================================== Test ====================================#

