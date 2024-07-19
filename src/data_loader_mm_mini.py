import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
np.set_printoptions(threshold=100000)

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
class ImageFolder(data.Dataset):
    def __init__(self, name_path,data_path, image_size=(128, 128), mode='train',augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.patient_paths = []
        self.patient_paths= os.listdir(name_path)
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        #self.transform = transform
        self.augmentation_prob = augmentation_prob
        self.data_path = data_path

        print("image count in {} path :{}".format(self.mode, len(self.patient_paths)))
    def __getitem__(self, index):

        patient_name = self.patient_paths[index]
        #print(patient_name)
        gray_image_path = self.data_path+ '/' + patient_name + '/gray.jpg'
        elastic_image_path = self.data_path + '/' + patient_name + '/elastic.jpg'
        img_names = [gray_image_path,elastic_image_path]
        label = np.load(self.data_path+ '/' + patient_name + '/label.npy')  # 显示所有源文件内容
        out_img = torch.zeros((6,self.image_size[0],self.image_size[1]))
        #print(out_img.shape)
        #print(image_paths)
        i=0
        Transform1 = []
        CropRange = 900
        Transform1.append(T.CenterCrop((CropRange, CropRange)))
        if (self.mode == 'train'):
            Transform1.append(T.RandomCrop((600,600)))
        else:
            Transform1.append(T.CenterCrop((600,600)))

        Transform1 = T.Compose(Transform1)
        p_transform = random.random()
        if (self.mode == 'train') and p_transform <= self.augmentation_prob:

            image = Image.open(gray_image_path)
            aspect_ratio = image.size[1] / image.size[0]
            Transform2 = []
            """RotationDegree = random.randint(0, 3)
            RotationDegree = self.RotationDegree[RotationDegree]
            if (RotationDegree == 90) or (RotationDegree == 270):
                aspect_ratio = 1 / aspect_ratio
            Transform2.append(T.RandomRotation((RotationDegree, RotationDegree)))"""
            RotationRange = random.randint(-15, 15)
            Transform2.append(T.RandomRotation((RotationRange, RotationRange)))
            Transform2 = T.Compose(Transform2)
            flip_random = random.random()
            Transform3 = []
            Transform3.append(T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02))
            Transform3 = T.Compose(Transform3)

        for imagename in img_names:
            #print(imagename)
            image = Image.open(imagename).convert('RGB')
            #print(image.size)
            image = Transform1(image)
            if (self.mode == 'train') and p_transform <= self.augmentation_prob:
                image = Transform2(image)
                if  flip_random< 0.5:
                    image = F.hflip(image)
                """if flip_random < 0.2:
                    image = F.vflip(image)"""
                #image = Transform3(image)
            Transform = []
            Transform.append(T.Resize(self.image_size))
            Transform.append(T.ToTensor())
            #Transform.append(AddGaussianNoise(mean=0.0, std=0.01))  # 添加随机高斯噪声
            Transform.append(T.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]))
            Transform = T.Compose(Transform)
            image = Transform(image)
            #print("image_shape:",image.shape)
            out_img[i*3+0, :, :] = image[0, :, :]
            out_img[i*3+1, :, :] = image[1, :, :]
            out_img[i*3+2, :, :] = image[2, :, :]
            i+=1
        label_np = np.array(float(label), np.float)
        label_tensor = torch.from_numpy(label_np)
        label_tensor = label_tensor.long()
            # image = torch.unsqueeze(image, dim=0)
        #print(out_img.shape)
        #return out_img[0:3],out_img[3:6], label_tensor,self.patient_paths[index]
        return out_img, label_tensor,self.patient_paths[index]

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.patient_paths)


def get_loader(name_path,data_path, image_size, batch_size,  num_workers=2, mode='train',augmentation_prob=0.4):
    """Builds and returns Dataloader."""
    dataset = ImageFolder(name_path = name_path,data_path=data_path,  image_size=image_size, mode=mode,
                          augmentation_prob=augmentation_prob)
    if augmentation_prob==0:
        data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
