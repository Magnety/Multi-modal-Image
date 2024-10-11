import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label, find_objects
from scipy.signal import convolve2d
class GaussianEdgeExtractor(nn.Module):
    def __init__(self,channel, kernel_size=5, sigma=1.0):
        super(GaussianEdgeExtractor, self).__init__()
        self.kernel_size = kernel_size

        gx = self.gaussian_derivative_x(kernel_size, sigma)
        gy = self.gaussian_derivative_y(kernel_size, sigma)

        #print(gx.shape)
        gx = gx.unsqueeze(0).unsqueeze(0)
        gy = gy.unsqueeze(0).unsqueeze(0)
        """
        gx = torch.Tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]]).view(1, 1, 3, 3)
        gy = torch.Tensor([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]]).view(1, 1, 3, 3)"""
        self.Gx_weight = nn.Parameter(gx.repeat(3,3,1,1),requires_grad=False)
        self.Gy_weight = nn.Parameter(gy.repeat(3,3,1,1),requires_grad=False)
        #print(self.Gx.weight.data.shape)
        self.act = nn.Tanh()

        # Freeze the weights (make them non-trainable)
        for param in self.parameters():
            param.requires_grad = False

    def gaussian_derivative_x(self, kernel_size, sigma):
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = (-xx / (2 * np.pi * sigma ** 4)) * np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        #kernel = kernel / kernel.sum()  # Normalize the kernel
        print(torch.from_numpy(kernel).float())
        return torch.from_numpy(kernel).float()
    def gaussian_derivative_y(self, kernel_size, sigma):
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = (-yy / (2 * np.pi * sigma ** 4)) * np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        #kernel = kernel / kernel.sum()  # Normalize the kernel
        print(torch.from_numpy(kernel).float())
        return torch.from_numpy(kernel).float()

    def forward(self, x):
        #print(x[0,0,:,:].max())
        #print(x[0,0,:,:].min())

        horizontal_edges = F.conv2d(x,self.Gx_weight,padding=self.kernel_size // 2)
        vertical_edges = F.conv2d(x,self.Gy_weight,padding=self.kernel_size // 2)

        #vertical_edges = self.Gy(x)
        #print(horizontal_edges[0,0,:,:].max())

        #print(vertical_edges[0,0,:,:])
        magnitude = torch.sqrt(horizontal_edges ** 2 + vertical_edges ** 2+1e-8)
        #print(magnitude[0,0,:,:].max())

        magnitude = self.act(magnitude)
        #magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
        #magnitude = (magnitude - 0.5) * 2
        #print(magnitude[0,0,:,:].max())

        return magnitude

def unravel_index(indices, shape):
    return tuple(torch.tensor(x) for x in np.unravel_index(indices, shape))
def compute_local_color_moments(feature_maps, window_size=5):
    padding = window_size // 2
    unfolded_maps = F.unfold(feature_maps, window_size, padding=padding).view(feature_maps.size(0),
                                                                              feature_maps.size(1),
                                                                              window_size * window_size,
                                                                              feature_maps.size(2),
                                                                              feature_maps.size(3))

    mean = unfolded_maps.mean(dim=2)
    var = unfolded_maps.var(dim=2)
    skewness = ((unfolded_maps - mean.unsqueeze(2)) ** 3).mean(dim=2) / (var.sqrt() ** 3 + 1e-6)
    comments = torch.cat((mean,var,skewness),1)
    return comments

class MutualAttention_register(nn.Module):
    def __init__(self,channel):
        super(MutualAttention_register, self).__init__()
        self.size = 128
        self.edge_extract = GaussianEdgeExtractor(channel=3)
        self.conv1 = nn.Conv2d(3,channel,kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(9,channel,kernel_size=3, padding=1)
        self.softmax = nn.Softmax()
        self.relu = nn.Sequential(nn.BatchNorm2d(channel),nn.ReLU6(inplace=True))
    def forward(self, warm_flag, feat1, feat2, cam1, cam2, f_g, f_e):
        # print(feat1.shape)
        # print(feat2.shape)
        # print(cam1.shape)
        cam1 = cam1.unsqueeze(1)
        cam1 = F.interpolate(cam1, size=(self.size,self.size), mode='bilinear', align_corners=False)
        cam2 = cam2.unsqueeze(1)
        cam2 = F.interpolate(cam2, size=(self.size,self.size), mode='bilinear', align_corners=False)
        # print('********',cam1.shape)
        mask2 = torch.zeros_like(feat2)
        mask1 = torch.zeros_like(feat1)
        max_indices1 = torch.zeros((cam1.shape[0], 2))
        max_indices2 = torch.zeros((cam2.shape[0], 2))
        if warm_flag == 1:
            B, C, H, W = cam1.shape
            # Iterate over each batch
            for i in range(B):
                F_CAM1 = cam1[i, 0, :, :]
                F_CAM2 = cam2[i, 0, :, :]
    
                H, W = F_CAM1.shape
    
    
                x_max, y_max = np.unravel_index(np.argmax(F_CAM1), F_CAM1.shape)
    
                gradient_x = np.gradient(F_CAM1, axis=0)
                gradient_y = np.gradient(F_CAM1, axis=1)
                gradient_magnitude = np.hypot(gradient_x, gradient_y)
    
                threshold = 0.2  # 
                mask = gradient_magnitude < threshold
    
                labeled_array, num_features = label(mask)
                label_at_max = labeled_array[x_max, y_max]
    
                if label_at_max != 0:
                    region_slice = find_objects(labeled_array == label_at_max)[0]
                    x1 = (region_slice[0].start + region_slice[0].stop) // 2
                    y1 = (region_slice[1].start + region_slice[1].stop) // 2
                else:
                    x1, y1 = x_max, y_max
    
    
                window_size = (H // 4, W // 4)  
                kernel = np.ones(window_size)
                conv_result = convolve2d(F_CAM2, kernel, mode='valid')
    
                x2_conv, y2_conv = np.unravel_index(np.argmax(conv_result), conv_result.shape)
                x2 = x2_conv + window_size[0] // 2
                y2 = y2_conv + window_size[1] // 2
    
    
    
                max_indices1[i][0] = x1
                max_indices1[i][1] = y1
    
                max_indices2[i][0] = x2
                max_indices2[i][1] = y2
    
                h_cam1, w_cam1 = F_CAM1.shape
                h_cam2, w_cam2 = F_CAM2.shape
    
                region_size1 = h_cam1 // 4
                region_size2 = h_cam2 // 4
    
                top1 = max(0, x1 - region_size1 // 2)
                bottom1 = min(h_cam1, x1 + region_size1 // 2)
                left1 = max(0, y1 - region_size1 // 2)
                right1 = min(w_cam1, y1 + region_size1 // 2)
    
                top2 = max(0, x2 - region_size2 // 2)
                bottom2 = min(h_cam2, x2 + region_size2 // 2)
                left2 = max(0, y2 - region_size2 // 2)
                right2 = min(w_cam2, y2 + region_size2 // 2)
    
                mask1[i, :, top1:bottom1, left1:right1] = 1
                mask2[i, :, top2:bottom2, left2:right2] = 1

        cam2_move = (max_indices2 - max_indices1) / (cam2.shape[2] // 2)
        cam1_move = (max_indices1 - max_indices2) / (cam1.shape[2] // 2)
        theta2 = torch.tensor(
            [[[1, 0, cam2_move[0, 0]], [0, 1, cam2_move[0, 1]]]])
        theta1 = torch.tensor(
            [[[1, 0, cam1_move[0, 0]], [0, 1, cam1_move[0, 1]]]])
        # print('*****',theta1.shape)
        for i in range(1, cam2.shape[0]):
            # print(i)
            theta2 = torch.cat((theta2, torch.tensor([[[1, 0, cam2_move[i, 0]], [0, 1, cam2_move[i, 1]]]])), dim=0)
            theta1 = torch.cat((theta1, torch.tensor([[[1, 0, cam1_move[i, 0]], [0, 1, cam1_move[i, 1]]]])), dim=0)
        # print('*****',theta1.shape)

        affine_grid1 = F.affine_grid(theta1, cam1.size())
        affine_grid2 = F.affine_grid(theta2, cam2.size())
        _feat2_color = compute_local_color_moments(f_e)

        _feat2_color_down = F.interpolate(_feat2_color, size=(self.size,self.size), mode='bilinear', align_corners=False)


        affine_grid2 = affine_grid2.to(_feat2_color_down.get_device())

        _feat1_edge = self.edge_extract(f_g)
        _feat1_edge_down = F.interpolate(_feat1_edge, size=(self.size, self.size), mode='bilinear', align_corners=False)

        affine_grid1 = affine_grid1.to(_feat1_edge_down.get_device())


        feat1_edge = self.conv1(_feat1_edge_down)
        feat1_edge = F.relu6(feat1_edge)
        feat2_color = self.conv2(_feat2_color_down)
        feat2_move =  F.grid_sample(feat2_color * mask2, affine_grid2)
        feat1_edge_move =  F.grid_sample(feat1_edge * mask1, affine_grid1)


        B,C,H,W = feat1.shape

        feat1_ = feat1.view(B,C,-1)
        feat2_ = feat2.view(B,C,-1)
        feat2_move = feat2_move.view(B,C,-1)
        feat1_edge_move = feat1_edge_move.view(B,C,-1)
        f2_attn = F.softmax(feat2_move@feat1_.transpose(-2, -1),-1)
        feat1_apply =(f2_attn@feat1_).contiguous()
        feat1_apply = feat1_apply.view(B,C,H,W)
        f1_attn = F.softmax(feat1_edge_move@feat2_.transpose(-2, -1),-1)


        feat2_apply =(f1_attn@feat2_).contiguous()
        feat2_apply = feat2_apply.view(B,C,H,W)
        feat1_apply = self.relu(feat1_apply)
        feat2_apply = self.relu(feat2_apply)


        if warm_flag==0:
            return feat1, feat2,affine_grid1,affine_grid2,feat1_edge,feat2_color,f1_attn,f2_attn,feat1, feat2

        else:
            return feat1_apply, feat2_apply,affine_grid1,affine_grid2,feat1_edge,feat2_color,f1_attn,f2_attn,feat1, feat2

class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):

        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x

class OursNet(nn.Module):

    """mobile net with simple bypass"""
    def __init__(self, class_num=100):

        super().__init__()
        self.stem_1 = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2_1 = Fire(96, 128, 16)
        self.fire3_1 = Fire(128, 128, 16)
        self.fire4_1 = Fire(128, 256, 32)
        self.fire5_1 = Fire(256, 256, 32)
        self.fire6_1 = Fire(256, 384, 48)
        self.fire7_1 = Fire(384, 384, 48)
        self.fire8_1 = Fire(384, 512, 64)
        self.fire9_1 = Fire(512, 512, 64)

        self.stem_2 = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2_2 = Fire(96, 128, 16)
        self.fire3_2 = Fire(128, 128, 16)
        self.fire4_2 = Fire(128, 256, 32)
        self.fire5_2 = Fire(256, 256, 32)
        self.fire6_2 = Fire(256, 384, 48)
        self.fire7_2 = Fire(384, 384, 48)
        self.fire8_2 = Fire(384, 512, 64)
        self.fire9_2 = Fire(512, 512, 64)
        self.muatt_1 = MutualAttention_register(128)

        self.conv10 = nn.Conv2d(512*2, class_num, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,warm_flag,x1,x2,cam1,cam2):
        f_g = x1
        f_e = x2

        x1 = self.stem_1(x1)
        f2_1 = self.fire2_1(x1)
        x2 = self.stem_2(x2)
        f2_2 = self.fire2_2(x2)
        #print(f2_2.shape)
        g_att_output_2,e_att_output_2,affin_grid2_1,affin_grid2_2,edge_2,color_2,f1_attn,f2_attn,g_att_output_2_before,e_att_output_2_before = self.muatt_1(warm_flag,f2_1, f2_2, cam1,cam2,f_g,f_e)



        f3_1 = self.fire3_1(g_att_output_2) + f2_1
        f4_1 = self.fire4_1(f3_1)
        f4_1 = self.maxpool(f4_1)
        f5_1 = self.fire5_1(f4_1) + f4_1
        f6_1 = self.fire6_1(f5_1)
        f7_1 = self.fire7_1(f6_1) + f6_1
        f8_1 = self.fire8_1(f7_1)
        f8_1 = self.maxpool(f8_1)


        f3_2 = self.fire3_2(e_att_output_2) + f2_2
        f4_2 = self.fire4_2(f3_2)
        f4_2 = self.maxpool(f4_2)
        f5_2 = self.fire5_2(f4_2) + f4_2
        f6_2 = self.fire6_2(f5_2)
        f7_2 = self.fire7_2(f6_2) + f6_2
        f8_2 = self.fire8_2(f7_2)
        f8_2 = self.maxpool(f8_2)

        f9_1 = self.fire9_1(f8_1)
        f9_2 = self.fire9_2(f8_2)

        output = torch.cat((f9_1, f9_2), 1)
        c10 = self.conv10(output)
        x = self.avg(c10)

        x = x.view(x.size(0), -1)
        x = self.softmax(x)

        return x,x1,x2,g_att_output_2,e_att_output_2,affin_grid2_1,affin_grid2_2,edge_2,color_2,f1_attn,f2_attn,g_att_output_2_before,e_att_output_2_before

def ours(class_num=2):
    return OursNet(class_num=class_num)
