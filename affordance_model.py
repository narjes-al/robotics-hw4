from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from common import draw_grasp
from collections import deque

def get_gaussian_scoremap(
        shape: Tuple[int, int], 
        keypoint: np.ndarray, 
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypont: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # TODO: (problem 2) complete this method and return the correct input and target as data dict
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================
        
        rgb_img = data['rgb'].detach().numpy()
        angle = float(data['angle'].detach().numpy())
        cent_pt = data['center_point'].detach().numpy()

        H = rgb_img.shape[0]
        W = rgb_img.shape[1]

        x_cent = cent_pt[0]
        y_cent = cent_pt[1]

        shape = rgb_img.shape

        rot = iaa.Affine(rotate=angle)
        kps = KeypointsOnImage([Keypoint(x=x_cent, y=y_cent)], shape=shape)
        aug_rgb, aug_kps = rot(image=rgb_img, keypoints=kps)

        aug_x_center = aug_kps.keypoints[0].x
        aug_y_center = aug_kps.keypoints[0].y

        aug_cent_pt = np.array([aug_x_center, aug_y_center])

        input_tensor = torch.from_numpy(aug_rgb).to(dtype=torch.float32)
        input_tensor = input_tensor/255.0
        input_tensor = input_tensor.permute(2,0,1)

        target_tensor = torch.tensor(get_gaussian_scoremap((H,W), aug_cent_pt), dtype=torch.float32)
        target_tensor = target_tensor.unsqueeze(2).permute(2,0,1)

        data = dict(input=input_tensor, target=target_tensor)

        return data 


class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, n_past_actions: int=0, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.past_actions = deque(maxlen=n_past_actions)

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img


    def predict_grasp(
        self, 
        rgb_obs: np.ndarray,  
    ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given an RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        # TODO: (problem 2) complete this method (prediction)
        # Hint: why do we provide the model's device here?
        # ===============================================================================
        
        
        best_point = (0,0)
        best_angle = 0
        best_img_idx = -1
        best_val = -float('inf')

        imgs = list()
    
        self.eval()

        with torch.no_grad():

            for i in range(8):

                angle = 22.5*i

                seq = iaa.Aggine(rotate=angle)

                aug_img = seq(image=rgb_obs)

                input_tensor = torch.from_numpy(aug_img).to(torch.float32)
                input_tensor = input_tensor/255.0
                input_tensor = input_tensor.permute(2,0,1).unsqueeze(0).to(device)

                prediction = self.predict(input_tensor)

                for y in range(prediction.shape[2]):
                    for x in range(prediction.shape[3]):

                        curr_val = float(prediction[0,0,y,x])
                        if curr_val > best_val:
                            best_point = (x,y)
                            best_angle = angle
                            bes_img_idx = i
                            best_val = curr_val

                pred_img = torch.squeeze(prediction, 0)
                pred_img = pred_img.detach().numpy()

                input_img = input_tensor.squeeze(0).detach().numpy()
                
                vis_img = self.visualize(input_img, pred_img)
                imgs.append(vis_img)

        coord, angle = None, None 

        angle = -1*best_angle

        kps = KeypointsOnImage([Keypoint(x=best_point[0], y=best_point[1])], shape=rgb_obs.shape)
        rot = iaa.Affine(rotate = angle)
        aug_kps = rot(Keypoints=kps)

        aug_x = int(aug_kps.keypoints[0].x)
        aug_y = int(aug_kps.keypoints[0].y)

        coord = (aug_x, aug_y)

        # ===============================================================================

        # TODO: (problem 3, skip when finishing problem 2) avoid selecting the same failed actions
        # ===============================================================================

        for max_coord in list(self.past_actions):  
            bin = max_coord[0] 
            # supress past actions and select next-best action
            """
            supression_map = get_gaussian_scoremap(shape=, keypoint=, sigma=4)
            affordance_map[bin] -=


        max_coord = 
        self.past_actions.append(max_coord)
    
        """

        # ===============================================================================
        
        # TODO: (problem 2) complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================
        vis_img = None

        draw_grasp(imgs[best_img_idx], best_point, 0)

        img_right = (np.concatenate(imgs[4:], axis=0)).astype(np.uint8)
        img_left = (np.concatenate(imgs[:4], axis=0)).astype(np.uint8)

        img = [img_left, img_right]
        vis_img = (np.concatenate(img, axis=1)).astype(np.uint8)

        # ===============================================================================
        return coord, angle, vis_img

