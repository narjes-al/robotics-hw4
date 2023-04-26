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

import cv2

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

        # checkout train.RGBDataset for the content of data
        rgb_img = data['rgb'].detach().numpy()
        angle = float(data['angle'].detach().numpy())
        cent_pt = data['center_point'].detach().numpy()

        # Bin the angle
        #angle_bins = np.arange(-157.5, 180, 22.5)
        #bin_idx = np.argmin(np.abs(angle - angle_bins))

        bin_size = 22.5
        binned_angle = round(angle / bin_size) * bin_size

        # Rotate the RGB image
        rotation = iaa.Rotate(binned_angle)
        aug_img = rotation(image=rgb_img)

        # Compute the target
        H, W, _ = rgb_img.shape
        
        kps = KeypointsOnImage([Keypoint(x=cent_pt[0], y=cent_pt[1])], shape=(H, W))
        aug_kps = rotation.augment_keypoints(kps)
        aug_cent_pt = np.array([aug_kps.keypoints[0].x, aug_kps.keypoints[0].y])
        scoremap = get_gaussian_scoremap((H, W), aug_cent_pt)

        # Convert to torch tensors
        input_tensor = torch.from_numpy(aug_img).to(dtype=torch.float32).permute(2, 0, 1) / 255.0
        target_tensor = torch.from_numpy(scoremap).to(dtype=torch.float32).unsqueeze(0)

        """
        # Binning the angle to 22.5-degree intervals
        bin_size = 22.5
        binned_angle = int(round(angle / bin_size)) * bin_size
        
        # Rotate the RGB image using the binned angle
        rotation = iaa.Rotate(angle=binned_angle)
        aug_data = rotation(image=rgb_img.numpy(), keypoints=center_point)

        # Extract augmented data
        aug_rgb_img = torch.from_numpy(aug_data['image']).permute(2, 0, 1).float() / 255.0
        aug_center_point = np.array([kp.x, kp.y], dtype=np.float32)

        seq = iaa.Sequential([
            iaa.Rotate((-binned_angle, -binned_angle))
        ])
        aug = seq(image=rgb.numpy())
        aug_tensor = torch.from_numpy(aug)
        
        # Convert center point to numpy array
        center_point = center_point.numpy()
        
        
        # Get the target scoremap using the rotated center point
        target = get_gaussian_scoremap(aug.shape[:2], center_point)
        target_tensor = torch.from_numpy(target)
        target_tensor = target_tensor.unsqueeze(0)
        
        # Scale the input tensor from [0, 255] to [0, 1]
        input_tensor = aug_tensor.permute(2, 0, 1).float() / 255.0
        
        # Return the input and target tensors as a dictionary
        data = {
            'input': input_tensor,
            'target': target_tensor
        }
        return data
        """

        """
        rgb_img = data['rgb'].detach().numpy()
        angle = float(data['angle'].detach().numpy())
        cent_pt = data['center_point'].detach().numpy()

        H, W, C = rgb_img.shape

        rot = iaa.Affine(rotate=angle)
        kps = KeypointsOnImage([Keypoint(x=cent_pt[0], y=cent_pt[1])], shape=rgb_img.shape)
        aug_img, aug_kps = rot(image=rgb_img, keypoints=kps)

        aug_x_center = aug_kps.keypoints[0].x
        aug_y_center = aug_kps.keypoints[0].y

        aug_cent_pt = np.array([aug_x_center, aug_y_center])

        input_tensor = torch.from_numpy(aug_img).to(torch.float32)
        input_tensor = input_tensor.permute(2,0,1) /255.0

        target_tensor = torch.from_numpy(get_gaussian_scoremap((H,W), aug_cent_pt)).to(torch.float32)
        target_tensor = target_tensor.unsqueeze(0)

        data = dict(input=input_tensor, target=target_tensor)

        """ 

        data = {'input': input_tensor, 'target': target_tensor}

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
        
        
        # Initialize variables
        NUM_ROTATIONS = 8   
        best_point = (0, 0)
        best_angle = 0
        best_val = -float('inf')
        imgs = []
        coord = (-1, -1)
        angle = -1

        affordance_map = None

        # Evaluate the model on each rotated image
        self.eval()

        with torch.no_grad():

            for i in range(NUM_ROTATIONS):

                angle = 22.5*i
                
                # Rotate the RGB image
                rotation = iaa.Rotate(angle)
                aug_img = rotation(image=rgb_obs)

                input_tensor = torch.from_numpy(aug_img).to(torch.float32)
                input_tensor /= 255.0
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                
                affordance_map = self.predict(input_tensor)
                affordance_map = torch.clip(affordance_map,0,1)

                # Find the best grasp point and angle
                idx = np.argmax(affordance_map)
                _,_,y,x = np.unravel_index(idx, affordance_map.shape)
                #affordance_map = affordance_map.cpu().numpy()
                curr_val = float(affordance_map[0, 0, y, x])

                if curr_val > best_val:
                    best_point = (x, y)
                    best_angle = angle
                    best_val = curr_val

                # Visualize the prediction
                pred_img = torch.squeeze(affordance_map, 0)
                pred_img = pred_img.detach().numpy()
                input_img = input_tensor.squeeze(0).detach().numpy()
                vis_img = self.visualize(input_img, pred_img)
                draw_grasp(vis_img, best_point, best_angle)

                imgs.append(vis_img)

        # Rotate the grasp point and angle back to the original image orientation
        
        angle = -best_angle
        
        rotation = iaa.Rotate(angle)

        H, W, _ = rgb_obs.shape
        
        kps = KeypointsOnImage([Keypoint(x=best_point[0], y=best_point[1])], shape=(H, W))
        aug_kps = rotation.augment_keypoints(kps)

        aug_x = int(aug_kps.keypoints[0].x)
        aug_y = int(aug_kps.keypoints[0].y)
        coord = (aug_x, aug_y)

        # ===============================================================================

        # TODO: (problem 3, skip when finishing problem 2) avoid selecting the same failed actions
        # ===============================================================================
        """
        score_map = prediction[0,0].cpu().numpy()
        affordance_map = dict()

        for max_coord in list(self.past_actions):  

            bin = max_coord[0] 
            # supress past actions and select next-best action
            x, y = max_coord[1]
            supression_map = get_gaussian_scoremap(shape=score_map.shape, keypoint=(y,x), sigma=4)
            affordance_map[bin] -= supression_map


        idx = np.argmax(affordance_map)
        y, x = np.unravel_index(idx, affordance_map.shape)
        curr_val = score_map[y,x]
        max_coord = (bin, (x, y))
        self.past_actions.append(max_coord)

        """

        # ===============================================================================
        
        # TODO: (problem 2) complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================

        # Visualize the prediction as a single image

        for img in imgs:
            cv2.line(img, (0, img.shape[0] - 1), (img.shape[1], img.shape[0] - 1), (127, 127, 127), 1)
        
        left_imgs = imgs[:NUM_ROTATIONS // 2]
        right_imgs = imgs[NUM_ROTATIONS // 2:]
        
        left_img = np.concatenate(left_imgs, axis=0).astype(np.uint8)
        right_img = np.concatenate(right_imgs, axis=0).astype(np.uint8)
        
        vis_img = np.concatenate([left_img, right_img], axis=1).astype(np.uint8)

        # ===============================================================================
        return coord, angle, vis_img

