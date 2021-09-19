from torch.utils.data import Dataset, DataLoader
from torchvision import utils, transforms
import os
import numpy as np
import torch
import cv2

normalize = transforms.Normalize(mean=[-0.057515066117048264, 0.08313903957605362],
                                     std=[0.07887575775384903, 0.04074226692318916])

Normalization = transforms.Compose([
    normalize
])

class MotionField_Data(Dataset):
    def __init__(self, Mix_dir, Global_dir, Mask_dir):
        self.Mix_dir = Mix_dir
        self.Global_dir = Global_dir
        self.Mask_dir = Mask_dir
        self.Mix = os.listdir(self.Mix_dir)
        self.Global = os.listdir(self.Global_dir)
        self.Mask = os.listdir(self.Mask_dir)


    def __getitem__(self, index):
        Mix_index = self.Mix[index]
        Global_index = self.Global[index]
        Mask_index = self.Mask[index]

        Mix_path = os.path.join(self.Mix_dir, Mix_index)
        Global_path = os.path.join(self.Global_dir, Global_index)
        Mask_path = os.path.join(self.Mask_dir, Mask_index)

        Mix = np.load(Mix_path).astype(np.float32)
        Global = np.load(Global_path).astype(np.float32)
        Mask = np.load(Mask_path).astype(np.float32)

        Mix = self.preprocess(Mix)
        Global = self.preprocess_label(Global)
        Mask = cv2.resize(Mask, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)

        sample = {'Mix':Mix, 'Global':Global, 'Mask':Mask}

        return sample

    @classmethod
    def preprocess(cls, img):
        # 将数据范围缩小到[-15,15]
        img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
        # 将阈值限制在[-15,15]
        # img = np.clip(img, -20, 20)
        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        # 数据归一化到[-1,1]
        img_trans = img_trans / 20

        img_trans = torch.from_numpy(img_trans)

        # img_trans = Normalization(img_trans)

        return img_trans

    @classmethod
    def preprocess_label(cls, img):
        img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
        # 将阈值限制在[-15,15]
        # img = np.clip(img, -15, 15)
        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        # 数据归一化到[-1,1]
        img_trans = img_trans / 20

        img_trans = torch.from_numpy(img_trans)

        # img_trans = Normalization(img_trans)

        return img_trans

    def __len__(self):
        return len(self.Mix)

if __name__ == '__main__':
    Mix_dir = '/disk1/yyc/U-net/data_new/least_square/train/mix_field/'
    Global_dir = '/disk1/yyc/U-net/data_new/least_square/train/global/'
    Mask_dir = '/disk1/yyc/U-net/data_new/least_square/train/ScoreMask/'

    train_data = MotionField_Data(Mix_dir, Global_dir, Mask_dir)
    dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    for batch, data in enumerate(dataloader):
        print(batch)
        print(data['Mix'])
        print(data['Global'])
        print(data['Mask'])

