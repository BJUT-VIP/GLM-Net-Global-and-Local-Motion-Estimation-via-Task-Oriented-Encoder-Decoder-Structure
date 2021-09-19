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

mean = [4.335279941558838, 1.3750430345535278]
std = [102.08905792236328, 9.452127456665039]

class MotionField_Data(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img = os.listdir(self.img_dir)
        self.label = os.listdir(self.mask_dir)


    def __getitem__(self, index):
        img_index = self.img[index]
        label_index = self.img[index]
        img_path = os.path.join(self.img_dir, img_index)
        mask_path = os.path.join(self.mask_dir, label_index)
        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        img = self.preprocess(img)
        mask = cv2.resize(mask, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
        # hash = {}
        # for i in range(64):
        #     for j in range(64):
        #         val = mask[i][j]
        #         hash[str(val)] = hash.get(str(val), 0) + 1

        # mask = self.preprocess_label(mask)

        sample = {'image':img, 'mask':mask}

        return sample

    @classmethod
    def preprocess(cls, img):
        # 将数据范围缩小到[-15,15]
        img = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)

        # img_x = img[:,:,0]
        # img_y = img[:,:,1]
        #
        # img_x -= mean[0]
        # img_y -= mean[1]
        #
        # img_x /= std[0]
        # img_y /= std[1]
        #
        # img_x = img_x[:, :, np.newaxis]
        # img_y = img_y[:, :, np.newaxis]
        #
        # img = np.concatenate((img_x, img_y), 2)

        # 将阈值限制在[-15,15]
        # img = np.clip(img, -20, 20)
        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        # 数据归一化到[-1,1]
        img_trans /= 20
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
        return len(self.img)

if __name__ == '__main__':
    img_dir = '/disk3/yyc/AutoEncoder/data/flow/'
    mask_dir = '/disk3/yyc/AutoEncoder/data/annoations/mask/'
    train_data = MotionField_Data(img_dir, mask_dir)
    dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    for batch, data in enumerate(dataloader):
        print(batch)
        print(data['image'])
        print(data['mask'])

