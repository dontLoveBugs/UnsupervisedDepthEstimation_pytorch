# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/28 10:31
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import os
from PIL import Image

from torch.utils.data import Dataset


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def readPathFiles(path):
    left_right_paths = []

    with open(path, 'rb') as f:
        lines = f.readlines()

        for line in lines:
            left_path = line.split()[0]
            right_path = line.split()[1]

            left_right_paths.append((left_path, right_path))

    return left_right_paths


class KittiFolder(Dataset):
    def __init__(self, root_dir, mode, transform=None, loader=pil_loader):
        self.root_dir = root_dir

        if mode == 'train':
            self.left_right_paths = readPathFiles('./eigen_train_files.txt')
        elif mode == 'val':
            self.left_right_paths = readPathFiles('./eigen_val_files.txt')
        elif mode == 'test':
            self.left_right_paths = readPathFiles('./eigen_test_files.txt')
        else:
            print('no dataloader mode named as ', mode)
            exit(-1)

        self.transform = transform
        self.mode = mode
        self.loader = loader

    def __len__(self):
        return len(self.left_right_paths)

    def __getitem__(self, idx):
        left_path, right_path = self.left_right_paths[idx]

        left_path = os.path.join(self.root_dir, left_path)
        right_path = os.path.join(self.root_dir, right_path)

        left_img = self.loader(left_path)
        right_img = self.loader(right_path)

        if self.transform is not None:
            left_img = self.transform(left_img)
            right_img = self.transform(right_path)

        return left_img, right_img
