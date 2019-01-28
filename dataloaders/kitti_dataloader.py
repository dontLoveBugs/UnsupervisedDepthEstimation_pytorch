# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/28 10:31
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import os
from PIL import Image

from torch.utils.data import Dataset


def readPathFiles(path):
    left_right_paths = []

    with open(path, 'rb') as f:
        lines = f.readlines()

        for line in lines:
            left_path = line.split()[0]
            right_path = line.split()[1]

            left_right_paths.append((left_path, right_path))

    return left_right_paths


class KittiLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None):
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

    def __len__(self):
        return len(self.left_right_paths)

    def __getitem__(self, idx):
        left_path, right_path = 
