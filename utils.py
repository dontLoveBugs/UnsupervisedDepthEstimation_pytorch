# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/28 10:30
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""

import torch
import collections
import os
from torch.utils.data import DataLoader, ConcatDataset

from network.models_resnet import Resnet18_md, Resnet50_md, ResnetModel
from dataloaders.kitti_dataloader import KittiFolder
from dataloaders.transforms import image_transforms


def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")


def get_model(model, input_channels=3, pretrained=False):
    if model == 'resnet50_md':
        out_model = Resnet50_md(input_channels)
    elif model == 'resnet18_md':
        out_model = Resnet18_md(input_channels)
    else:
        out_model = ResnetModel(input_channels, encoder=model, pretrained=pretrained)
    return out_model


def prepare_dataloader(root_dir, mode, augment_parameters,
                       do_augmentation, batch_size, size, num_workers):
    data_transform = image_transforms(
        mode=mode,
        augment_parameters=augment_parameters,
        do_augmentation=do_augmentation,
        size=size)

    if mode == 'train':
        data_set = KittiFolder(root_dir=root_dir, mode=mode, transform=data_transform)
        loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers, pin_memory=True)
        n_img = len(data_set)

    else:
        data_set = KittiFolder(root_dir=root_dir, mode=mode, transform=data_transform)
        loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False,
                                             num_workers=num_workers, pin_memory=True)
        n_img = len(data_set)
    return n_img, loader
