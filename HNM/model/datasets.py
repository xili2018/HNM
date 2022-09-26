# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
from sklearn import preprocessing
from utils import open_file
from utils import open_file, normalise_image

DATASETS_CONFIG = {
       
        'IndianPines': {
            'img': 'Indian_pines_corrected.mat',
            'gt': 'Indian_pines_gt.mat'
            },
        'Houston':{
            'img':'Houston.mat',
            'gt':'Houston_gt.mat',
            },
        'WHU_Hi_HanChuan':{
            'img':'WHU_Hi_HanChuan.mat',
            'gt':'WHU_Hi_HanChuan_gt.mat',
            },
                 }

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG, normalization_method='None',):
    
    """ 
    Gets the dataset specified by name and return the related components.
 
    """
    
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    folder = target_folder + datasets[dataset_name].get('folder', dataset_name + '/')
    
    if dataset_name == 'IndianPines':
        # Load the image
        img = open_file(folder + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']
        gt = open_file(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]

        ignored_labels = [0]

    elif dataset_name == 'Houston':
        # Load the image
        img = open_file(folder + 'Houston.mat')['Houston']
        gt = open_file(folder + 'Houston_gt.mat')['Houston_gt']
        label_values = ["Undefined", "Healthy grass", "Stressed grass",
                        "Synthetic grass", "Trees",
                        "Soil", "Water",
                        "Residential", "Commercial", "Road",
                        "Highway", "Railway", "Parking Lot 1", "Parking Lot 2",
                        "Tennis Court","Running Track"]

        ignored_labels = [0]


    elif dataset_name == 'WHU_Hi_HanChuan':
        # Load the image
        img = open_file(folder + 'WHU_Hi_HanChuan.mat')['WHU_Hi_HanChuan']
        gt = open_file(folder + 'WHU_Hi_HanChuan_gt.mat')['WHU_Hi_HanChuan_gt']
        label_values = ["Undefined", "Strawberry ", "Cowpea ",
                        "Soybean ", "Sorghum ",
                        "Water spinach ", "Watermelon ","Greens ",
                        "Trees ", "Grass ", "Red roof ",
                        "Gray roof ", "Plastic ", "Bare soil ", "Road "," Bright object ","Water "
                        ]
        ignored_labels = [0]


        
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
       print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)
    ignored_labels = list(set(ignored_labels))
    img = np.asarray(img, dtype='float32') 
    data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:])) 
    data  = preprocessing.minmax_scale(data) 
    img = data.reshape(img.shape)

    if normalization_method != 'None':
        img = normalise_image(img, method=normalization_method)
    else:
        img = img



    return img, gt, label_values, ignored_labels




class HSI_data(torch.utils.data.Dataset):

    def __init__(self, data, gt, patch_size = 5, **hyperparams):

        super(HSI_data, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams['dataset']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.center_pixel = hyperparams['center_pixel']
        self.patch_size = patch_size
        mask = np.ones_like(gt) 
        for l in self.ignored_labels:
            mask[gt == l] = 0

        x_pos, y_pos = np.nonzero(mask) 
        p = self.patch_size // 2 
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        np.random.shuffle(self.indices) 

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32') 
        label = np.asarray(np.copy(label), dtype='int64')
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        if self.patch_size > 1:
            data = data.unsqueeze(0) 
            
        return data, label
