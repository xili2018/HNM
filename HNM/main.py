# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import torch
import torch.utils.data as data
import numpy as np
from utils import metrics, sample_gt, get_device
from model.datasets import get_dataset, HSI_data, open_file, DATASETS_CONFIG
from model.module import get_model, train
import argparse


dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None, choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--folder', type=str, help="Folder where to store the "
                    "datasets (defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--restore', type=str, default="weights/IndianPines/IP_weight.pth",
                    help="Weights to use for initialization, e.g. a checkpoint")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=float, default=0.1,
                    help="Percentage of samples to use for training (default: 10%)")
group_dataset.add_argument('--train_set', type=str, default=None,
                    help="Path to the train ground truth (optional, this "
                    "supersedes the --sampling_mode option)")

# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int, help="Training epochs (optional, if"
                    " absent will be set by the model)")
group_train.add_argument('--lr', type=float,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--batch_size', type=int,
                    help="Batch size (optional, if absent will be set by the model")


group_dataset.add_argument(
    "--normalization",
    type=str,
    default='None',
    help="Normalization method to use for image preprocessing. Available:\n"
    "None : Applying none preprocessing."
    "SNB (StandardNormalize for each Band): Normalizing first- and second-order moments along each band."
)





# python main.py  --dataset IndianPines  --training_sample 0.1 --cuda 1 --epoch 200

args = parser.parse_args()
CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Dataset name
DATASET = args.dataset
# Target folder to store/load the datasets
FOLDER = args.folder
# Number of epochs
EPOCH = args.epoch
# weights to restore
CHECKPOINT = args.restore
# Learning rate
LEARNING_RATE = args.lr
# Training ground truth file
TRAIN_GT = args.train_set
# Normalization method
NORM_METHOD = args.normalization

hyperparams = vars(args) 
img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(DATASET, FOLDER,normalization_method=NORM_METHOD)
N_CLASSES = len(LABEL_VALUES)
N_BANDS = img.shape[-1]
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

if TRAIN_GT is not None :
    train_gt = open_file(TRAIN_GT)
elif TRAIN_GT is not None:
    train_gt = open_file(TRAIN_GT)

else:

    train_gt, _ = sample_gt(gt, SAMPLE_PERCENTAGE)

# Neural network
model, optimizer, loss, hyperparams = get_model(**hyperparams)

# Split train set in train/val
train_gt, val_gt = sample_gt(train_gt, 0.95)  

# Generate the dataset
train_dataset = HSI_data(img, train_gt, **hyperparams) 
train_loader = data.DataLoader(train_dataset,                         
                                batch_size=hyperparams['batch_size'],
                                shuffle=True,drop_last=True)
val_dataset = HSI_data(img, val_gt, **hyperparams)
val_loader = data.DataLoader(val_dataset,
                                batch_size=hyperparams['batch_size'])


if CHECKPOINT is not None:
    model.load_state_dict(torch.load(CHECKPOINT))

try:
    train(model, optimizer, loss, train_loader, hyperparams['epoch'],
        scheduler=hyperparams['scheduler'], device=hyperparams['device'],
        val_loader=val_loader,
        )
except KeyboardInterrupt:
    pass



