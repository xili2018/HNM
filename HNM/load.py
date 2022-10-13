# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import torch
import torch.utils.data as data
import numpy as np
from utils import metrics, sample_gt, get_device
from model.datasets import get_dataset, HSI_data, open_file, DATASETS_CONFIG
from model.module import get_model, test
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
group_dataset.add_argument('--test_set', type=str, default=None,
                    help="Path to the test set (optional, by default "
                    "the test_set is the entire ground truth minus the training)")


# Test options
group_train = parser.add_argument_group('Test')
group_train.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")


group_dataset.add_argument(
    "--normalization",
    type=str,
    default='None',
    help="Normalization method to use for image preprocessing. Available:\n"
    "None : Applying none preprocessing."
    "SNB (StandardNormalize for each Band): Normalizing first- and second-order moments along each band."
)


args = parser.parse_args()
CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Dataset name
DATASET = args.dataset
# Target folder to store/load the datasets
FOLDER = args.folder
# weights to restore
CHECKPOINT = args.restore
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride
# Normalization method
NORM_METHOD = args.normalization

hyperparams = vars(args) 
img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(DATASET, FOLDER,normalization_method=NORM_METHOD)
N_CLASSES = len(LABEL_VALUES)
N_BANDS = img.shape[-1]
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

if TEST_GT is not None:
    test_gt = open_file(TEST_GT)
elif TEST_GT is not None:
    test_gt = open_file(TEST_GT)
else:

    _, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE)

# Neural network
model, optimizer, loss, hyperparams = get_model(**hyperparams)

test_dataset = HSI_data(img, test_gt, **hyperparams)
test_loader = data.DataLoader(test_dataset,
                                batch_size=hyperparams['batch_size'])

model.load_state_dict(torch.load(CHECKPOINT))
probabilities = test(model, img, hyperparams)
prediction = np.argmax(probabilities, axis=-1)
run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)


