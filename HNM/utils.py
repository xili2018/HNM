# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import itertools
import spectral
from scipy import io, misc
import imageio
import os
import re
import torch
from sklearn.preprocessing._data import _handle_zeros_in_scale as handle_zeros


def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))





def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples

    """
    samples = []
    labels = []
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)


def get_random_pos(img, window_shape):
    """ 
    Return the corners of a random window in the input image

    """
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    """
    Sliding window generator over an input image.
    
    """
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def normalise_image(image, method='None'):
    """
    Normalise an image.

    """
    allowed_methods = ['NONE', 'SNB',]
    method = method.upper()
    if method not in allowed_methods:
        raise ValueError('unknown normalization method "%s"' % (method,))

    
    shp = image.shape
    if len(shp) >= 3:
        image = image.reshape(-1, shp[-1])
    elif method == 'SNB':
        scale = handle_zeros(np.std(image, axis=0, keepdims=True))
        image = (image - np.mean(image, axis=0, keepdims=True)) / scale

    
    return np.reshape(image, shp) 



def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True

    ignored_mask = ~ignored_mask 
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]
    n_classes = np.max(target) + 1 if n_classes is None else n_classes
    cm = confusion_matrix(
    target,
    prediction,
    labels=range(n_classes))
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    # print("accuracy -sum:",accuracy)
    accuracy *= 100 / float(total)

    print("accuracy:", accuracy)
    
def sample_gt(gt, train_size,seed = 1):
    """
    Extract a fixed percentage of samples from an array of labels.
    
    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) 
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
    
    train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y, random_state =seed) 
    train_indices = [list(t) for t in zip(*train_indices)]
    test_indices = [list(t) for t in zip(*test_indices)]
    train_gt[train_indices] = gt[train_indices]
    test_gt[test_indices] = gt[test_indices]

    return train_gt, test_gt

