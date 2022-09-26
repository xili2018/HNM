# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch
import torch.optim as optim
import os
import numpy as np
import joblib
from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window
from model.HNMnet import HNM

def get_model(**kwargs):
    """
    obtain the model

    """
    device = kwargs.setdefault('device', torch.device('cpu'))
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(device)
    weights = kwargs.setdefault('weights', weights)

    # model
    patch_size = 5
    center_pixel = True
    model = HNM(n_bands, n_classes, n_planes=16, patch_size=patch_size)
    lr = kwargs.setdefault('learning_rate', 0.01)
    optimizer = optim.SGD(model.parameters(),
                            lr=lr, momentum=0.9, weight_decay=0.0005)
    epoch = kwargs.setdefault('epoch', 200)
    criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])
    model = model.to(device)
    epoch = kwargs.setdefault('epoch', 100)
    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=epoch//4, verbose=True))
    kwargs.setdefault('batch_size', 100)
    kwargs['center_pixel'] = center_pixel
    
    return model, optimizer, criterion, kwargs



def train(net, optimizer, criterion, data_loader, epoch, scheduler=None,
          device=torch.device('cpu'), display=None,
          val_loader=None):
    """
    Training the model

    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")
    net.to(device)
    save_epoch = 20
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.
        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output+1e-10, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])
            iter_ += 1
            del(data, target, loss, output)
        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss
            
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        if e % save_epoch == 0:
            save_model(net, data_loader.dataset.name,epoch=e, metric=abs(metric))
        

def save_model(model, dataset_name, **kwargs):
     model_dir = './checkpoints/' +  "/" + dataset_name + "/"
     if not os.path.isdir(model_dir):
         os.makedirs(model_dir, exist_ok=True)
     if isinstance(model, torch.nn.Module):
         filename = str('weight') + "{epoch}_{metric:.4f}_".format(**kwargs) 
         tqdm.write("Saving neural network weights in {}".format(filename))
         torch.save(model.state_dict(), model_dir + filename + '.pth')
     else:
         filename = str('weight')
         tqdm.write("Saving model params in {}".format(filename))
         joblib.dump(model, model_dir + filename + '.pkl')


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = 5
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']
    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))
    iterations = count_sliding_window(img, **kwargs) // batch_size
    
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                     total=(iterations),
                     desc="Inference on the image"
                     ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]

            data = data.to(device)

            output = net(data)
            if isinstance(output, tuple): 
                output = output[0]
            output = output.to('cpu')

            if patch_size == 1 or center_pixel:
                output = output.numpy()
              
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))

            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x:x + w, y:y + h] += out

    return probs

def val(net, data_loader, device='cpu', supervision='full'):
# TODO : fix me using metrics()
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            
            data, target = data.to(device), target.to(device)
            output = net(data)
            _, output = torch.max(output, dim=1)
            for pred, out in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
                    
    return accuracy / total
