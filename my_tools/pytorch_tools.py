# --------------------------------------------------------------------------------------------------------
# 2019/12/29
# src - pytorch_tools.py
# md
# --------------------------------------------------------------------------------------------------------
import random
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torchvision as thv
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from my_tools.python_tools import print_file


def set_random_seed(seed):
    # Reproducability problems with pooling (AdaptiveMaxPool2d, AdaptiveAvgPool2d, ...)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # for multiGPUs.
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


# MODELS
def set_requires_grad(model, names, requires_grad, to_file=''):
    """
    Set the requires_grad on parameters from model based on name

    Args:
        model: model containing parameters that need to set requires_grad
        names: ('all', str, int of list[str]) Parameter name or part of a parameter name.
            if 'all: then all requires_grad will be set for parameters
            If part, then all parameters with that part in their name will have requires_grad set.
        requires_grad: (bool) Sets the requires_grad of the parameter
        to_file: (string) if not '' then print also print to file to_file
    """
    if names == 'all': names = ['.']
    if isinstance(names, str): names = [names]
    if isinstance(names, int): names = [str(names)]
    # print_file(f'Setting requires_grad:', to_file, False)
    for n in names:
        for name, param in [(name, param) for name, param in model.named_parameters() if n in name]:
            param.requires_grad = requires_grad
            # print_file(f'\t{name:40}: {requires_grad}', to_file, True)


def set_lr(model, names, lr):
    """
    Sets the learning rate for parameters defined by name to be used in optimizers
    Args:
        model: model containing parameters that need to set requires_grad
        names: (str, int of list[str]) Parameter name or part of a parameter name.
            If part, then all parameters with that part in their name will have lr set.
        lr: learning rate for the parameter(s)

    Returns:
        List of dictionaries of the form [{'params': param, 'lr': lr}, ...] to be used in optimizer
    """
    if isinstance(names, str): names = [names]
    if isinstance(names, int): names = [str(names)]
    params = []
    for name in names:
        params += [{'name': n, 'params': p, 'lr': lr} for n, p in model.named_parameters() if name in n]  # name added for debugging
    return params


# DATA
def random_split_train_valid(dataset, valid_frac):
    """
    Randomly splits a dataset into a train and valid dataset subset.
    Important: Random split doesn't guarantee that the train and valid dataset subsets have the same the class distributions.

    Args:
        dataset: The dataset to split into a train and valid dataset subset.
        valid_frac: Between 0 and 1. The fraction of the dataset that will be split into a valid dataset subset.

    Return:
        tuple: (train_ds, valid_ds) where train_ds is the train dataset subset and valid_ds is the valid dataset subset.
    """
    assert 0 < valid_frac < 1, "valid_frac must be bigger than 0 and smaller than 1"
    data = dataset.data
    targets = dataset.targets
    train, valid = deepcopy(dataset), deepcopy(dataset)
    train.data, valid.data, train.targets, valid.targets = train_test_split(data, targets, test_size=valid_frac, stratify=None)
    return train, valid


def stratified_split_train_valid(dataset, valid_frac):
    """
    Split a dataset into a train and valid dataset subset using stratification. Stratification means that the split will try
    to approximate the class distributions from the dataset to the train and valid datasets.

    Args:
        dataset: The dataset to split into a train and valid dataset subset.
        valid_frac: Between 0 and 1. The fraction of the dataset that will be split into a valid dataset subset.

    Return:
        tuple: (train_ds, valid_ds) where train_ds is the train dataset subset and valid_ds is the valid dataset subset.
    """
    assert 0 < valid_frac < 1, "valid_frac must be bigger than 0 and smaller than 1"
    data = dataset.data
    targets = dataset.targets
    train, valid = deepcopy(dataset), deepcopy(dataset)
    train.data, valid.data, train.targets, valid.targets = train_test_split(data, targets, test_size=valid_frac, stratify=targets)
    return train, valid


def get_class_distribution(dataset):
    """
    Calculates the class distribution, ie the number of samples per class.

    Args:
        dataset: a data_process.standard_datasets dataset that implemented the classes attibute

    Return:
        pandas.DataFrame with index=class index and columns=['class', 'n_samples'] sorted by index
    """
    classes = dataset.classes
    index = dataset.targets
    class_distribution = pd.DataFrame(columns=['class', 'n_samples', 'normalised'])
    class_distribution['n_samples'] = pd.Series(index).value_counts().sort_index()
    class_distribution['normalised'] = class_distribution['n_samples'] / max(class_distribution['n_samples'])
    class_distribution['class'] = classes
    return class_distribution


def get_mean_and_std2(dataset):  # Todo: remove because it loads full dataset in memory. Calculate per image
    #                                    doesn't work if images are rectangualar and moi squares
    print('This method loads the full dataset in memeory !!!! ')
    imgs = [dataset[i][0] for i in range(len(dataset.data))]
    imgs = th.stack(imgs)
    mean = imgs.mean()
    std = imgs.std()
    return mean, std


def get_mean_and_std(dataset):  # Todo: std calculation is wrong
    """ Compute the mean and std value of dataset. From: https://github.com/isaykatsman/pytorch-cifar/blob/master/utils.py """
    dataloader = th.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = th.zeros(3)
    std = th.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


# LOGGING

def summary(model, input_size, batch_size=-1, device="cuda", to_file=None):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += th.prod(th.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += th.prod(th.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if not isinstance(module, th.nn.Sequential) and not isinstance(module, th.nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in ["cuda", "cpu", ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and th.cuda.is_available():
        dtype = th.cuda.FloatTensor
    else:
        dtype = th.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [th.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks: h.remove()

    print_file("--------------------------------------------------------------------------", to_file, append=False)
    line_new = "{:>20}  {:>25} {:>15}  {:>8}".format("Layer (type)", "Output Shape", "Param #", 'Unfrozen')
    print_file(line_new, to_file)
    print_file("==========================================================================", to_file)

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(layer,
                                                  str(summary[layer]["output_shape"]),
                                                  "{0:,}".format(summary[layer]["nb_params"]),
                                                  )
        if "trainable" in summary[layer]:
            line_new += "   {}".format(summary[layer]["trainable"])
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]: trainable_params += summary[layer]["nb_params"]
        print_file(line_new, to_file)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print_file("==========================================================================", to_file)
    print_file("Total params: {0:,}".format(total_params), to_file)
    print_file("Trainable params: {0:,}".format(trainable_params), to_file)
    print_file("Non-trainable params: {0:,}".format(total_params - trainable_params), to_file)
    print_file("--------------------------------------------------------------------------", to_file)
    print_file("Input size (MB): %0.2f" % total_input_size, to_file)
    print_file("Forward/backward pass size (MB): %0.2f" % total_output_size, to_file)
    print_file("Params size (MB): %0.2f" % total_params_size, to_file)
    print_file("Estimated Total Size (MB): %0.2f" % total_size, to_file)
    print_file("--------------------------------------------------------------------------", to_file)


def create_tb_summary_writer(model, data_loader, log_dir, device='cuda'):
    """
    Creates tensorboard summary writer, adds the model's graph to tensorboard and returns the writer
    """
    writer = SummaryWriter(log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    x, y = x.to(device), y.to(device)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def model_to_sequential(model):
    """
    Receives a model, removes all sequential layers and returns one sequential with all layers
    Todo: This doesn't work if the model is not sequentiable, like Resnet. Does it work with VGG16???
    """

    def remove_sequential(model):
        for layer in model.children():
            if not list(layer.children()):  # if leaf node, add it to list
                all_layers.append(layer)
            else:  # if sequence, remove it recursively
                remove_sequential(layer)
        return all_layers

    all_layers = []
    remove_sequential(model)
    return nn.Sequential(*all_layers)


# Tools
# See https://github.com/fastai/fastai/blob/99c2c269b58349e8edc3025468bfc448b25e9364/old/fastai/core.py
def to_np(v):  # from fastai. # Todo:test
    """returns an np.array object given an input of np.array, list, tuple, torch variable or tensor."""
    if isinstance(v, float): return np.array(v)
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list, tuple)): return [to_np(o) for o in v]
    if isinstance(v, th.Tensor): v = v.data
    if th.cuda.is_available():
        if is_half_tensor(v): v = v.float()
    if isinstance(v, th.FloatTensor): v = v.float()
    return v.cpu().numpy()


def is_half_tensor(v):  # todo: test
    return isinstance(v, th.cuda.HalfTensor)


class DeNormalize(thv.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = th.as_tensor(mean)
        std_inv = th.as_tensor(1.) / (th.as_tensor(std) + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


if __name__ == '__main__':
    set_random_seed(1)
