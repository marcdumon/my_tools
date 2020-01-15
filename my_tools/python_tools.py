# --------------------------------------------------------------------------------------------------------
# 2019/12/27
# src - python_tools.py
# md
# --------------------------------------------------------------------------------------------------------
import random
from datetime import datetime
from math import sqrt, ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th


# COMMON


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def now_str(pattern='yyyymmdd_hhmmss'):
    """
    The currect datetime according to pattern
    Args:
        pattern: string indicating the format of the returned datetime.
    Return:
        datetime string in the format
    """
    now = datetime.now()
    if pattern == 'yyyymmdd_hhmmss': return f'{now:%Y%m%d_%H%M%S}'
    if pattern == 'yymmdd_hhmmss': return f'{now:%y%m%d_%H%M%S}'
    if pattern == 'yyyy-mm-dd hh:mm:ss': return f'{now:%Y-%m-%d %H:%M:%S}'
    if pattern == 'mm-dd hh:mm:ss': return f'{now:%m-%d %H:%M:%S}'
    return 'Pattern not implemented!'


def print_file(txt, file='', append=True):
    """
    Print to console. If file given then also print to file

    Args:
        txt: text to print or to save in file.
        file: if given, txt will be saved in file
        append: if false then a new file will be created, if tre then txt will be appended to an existing file.
    """
    print(txt)
    mode = 'w'
    if append: mode = 'a'
    if file: print(txt, file=open(file, mode))


# CHARTING
def show_mpl_grid(images, titles=None, figsize=(10, 7), gridshape=(0, 0), cm='gray'):
    """
    Shows images in a grid. Uses matplotlib pyplot

    Args:
        images: list of images to show in a grid
        titles: list of titles for each image
        figsize: (horizontal, vertical) a tuple passing to plt figuresize
        gridshape: (rows, columns) the shape of the grid. The shape will be automatically calculated when it's not provided
        cm: matplotlib cmap

    Returns:
        Shows a matplotlib grig of images
    """

    # Matplotlib needs grayscal images of shape (M,N), not (M,N,1)
    # if images.shape[-1] == 1: images = images[:, :, :, 0]
    # Todo:
    '''
    Traceback (most recent call last):
        File "/media/md/Development/My_Projects/experiments_mnist/src/filter_visualisation.py", line 136, in <module>
        show_mpl_grid(images)
        File "/home/md/Miniconda3/envs/ai/lib/python3.7/site-packages/my_tools/python_tools.py", line 73, in show_mpl_grid
        if images.shape[-1] == 1: images = images[:, :, :, 0]
    AttributeError: 'list' object has no attribute 'shape'
    '''
    if gridshape == (0, 0):
        l = len(images)
        r = int(sqrt(l))
        c = int(ceil(l / r))
        gridshape = (r, c)
    fig = plt.figure(figsize=figsize)
    for i in range(len(images)):
        ax = plt.subplot(gridshape[0], gridshape[1], 1 + i)
        ax.imshow(images[i], cmap=cm)
        if titles: ax.title.set_text(titles[i])
    # plt.tight_layout(1.08)
    plt.show(block=False)
    # plt.pause(2)
    plt.waitforbuttonpress(0)
    plt.close()


# DISK OPERATIONS
def create_path(path: str):
    """
    Creates a path if it doesn't already exists
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)


def copy_file():
    pass


if __name__ == '__main__':
    pass
