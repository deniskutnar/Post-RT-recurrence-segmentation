import os
import torch
import torchvision
import numpy as np
import SimpleITK as sitk
import torch.nn as nn
import nibabel as nib
import shutil
import h5py
import pathlib
import json
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter
import math
from torch.autograd import Variable
import scipy.spatial



def read_image(path):
    img = sitk.ReadImage(path)
    img_as_numpy = sitk.GetArrayFromImage(img).astype('float32')
    return img_as_numpy



def resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkNearestNeighbor):
    """
    use itk Method to convert the original image resample To be consistent with the target image
    :param ori_img: Original alignment required itk image
    :param target_img: Target to align itk image
    :param resamplemethod: itk interpolation method : sitk.sitkLinear-linear  sitk.sitkNearestNeighbor-Nearest neighbor
    :return:img_res_itk: Resampling okay itk image
    """
    target_Size = target_img.GetSize()      # Target image size [x,y,z]
    target_Spacing = target_img.GetSpacing()   # Voxel block size of the target [x,y,z]
    target_origin = target_img.GetOrigin()      # Starting point of target [x,y,z]
    target_direction = target_img.GetDirection()  # Target direction [crown, sagittal, transverse] = [z,y,x]

    # The method of itk is resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # Target image to resample
    # Set the information of the target image
    resampler.SetSize(target_Size)		# Target image size
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # Set different dype according to the need to resample the image
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt16)   # Nearest neighbor interpolation is used for mask, and uint16 is saved
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # Linear interpolation is used for PET/CT/MRI and the like, and float32 is saved
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))    
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # Get the resampled image
    return itk_img_resampled



def normalize_ct(img):
    norm_img = np.clip(img, -1024, 1024) / 1024
    return norm_img

    
def normalize_pt(img):
    mean = np.mean(img)
    std = np.std(img)
    return (img - mean) / (std + 1e-3)



def calculate_dice(A, B, zero_div_is_zero=False, perfect_empty_is_one=True,
                   foreground_class=1):
    """Calculates the dice score between two masks A and B

    Parameters
    ----------
    A : array-like
        Foreground mask
    B : array-like
        Background mask
    zero_div_is_zero : bool, optional
        If True, empty foreground and background masks will result
        in zero Dice score. Else will return NaN.
    perfect_empty_is_one : bool, optional
        If True, will return 1.0 if both masks are empty. Else will return
        either NaN or 0.0, depending on above option.
    foreground_class : int, optional
        Value of the foreground mask. By default it is '1'.

    Returns
    -------
    float
        Dice score
    """
    A_elts = np.argwhere(A.flat == foreground_class)
    B_elts = np.argwhere(B.flat == foreground_class)
    overlap_elts = np.intersect1d(A_elts, B_elts)
    overlap = overlap_elts.size
    total_sum = (A_elts.size + B_elts.size)
    if total_sum == 0 and overlap == 0:
        if perfect_empty_is_one:
            score = 1.0
            return score

    try:
        score = 2 * overlap / total_sum
    except ZeroDivisionError:
        if zero_div_is_zero:
            score = 0
        else:
            score = np.NaN
    return score

  
        
        
def plot_training_loss(training_losses,
                  validation_losses,
                  learning_rate,
                  gaussian=True,
                  sigma=2,
                  figsize=(8, 6)
                  ):
    """
    Returns a loss plot with training loss, validation loss and learning rate.
    """
    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values
    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    subfig1 = fig.add_subplot(grid[0, 0])
    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines['top'].set_visible(False)
        subfig.spines['right'].set_visible(False)

    if gaussian:
        training_losses_gauss = gaussian_filter(training_losses, sigma=sigma)
        validation_losses_gauss = gaussian_filter(validation_losses, sigma=sigma)
        linestyle_original = '.'
        color_original_train = 'lightcoral'
        color_original_valid = 'lightgreen'
        color_smooth_train = 'red'
        color_smooth_valid = 'green'
        alpha = 0.25
    else:
        linestyle_original = '-'
        color_original_train = 'red'
        color_original_valid = 'green'
        alpha = 1.0

    # Subfig 1
    subfig1.plot(x_range, training_losses, linestyle_original, color=color_original_train, label='Training',
                 alpha=alpha)
    subfig1.plot(x_range, validation_losses, linestyle_original, color=color_original_valid, label='Validation',
                 alpha=alpha)
    if gaussian:
        subfig1.plot(x_range, training_losses_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)
        subfig1.plot(x_range, validation_losses_gauss, '-', color=color_smooth_valid, label='Validation', alpha=0.75)
    subfig1.title.set_text('Training & validation (pre-trained) loss')
    subfig1.set_xlabel('Epoch')
    subfig1.set_ylabel('Loss')
    subfig1.legend(loc='upper right')
    return fig




def plot_training_dice(training_dice,
                  validation_dice,
                  learning_rate,
                  gaussian=True,
                  sigma=2,
                  figsize=(8, 6)
                  ):
    """
    Returns a loss plot with training loss, validation loss and learning rate.
    """
    list_len = len(training_dice)
    x_range = list(range(1, list_len + 1))  # number of x values
    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    subfig1 = fig.add_subplot(grid[0, 0])
    subfigures = fig.get_axes()

    for i, subfig in enumerate(subfigures, start=1):
        subfig.spines['top'].set_visible(False)
        subfig.spines['right'].set_visible(False)

    if gaussian:
        training_dice_gauss = gaussian_filter(training_dice, sigma=sigma)
        validation_dice_gauss = gaussian_filter(validation_dice, sigma=sigma)
        linestyle_original = '.'
        color_original_train = 'lightcoral'
        color_original_valid = 'lightgreen'
        color_smooth_train = 'red'
        color_smooth_valid = 'green'
        alpha = 0.25
    else:
        linestyle_original = '-'
        color_original_train = 'red'
        color_original_valid = 'green'
        alpha = 1.0

    subfig1.plot(x_range, training_dice, linestyle_original, color=color_original_train, label='Training',
                 alpha=alpha)
    subfig1.plot(x_range, validation_dice, linestyle_original, color=color_original_valid, label='Validation',
                 alpha=alpha)
    if gaussian:
        subfig1.plot(x_range, training_dice_gauss, '-', color=color_smooth_train, label='Training', alpha=0.75)
        subfig1.plot(x_range, validation_dice_gauss, '-', color=color_smooth_valid, label='Validation', alpha=0.75)
    subfig1.title.set_text('Training & validation plot')
    subfig1.set_xlabel('Epoch')
    subfig1.set_ylabel('DSC')
    subfig1.legend(loc='upper right')
    return fig


   

def getDSC(testImage, predImage):    
    testArray   = testImage.flatten()
    predArray = predImage.flatten()
    return 1.0 - scipy.spatial.distance.dice(testArray, predArray) 



class EarlyStoppingLoss:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='UNet-model.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss





class EarlyStoppingMetric:
    """Early stops the training if validation dice doesn't improve after a given patience."""
    def __init__(self, patience=60, verbose=False, delta=0, path='UNet-model.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_dsc_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_dsc, model):

        score = val_dsc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_dsc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_dsc, model)
            self.counter = 0

    def save_checkpoint(self, val_dsc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation dice Incerased ({self.val_dsc_max:.6f} --> {val_dsc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_dsc_max = val_dsc
