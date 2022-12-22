import os
import argparse
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, utils
import SimpleITK as sitk
import glob
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as F
import pathlib

from patchgen import *
from utils import *



argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--data_path", help="path to the dataset location")
args = argParser.parse_args()
train_path = args.data_path + "train_dataset.hdf5"
val_path =  args.data_path + "val_dataset.hdf5"
h5file_train = h5py.File(train_path, 'r')
h5file_val = h5py.File(val_path, 'r')
train_dataset = RelapseDatasetSUV(h5file_train, num_samples=1, transform=None)
val_dataset = RelapseDatasetSUV(h5file_val, num_samples=1, transform=None)
dataset = train_dataset + val_dataset 


lst = list(range(1,101))
final_means = []
for rr in range(len(lst)):

    DSC_arr = []
    for f in range(len(dataset)):
        # Get PET and GT
        x, gtv, relapse = dataset[f]
        pet = x[1,:,:,:]
        GT = np.array(relapse)
        loc3D = np.nonzero(relapse.squeeze(0))

        # SUV max
        SUVmax = int(pet.max() * (lst[rr] / 100))

        # Threshold image 
        pet_itk = sitk.GetImageFromArray(pet)
        Im = pet_itk
        BinThreshImFilt = sitk.BinaryThresholdImageFilter()
        BinThreshImFilt.SetLowerThreshold(SUVmax)
        BinThreshImFilt.SetUpperThreshold(10000000)
        BinThreshImFilt.SetOutsideValue(0)
        BinThreshImFilt.SetInsideValue(1)
        BinIm = BinThreshImFilt.Execute(Im)
        pet_SUV = sitk.GetArrayFromImage(BinIm)

        # Calculate DSC 
        DSC = getDSC(GT, pet_SUV)
        DSC_arr.append(DSC)           
    
    DSC_arr = np.array(DSC_arr)
    mean = DSC_arr.mean()            
    final_means.append(mean)          

final_means = np.array(final_means)
print(final_means)





