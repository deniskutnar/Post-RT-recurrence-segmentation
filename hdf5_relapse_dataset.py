import os
import argparse
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import SimpleITK as sitk
from glob import glob
import torch.nn as nn
import nibabel as nib
import shutil
import h5py
import random

from utils import * 


argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--data_path", help="path to the dataset location")
args = argParser.parse_args()
ct_dirs = glob(data_path+ "/*/Processed/CT_CT*.nii.gz")

### Shuffle dirs
random.shuffle(ct_dirs)

pet_dirs = []
gtv_dirs = []
relapse_dirs = []
for f in range(len(ct_dirs)):
    base = ct_dirs[f][:-12]
    pet = base + "PET_PET.nii.gz"
    gtv = base + "GTV_f.nii.gz"
    relapse = base + "Relapse_f.nii.gz"

    if os.path.exists(pet):
        pet_dirs.append(pet)
    if os.path.exists(gtv):
        gtv_dirs.append(gtv)
    if os.path.exists(relapse):
        relapse_dirs.append(relapse)


### Split: 23 : 7 : 7 (37)

ct_dirs_train = ct_dirs[:23]
pet_dirs_train = pet_dirs[:23]
gtv_dirs_train = gtv_dirs[:23]
relapse_dirs_train = relapse_dirs[:23]

ct_dirs_val = ct_dirs[23:30]
pet_dirs_val = pet_dirs[23:30]
gtv_dirs_val = gtv_dirs[23:30]
relapse_dirs_val = relapse_dirs[23:30]

ct_dirs_test = ct_dirs[30:]
pet_dirs_test = pet_dirs[30:]
gtv_dirs_test = gtv_dirs[30:]
relapse_dirs_test = relapse_dirs[30:]


### HDF5 Train
f = h5py.File(data_path + "/datasets/train_dataset.hdf5", "w")
ptg = f.create_group('patients')

for i in range(len(ct_dirs_train)):
    # Create datastructure inside the HDF5
    pt_fol = ptg.create_group('{:03d}'.format(i))
    pt_mask = pt_fol.create_group('masks')
    pt_img = pt_fol.create_group('images')
    pt_points = pt_fol.create_group('points')
    
    ## resample PET --> CT
    t_img = sitk.ReadImage(ct_dirs_train[i])
    o_img = sitk.ReadImage(pet_dirs_train[i])
    reg_pet = resize_image_itk(o_img, t_img, sitk.sitkLinear)
    
    ### loop to go over all file paths
    # read ct, pet, gtv, relapse
    ct  = sitk.GetArrayFromImage(t_img).astype('float32')
    pet = sitk.GetArrayFromImage(reg_pet).astype('float32')
    gtv = sitk.GetArrayFromImage(sitk.ReadImage(gtv_dirs_train[i]))
    relapse = sitk.GetArrayFromImage(sitk.ReadImage(relapse_dirs_train[i]))

    ## Normalise data 
    ct_norm = normalize_ct(ct)
    pet_norm = normalize_pt(pet)
    
    ## Create points
    gtv_loc = np.transpose(np.nonzero(gtv))
    relapse_loc = np.transpose(np.nonzero(relapse))
    
    pt_img.create_dataset('ct', data=ct, chunks=True, compression="lzf")
    pt_img.create_dataset('pet', data=pet, chunks=True, compression="lzf")
    pt_img.create_dataset('ct_norm', data=ct_norm, chunks=True, compression="lzf")
    pt_img.create_dataset('pet_norm', data=pet_norm, chunks=True, compression="lzf")
    pt_mask.create_dataset('gtv', data=gtv, chunks=True, compression="lzf")
    pt_mask.create_dataset('relapse', data=relapse, chunks=True, compression="lzf")
    pt_points.create_dataset('gtv_loc', data=gtv_loc, chunks=True, compression="lzf")
    pt_points.create_dataset('relapse_loc', data=relapse_loc, chunks=True, compression="lzf")
    print(i)
f.close()




### HDF5 Val
f = h5py.File(data_path + "/datasets/val_dataset.hdf5", "w")
ptg = f.create_group('patients')

for i in range(len(ct_dirs_val)):
    # Create datastructure inside the HDF5
    pt_fol = ptg.create_group('{:03d}'.format(i))
    pt_mask = pt_fol.create_group('masks')
    pt_img = pt_fol.create_group('images')
    pt_points = pt_fol.create_group('points')
    
    ## resample PET --> CT
    t_img = sitk.ReadImage(ct_dirs_val[i])
    o_img = sitk.ReadImage(pet_dirs_val[i])
    reg_pet = resize_image_itk(o_img, t_img, sitk.sitkLinear)
    
    ### loop to go over all file paths
    # read ct, pet, gtv, relapse
    ct  = sitk.GetArrayFromImage(t_img).astype('float32')
    pet = sitk.GetArrayFromImage(reg_pet).astype('float32')
    gtv = sitk.GetArrayFromImage(sitk.ReadImage(gtv_dirs_val[i]))
    relapse = sitk.GetArrayFromImage(sitk.ReadImage(relapse_dirs_val[i]))

    ## Normalise data 
    ct_norm = normalize_ct(ct)
    pet_norm = normalize_pt(pet)
    
    ## Create points
    gtv_loc = np.transpose(np.nonzero(gtv))
    relapse_loc = np.transpose(np.nonzero(relapse))
    
    pt_img.create_dataset('ct', data=ct, chunks=True, compression="lzf")
    pt_img.create_dataset('pet', data=pet, chunks=True, compression="lzf")
    pt_img.create_dataset('ct_norm', data=ct_norm, chunks=True, compression="lzf")
    pt_img.create_dataset('pet_norm', data=pet_norm, chunks=True, compression="lzf")
    pt_mask.create_dataset('gtv', data=gtv, chunks=True, compression="lzf")
    pt_mask.create_dataset('relapse', data=relapse, chunks=True, compression="lzf")
    pt_points.create_dataset('gtv_loc', data=gtv_loc, chunks=True, compression="lzf")
    pt_points.create_dataset('relapse_loc', data=relapse_loc, chunks=True, compression="lzf")
    print(i)
f.close()



### HDF5 Test
f = h5py.File(data_path +"/datasets/test_dataset.hdf5", "w")
ptg = f.create_group('patients')

for i in range(len(ct_dirs_test)):
    # Create datastructure inside the HDF5
    pt_fol = ptg.create_group('{:03d}'.format(i))
    pt_mask = pt_fol.create_group('masks')
    pt_img = pt_fol.create_group('images')
    pt_points = pt_fol.create_group('points')
    
    ## resample PET --> CT
    t_img = sitk.ReadImage(ct_dirs_test[i])
    o_img = sitk.ReadImage(pet_dirs_test[i])
    reg_pet = resize_image_itk(o_img, t_img, sitk.sitkLinear)
    
    ### loop to go over all file paths
    # read ct, pet, gtv, relapse
    ct  = sitk.GetArrayFromImage(t_img).astype('float32')
    pet = sitk.GetArrayFromImage(reg_pet).astype('float32')
    gtv = sitk.GetArrayFromImage(sitk.ReadImage(gtv_dirs_test[i]))
    relapse = sitk.GetArrayFromImage(sitk.ReadImage(relapse_dirs_test[i]))

    ## Normalise data 
    ct_norm = normalize_ct(ct)
    pet_norm = normalize_pt(pet)
    
    ## Create points
    gtv_loc = np.transpose(np.nonzero(gtv))
    relapse_loc = np.transpose(np.nonzero(relapse))
    
    pt_img.create_dataset('ct', data=ct, chunks=True, compression="lzf")
    pt_img.create_dataset('pet', data=pet, chunks=True, compression="lzf")
    pt_img.create_dataset('ct_norm', data=ct_norm, chunks=True, compression="lzf")
    pt_img.create_dataset('pet_norm', data=pet_norm, chunks=True, compression="lzf")
    pt_mask.create_dataset('gtv', data=gtv, chunks=True, compression="lzf")
    pt_mask.create_dataset('relapse', data=relapse, chunks=True, compression="lzf")
    pt_points.create_dataset('gtv_loc', data=gtv_loc, chunks=True, compression="lzf")
    pt_points.create_dataset('relapse_loc', data=relapse_loc, chunks=True, compression="lzf")
    print(i)
f.close()





