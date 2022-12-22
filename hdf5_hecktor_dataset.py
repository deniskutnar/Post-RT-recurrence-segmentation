import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import SimpleITK as sitk
import glob
import torch.nn as nn
import nibabel as nib
import shutil
import h5py
import random

from utils import * 


anno_dirs = glob.glob('hecktor2022/labelsTr/*nii.gz')

## For simplicity, remove CT with diff resolution than 512 x 512 
odd = []
for f in range(len(anno_dirs)):
    mask = read_image(anno_dirs[f])
    x = mask.shape[1]
    y = mask.shape[2]
    
    if ((x!=512 or y!=512) or mask.max()==0 ):
        odd.append(anno_dirs[f])


new_anno_set = set(anno_dirs) - set(odd)
anno_dirs = list(new_anno_set)
random.shuffle(anno_dirs)

pet_dirs = []
ct_dirs = []
for idx,f in enumerate(anno_dirs):
    pid = os.path.basename(f).split('.nii.gz')[0]
    
    pet = os.path.join(os.path.dirname(os.path.dirname(f)), 'imagesTr/'+ pid + '__PT.nii.gz')
    ct = os.path.join(os.path.dirname(os.path.dirname(f)), 'imagesTr/'+ pid + '__CT.nii.gz')
    if os.path.exists(pet):
        pet_dirs.append(pet)
    if os.path.exists(ct):
        ct_dirs.append(ct)


print(len(pet_dirs), len(ct_dirs), len(anno_dirs))

# Split the data
ct_dirs_train = ct_dirs[:412]
pet_dirs_train = pet_dirs[:412]
anno_dirs_train = anno_dirs[:412]

ct_dirs_val = ct_dirs[412:]
pet_dirs_val = pet_dirs[412:]
anno_dirs_val = anno_dirs[412:]



# Create files
f = h5py.File("train_dataset.hdf5", "w")
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

    ## resample Mask --> CT
    t_img = sitk.ReadImage(ct_dirs_train[i])
    o_img = sitk.ReadImage(anno_dirs_train[i])
    reg_mask = resize_image_itk(o_img, t_img, sitk.sitkNearestNeighbor)
    
    ### loop to go over all file paths
    ### read ct,pet,mask
    ct = img_as_numpy = sitk.GetArrayFromImage(t_img).astype('float32')
    pet = img_as_numpy = sitk.GetArrayFromImage(reg_pet).astype('float32')
    #mask = img_as_numpy = sitk.GetArrayFromImage(reg_mask)
    
    ## Threshold GTVp ==1, GTVn ==2
    GTVp_itk = sitk.BinaryThreshold(reg_mask, 1,  1, 1, 0) # GTVp == 1
    GTVp = sitk.GetArrayFromImage(GTVp_itk)
    GTVn_itk = sitk.BinaryThreshold(reg_mask, 2,  2, 1, 0) # GTVn == 2
    GTVn = sitk.GetArrayFromImage(GTVn_itk)
    
    
    ## Normalise data 
    ct = normalize_ct(ct)
    pet = normalize_pt(pet)
    
    GTVp_loc = np.transpose(np.nonzero(GTVp))
    GTVn_loc = np.transpose(np.nonzero(GTVn))
    
    pt_img.create_dataset('ct', data=ct, chunks=True, compression="lzf")
    pt_img.create_dataset('pet', data=pet, chunks=True, compression="lzf")
    pt_mask.create_dataset('GTVp', data=GTVp, chunks=True, compression="lzf")
    pt_mask.create_dataset('GTVn', data=GTVn, chunks=True, compression="lzf")
    pt_points.create_dataset('GTVp_loc', data=GTVp_loc, chunks=True, compression="lzf")
    pt_points.create_dataset('GTVn_loc', data=GTVn_loc, chunks=True, compression="lzf")
    print(i)
f.close()



f = h5py.File("val_dataset.hdf5", "w")
ptg = f.create_group('patients')


for i in range(len(ct_dirs_val)):
    # Create datastructure inside the HDF5
    pt_fol = ptg.create_group('{:03d}'.format(i))
    pt_mask = pt_fol.create_group('masks')
    pt_img = pt_fol.create_group('images')
    pt_points = pt_fol.create_group('points')
    
    ## Resample PET --> CT
    t_img = sitk.ReadImage(ct_dirs_val[i])
    o_img = sitk.ReadImage(pet_dirs_val[i])
    reg_pet = resize_image_itk(o_img, t_img, sitk.sitkLinear)
    
    ## Resample Mask --> CT
    t_img = sitk.ReadImage(ct_dirs_val[i])
    o_img = sitk.ReadImage(anno_dirs_val[i])
    reg_mask = resize_image_itk(o_img, t_img, sitk.sitkNearestNeighbor)
    
    ### loop to go over all file paths
    ### read ct,pet,mask
    ct = img_as_numpy = sitk.GetArrayFromImage(t_img).astype('float32')
    pet = img_as_numpy = sitk.GetArrayFromImage(reg_pet).astype('float32')
    #mask = img_as_numpy = sitk.GetArrayFromImage(reg_mask)
    
    ## Threshold GTVp ==1, GTVn ==2
    GTVp_itk = sitk.BinaryThreshold(reg_mask, 1,  1, 1, 0) # GTVp == 1
    GTVp = sitk.GetArrayFromImage(GTVp_itk)
    GTVn_itk = sitk.BinaryThreshold(reg_mask, 2,  2, 1, 0) # GTVn == 2
    GTVn = sitk.GetArrayFromImage(GTVn_itk)
    
    
    ## Normalise data 
    ct = normalize_ct(ct)
    pet = normalize_pt(pet)
    
    GTVp_loc = np.transpose(np.nonzero(GTVp))
    GTVn_loc = np.transpose(np.nonzero(GTVn))
    
    pt_img.create_dataset('ct', data=ct, chunks=True, compression="lzf")
    pt_img.create_dataset('pet', data=pet, chunks=True, compression="lzf")
    pt_mask.create_dataset('GTVp', data=GTVp, chunks=True, compression="lzf")
    pt_mask.create_dataset('GTVn', data=GTVn, chunks=True, compression="lzf")
    pt_points.create_dataset('GTVp_loc', data=GTVp_loc, chunks=True, compression="lzf")
    pt_points.create_dataset('GTVn_loc', data=GTVn_loc, chunks=True, compression="lzf")
    print(i)
f.close()

