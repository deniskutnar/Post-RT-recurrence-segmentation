import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import glob
import torch.nn as nn
import nibabel as nib
import shutil
import h5py
from random import randrange
import random


# All images have the same x,y dim = 512 x 512,
# so we start cropp from the midle point (x = 256, y = 256)
# Based on the distance analysis the min area that covers both gtv & relapse is:
# X: 189 - 327 ====> 160 px (15% padding)
# Y: 213 - 427 ====> 224 px (15% padding + to next even) 
# Z: 43 - 155  ====> 128 px (15% padding) 

# Final ROI + padding = 128 x 224 x 160 (z,y,x)

class RelapseDataset(Dataset):
    """Train & Val dataset.
    Returns patch (128 x 224 x 160)
    """
    
    def __init__(self, h5_file, num_samples, transform=None):
        self.ptg = h5_file['patients']
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return len(self.ptg) * self.num_samples

    def __getitem__(self, idx):
        patient_index = idx // self.num_samples
        pids = [key for key in self.ptg.keys()]
        ct = self.ptg[pids[patient_index]]['images/ct_norm']
        pet = self.ptg[pids[patient_index]]['images/pet_norm']
        gtv = self.ptg[pids[patient_index]]['masks/gtv'] 
        relapse = self.ptg[pids[patient_index]]['masks/relapse'] 
        gtvloc = self.ptg[pids[patient_index]]['points/gtv_loc'] 
        relapseloc = self.ptg[pids[patient_index]]['points/relapse_loc'] 
        
        ct = np.array(ct)
        ct = torch.from_numpy(ct)
        ct = ct.unsqueeze(0)
        ct = ct.unsqueeze(0)
        
        pet = np.array(pet)
        pet = torch.from_numpy(pet)
        pet = pet.unsqueeze(0)
        pet = pet.unsqueeze(0)
        
        gtv_np = np.array(gtv)
        gtv = torch.from_numpy(gtv_np)
        gtv = gtv.unsqueeze(0)
        gtv = gtv.unsqueeze(0)

        relapse = np.array(relapse)
        relapse = torch.from_numpy(relapse)
        relapse = relapse.unsqueeze(0)
        relapse = relapse.unsqueeze(0)
        
        gtvloc = np.array(gtvloc)
        tensor = torch.cat((ct, pet, relapse),1)

        # Flip the tensor so the head starts with slice 0
        tensor = torch.flip(tensor, [2])
        cropped = tensor[:,:, 35:163, 208:432 , 176:336]
        cropped = cropped.squeeze(0)

        if self.transform:
             cropped = self.transform(cropped)

        cropped = cropped.unsqueeze(0)
        train_tensor = cropped[:, :2, :, :, :]
        train_tensor = train_tensor.squeeze(0)
        target_tensor = cropped[:, 2, :, :, :]
        
        return train_tensor, target_tensor





class RelapseDatasetSUV(Dataset):
    """Train & Val dataset.
    Return non normalised patch (128 x 224 x 160)
    """

    def __init__(self, h5_file, num_samples, transform=None):
        self.ptg = h5_file['patients']
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return len(self.ptg) * self.num_samples

    def __getitem__(self, idx):
        patient_index = idx // self.num_samples
        pids = [key for key in self.ptg.keys()]
        ct = self.ptg[pids[patient_index]]['images/ct']         
        pet = self.ptg[pids[patient_index]]['images/pet']       
        gtv = self.ptg[pids[patient_index]]['masks/gtv'] 
        relapse = self.ptg[pids[patient_index]]['masks/relapse'] 
        gtvloc = self.ptg[pids[patient_index]]['points/gtv_loc'] 
        relapseloc = self.ptg[pids[patient_index]]['points/relapse_loc'] 
        
        ct = np.array(ct)
        ct = torch.from_numpy(ct)
        ct = ct.unsqueeze(0)
        ct = ct.unsqueeze(0)
        
        pet = np.array(pet)
        pet = torch.from_numpy(pet)
        pet = pet.unsqueeze(0)
        pet = pet.unsqueeze(0)
        
        gtv_np = np.array(gtv)
        gtv = torch.from_numpy(gtv_np)
        gtv = gtv.unsqueeze(0)
        gtv = gtv.unsqueeze(0)

        relapse = np.array(relapse)
        relapse = torch.from_numpy(relapse)
        relapse = relapse.unsqueeze(0)
        relapse = relapse.unsqueeze(0)
        
        gtvloc = np.array(gtvloc)
        tensor = torch.cat((ct, pet, gtv, relapse),1)

        # Flip the tensor so the head starts with slice 0
        tensor = torch.flip(tensor, [2])
        cropped = tensor[:,:, 35:163, 208:432 , 176:336]
        cropped = cropped.squeeze(0)

        if self.transform:
             cropped = self.transform(cropped)

        cropped = cropped.unsqueeze(0)
        train_tensor = cropped[:, :2, :, :, :]
        train_tensor = train_tensor.squeeze(0)
        gtv_ten = cropped[:, 2, :, :, :]
        relapse_ten = cropped[:, 3, :, :, :]

        return train_tensor, gtv_ten, relapse_ten




class HecktorDataset(Dataset):
    def __init__(self, h5_file, num_samples, transform=None):
        self.ptg = h5_file['patients']
        self.num_samples = num_samples
        self.transform = transform
        

    def __len__(self):
        return len(self.ptg) * self.num_samples
        

    def __getitem__(self, idx):
        patient_index = idx // self.num_samples
        pids = [key for key in self.ptg.keys()]
        ct = self.ptg[pids[patient_index]]['images/ct']
        pet = self.ptg[pids[patient_index]]['images/pet']
        mask = self.ptg[pids[patient_index]]['masks/GTVp'] 
        lesionloc = self.ptg[pids[patient_index]]['points/GTVp_loc'] 
        
        ct = np.array(ct)
        ct = torch.from_numpy(ct)
        ct = ct.unsqueeze(0)
        ct = ct.unsqueeze(0)
        
        pet = np.array(pet)
        pet = torch.from_numpy(pet)
        pet = pet.unsqueeze(0)
        pet = pet.unsqueeze(0)
        
        mask = np.array(mask)
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(0)  
        
        lesionloc = np.array(lesionloc)
        tensor = torch.cat((ct, pet, mask),1)
        
        # Flip the tensor so the head starts with slice 0
        tensor = torch.flip(tensor, [2])
        
        zcount = int(len(tensor[0,0,:,:,:]))
        # Tensor smaller than 128 will be pad to 128
        if(zcount < 128):
            tensor = F.pad(input=tensor, pad=(0, 0, 0, 0, 128-zcount, 0), mode='reflect')
            cropped = tensor[:,:, : , 86:310, 167:327]

        # Tensor larger than 128 will be cropped to 128
        else:
            cropped = tensor[:,:, 0:128 , 86:310, 167:327]
  
        cropped = cropped.squeeze(0)
        if self.transform:
            cropped = self.transform(cropped)
        cropped = cropped.unsqueeze(0)
        
        train_tensor = cropped[:, :2, :, :, :]
        train_tensor = train_tensor.squeeze(0)
        target_tensor = cropped[:, 2, :, :, :]
        return train_tensor, target_tensor
    
    
    


