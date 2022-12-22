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
import medpy
from medpy.metric import binary 
from torch.autograd import Variable
import scipy.spatial
import math
import numpy.ma as ma
import matplotlib.patches as mpatches
import cv2
import cc3d
import scipy.ndimage
from scipy import ndimage

from netgen import *
from patchgen import *
from utils import *

# For each test case print: DSC, Rec, Prec, detected POs, volume 


argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--data_path", help="path to the dataset location")
args = argParser.parse_args()
h5file_test = h5py.File(args.data_path+"/test_dataset.hdf5", 'r') 
test_dataset = RelapseDatasetSUV(h5file_test, num_samples=1, transform=None)
# For CNN models we need normalised data input for the infer
test_dataset_norm = RelapseDataset(h5file_test, num_samples=1, transform=None) 
print("Len dataset: ", len(test_dataset))



# I run this script 6 times with j from 0-6
j = 6 #                                     =======> change me 
x, gtv, relapse = test_dataset[j]
x_train, y_target = test_dataset_norm[j]
pet = x[1,:,:,:]
ct = x[0,:,:,:]

# 1. Relapse volume
relapse_volume = np.nonzero(relapse.squeeze(0))
relapse_volume = relapse_volume[:,0]
print("Relapse Volume: ",relapse_volume.shape)

# 2. Number of relapses
labels_out, N = cc3d.connected_components(np.array(relapse.squeeze(0)), return_N=True)
print("Number of Relapse volumes: ",N)


# 2.A  Get each individual relapses 
results = []
for i in range(N):
    o1 = (labels_out ==(i+1)).astype(np.float32) 
    results.append(o1)

# 2.B  Binary erosion 
# eroded relapses by 3x3x3 SE
erorded_res = []
for i in range(len(results)):
    er = ndimage.binary_erosion(results[i], structure=np.ones((3,3,3))).astype(results[i].dtype)
    erorded_res.append(er)

print("")



###################################
# Method 1 (Relapse + AI Random)  #
###################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=2,
        out_channels=1,
        n_blocks=5,
        start_filters=32,
        activation='leaky',
        normalization='instance',
        conv_mode='same',
        dim=3)

model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.load_state_dict(torch.load('models/UNet-model-rand.pt'))
model.cuda()

def Unet_pred(inp_cube):
    model.eval()
    cube = inp_cube
    cube = cube.unsqueeze(0)
    cube = Variable(cube.cuda())
    y_pred = model(cube)
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.squeeze(0)
    y_pred = y_pred.squeeze(0)
    y_pred = y_pred.cpu().data.numpy() 
    y_pred = (y_pred >=0.5).astype(np.float32)   
    return y_pred

pred = Unet_pred(x_train)
DSC1 = getDSC(relapse, pred)
recall1 = medpy.metric.binary.recall(pred, np.array(relapse))
precision1 = medpy.metric.binary.precision(pred, np.array(relapse))

pred1_volume = torch.tensor(pred)
pred1_volume = np.nonzero(pred1_volume)
pred1_volume = pred1_volume[:,0]

print("Method 1 DSC: ", DSC1)
print("Method 1 Recall: ", recall1)
print("Method 1 Precision: ", precision1)
print("Method 1 predicted volume: ",pred1_volume.shape)

# Now compare the eroded relapses with the prediction
# we will add 2 arrays together, if the PO and Relapse 
# overlay, the max == 2
scores1 = []
for i in range(len(erorded_res)):
    hit = erorded_res + pred
    hit = hit.max()
    if(hit.max()==2):
        scores1.append("YES")
    else:
         scores1.append("NO")
print("Detected relapse PO: ", scores1)
print("")





#####################################
# Method 2 (Relapse + AI finetune)  #
#####################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=2,
        out_channels=1,
        n_blocks=5,
        start_filters=32,
        activation='leaky',
        normalization='instance',
        conv_mode='same',
        dim=3)

model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.load_state_dict(torch.load('models/UNet-model-finetune.pt'))
model.cuda()

def Unet_pred(inp_cube):
    model.eval()
    cube = inp_cube
    cube = cube.unsqueeze(0)
    cube = Variable(cube.cuda())
    y_pred = model(cube)
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.squeeze(0)
    y_pred = y_pred.squeeze(0)
    y_pred = y_pred.cpu().data.numpy() 
    y_pred = (y_pred >=0.5).astype(np.float32)   
    return y_pred

pred = Unet_pred(x_train)
DSC2 = getDSC(relapse, pred)
recall2 = medpy.metric.binary.recall(pred, np.array(relapse))
precision2 = medpy.metric.binary.precision(pred, np.array(relapse))

pred2_volume = torch.tensor(pred)
pred2_volume = np.nonzero(pred2_volume)
pred2_volume = pred2_volume[:,0]

print("Method 2 DSC: ", DSC2)
print("Method 2 Recall: ", recall2)
print("Method 2 Precision: ", precision2)
print("Method 2 predicted volume: ",pred2_volume.shape)

# Now compare the eroded relapses with the prediction
# we will add 2 arrays together, if the PO and Relapse 
# overlay, the max == 2
scores2 = []
for i in range(len(erorded_res)):
    hit = erorded_res + pred
    hit = hit.max()
    if(hit.max()==2):
        scores2.append("YES")
    else:
         scores2.append("NO")
print("Detected relapse PO: ", scores2)
print("")







###############################
# Method 3 (Relapse + SUVmax) #
###############################
SUVmax40 = int(pet.max() * 0.5)

# Threshold image 
pet_itk = sitk.GetImageFromArray(pet)
Im = pet_itk
BinThreshImFilt = sitk.BinaryThresholdImageFilter()
BinThreshImFilt.SetLowerThreshold(SUVmax40)
BinThreshImFilt.SetUpperThreshold(10000000)
BinThreshImFilt.SetOutsideValue(0)
BinThreshImFilt.SetInsideValue(1)
BinIm = BinThreshImFilt.Execute(Im)
pet_SUV = sitk.GetArrayFromImage(BinIm)

DSC3 = getDSC(relapse, pet_SUV)
recall3 = medpy.metric.binary.recall(pet_SUV, np.array(relapse))
precision3 = medpy.metric.binary.precision(pet_SUV, np.array(relapse))

pred3_volume = torch.tensor(pet_SUV)
pred3_volume = np.nonzero(pred3_volume)
pred3_volume = pred3_volume[:,0]

print("Method 3 DSC: ", DSC3)
print("Method 3 Recall: ", recall3)
print("Method 3 Precision: ", precision3)
print("Method 3 predicted volume: ",pred3_volume.shape)

# Now compare the eroded relapses with the prediction
# we will add 2 arrays together, if the PO and Relapse 
# overlay, the max == 2
scores3 = []
for i in range(len(erorded_res)):
    hit = erorded_res + pet_SUV
    hit = hit.max()
    if(hit.max()==2):
        scores3.append("YES")
    else:
         scores3.append("NO")
print("Detected relapse PO: ", scores3)
print("")




############################
# Method 4 (GTV + Relapse) #
############################
DSC4 = getDSC(relapse, gtv)
recall4 = medpy.metric.binary.recall(np.array(gtv), np.array(relapse))
precision4 = medpy.metric.binary.precision(np.array(gtv), np.array(relapse))
pred4_volume = np.nonzero(gtv.squeeze(0))
pred4_volume = pred4_volume[:,0]

print("Method 4 DSC: ", DSC4)
print("Method 4 Recall: ", recall4)
print("Method 4 Precision: ", precision4)
print("Method 4 predicted volume: ",pred4_volume.shape)

# Now compare the eroded relapses with the prediction
# we will add 2 arrays together, if the PO and Relapse 
# overlay, the max == 2
scores4 = []
for i in range(len(erorded_res)):
    hit = erorded_res + np.array(gtv.squeeze(0))
    hit = hit.max()
    if(hit.max()==2):
        scores4.append("YES")
    else:
         scores4.append("NO")
print("Detected relapse PO: ", scores4)
print("")







