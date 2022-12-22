import os
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
from matplotlib import gridspec
from scipy.ndimage import gaussian_filter
import argparse
import logging
import torchio.transforms as transforms
import random
from torch.autograd import Variable
from monai.losses.dice import *  
from monai.losses.dice import DiceLoss
from random import randrange

from netgen import *
from patchgen import *
from utils import *


def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--data_path", help="path to the dataset location")
    args = argParser.parse_args()

    train_path = args.data_path + "train_dataset.hdf5"
    val_path =  args.data_path + "val_dataset.hdf5"
    model_path = "models"
    stats_path = "stats"

    h5file_train = h5py.File(train_path, 'r')
    h5file_val = h5py.File(val_path, 'r')
    print("Loading data completed")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    isExist = os.path.exists(model_path)
    if not isExist:
        os.makedirs(model_path)
     
    isExist = os.path.exists(stats_path)
    if not isExist:
        os.makedirs(stats_path)
     

    model = UNet(in_channels=2,
                 out_channels=1,
                 n_blocks=5,
                 start_filters=32,
                 activation='leaky',
                 normalization='instance',
                 conv_mode='same',
                 dim=3)

    # Multi GPU setup
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.cuda()
    
    
    
    class Trainer:
        def __init__(self,
                     model: torch.nn.Module,
                     device: torch.device,
                     criterion: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     training_DataLoader: torch.utils.data.Dataset,
                     validation_DataLoader: torch.utils.data.Dataset = None,
                     lr_scheduler: torch.optim.lr_scheduler = None,
                     epochs: int = 100,
                     epoch: int = 0,
                     notebook: bool = False
                     ):
            self.model = model
            self.criterion = criterion
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
            self.training_DataLoader = training_DataLoader
            self.validation_DataLoader = validation_DataLoader
            self.device = device
            self.epochs = epochs
            self.epoch = epoch
            self.notebook = notebook
            self.training_loss = []
            self.validation_loss = []
            self.learning_rate = []
            self.training_dice = []
            self.validation_dice = []



        def run_trainer(self):
            early_stopping = EarlyStoppingMetric(patience=patience_rate, verbose=True, delta=1e-4, path='models/UNet-model-random.pt')
            if self.notebook:
                from tqdm.notebook import tqdm, trange
            else:
                from tqdm.auto import tqdm, trange
            progressbar = trange(self.epochs, desc='Progress')
            for i in progressbar:
                print(i)
                """Epoch counter"""
                self.epoch += 1  # epoch counter
                """Training block"""
                self._train()
                """Validation block"""
                if self.validation_DataLoader is not None:
                    self._validate()
                    early_stopping(self.valid_DSC, model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                """Learning rate scheduler block"""
                if self.lr_scheduler is not None:
                    if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__=='ReduceLROnPlateau':
                        self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                    else:
                        self.lr_scheduler.batch()  # learning rate scheduler step
            return self.training_loss, self.validation_loss, self.learning_rate, self.training_dice, self.validation_dice

        
        
        
        
        
        
        
        
        def _train(self):
            if self.notebook:
                from tqdm.notebook import tqdm, trange
            else:
                from tqdm import tqdm, trange
            self.model.train()  # train mode
            train_losses = []  # accumulate the losses here
            train_dice = []
            batch_iter = tqdm(enumerate(self.training_DataLoader), 'Training', total=len(self.training_DataLoader),
                              leave=False)
            for i, (x, y) in batch_iter:
                input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
                self.optimizer.zero_grad()  # zerograd the parameters
                out = self.model(input)  # one forward pass
                prediction = torch.sigmoid(out)
                prediction = prediction.cpu().data.numpy() 
                prediction = (prediction >=0.5).astype(np.float32)    
                targetnp = target.cpu().numpy()
                dice = calculate_dice(prediction, targetnp)
                train_dice.append(dice)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                train_losses.append(loss_value)
                loss.backward()  # one backward pass
                self.optimizer.step()  # update the parameters
                batch_iter.set_description(f'Training: (loss {loss_value:.4f})')  # update progressbar
            self.training_loss.append(np.mean(train_losses))
            self.training_dice.append(np.mean(train_dice))
            self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
            mean_dice = np.mean(train_dice)
            print(' Train loss:', np.mean(train_losses), ' DICE:', mean_dice)
            batch_iter.close()




        def _validate(self):
            if self.notebook:
                from tqdm.notebook import tqdm, trange
            else:
                from tqdm import tqdm, trange
            self.model.eval()  # evaluation mode
            valid_losses = []  # accumulate the losses here
            val_dice = []
            batch_iter = tqdm(enumerate(self.validation_DataLoader), 'Validation', total=len(self.validation_DataLoader),
                              leave=False)
            for i, (x, y) in batch_iter:
                input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
                with torch.no_grad():
                    out = self.model(input)
                    prediction = torch.sigmoid(out)
                    prediction = prediction.cpu().data.numpy() 
                    prediction = (prediction >=0.5).astype(np.float32)
                    targetnp = target.cpu().numpy()
                    dice = calculate_dice(prediction, targetnp)
                    val_dice.append(dice)
                    loss = self.criterion(out, target)
                    loss_value = loss.item()
                    valid_losses.append(loss_value)
                    batch_iter.set_description(f'Validation: (loss {loss_value:.4f})')

            self.validation_loss.append(np.mean(valid_losses))
            self.validation_dice.append(np.mean(val_dice))
            mean_dice = np.mean(val_dice)
            print(' val loss:', np.mean(valid_losses), ' DICE:', mean_dice)
            self.valid_LL = np.mean(valid_losses)
            self.valid_DSC = np.mean(val_dice)
            batch_iter.close()
        

    patience_rate = 40
    criterion = DiceLoss(sigmoid=True, reduction='none', batch=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    transformations = transforms.Compose([transforms.RandomFlip(axes=('LR',), flip_probability= 0.001)])
    train_dataset = RelapseDataset(h5file_train, num_samples=1, transform=transformations)
    val_dataset = RelapseDataset(h5file_val, num_samples=1, transform=None)
    
    
    dataloader_training = DataLoader(train_dataset, batch_size=2, num_workers=12, shuffle=True)
    dataloader_validation = DataLoader(val_dataset, batch_size=2, num_workers=12, shuffle=True)
    print(len(train_dataset),len(val_dataset))

    
    trainer = Trainer(model=model,
                      device=device,
                      criterion=criterion,
                      optimizer=optimizer,
                      training_DataLoader=dataloader_training,
                      validation_DataLoader=dataloader_validation,
                      lr_scheduler=None,
                      epochs=1000,
                      epoch=0,
                      notebook=False)
    

    training_losses, validation_losses, lr_rates, training_dice, validation_dice = trainer.run_trainer()
    print("Final result (Val DSC) : ", np.max(validation_dice))
    torch.save([training_losses, validation_losses, lr_rates, training_dice, validation_dice], 'stats/trainer_stats_random.pt')
    
  
    
    
if __name__ == '__main__':
    try:
        main()
    except Exception as exception:
        print(exception)
        raise
        
