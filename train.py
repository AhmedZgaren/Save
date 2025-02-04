'''
Author: Ahmed Zgaren
Date: Feb 2024
'''

import wandb
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch

from model.trcount import TrCount
from utils.utils import CustomImageDataset
from utils.utils import *

# Set seed for NumPy
import numpy as np
np.random.seed(42)

# Set seed for Torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior for CUDA operations
torch.backends.cudnn.benchmark = True

def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    # Here, we use tqdm(train_dataloader) instead of
    # iter(training_loader) so that we can track the batch
    for local_batch, local_labels in tqdm(train_dataloader):
        
        #Preprocess the batch and load to GPU
        local_batch = local_batch.permute((0,3,1,2)).float() / 255
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        
        #initialize the optimizer
        optimizer.zero_grad()
        
        # Make predictions for this batch
        outputs = model(local_batch)
        
        # Compute the loss and its gradients
        loss = loss_fn(outputs.squeeze(1), local_labels)
        
        #one backward step over the model
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
    print('loss')

    return running_loss / len(train_dataloader)

def validate():
    running_vloss = 0.0
    # #We don't need gradients on to do reporting
    # #Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for vlocal_batch, vlocal_labels in tqdm(val_dataloader):
        
            #preprocess the data 
            vlocal_batch = vlocal_batch.permute((0,3,1,2)).float() / 255
            vlocal_batch, vlocal_labels = vlocal_batch.to(device), vlocal_labels.to(device)

            voutputs = model(vlocal_batch)
            vloss = loss_fn(voutputs.squeeze(1), vlocal_labels)
            running_vloss += vloss

    return running_vloss / len(val_dataloader)  


if __name__ == "__main__":
    
    #if you want to use a different gpu change cuda:0 to cuda:2 or different number
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    
    # Model parameters
    d_model = 768 #dimension of the embedding

    enc_in = 256 #Output channel dimension of the backbone

    epochs = 100
    lr = 0.0001393 # learning rate
    batchsize = 32
    
    #parameters for validation dataloader
    paramsv = {'batch_size': 1,
      'shuffle': True,
      'num_workers': 1}
    #parameters for training dataloader
    params = {'batch_size': batchsize,
          'shuffle': True,
          'num_workers': 32}
    
    #WandB initialization for training progress log
    wandb.login()
    wandb.init()
   
    #Directory to save the best and last weights of the trained model 
    SAVEDIR = f"{wandb.run.dir}/chkpt/"
    if  not os.path.isdir(SAVEDIR):
        os.mkdir(SAVEDIR)

    # Load Model  architecture

    model = TrCount(d_model, enc_in)
    model.to(device)

    # Dataset preparation
    
    tr_d = CustomImageDataset('data/train.csv', 'data/images/') 
    val_d = CustomImageDataset('data/valid.csv', 'data/images/')
    
    # train and valid dataloader
    train_dataloader = DataLoader(tr_d, **params)
    val_dataloader = DataLoader(val_d, **paramsv)
    

    

    #Optimizer and Loss function
    loss_fn = torch.nn.L1Loss()
    optimizer =  torch.optim.Adam(model.parameters(), lr=lr)

    epoch_number = 0 #initialize the epoch number
    
    best_vloss = 1_000_000.
    
    # training loop        
    for epoch in range(1, epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        # #Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss = train_one_epoch()
        
        # #Make sure that gradient is off, and do a pass over the validation data
        model.eval()
        avg_vloss = validate()

        #log the model with wandb
        wandb.log({"tr_loss": avg_loss, "val_loss": avg_vloss})
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # #Track best performance, and save the model's state

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(SAVEDIR, 'best.pt')
            torch.save(model.state_dict(), model_path)

            

        # Save last model
        model_path = os.path.join(SAVEDIR, 'last.pt')
        torch.save(model.state_dict(), model_path)

        epoch_number += 1