''''
Author: Ahmed Zgaren
Date: February 2024
Script to test the model on the validation set of FSC147
'''


from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
from sklearn.metrics import  mean_absolute_error as mae
from sklearn.metrics import  mean_squared_error as rmse
import argparse

from model.trcount import TrCount
from utils.utils import CustomImageDataset
from utils.utils import *

# Set seed for NumPy
import numpy as np
np.random.seed(42)

# Set seed for Torch
torch.manual_seed(42)


def validate():
    criterion = torch.nn.L1Loss()
    
    countfinal = 0
    pred = torch.tensor(torch.zeros((1))).to(device)
    lab = torch.tensor(torch.zeros((1))).to(device)
    model.eval()
    for vlocal_batch, vlocal_labels in val_dataloader:
        vlocal_batch = vlocal_batch.permute((0,3,1,2)).float()/255
        vlocal_batch, vlocal_labels = vlocal_batch.to(device), vlocal_labels.to(device)
        
        with torch.no_grad():
            count = model(vlocal_batch)
            countfinal += criterion(count.squeeze(1),vlocal_labels)
            pred = torch.cat([pred,count.squeeze(1)], dim = 0)
            lab = torch.cat([lab,vlocal_labels])
    
    rmseror =  rmse(pred[1:].cpu().detach().numpy(), lab[1:].cpu().detach().numpy(), squared= False)
            
    print('L1loss = ', countfinal.cpu().detach().numpy()/ len(val_dataloader) )
    print('RMSError = ', rmseror)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAVE Demo code")
    parser.add_argument("-valid", "--valid", type=bool, default=False, help="Validation set")
    parser.add_argument("-test", "--test", type=bool, default=False, help="Test set")
    parser.add_argument("-vcoco", "--val-coco", type=bool, default=False, help="Coco validation set")
    parser.add_argument("-tcoco", "--test-coco", type=bool, default=False, help="Coco test set")

    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    paramsv = {'batch_size': 1,
      'shuffle': False}
    
    model = TrCount(d_model=768)
    model_path = r"pretrained\\save.pt"
    model.load_state_dict(torch.load(model_path, map_location= device))
    model.to(device)

    loss_fn = torch.nn.L1Loss()

    if args.valid:
        val_d = CustomImageDataset('data/valid.csv', 'data/images')
    elif args.test:
        val_d = CustomImageDataset('data/test.csv', 'data/images/')
    elif args.val_coco:
        val_d = CustomImageDataset('data/val_coco.csv', 'data/images/')
    elif args.test_coco:
        val_d = CustomImageDataset('data/test_coco.csv', 'data/images/')
    val_dataloader = DataLoader(val_d, **paramsv)
    
    validate()

    