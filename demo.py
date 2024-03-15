'''
Author: 
Date: Feb 2024
'''
import torch
import cv2
import torchvision.transforms as transforms
import argparse

from model.trcount import TrCount
from utils.utils import *





parser = argparse.ArgumentParser(description="SAVE Demo code")
parser.add_argument("-i", "--input-image", type=str, required=True, help="/Path/to/input/image/file/")

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = TrCount(d_model=768)
model_path = r"pretrained\\save.pt"
model.load_state_dict(torch.load(model_path, map_location= device))
model.to(device)

start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

im0 = cv2.imread(args.input_image)  # BGR
im0 =  cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
shape = im0.shape
if shape != (640, 640):
    im,ratio,(dw,dh) = letterbox(im0)
# Define a transform to convert the image to tensor
transform = transforms.ToTensor()

# Convert the image to PyTorch tensor
im1 = transform(im)
im1 = im1.unsqueeze(0)
input = im1.float()
input = input.to(device)

model.eval()

count = model(input)


print('Count :', count[0][0].cpu().detach().numpy())
