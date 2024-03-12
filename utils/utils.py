import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
import torch


import torchvision
from PIL import Image
import numpy as np
import imageio

from datetime import datetime
import wandb
from torch.utils.tensorboard.writer import SummaryWriter
#from ultralytics import YOLO
import cv2
from pathlib import Path
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.data.dataloaders.stream_loaders import (LOADERS, LoadImages, LoadPilAndNumpy, LoadScreenshots,
                                                              LoadStreams, LoadTensor, SourceTypes, autocast_list)

from ultralytics.yolo.data.augment import LetterBox, classify_transforms




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def preprocess(im):
    """Prepares input image before inference.

    Args:
        im (torch.Tensor | List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
    """
    if not isinstance(im, torch.Tensor):
        im = np.stack(pre_transform(im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)
    # NOTE: assuming im with (b, 3, h, w) if it's a tensor
    
    img = im.to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    return img
def pre_transform(im):
        """Pre-tranform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        
        return [LetterBox()(image=x) for x in im]
def load_inference_source(source=None, imgsz=640, vid_stride=1):
    """
    Loads an inference source for object detection and applies necessary transformations.

    Args:
        source (str, Path, Tensor, PIL.Image, np.ndarray): The input source for inference.
        imgsz (int, optional): The size of the image for inference. Default is 640.
        vid_stride (int, optional): The frame interval for video sources. Default is 1.

    Returns:
        dataset (Dataset): A dataset object for the specified input source.
    """
    source, webcam, screenshot, from_img, in_memory, tensor = check_source(source)
    source_type = source.source_type if in_memory else SourceTypes(webcam, screenshot, from_img, tensor)

    # Dataloader
    if tensor:
        dataset = LoadTensor(source)
    elif in_memory:
        dataset = source
    elif webcam:
        dataset = LoadStreams(source, imgsz=imgsz, vid_stride=vid_stride) # type: ignore
    elif screenshot:
        dataset = LoadScreenshots(source, imgsz=imgsz)
    elif from_img:
        dataset = LoadPilAndNumpy(source, imgsz=imgsz)
    else:
        dataset = LoadImages(source, imgsz=imgsz, vid_stride=vid_stride)

    # Attach source types to the dataset
    setattr(dataset, 'source_type', source_type)

    return dataset
def check_source(source):
    """Check source type and return corresponding flag values."""
    webcam, screenshot, from_img, in_memory, tensor = False, False, False, False, False
    if isinstance(source, (str, int, Path)):  # int for local usb camera
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('https://', 'http://', 'rtsp://', 'rtmp://'))
        webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
        screenshot = source.lower().startswith('screen')
        if is_url and is_file:
            source = check_file(source)  # download
    elif isinstance(source, tuple(LOADERS)):
        in_memory = True
    elif isinstance(source, (list, tuple)):
        source = autocast_list(source)  # convert all list elements to PIL or np arrays
        from_img = True
    elif isinstance(source, (Image.Image, np.ndarray)):
        from_img = True
    elif isinstance(source, torch.Tensor):
        tensor = True
    else:
        raise TypeError('Unsupported image type. For supported types see https://docs.ultralytics.com/modes/predict')

    return source, webcam, screenshot, from_img, in_memory, tensor
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, size = None, transform=None, target_transform=None):
        self.img_labels0 = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        if size != None:
            self.img_labels = self.img_labels0.iloc[:size] #control the size of data for training
        else:
            self.img_labels = self.img_labels0

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        
        # dataset = load_inference_source(img_path)
        # p, im, c,s = next(iter(dataset))
        im0 = cv2.imread(img_path)  # BGR
        im0 =  cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        shape = im0.shape
        if shape != (640, 640):
            self.im,ratio,(dw,dh) = letterbox(im0)
        else:
            self.im = im0
        self.label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(self.im)
            
            return image, label
        if self.target_transform:
            label = self.target_transform(label)
        return self.im, self.label
    
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return im, ratio, (dw, dh)