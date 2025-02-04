''''
Author: Ahmed Zgaren
Date: February 2024
'''
from torch import nn
from .backbone import YOLOFeatures
from .transformer import Trans

class TrCount(nn.Module):
    '''
    SAVE implementation
    
    '''
    def __init__(self, d_model=768,h = 8, d_ff = 2048, num_layers=1, enc_in = 256, drop = 0.1, visualize = False):
        super(TrCount, self).__init__()

         # Backbone
        self.yolo = YOLOFeatures(r'data\yolov8l.yaml', task = 'detect')
            #Uncomment next line for training from scratch
        #self.yolo.load(r'pretrained\backbone.pt') 
        self.yolo = self.yolo.model
         #freeze yolo parameters
        for name, para in self.yolo.named_parameters(): 
            para.requires_grad = False
         # VEM, SAMM, and CRM
        self.trans = Trans(d_model,h, d_ff, num_layers, enc_in, drop)
        self.vis = visualize

    def forward(self, x):
        out0 = self.yolo(x) 
        out, tmap, cot = self.trans(out0)
        if self.vis:
            return out, tmap, out0, cot #out: count, tmap: output sequence of the SAMM, cot: VEM output map 
        else:
            return out