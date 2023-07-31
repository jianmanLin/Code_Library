import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import argparse

from iresnet import *
from torch.nn.utils import spectral_norm
from torchvision import models, utils

sys.path.append("/data2/JM/code/code_library/Arcface")
from arcface.iresnet import *

class BlendFace(nn.Module):
    def __init__(self, opts):
        super().__init__()
        device='cuda' if torch.cuda.is_available() else 'cpu'
        self.blendface = iresnet100(pretrained=False, fp16=False)
        self.blendface.load_state_dict(torch.load(opts.blendface_model_path, map_location='cpu'))
        self.blendface= self.blendface.eval()
    def forward(self, x):
        
        return self.blendface(x)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--blendface_model_path', type=str, default="/data2/JM/code/code_library/pretrain_model/blendface.pt")
    opts = parser.parse_args()
    arcface_encoder = BlendFace(opts=opts)
    x = torch.randn((1, 3, 112, 112))
    result = arcface_encoder(x)
    print(result.shape)
    
