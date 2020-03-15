import torch
from torch import nn
import numpy as np
import time
import typing
import collections


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        
        
        #Added:
        #Feature Extractor 0:
        #NOTE: control that output_feature_size is correct
        self.feature_extractor_0 = nn.Sequential(
            nn.Conv2d(image_channels, 32,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            
            nn.Conv2d(32, 64,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            
            nn.Conv2d(64, 64,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.ReLU(),
            
            #Extract this to out_feature[0]
            nn.Conv2d(64, self.output_channels[0],kernel_size = 3,stride = 2,padding = 1,bias = True),
        )
        
        #Feature Extractor 1:
        self.feature_extractor_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[0], 128,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.ReLU(),
            nn.Conv2d(128, self.output_channels[1],kernel_size = 3,stride = 2,padding = 1,bias = True),
        )
        
        #Feature Extractor 2:
        self.feature_extractor_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[1], 256,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.ReLU(),
            nn.Conv2d(256, self.output_channels[2],kernel_size = 3,stride = 2,padding = 1,bias = True),
        )
        
        #Feature Extractor 3:
        self.feature_extractor_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[2], 128,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.ReLU(),
            nn.Conv2d(128, self.output_channels[3],kernel_size = 3,stride = 2,padding = 1,bias = True),
        )
        
        #Feature Extractor 4:
        self.feature_extractor_4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[3], 128,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.ReLU(),
            nn.Conv2d(128, self.output_channels[4],kernel_size = 3,stride = 2,padding = 1,bias = True),
        )
        
        #Feature Extractor 5:
        self.feature_extractor_5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.output_channels[4], 128,kernel_size = 3,stride = 1,padding = 1,bias = True),
            nn.ReLU(),
            nn.Conv2d(128, self.output_channels[5],kernel_size = 3,stride = 1,padding = 0,bias = True),
        )
    
    
    
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[4], 3, 3),
            shape(-1, output_channels[5], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        #batch_size = x.shape[0]
        out_features = []
        out_features.append(self.feature_extractor_0(x))
        out_features.append(self.feature_extractor_1(out_features[0]))
        out_features.append(self.feature_extractor_2(out_features[1]))
        out_features.append(self.feature_extractor_3(out_features[2]))
        out_features.append(self.feature_extractor_4(out_features[3]))
        out_features.append(self.feature_extractor_5(out_features[4]))
        
        actual_feature_map_size = [38,19,10,5,3,1]
        for idx, feature in enumerate(out_features):
            expected_shape = (output_channels[idx], actual_feature_map_size[idx], actual_feature_map_size[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

