import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import vggnet
from models.utils import *

class CorrNet(nn.Module):
    def __init__(self, args, pretrained=False, pool2None=[], scale=1):
        super().__init__()
        self.args = args
        self.use_gpu = True
        self.scale = scale
        self.iscenterL2Norm = True
        self.relu = nn.ReLU()

        """ choose backcbone """
        self.encoder = vggnet.vggnet19(pretrained=pretrained, lastBlock=3)
        self.encoder.modify(pool2None=pool2None)


    def getWarpedTensor(self, target_features, correlation_map):
        b, d, h, w = target_features.shape
        """ resolution 128, 256 is too big to calculate correlation map. : cause memory issue. """
        if h > 64:
            h=64
            w=64
            target_features = F.interpolate(target_features, (h,w), mode="bilinear")
        
        """ correlationdp temperature. controlling sharpness """
        softtmax_correlation_map = F.softmax(correlation_map/self.args.corr_temp, dim=-1)
        
        """ warping """
        warped_t2s_output = torch.bmm(softtmax_correlation_map, target_features.view(b,d,h*w).permute(0,2,1)).permute(0,2,1) # (b,s*s,d) -> (b, d. s*s)
        warped_t2s_output = warped_t2s_output.view(b,d,h,w)
        # output shape: (b,d,h,w)
        return warped_t2s_output
    
    def getCorrelation(self, source_features, target_features, kernel_size=3, is_unfold=True):
        b, d, h, w = source_features.shape

        """ resolution 128, 256 is too big to calculate correlation map. : cause memory issue. """
        if h > 64:
            h=64
            w=64
            source_features = F.interpolate(source_features, (h,w), mode="bilinear")
            target_features = F.interpolate(target_features, (h,w), mode="bilinear")
        
        """ unfold. 3x3 neural patch """
        if is_unfold:
            c_patch_features = F.unfold(source_features, kernel_size=kernel_size, padding=int(kernel_size//2))
            t_patch_features = F.unfold(target_features, kernel_size=kernel_size, padding=int(kernel_size//2))
        else:
            c_patch_features = source_features
            t_patch_features = target_features

        """ normalize and change the shape """
        if self.iscenterL2Norm:
            s_norm = centerL2Norm(c_patch_features, dim=1).view(b,-1,h*w) # (b,d,h*w)
            t_norm = centerL2Norm(t_patch_features, dim=1).view(b,-1,h*w) # (b,d,h*w)
        else:
            s_norm = normalize(c_patch_features, dim=1).view(b,-1,h*w) # (b,d,h*w)
            t_norm = normalize(t_patch_features, dim=1).view(b,-1,h*w) # (b,d,h*w)
        
        """ style -> content """
        """ corrleation map"""
        correlation_map = torch.bmm( s_norm.permute(0,2,1), t_norm) # shape: (batch, S*S, S*S)
        return correlation_map

    def forward(self, content, style, prev_c_feat=None, step=0):
        # dict.keys() : ['r12', 'r22', 'r34', 'r44', 'r54']
        c_feature_clone = None
        
        if prev_c_feat is None:
            c_features_dict = self.encoder(content) # type: dict (b, dim, s, s)
            s_features_dict = self.encoder(style) # type: dict  (b, dim, s, s)
        else:            
            c_features_dict = self.encoder(content) # type: dict (b, dim, s, s)
            s_features_dict = self.encoder(style) # type: dict  (b, dim, s, s)
            c_feature_clone = c_features_dict[self.args.keys[2]].clone()

            """ feature moving average """
            if prev_c_feat is not None and self.args.warpMv:
                for i, key in enumerate(c_features_dict.keys()):
                    #if i<3:
                    if i==2:
                        c_features_dict[key] = (1-self.args.mv_wt)*c_features_dict[key] + (self.args.mv_wt)*(prev_c_feat[key])

        keys = list(c_features_dict.keys())
        depth = len(keys)

        """ down sampling """
        down_c = F.adaptive_avg_pool2d(content, 256//self.scale) # imgSize/downScale, (b, 3, s, s)
        down_s = F.adaptive_avg_pool2d(style, 256//self.scale) # imgSize/downScale, (b, 3, s, s)     
        
        s2c_corr = self.getCorrelation( c_features_dict[keys[2]], s_features_dict[keys[2]]) # get correlation map
        
        warped_s2c_feats = self.getWarpedTensor(s_features_dict[keys[2]], s2c_corr) # get warped feature (style to content)
        warped_c2s_feats = self.getWarpedTensor(c_features_dict[keys[2]], s2c_corr.permute(0,2,1)) # get warped feature (content to style)
    
        """ get warped image in layer 3 """
        # (b,s*s,3)->(b,3,s*s)->(b,3,s,s)
        b,d,h,w = c_features_dict[keys[2]].shape

        warped_s2c_imgs = self.getWarpedTensor( down_s, s2c_corr  )
        warped_c2s_imgs = self.getWarpedTensor( down_c, s2c_corr.permute(0,2,1).contiguous()  )

        warped_s2c2s_feats = self.getWarpedTensor( warped_s2c_feats, s2c_corr.permute(0,2,1).contiguous())
        warped_s2c2s_imgs = self.getWarpedTensor( warped_s2c_imgs, s2c_corr.permute(0,2,1).contiguous())

        warped_c2s2c_feats = self.getWarpedTensor( warped_c2s_feats, s2c_corr)
        warped_c2s2c_imgs = self.getWarpedTensor( warped_c2s_imgs, s2c_corr)

        if c_feature_clone is not None:
            c_features_dict[keys[2]] = c_feature_clone
        return c_features_dict, s_features_dict, warped_s2c_feats, warped_s2c_imgs, warped_s2c2s_feats, warped_c2s2c_feats, warped_s2c2s_imgs, warped_c2s2c_imgs