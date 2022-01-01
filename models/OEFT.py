import os
import sys
import random
import numpy as np
import cv2

from models.encoder import vggnet
from models.corrnet import CorrNet
from models.decoder.vggDecoder import vggDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import *

class OEFT(nn.Module):
    def __init__(self, args, pretrained=False):
        super().__init__()
        self.args = args

        if "cuda" in args.device:
            self.vggMean = torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape(1,3,1,1) ).type(torch.cuda.FloatTensor).to(args.device)
            self.vggStd =  torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape(1,3,1,1) ).type(torch.cuda.FloatTensor).to(args.device)
        else:
            self.vggMean = torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape(1,3,1,1) ).type(torch.FloatTensor).to(args.device)
            self.vggStd =  torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape(1,3,1,1) ).type(torch.FloatTensor).to(args.device)
        self.vggMean.requires_grad = False
        self.vggStd.requires_grad  = False

        """ define corrNet(get correlation and warped features&imgs) """
        pool2None= [] #["4", "5"] # 4,5=32 (32//4) // 5=16 (32//2)// ''=8 (32//1)
        self.corrNet = CorrNet(args=self.args, pretrained=pretrained, scale=32//8)

        self.pretrained_vgg19 = self.corrNet.encoder # define encoder

        """ decoder zero initialization """
        self.decoder = vggDecoder(self.args) # define decoder

    def vggDeNorm(self, x):
        x = (x*self.vggStd)+self.vggMean
        return x

    def vggNorm(self, x):
        x = (x-self.vggMean)/self.vggStd
        return x

    def forward(self, styles, contents, prev_warped_feature=None, step=0):
        """
        # shape(256,128,64,32,16)
        content loss(c_features_dict, s2c_img_decs_feats_dict): r34
        style loss(s_features_dict, s2c_img_decs_feats_dict): r12, r22, r34, r44, r54
        """

        if not self.args.warpMv:
            del prev_warped_feature
            prev_warped_feature = None

        """ get warped feature,image and decoded output """
        """
            c_features_dict  : content feature dictionary for nce loss
            s_features_dict  : content feature dictionary for NN style loss
            warped_s2c_feats : for decoder input
            warped_s2c_imgs  : for residual connection
            warped_s2c2s_feats, warped_c2s2c_feats : for cycle consistency loss
            warped_s2c2s_imgs,  warped_c2s2c_imgs  : for cycle consistency loss(warped image residual connection)
        """
        c_features_dict, s_features_dict, warped_s2c_feats, warped_s2c_imgs , warped_s2c2s_feats, warped_c2s2c_feats, warped_s2c2s_imgs, warped_c2s2c_imgs = self.corrNet(contents, styles, prev_c_feat = prev_warped_feature, step=step)


        """ get keys """
        keys = list(c_features_dict.keys())
        depth = len(keys)

        s2c_img_decs = self.decoder(warped_s2c_feats)

        if self.args.warpRes:
            s2c_img_decs = ( s2c_img_decs *(1-self.args.res_wt) + F.interpolate(warped_s2c_imgs, (256,256), mode='bilinear')*(self.args.res_wt)     )#*1/2 # index2=64x64 , [64(256), 64(128), 64, 32, 16]
        else:
            s2c_img_decs = ( s2c_img_decs )

        """ for reconstruction loss """
        s2c2s_img_decs = self.decoder(  warped_s2c2s_feats  )
        c2s2c_img_decs = self.decoder(  warped_c2s2c_feats )


        """ residual connection """
        if self.args.warpRes:
            s2c2s_img_decs = (s2c2s_img_decs *(1-self.args.res_wt) + F.interpolate(warped_s2c2s_imgs, (256,256), mode='bilinear')*(self.args.res_wt)     )
            c2s2c_img_decs = (c2s2c_img_decs *(1-self.args.res_wt) + F.interpolate(warped_c2s2c_imgs, (256,256), mode='bilinear')*(self.args.res_wt)     )
        else:
            s2c2s_img_decs = ( s2c2s_img_decs )
            c2s2c_img_decs = ( c2s2c_img_decs )        

        """ pass the decoded image to the encoder start!!!!!!!!"""  ## 여기부터!!
        s2c_img_decs_feats_dict = self.pretrained_vgg19(s2c_img_decs) # type: dict        
         
        """ !!!!!!!!!!!!!!!! get loss !!!!!!!!!!!!!!!!!"""
        loss_recon     = 0.
        loss_content   = 0.
        loss_style     = 0.
        loss_wta_style = 0.
        loss_cycle     = 0.

        """  cycle loss """
        loss_cycle = 0.
        if self.args.cycle:
            loss_cycle += F.mse_loss( s2c2s_img_decs, styles) * self.args.cycle_wt
            loss_cycle += F.mse_loss( c2s2c_img_decs, contents) * self.args.cycle_wt


        """ compute content loss """
        patch_id_1 =patchLocalizer(c_features_dict[keys[0]].shape)
        patch_id_2 =patchLocalizer(c_features_dict[keys[1]].shape)
        patch_id_3 =patchLocalizer(c_features_dict[keys[2]].shape)

        patch_id_4 =patchLocalizer(c_features_dict[keys[3]].shape)
        patch_id_5 =patchLocalizer(c_features_dict[keys[4]].shape)

        patches_list_1 = patchSampler(self.args, c_features_dict[keys[0]], s2c_img_decs_feats_dict[keys[0]], patch_id_1)
        patches_list_2 = patchSampler(self.args, c_features_dict[keys[1]], s2c_img_decs_feats_dict[keys[1]], patch_id_2)
        patches_list_3 = patchSampler(self.args, c_features_dict[keys[2]], s2c_img_decs_feats_dict[keys[2]], patch_id_3)
        #patches_list_4 = patchSampler(self.args, c_features_dict[keys[3]], s2c_img_decs_feats_dict[keys[3]], patch_id_4)
        #patches_list_5 = patchSampler(self.args, c_features_dict[keys[4]], s2c_img_decs_feats_dict[keys[4]], patch_id_5)

        loss_nce_content_1 = NCELoss(self.args, patches_list_1[0], patches_list_1[1]) 
        loss_nce_content_2 = NCELoss(self.args, patches_list_2[0], patches_list_2[1])
        loss_nce_content_3 = NCELoss(self.args, patches_list_3[0], patches_list_3[1])
        loss_nce_content_4 = 0.#NCELoss(self.args, patches_list_4[0], patches_list_4[1])
        loss_nce_content_5 = 0.#NCELoss(self.args, patches_list_5[0], patches_list_5[1])
        
        loss_nce_content = ( self.args.nce_wt[0]*loss_nce_content_1 + 
                             self.args.nce_wt[1]*loss_nce_content_2 +
                             self.args.nce_wt[2]*loss_nce_content_3 +
                             self.args.nce_wt[3]*loss_nce_content_4 +
                             self.args.nce_wt[4]*loss_nce_content_5
                        )

        """ compute style loss """
        """ compute non-parametric loss """
        patches_style_list_1 = patchSampler(self.args, s_features_dict[keys[0]], s2c_img_decs_feats_dict[keys[0]], patch_id_1,  is_unfold=True, kernel_size=3)
        patches_style_list_2 = patchSampler(self.args, s_features_dict[keys[1]], s2c_img_decs_feats_dict[keys[1]], patch_id_2,  is_unfold=True, kernel_size=3)
        patches_style_list_3 = patchSampler(self.args, s_features_dict[keys[2]], s2c_img_decs_feats_dict[keys[2]], patch_id_3,  is_unfold=True, kernel_size=3)

        patches_style_list_4 = patchSampler(self.args, s_features_dict[keys[3]], s2c_img_decs_feats_dict[keys[3]], patch_id_4,  is_unfold=True, kernel_size=3)
        patches_style_list_5 = patchSampler(self.args, s_features_dict[keys[4]], s2c_img_decs_feats_dict[keys[4]], patch_id_5)

        loss_wta_style_1  = WTAStyleLoss(self.args, patches_style_list_1[0], patches_style_list_1[1], stylePatchSample=True) 
        loss_wta_style_2  = WTAStyleLoss(self.args, patches_style_list_2[0], patches_style_list_2[1], stylePatchSample=True)
        loss_wta_style_3  = WTAStyleLoss(self.args, patches_style_list_3[0], patches_style_list_3[1], stylePatchSample=True )
        loss_wta_style_4  = WTAStyleLoss(self.args, patches_style_list_4[0], patches_style_list_4[1], stylePatchSample=True )
        loss_wta_style_5  = WTAStyleLoss(self.args, patches_style_list_5[0], patches_style_list_5[1], stylePatchSample=True )
        
        loss_wta_style =  ( self.args.nns_wt[0] * loss_wta_style_1 + 
                            self.args.nns_wt[1] * loss_wta_style_2 +
                            self.args.nns_wt[2] * loss_wta_style_3 + 
                            self.args.nns_wt[3] * loss_wta_style_4 +
                            self.args.nns_wt[4] * loss_wta_style_5
                          )

        """ final loss """
        loss_style   = loss_wta_style
          
        loss_content = loss_nce_content
        loss = (1-self.args.content_style_wt)*loss_content + (self.args.content_style_wt)*loss_style + loss_cycle

        loss_dict = {"L_style": (self.args.content_style_wt)*loss_style, "L_content": (1-self.args.content_style_wt)*loss_content, "L_cycle": loss_cycle}
        nce_dict = {"nce1":self.args.nce_wt[0]*loss_nce_content_1 , "nce2":self.args.nce_wt[1]*loss_nce_content_2, "nce3":self.args.nce_wt[2]*loss_nce_content_3}
        nns_dict = {"nns1":self.args.nns_wt[0] *loss_wta_style_1 , "nns2":self.args.nns_wt[1] *loss_wta_style_2, "nns3":self.args.nns_wt[2] *loss_wta_style_3
                     , "nns4":self.args.nns_wt[3] *loss_wta_style_4, "nns5":self.args.nns_wt[4] *loss_wta_style_5}
        return s2c_img_decs_feats_dict, warped_s2c_imgs, s2c_img_decs, loss, loss_dict, nce_dict, nns_dict