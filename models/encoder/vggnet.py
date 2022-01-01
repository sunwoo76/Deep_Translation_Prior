# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torchvision.models.vgg as torch_vgg
from torchvision.models.vgg import make_layers

model_urls = {  'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
                'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(torch_vgg.VGG):
    def __init__(self, lastBlock, num_classes, *args, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)
        self.lastBlock=lastBlock
        self.avgpool2D = nn.AvgPool2d(kernel_size=2, stride=2)

    def modify(self, pool2None=[]):
        remove_layers=[]
        filter_layers = lambda x: [l for l in x if getattr(self, l) is not None]
        """
        # Set stride of layer3 and layer 4 to 1 (from 2)
        setattr(self, self.features[4], None) # first pool to None
        setattr(self, self.features[9], None) # second pool to None
        setattr(self, self.features[18], None) # third pool to None
        """
        if "4" in pool2None:
            self.features[27] = None # fourth pool to None
            if "5" in pool2None:
                self.features[36] = None # fifth pool to None

        # Remove extraneous layers
        remove_layers += ['avgpool', 'classifier']
        for layer in filter_layers(remove_layers):
            setattr(self, layer, None)
        
    def forward(self, x):
        #print("!! networks.dir(correspondence.py, VGG19_feature_color_torchversione class): function forward")
        #print("@@@@@ x.shape: ", x.shape)
        outputs = {}
        """
        out = {}

        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        """
        # input shape 256
        
        r11 = self.features[1]( self.features[0](x) )
        r12 = self.features[3]( self.features[2](r11) )
        outputs['r12'] = r12
        #p1 = self.avgpool2D(r12)
        p1 = r12 if self.features[4] is None else self.features[4](r12)
        
        # input shape 128
        
        r21 = self.features[6]( self.features[5](p1) )
        r22 = self.features[8]( self.features[7](r21) )
        outputs['r22'] = r22
        #p2 = self.avgpool2D(r22)
        p2 = r22 if self.features[9] is None else self.features[9](r22) 

        
        # input shape64
        r31 = self.features[11]( self.features[10](p2) )
        r32 = self.features[13]( self.features[12](r31) )
        r33 = self.features[15]( self.features[14](r32) )
        r34 = self.features[17]( self.features[16](r33) )
        outputs['r34'] = r34
        #p3 = self.avgpool2D(r34)
        p3 = r34 if self.features[18] is None else self.features[18](r34) 

        
        # input shape 32
        r41 = self.features[20]( self.features[19](p3) )
        r42 = self.features[22]( self.features[21](r41) )
        r43 = self.features[24]( self.features[23](r42) )
        r44 = self.features[26]( self.features[25](r43) )
        outputs['r44'] = r44
        #p4 = self.avgpool2D(r44)
        p4 = r44 if self.features[27] is None else self.features[27](r44) 

        
        # input shape 16
        r51 = self.features[29]( self.features[28](p4) )
        r52 = self.features[31]( self.features[30](r51) )
        r53 = self.features[33]( self.features[32](r52) )
        r54 = self.features[35]( self.features[34](r53) )
        outputs['r54'] = r54
        #p5 = self.avgpool2D(r54)
        p5  = r54 if self.features[36] is None else  self.features[36](r54) 

        
        return outputs
        #return [out[key] for key in out_keys] # key 값에 해당하는 layer의 출력값을 받는다.      


def _vgg(arch, cfg, pretrained, progress, batch_norm, lastBlock, num_classes, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    
    model = VGG(lastBlock, num_classes, make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    #print(model.state_dict)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def vggnet19(lastBlock=5, pretrained=False, progress=True, **kwargs):
    return _vgg('vgg19', cfg='E', pretrained=pretrained, progress=progress, batch_norm=False,
                 lastBlock=lastBlock, num_classes=1000, **kwargs)
