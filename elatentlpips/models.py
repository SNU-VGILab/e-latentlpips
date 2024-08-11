### Some parts of the code are adapted from: https://github.com/richzhang/PerceptualSimilarity

import torch
import torch.nn as nn
from .augmentations import ada_aug

cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16_Latent': [64, 64, 128, 128, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
}
ada_augpipe = {
    'blit':   dict(xflip=1, rotate90=1, xint=1),
    'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
    'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
    'filter': dict(imgfilter=1),
    'noise':  dict(noise=1),
    'cutout': dict(cutout=1),
    'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
    'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
    'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
    'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
    'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    'bg+cutout': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, cutout=1),
    'bgc+cutout': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, cutout=1),
    'bgcf+cutout': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, cutout=1),
}


########### (Latent) VGG Model ###########
class VGGBase(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGGBase, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def _make_layers(cfg, in_channels=4, norm_type='none', num_groups=32):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if norm_type == 'batch':
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            elif norm_type == 'group':
                layers += [conv2d, nn.GroupNorm(num_groups, v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def VGG16(num_classes=1000):
    return VGGBase(_make_layers(cfg['VGG16'], in_channels=3, norm_type='none'), num_classes)

def VGG16_BN(num_classes=1000):
    return VGGBase(_make_layers(cfg['VGG16'], in_channels=3, norm_type='batch'), num_classes)

def VGG16_GN(num_classes=1000, num_groups=32):
    return VGGBase(_make_layers(cfg['VGG16'], in_channels=3, norm_type='group', num_groups=num_groups), num_classes)

def VGG16_Latent(num_classes=1000, in_channels=4):
    return VGGBase(_make_layers(cfg['VGG16_Latent'], in_channels=in_channels, norm_type='none'), num_classes)

def VGG16_Latent_BN(num_classes=1000, in_channels=4):
    return VGGBase(_make_layers(cfg['VGG16_Latent'], in_channels=in_channels, norm_type='batch'), num_classes)

def VGG16_Latent_GN(num_classes=1000, num_groups=32, in_channels=4):
    return VGGBase(_make_layers(cfg['VGG16_Latent'], in_channels=in_channels, norm_type='group', num_groups=num_groups), num_classes)


########### Latent LPIPS Model ###########
class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def normalize_tensor(in_feat,eps=1e-8):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True) + eps)
    return in_feat/(norm_factor)

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

class LatentLPIPS(nn.Module):
    def __init__(self, backbone_type="VGG16_Latent_GN", backbone_in_channels=4,
                 pretrained_backbone_path=None, pretrained_full_model_path=None, 
                 use_dropout=True, eval_mode=True):
        super(LatentLPIPS, self).__init__()
        
        if backbone_type == "VGG16_Latent":
            backbone = VGG16_Latent(in_channels=backbone_in_channels)
        elif backbone_type == "VGG16_Latent_BN":
            backbone = VGG16_Latent_BN(in_channels=backbone_in_channels)
        elif backbone_type == "VGG16_Latent_GN":
            backbone = VGG16_Latent_GN(in_channels=backbone_in_channels)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")
        
        if pretrained_backbone_path is not None:
            if pretrained_backbone_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                backbone.load_state_dict(load_file(pretrained_backbone_path))
            else:
                backbone.load_state_dict(torch.load(pretrained_backbone_path, map_location='cpu'))
        
        for param in backbone.parameters():
            param.requires_grad = False

        self.slice1 = backbone.features[0:6]
        self.slice2 = backbone.features[6:12]
        self.slice3 = backbone.features[12:21]
        self.slice4 = backbone.features[21:31]
        self.slice5 = backbone.features[31:41]
        self.slices = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]

        self.linear1 = NetLinLayer(64, use_dropout=use_dropout)
        self.linear2 = NetLinLayer(128, use_dropout=use_dropout)
        self.linear3 = NetLinLayer(256, use_dropout=use_dropout)
        self.linear4 = NetLinLayer(512, use_dropout=use_dropout)
        self.linear5 = NetLinLayer(512, use_dropout=use_dropout)
        self.linear_layers = [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]

        if pretrained_full_model_path is not None:
            if pretrained_full_model_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                self.load_state_dict(load_file(pretrained_full_model_path))
            else:
                self.load_state_dict(torch.load(pretrained_full_model_path, map_location='cpu'))
        
        if eval_mode:
            self.eval()
    
    def forward(self, in0, in1, retPerLayer=False):
        outs0, outs1 = [], []
        for i in range(len(self.slices)):
            out0, out1 = self.slices[i](in0), self.slices[i](in1)
            outs0.append(out0)
            outs1.append(out1)
            in0, in1 = out0, out1
        
        diffs = []
        for i in range(len(self.linear_layers)):
            diffs.append((normalize_tensor(outs0[i])-normalize_tensor(outs1[i]))**2)

        res = [spatial_average(self.linear_layers[i](diffs[i]), keepdim=True) for i in range(len(self.linear_layers))]

        val = 0
        for i in range(len(res)):
            val += res[i]
        
        if retPerLayer:
            return val, res
        else:
            return val


########### E Latent LPIPS Model ###########
class ELatentLPIPS(nn.Module):
    def __init__(self, backbone_type="VGG16_Latent_GN", backbone_in_channels=4,
                 pretrained_latent_lpips_path=None, aug_type="bgc+cutout", aug_strength=1.0, 
                 use_dropout=True, eval_mode=True):
        super(ELatentLPIPS, self).__init__()
        
        if backbone_type == "VGG16_Latent":
            backbone = VGG16_Latent(in_channels=backbone_in_channels)
        elif backbone_type == "VGG16_Latent_BN":
            backbone = VGG16_Latent_BN(in_channels=backbone_in_channels)
        elif backbone_type == "VGG16_Latent_GN":
            backbone = VGG16_Latent_GN(in_channels=backbone_in_channels)
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")
                
        for param in backbone.parameters():
            param.requires_grad = False

        self.slice1 = backbone.features[0:6]
        self.slice2 = backbone.features[6:12]
        self.slice3 = backbone.features[12:21]
        self.slice4 = backbone.features[21:31]
        self.slice5 = backbone.features[31:41]
        self.slices = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]

        self.linear1 = NetLinLayer(64, use_dropout=use_dropout)
        self.linear2 = NetLinLayer(128, use_dropout=use_dropout)
        self.linear3 = NetLinLayer(256, use_dropout=use_dropout)
        self.linear4 = NetLinLayer(512, use_dropout=use_dropout)
        self.linear5 = NetLinLayer(512, use_dropout=use_dropout)
        self.linear_layers = [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5]

        if pretrained_latent_lpips_path is not None:
            if pretrained_latent_lpips_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                self.load_state_dict(load_file(pretrained_latent_lpips_path))
            else:
                self.load_state_dict(torch.load(pretrained_latent_lpips_path, map_location='cpu'))

        if eval_mode:
            self.eval()

        if aug_type == "none":
            self.augment_pipe = None
        else:
            self.augment_pipe = ada_aug.AdaAugment(**ada_augpipe[aug_type]).requires_grad_(True)
            self.augment_pipe.p = torch.tensor(aug_strength).float()
    
    def forward(self, in0, in1, retPerLayer=False):
        in0, in1 = self.augment_pipe(in0, in1) if self.augment_pipe is not None else (in0, in1)
        outs0, outs1 = [], []
        for i in range(len(self.slices)):
            out0, out1 = self.slices[i](in0), self.slices[i](in1)
            outs0.append(out0)
            outs1.append(out1)
            in0, in1 = out0, out1
        
        diffs = []
        for i in range(len(self.linear_layers)):
            diffs.append((normalize_tensor(outs0[i])-normalize_tensor(outs1[i]))**2)

        res = [spatial_average(self.linear_layers[i](diffs[i]), keepdim=True) for i in range(len(self.linear_layers))]

        val = 0
        for i in range(len(res)):
            val += res[i]
        
        if retPerLayer:
            return val, res
        else:
            return val


########### VGG16_Latent Details ###########
'VGG16'
# Conv1: 3x224x224 -> 64x224x224, Conv2: 64x224x224 -> 64x224x224, MaxPool: 64x224x224 -> 64x112x112
# Conv3: 64x112x112 -> 128x112x112, Conv4: 128x112x112 -> 128x112x112, MaxPool: 128x112x112 -> 128x56x56
# Conv5: 128x56x56 -> 256x56x56, Conv6: 256x56x56 -> 256x56x56, Conv7: 256x56x56 -> 256x56x56, MaxPool: 256x56x56 -> 256x28x28
# Conv8: 256x28x28 -> 512x28x28, Conv9: 512x28x28 -> 512x28x28, Conv10: 512x28x28 -> 512x28x28, MaxPool: 512x28x28 -> 512x14x14
# Conv11: 512x14x14 -> 512x14x14, Conv12: 512x14x14 -> 512x14x14, Conv13: 512x14x14 -> 512x14x14, MaxPool: 512x14x14 -> 512x7x7
# Linear1: 512x7x7 -> 4096, Linear2: 4096 -> 4096, Linear3: 4096 -> 1000

# 'VGG16_Latent' (in_channels=4 --> could be changed to 16 for sd3 and flux)
# Conv1: 4x64x64 -> 64x64x64, Conv2: 64x64x64 -> 64x64x64
# Conv3: 64x64x64 -> 128x64x64, Conv4: 128x64x64 -> 128x64x64
# Conv5: 128x64x64 -> 256x64x64, Conv6: 256x64x64 -> 256x64x64, Conv7: 256x64x64 -> 256x64x64 # MaxPool: 256x64x64 -> 256x32x32
# Conv8: 256x32x32 -> 512x32x32, Conv9: 512x32x32 -> 512x32x32, Conv10: 512x32x32 -> 512x32x32 # MaxPool: 512x32x32 -> 512x16x16
# Conv11: 512x16x16 -> 512x16x16, Conv12: 512x16x16 -> 512x16x16, Conv13: 512x16x16 -> 512x16x16
# AdaptiveAvgPool: 512x16x16 -> 512x7x7 Linear1: 512x7x7 -> 4096, Linear2: 4096 -> 4096, Linear3: 4096 -> 1000