import math
import torch.nn as nn
import torch.nn.init as init

__all__ = ['VGG', 'vgg11', 'vgg11_bw', 'vgg11_tau_bias', 'vgg11_100',  'vgg16', 'vgg16_bw', 'vgg16_tau', 'vgg16_tau_bias', 'vgg16_100', 'vgg19_tau', 'vgg11_tau_ff', 'vgg16_tau_ff', 'vgg11_tau_ff_bias', 'vgg11_tau_adaptive', 'vgg16_tau_adaptive', 'vgg11_bw_adaptive', 'vgg11_bw_adaptive_50', 'vgg11_bw_wide_adapt', 'vgg11_wide']

class VGG(nn.Module):
    def __init__(self, features, num_classes=10, bias=False, full_ff_layers=False, adaptive_ff_layers=False, wide=False, adaptive=False):
        super(VGG, self).__init__()
        self.features = features
        self.adaptive_ff_layers = adaptive_ff_layers
        self.adaptive = adaptive
        self.wide = wide
        if full_ff_layers:
            self.classifier = nn.Sequential(
                nn.Linear(512, 512, bias=bias),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512, bias=bias),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes, bias=bias),
            )
        elif adaptive_ff_layers:
            d = 2 if not self.wide else 1
            self.avgpool = nn.AdaptiveAvgPool2d((d, d)) 
            self.classifier = nn.Sequential(
                nn.Linear(512*d*d if not self.wide else 1024 * d * d, 1024, bias=bias),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024, bias=bias),
                nn.ReLU(inplace=True),
                nn.Linear(1024, num_classes, bias=bias),
            )
            # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            # self.classifier = nn.Sequential(
            #     nn.Linear(512*7*7, 4096, bias=bias),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(4096, 4096, bias=bias),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(4096, num_classes, bias=bias),
            # )
        elif adaptive:
            d = 2 if not self.wide else 1
            self.avgpool = nn.AdaptiveAvgPool2d((d, d)) 
            self.classifier = nn.Sequential(
                nn.Linear(512*d*d if not self.wide else 1024 * d * d, num_classes, bias=bias),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes, bias=bias),
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        x = self.features(x)
        try:
            if self.adaptive_ff_layers or self.adaptive:
                x = self.avgpool(x)
        except:
            # For backward compatibility
            self.adaptive = False
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, bw=False, bias=False):
    layers = []
    if bw:
        in_channels = 1
    else:
        in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=bias)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'F': [64,      128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # added for mnist
    'Awide': [128, 'M', 256, 'M', 512, 512, 'M', 1024, 1024, 'M', 1024, 1024, 'M'], # added for mnist
    'Fwide': [128,      256, 'M', 512, 512, 'M', 1024, 1024, 'M', 1024, 1024, 'M'], # added for mnist
    'V16'   : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'V16BW' : [64, 64     , 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def vgg11():
    return VGG(make_layers(cfgs['A']))

def vgg11_wide():
    return VGG(make_layers(cfgs['Awide'], bw=False), adaptive=True, wide=True)

def vgg11_bw():
    return VGG(make_layers(cfgs['F'], bw=True))

def vgg11_tau():
    return VGG(make_layers(cfgs['A'], bw=True))

def vgg11_tau_ff():
    return VGG(make_layers(cfgs['A'], bw=True), full_ff_layers=True)

def vgg11_tau_adaptive():
    return VGG(make_layers(cfgs['A'], bw=True), adaptive_ff_layers=True)

def vgg11_bw_adaptive():
    return VGG(make_layers(cfgs['F'], bw=True), adaptive_ff_layers=True)

def vgg11_bw_wide_adapt():
    return VGG(make_layers(cfgs['Fwide'], bw=True), adaptive_ff_layers=True, wide=True)

def vgg11_bw_wide_adapt_50():
    return VGG(make_layers(cfgs['Fwide'], bw=True), adaptive_ff_layers=True, wide=True, num_classes=50)
    

def vgg11_bw_adaptive_50():
    return VGG(make_layers(cfgs['F'], bw=True), adaptive_ff_layers=True, num_classes=50)

def vgg11_tau_bias():
    return VGG(make_layers(cfgs['A'], bw=True, bias=True), bias=True)

def vgg11_tau_ff_bias():
    return VGG(make_layers(cfgs['A'], bw=True, bias=True), bias=True, full_ff_layers=True)


def vgg11_100():
    return VGG(make_layers(cfgs['A']), num_classes=100)


def vgg16():
    return VGG(make_layers(cfgs['V16']))

def vgg16_bw():
    return VGG(make_layers(cfgs['V16BW'], bw=True))

def vgg16_tau():
    return VGG(make_layers(cfgs['V16'], bw=True))

def vgg16_tau_adaptive():
    return VGG(make_layers(cfgs['V16'], bw=True), adaptive_ff_layers=True)


def vgg16_tau_ff():
    return VGG(make_layers(cfgs['V16'], bw=True), full_ff_layers=True)

def vgg16_tau_bias():
    return VGG(make_layers(cfgs['V16'], bw=True, bias=True), bias=True)

def vgg16_100():
    return VGG(make_layers(cfgs['V16']), num_classes=100)



def vgg19_tau():
    return VGG(make_layers(cfgs['E'], bw=True))
def vgg19_tau_ff():
    return VGG(make_layers(cfgs['E'], bw=True), full_ff_layers=True)