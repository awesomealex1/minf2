from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, resnet18
from wide_res_net import Wide_ResNet
from pyramid_net import PyramidNet
from torch import nn

def get_efficient_net_s():
    return efficientnet_v2_s()

def get_efficient_net_m():
    return efficientnet_v2_m()

def get_efficient_net_l():
    return efficientnet_v2_l()

def get_wide_res_net(depth, widen_factor, dropout_rate, num_classes):
    return Wide_ResNet(depth, widen_factor, dropout_rate, num_classes)

def get_pyramid_net(dataset, depth, alpha, num_classes, bottleneck=False):
    return PyramidNet(dataset, depth, alpha, num_classes, bottleneck=False)

def get_res_net_18(one_channel=False):
    model = resnet18()
    if one_channel:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model