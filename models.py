from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, resnet18, mobilenet_v3_large, mobilenet_v3_small, vgg16
from wide_res_net import Wide_ResNet
from pyramid_net import PyramidNet
from torch import nn

def get_efficient_net_s(dataset):
    model = efficientnet_v2_s()
    num_ftrs = model.classifier[1].in_features  # Get the input features of the classifier layer
    if dataset == "cifar10":
        model.classifier[1] = nn.Linear(num_ftrs, 10)
    elif dataset == "cifar100":
        model.classifier[1] = nn.Linear(num_ftrs, 100)
    return model

def get_efficient_net_m(dataset):
    model = efficientnet_v2_m()
    num_ftrs = model.classifier[1].in_features  # Get the input features of the classifier layer
    if dataset == "cifar10":
        model.classifier[1] = nn.Linear(num_ftrs, 10)
    elif dataset == "cifar100":
        model.classifier[1] = nn.Linear(num_ftrs, 100)
    return model    

def get_efficient_net_l(dataset):
    model = efficientnet_v2_l()
    num_ftrs = model.classifier[1].in_features  # Get the input features of the classifier layer
    if dataset == "cifar10":
        model.classifier[1] = nn.Linear(num_ftrs, 10)
    elif dataset == "cifar100":
        model.classifier[1] = nn.Linear(num_ftrs, 100)
    return model

def get_vgg16():
    return vgg16()

def get_mobilenet_v3_s(dataset):
    model = mobilenet_v3_small()
    return model

def get_mobilenet_v3_l(dataset):
    model = mobilenet_v3_large()
    return model

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