from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

def get_efficient_net_s():
    return efficientnet_v2_s()

def get_efficient_net_m():
    return efficientnet_v2_m()

def get_efficient_net_l():
    return efficientnet_v2_l()