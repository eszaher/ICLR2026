import torch
from robustness.model_utils import make_and_restore_model
from robustness.datasets import ImageNet
import config as cfg
import os   
import torch.nn as nn

class RobustResNetFeatures(nn.Module):
    def __init__(self, attacker_model):
        super().__init__()
        
        base = attacker_model.model
        
        # Standard ResNet-50 layers:
        self.layer0  = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.conv1 =  base.conv1
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        # you can skip fc, avgpool if you only want convolution features

    def forward(self, x):
        f0 = self.conv1(x)
        #f0 = self.layer0(x)
        x  = self.layer0(x)
        f1 = self.layer1(x)
        #f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]

class VanillatResNetFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        # attacker_model.model is the actual ResNet
        base  = models.resnet50(pretrained=True)
        
        # Standard ResNet-50 layers:
        self.layer0  = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.conv1 =  base.conv1
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        # you can skip fc, avgpool if you only want convolution features

    def forward(self, x):
        f0 = self.conv1(x)
        x  = self.layer0(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]

class VGGPerceptual(nn.Module):
    """
    Extract VGG-19 features from conv1_2, conv2_2, conv3_2, conv4_2, conv5_2.
    """
    def __init__(self):
        super().__init__()
        vgg_pretrained = models.vgg19(pretrained=True).features
        # Slice the network at relevant layers:
        # conv1_2 ends around index 3,
        # conv2_2 ends around index 8,
        # conv3_2 ends around index 17,
        # conv4_2 ends around index 26,
        # conv5_2 ends around index 35.
        self.slice1 = vgg_pretrained[:4]   # up to conv1_2
        self.slice2 = vgg_pretrained[4:9]  # up to conv2_2
        self.slice3 = vgg_pretrained[9:18] # up to conv3_2
        self.slice4 = vgg_pretrained[18:27]# up to conv4_2
        self.slice5 = vgg_pretrained[27:36]# up to conv5_2

        # VGG was trained on ImageNet means, freeze params:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x assumed to be in [0,1] or [-1,1] range, you might need to adjust
        # Typically: subtract ImageNet means or scale
        # But let's keep it simple for demonstration:
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return [h1, h2, h3, h4, h5]
    

IMAGENET_PATH = cfg.IMAGENET_path  # Path to the ImageNet dataset for loading the robust model, we use a mock path for just loading the model
ds = ImageNet(IMAGENET_PATH)

# 2) Model kwargs: specify 'arch' matching the checkpoint. 'resnet50' is typical
model_kwargs = {
    'arch': cfg.arch_robust,  # architecture
    'dataset': ds        # dataset instance
}

# 3) The local checkpoint file to load. 
resume_path = "imagenet_l2_3_0.pt" #"os.path.expanduser('resnet50_linf_4.pth.tar')

# 4) Build the model and load checkpoint
def make_restore_model(model_kwargs, resume_path):
    """
    Make and restore a model from a checkpoint.
    """
    res50_model_robust, _ = make_and_restore_model(**model_kwargs, resume_path=resume_path)
    res50_model_robust.eval()
    return res50_model_robust

def perceptual_model(model="robust_resnet50"):
    if model == "vanilla_resnet50":
        return VanillatResNetFeatures().eval().to(cfg.device)
    elif model == "vgg_perceptual":
        return VGGPerceptual().eval().to(cfg.device)
    elif model == "robust_resnet50":
        res50_backbone = make_restore_model(model_kwargs, resume_path)
        robust_feats = RobustResNetFeatures(res50_backbone).eval().to(cfg.device)
        return robust_feats

