import torch 
# Dataset
dataset = "AFHQ"
dataset_path = "None"

# Artifacts
output_path = "output"

# StyleGAN2 parameters
arch = "stylegan2"  # (stylegan2 | autoencoder | StyleEx)
size = 256
latent = 512
batch = 32
iter = 800000
persistent_workers = True 
pin_memory = True
mean=[0.5, 0.5, 0.5]
std=[0.5, 0.5, 0.5]
mlp = 8
ckpt = None
lr = 0.002

# CLF
mean_clf = [0.485, 0.456, 0.406]
std_clf = [0.229, 0.224, 0.225]
lr_clf = 0.001

# CUDA 
device = "cuda" if torch.cuda.is_available() else "cpu"


#PCG
is_latent = False

# Others

IMAGENET_path = '/path/to/imagenet'  # a mock path for loading the robust model
arch_robust = 'resnet50' # architecture for the robust model

