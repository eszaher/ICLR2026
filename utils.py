import torch
import torch.nn as nn
import torch.optim as optim
import config as cfg
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.nn import functional as F

def interpolate(start, end, steps):
    """Generate interpolated vectors between start and end."""
    interpolation = [start + (float(i)/steps)*(end-start) for i in range(0, steps)]
    return interpolation

def denormalize_image(image, mean, std):
    mean = torch.tensor(mean).view(1, 1, -1)
    std = torch.tensor(std).view(1, 1, -1)
    image = image * std + mean  # Reverse normalization
    return image
def visualize_transformed_samples(dataloader, num_samples=9):
    # Create a grid for visualization
    plt.figure(figsize=(10, 10))
    
    # Extract a batch from the DataLoader
    images, labels = next(iter(dataloader))
    
    for i in range(min(num_samples, len(images))):
        # Extract one image and its label
        image = images[i].permute(1, 2, 0)  # Permute dimensions for plotting
        #label = labels[i].item()
        
        # Undo normalization if applied
        #if image.min() < 0:  # Assuming normalization to [-1, 1]
        #    image = (image + 1) / 2  # Scale back to [0, 1]
        image = denormalize_image(image, mean, std)
        image = image.numpy()
        image = image.clip(0, 1)
        # Plot the image
        plt.subplot(3, 3, i + 1)
        plt.imshow(image)
        plt.title(f"Label: not loaded")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def show_tensor_images(tensor, nrow=8, title=None):
    """
    tensor: a PyTorch tensor of shape (B, 3, H, W) with values in [-1, 1].
    nrow : how many images to display per row.
    title: optional title for the figure.
    """
    # Denormalize and create a grid
    grid_image = make_grid(tensor, nrow=nrow, normalize=True)
    
    # Convert the tensor to numpy for displaying via matplotlib
    np_grid = grid_image.permute(1, 2, 0).cpu().numpy()
    
    # Display the image
    plt.figure(figsize=(16, 16))
    if title is not None:
        plt.title(title)
    plt.imshow(np_grid)
    plt.axis('off')
    plt.show()
    
def normalize_img_bc(img):
    """
    Per-image brightness/contrast normalization:
    Subtract mean, divide by std, per sample.
    img: shape (B, C, H, W)
    """
    B, C, H, W = img.shape
    # Flatten the spatial dims for mean/std
    reshaped = img.view(B, C, -1)
    means = reshaped.mean(dim=2, keepdim=True)
    stds  = reshaped.std(dim=2,  keepdim=True) + 1e-8
    normalized = (reshaped - means) / stds
    return normalized.view(B, C, H, W)

# For Percepetual Loss
class VGG19(nn.Module):
    """
     Simplified version of the VGG19 "feature" block
     This module's only job is to return the "feature loss" for the inputs
    """
    def __init__(self, channel_in=3, width=64):
        super(VGG19, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, width, 3, 1, 1)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)

        self.conv3 = nn.Conv2d(width, 2 * width, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * width, 2 * width, 3, 1, 1)

        self.conv5 = nn.Conv2d(2 * width, 4 * width, 3, 1, 1)
        self.conv6 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv7 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv8 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)

        self.conv9 = nn.Conv2d(4 * width, 8 * width, 3, 1, 1)
        self.conv10 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv11 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv12 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.conv13 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv14 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv15 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv16 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.load_params_()

    def load_params_(self):
        # Download and load Pytorch's pre-trained weights
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        for ((name, source_param), target_param) in zip(state_dict.items(), self.parameters()):
            target_param.data = source_param.data
            target_param.requires_grad = False

    def feature_loss(self, x):
        return (x[:x.shape[0] // 2] - x[x.shape[0] // 2:]).pow(2).mean()

    def forward(self, x):
        """
        :param x: Expects x to be the target and source to concatenated on dimension 0
        :return: Feature loss
        """
        x = self.conv1(x)
        loss = self.feature_loss(x)
        x = self.conv2(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 64x64

        x = self.conv3(x)
        loss += self.feature_loss(x)
        x = self.conv4(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 32x32

        x = self.conv5(x)
        loss += self.feature_loss(x)
        x = self.conv6(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv7(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv8(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 16x16

        x = self.conv9(x)
        loss += self.feature_loss(x)
        x = self.conv10(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv11(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv12(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 8x8

        x = self.conv13(x)
        loss += self.feature_loss(x)
        x = self.conv14(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv15(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv16(self.relu(x))
        loss += self.feature_loss(x)

        return loss/16
    

vgg_preprocessing = transforms.Compose([
    transforms.Resize((224, 224)),                           # Resize to 224x224
    transforms.Normalize(mean=cfg.mean_clf,         # ImageNet mean
                         std=cfg.std_clf)          # ImageNet std
])

vgg_preprocessing_256 = transforms.Compose([
    transforms.Resize((cfg.size, cfg.size)),                           # Resize to 256x256
    transforms.Normalize(mean=cfg.mean_clf,       
                         std=cfg.std_clf)       
])

def compute_feature_loss(x1, x2, vgg_model, normalize_brightness_contrast=True):
    """
    Computes a feature-based distance using the sum of L2 differences in selected VGG layers.
    Optionally normalizes brightness and contrast for each image before passing to VGG.
    """
    # Optional brightness/contrast normalization:
    if normalize_brightness_contrast:
        x1 = normalize_img_bc(x1)
        x2 = normalize_img_bc(x2)

    # Extract features
    feats1 = vgg_model(x1)
    feats2 = vgg_model(x2)

    # Compute MSE in each layer, average across layers
    dist = 0.0
    for f1, f2 in zip(feats1, feats2):
        # f1, f2 shape is [B, C, H, W]; typically B=1 in an interpolation scenario
        #  -- If you do multi-batch, this still works the same, N_j = B*C*H*W
        #sum_sq = torch.nn.functional.l1_loss(f1, f2, reduction='sum')
        sum_sq = F.mse_loss(f1, f2, reduction='sum')  # sum of (f1 - f2)^2
        N_j = f1.numel()  # total number of scalars in that feature map
        layer_dist = sum_sq / float(N_j)
        dist += layer_dist

    return dist

def perceptual_distance(x1, x2, perceptual_model, normalize_brightness_contrast=True):
    """
    Computes a feature-based distance using the sum of L2 differences in selected VGG layers.
    Optionally normalizes brightness and contrast for each image before passing to VGG.
    """
    # Optional brightness/contrast normalization:
    x1 = (x1 + 1) / 2  # Rescale to [0, 1] from [-1, 1]
    x1 = vgg_preprocessing_256(x1)  # Apply preprocessing for VGG]
    x2 = (x2 + 1) / 2  # Rescale to [0, 1] from [-1, 1]
    x2 = vgg_preprocessing_256(x2)  # Apply preprocessing for VGG]
    
    if normalize_brightness_contrast:
        x1 = normalize_img_bc(x1)
        x2 = normalize_img_bc(x2)

    # Extract features
    feats1 = perceptual_model(x1)
    feats2 = perceptual_model(x2)

    # Compute MSE in each layer, average across layers
    dist = 0.0
    for f1, f2 in zip(feats1, feats2):
        # f1, f2 shape is [B, C, H, W]; typically B=1 in an interpolation scenario
        #  -- If you do multi-batch, this still works the same, N_j = B*C*H*W
        #sum_sq = torch.nn.functional.l1_loss(f1, f2, reduction='sum')
        sum_sq = F.mse_loss(f1, f2, reduction='sum')  # sum of (f1 - f2)^2
        N_j = f1.numel()  # total number of scalars in that feature map
        layer_dist = sum_sq / float(N_j)
        dist += layer_dist
        
    return dist
    

def feature_energy(path, generator, perceptual_model, is_latent):
    """
    path: tensor shape [t+1, latent_dim].
          path[0] is z0 (fixed or not, depending on your design).
    
    We sum over segments i=0..t-1:
        norm( Jv(z_i, z_{i+1}-z_i) )^2
      = < (z_{i+1}-z_i),  G(z_i)  (z_{i+1}-z_i) >
      where G(z_i) = J(z_i)^T J(z_i).

    We'll use matrix-free approach:
       If r_i = z_{i+1}-z_i,
       then segment energy = || Jv(z_i, r_i) ||^2.

    Returns: a scalar (PyTorch tensor) that can be backprop'ed.
    """
    total_energy = 0.0
    t = path.shape[0] - 1
    dt = 1/t
    
    for i in range(t):
        #z_i = path[i]     # shape [latent_dim]
        #z_ip1 = path[i+1]
        xi,_ = generator([path[i] ], input_is_latent=is_latent, randomize_noise=False)
        xip1,_ = generator([path[i+1]], input_is_latent=is_latent, randomize_noise=False)
        #diff = z_ip1 - z_i
        # Jv => shape [3,32,32]
        #u = Jv(generator, z_i, diff, create_graph=True)
        u = perceptual_distance(xi, xip1, perceptual_model)
        seg_energy = 0.5*dt*(u ** 2).sum()  # scalar
        total_energy = total_energy + seg_energy
    
    return total_energy

def classifier_loss_with_softmax(generator, classifier, z_end, target_label):
    """
    Computes loss using softmax probabilities.
    
    This function calculates the negative log probability of the target label
    after applying a softmax to the logits output by the classifier.
    
    target_label: int in [0..9] (or 0..1 for binary classification).
    """
    if z_end.dim() == 1:
        z_end = z_end.unsqueeze(0)  # shape: [1, latent_dim]
        
    x_end, _ = generator([z_end], input_is_latent=False, randomize_noise=False)  # shape: [1, 3, 256, 256]
    x_end = (x_end + 1) / 2  # Rescale to [0, 1] from [-1, 1]
    
    x_end = vgg_preprocessing(x_end)  # Apply preprocessing for VGG
    
    logits = classifier(x_end)  # shape: [1, num_classes]
    
    # Compute softmax to obtain probabilities
    probabilities = F.softmax(logits, dim=1)  # shape: [1, num_classes]
    
    # Create target tensor
    target = torch.tensor([target_label], dtype=torch.long, device=z_end.device)
    
    # Compute negative log likelihood loss manually:
    # For batch size 1, extract the probability for the target label.
    loss_val = -torch.log(probabilities[0, target])
    # If batch size > 1, you would typically use gather:
    # loss_val = -torch.log(probabilities.gather(1, target.unsqueeze(1))).mean()
    
    return loss_val
    
def classifier_loss(generator, classifier, z_end, target_label):
    """
    Simple cross-entropy at the final latent code z_end.
    target_label: int in [0..9].
    """
    if z_end.dim() == 1:
        z_end = z_end.unsqueeze(0)  # => [1,latent_dim]
        
    x_end,_ = generator([z_end], input_is_latent=False, randomize_noise=False)  # shape: [1, 3, 256, 256]
    x_end = (x_end + 1) / 2  # Rescale to [0, 1] from [-1, 1]
    
    #x = (x + 1) / 2  # Rescale to [0, 1] from [-1, 1]
    x_end = vgg_preprocessing(x_end)  # Apply preprocessing for VGG]
    
    logits = classifier(x_end)      # => [1,10]
    target = torch.tensor([target_label], dtype=torch.long, device=z_end.device)
    #print("logits.shape =", logits.shape)  # e.g. [1, 10] or [1, 1000], etc.
    #print("target_label =", target_label)
    loss_val = F.cross_entropy(logits, target)
    return loss_val
    

def full_path_loss(path, generator, classifier, perceptual_model, target_label, is_latent=cfg.is_latent, lam=1.0):
    """
    Combine discrete geodesic energy + lam * classifier loss on final point.
    """
    #geo = discrete_geodesic_energy(path, generator, is_latent)
    geo = feature_energy(path, generator, perceptual_model, is_latent)
    clf = classifier_loss(generator, classifier, path[-1], target_label)
    
    return geo + lam * clf

# Path Energy

def Jv(generator, z, v, is_latent=cfg.is_latent, create_graph=False):
    """
    Compute the Jacobian-vector product J(z) * v
    where J(z) = d g(z) / d z, and v is the direction in latent space.
    
    Return shape = same as generator(z) => e.g. [3,32,32].
    
    Implementation approach:
     1) We'll do a forward pass to get g(z).
     2) We'll compute the gradient of g(z) wrt. z, but we only want
        the product with v, not the full Jacobian. This can be done
        by treating 'v' as a 'virtual' gradient from the output side.
    """
    # Ensure z and v are both differentiable:
    def g_wrap(z):
        out = generator([z], input_is_latent=is_latent, randomize_noise=False)
        return out[0]

        
    z = z.detach().requires_grad_(True)
    
    # Forward pass
    x = generator([z], input_is_latent=is_latent, randomize_noise=False) # shape [3,32,32]
    
    # We'll flatten x so that we can feed a single 'vector' as grad_outputs
    # but we can keep x in shape [3,32,32] if we prefer, as long as 
    # we do a matching shape for grad_outputs.
    x_flat = x[0].view(-1)  # shape [3*32*32]


    jvp_result = torch.autograd.functional.jvp(
        func=g_wrap,
        inputs=z,
        v=v,
        create_graph=True
    )[1]
    # shape of jvp_result: same as g(z) => [3,32,32]

    return jvp_result.squeeze(dim=0)