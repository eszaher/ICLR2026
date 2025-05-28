import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from torchvision.datasets import MNIST
import os
import numpy as np
import random

def _color_digit(img_gray, color_label):
    """
    Convert a single-channel grayscale image to an RGB image, 
    coloring the digit either green (C=0) or red (C=1).
    
    Args:
        img_gray (Tensor): shape (28, 28), pixel intensity in [0,1].
        color_label (int): 0 => green, 1 => red.
        
    Returns:
        img_rgb (Tensor): shape (3, 28, 28).
    """
    # Convert to (H,W) -> (1,H,W) for broadcasting
    if len(img_gray.shape) == 2:
        img_gray = img_gray.unsqueeze(0)  # shape: (1,28,28)
    
    # Create an empty 3-channel image
    img_rgb = torch.zeros(3, img_gray.shape[1], img_gray.shape[2])
    
    # For demonstration, we threshold the digit region
    # If a pixel in grayscale > 0, we color it
    digit_mask = (img_gray > 0).float()
    
    if color_label == 0:
        # color = green => (R=0, G=1, B=0) * intensity
        img_rgb[0] = 0.0
        img_rgb[1] = digit_mask * img_gray
        img_rgb[2] = 0.0
    else:
        # color = red => (R=1, G=0, B=0) * intensity
        img_rgb[0] = digit_mask * img_gray
        img_rgb[1] = 0.0
        img_rgb[2] = 0.0
    
    # For background: just leave it black
    return img_rgb


def _add_bar(img_rgb, bar_label, bar_thickness=3):
    """
    Adds a horizontal blue bar (B=1 => bar) at the top rows of the image.
    
    Args:
        img_rgb (Tensor): shape (3, H, W).
        bar_label (int): 0 => no bar, 1 => has bar.
        bar_thickness (int): how many rows to color in blue.
    
    Returns:
        img_rgb (Tensor): shape (3, H, W), possibly with top bar in blue.
    """
    if bar_label == 1:
        # color top 'bar_thickness' rows entirely blue (R=0, G=0, B=1)
        img_rgb[0, :bar_thickness, :] = 0.0
        img_rgb[1, :bar_thickness, :] = 0.0
        img_rgb[2, :bar_thickness, :] = 1.0
    return img_rgb
class ColoredMNISTwithBarsBackdoor(Dataset):
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform=None,
        download: bool = True,
        bar_thickness: int = 3,
        seed: int = 42
    ):
        """
        Custom MNIST dataset that applies the 'Backdoor' SCM for color + bar generation.
        
        Args:
            root (str): path to store MNIST.
            train (bool): load train or test split.
            transform: optional transform on final colored image (e.g., ToTensor()).
            download (bool): whether to download MNIST if not present.
            bar_thickness (int): how thick the horizontal bar is at the top.
            seed (int): random seed to control reproducibility.
        """
        super().__init__()
        
        # Set random seed for reproducibility
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        # We first load raw MNIST
        self.mnist_data = MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor()  # We want raw tensor shape (1,28,28)
        )
        
        self.transform = transform
        self.bar_thickness = bar_thickness
        
        # We will store the augmented images and labels
        self.data = []
        self.targets = []
        
        # Pre-generate the entire dataset in memory
        self._prepare_data()
    
    
    def _prepare_data(self):
        """
        For each sample in MNIST, apply the Backdoor SCM to get {D, C, B} 
        and then generate the colored digit with or without a bar.
        """
        for idx in range(len(self.mnist_data)):
            img_gray, digit = self.mnist_data[idx]  # digit D in [0..9]
            
            # === 1) Sample C ===
            # Probability that C=0 (green) = (0.95 - 0.1 * D)
            # Probability that C=1 (red)   = 1 - (above)
            prob_green = 0.95 - 0.1 * digit
            # clamp probabilities to [0,1]
            prob_green = np.clip(prob_green, 0.0, 1.0)
            c = 0 if self.rng.rand() < prob_green else 1
            
            # === 2) Sample exogenous variables U1, U2, U3 ===
            u1 = 1 if self.rng.rand() < 0.8 else 0
            u2 = 1 if self.rng.rand() < 0.9 else 0
            u3 = 1 if self.rng.rand() < 0.75 else 0
            
            # === 3) Compute B from the formula:
            # B = ([D>=5] XOR U1) OR ((C XOR U2) AND U3)
            part1 = (1 if digit >= 5 else 0) ^ u1  # XOR
            part2 = (c ^ u2) and u3
            b = part1 or part2  # OR
            
            # === 4) Color the digit according to c, add bar if b==1
            img_rgb = _color_digit(img_gray, c)  # shape: (3,28,28)
            img_rgb = _add_bar(img_rgb, b, self.bar_thickness)
            
            # Optionally apply a final transform (e.g., normalization)
            if self.transform is not None:
                img_rgb = self.transform(img_rgb)
            
            # We store:
            # - The final image (3,28,28)
            # - The triple (D, C, B)
            self.data.append(img_rgb)
            self.targets.append((digit, c, b))
    
    
    def __getitem__(self, index):
        """
        Returns: (image, (D, C, B))
          where 'image' is a Tensor of shape (3, 28, 28) or possibly transformed,
          and (D, C, B) are integers.
        """
        img = self.data[index]
        digit, c, b = self.targets[index]
        #print("digit", type(digit))
        return img, (digit, c, b)
    
    
    def __len__(self):
        return len(self.data)
    
class CelebADataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory with images.
            transform (callable, optional): Transformations to apply to images.
        """
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, fname)for fname in os.listdir(image_dir) if fname.lower().endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB format

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Return image and dummy label (since no labels are used)
        return image
    
def get_dataset(args):
    """
    Loads either MNIST or CelebA from torchvision's standard datasets.
    `args` is assumed to have the attributes:
      - args.dataset_name: "mnist" or "celeba"
      - args.dataset_path: path to download/store the dataset
      - args.size: desired output image size (if you want to resize images)
    """

    # Common transforms
    if args.dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.Resize(args.size),  # optional resize
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # standard MNIST mean/std
        ])

        dataset = datasets.MNIST(
            root=args.dataset_path,
            train=True,             # or False if you want test set
            transform=transform,
            download=True,
        )
        
    elif args.dataset_name.lower() == "mnist_colored":
        transform = transforms.Compose([
            transforms.Resize(args.size),  # optional resize
            #transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # standard MNIST mean/std
        ])
        dataset = ColoredMNISTwithBarsBackdoor(
            root=args.dataset_path,
            train=True,
            transform=transform,
            download=True,
            bar_thickness=2,
            seed=32
        )

    elif args.dataset_name.lower() == "celeba":
        transform = transforms.Compose([
            transforms.Resize(args.size),     # optional resize
            transforms.CenterCrop(args.size), # CelebA images are 178x218, so many folks crop square
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),  # typical CelebA normalization
        ])
        dataset = CelebADataset(image_dir=args.dataset_path,transform=transform)

        """dataset = datasets.CelebA(
            root=args.dataset_path,
            split="test",          # or "valid" / "test" if needed
            target_type='attr',
            transform=transform,
            download=False,
        )"""

    else:
        raise ValueError("Unsupported dataset. Use 'mnist' or 'celeba'.")

    return dataset


def data_sampler(dataset, shuffle=True, distributed=False):
    """Returns the appropriate PyTorch sampler depending on `distributed` and `shuffle`."""
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    else:
        return data.RandomSampler(dataset) if shuffle else data.SequentialSampler(dataset)


def get_dataloader(args):
    """
    Builds a DataLoader using the dataset fetched by `get_dataset()`.
    `args` is assumed to have the attributes:
      - args.batch: batch size
      - args.distributed: boolean, for distributed training or not
    """

    dataset = get_dataset(args)
    sampler = data_sampler(dataset, shuffle=True, distributed=args.distributed)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=sampler,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    return dataloader