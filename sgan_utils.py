from lpips import LPIPS, lpips
import torch

from torch import nn, autograd, optim
from torch.nn import functional as F
from torchvision import utils, models
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageColor
import random
from ops import *
import os

@torch.no_grad()
def get_image(tensor, **kwargs):
    grid = utils.make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


@torch.no_grad()
def concat_image_by_height(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def save_sample_images(g_ema, args, itr_idx, sample_z, sample_z_label=None):
    with torch.no_grad():
        filename = os.path.join(args.output_path, f"sample/{args.dataset_name}/{str(itr_idx).zfill(6)}.png")

        g_ema.eval()
        
        nof_images = sample_z.shape[0]
        minv = -1
        maxv = 1
        if sample_z_label is None or sample_z_label.shape[2] != 2:
            sample, _ = g_ema([sample_z])
            minv = torch.min(sample)
            maxv = torch.max(sample)

            utils.save_image(
                sample,
                filename,
                nrow=int(nof_images ** 0.5),
                normalize=True,
                value_range=(minv, maxv))
            return
        else:
            # For Conditional GAN, draw borders around each image according to its label
            sample, _ = g_ema([sample_z], labels=sample_z_label)
            minv = torch.min(sample)
            maxv = torch.max(sample)
            sample_z_label = sample_z_label.cpu().squeeze()
            im = get_image(
                sample,
                nrow=int(nof_images ** 0.5),
                normalize=True,
                value_range=(minv, maxv),
                padding=0)
            border_size = 4
            draw = ImageDraw.Draw(im)
            # we have to iterate, each image get a padding fill value of its own
            for i in range(nof_images):
                
                label = sample_z_label[i].numpy()[0].item()

                if label == 0:
                    row = i // 4
                    column = i % 4
                    # img = sample[i, :]
                    draw.rectangle(
                        [column*256,row*256,
                        (column+1)*256,(row+1)*256],
                        outline=ImageColor.getrgb("darkred"),
                        width=border_size)
               
            im.save(filename)

def save_real_vs_encoded(generator, encoder, args, i, sample_images):
    with torch.no_grad():
        latents = encoder(sample_images)
        encoded_imgs, _ = generator([latents], input_is_latent=False)

        # save real vs. encoded images
        minv = torch.min(encoded_imgs)
        maxv = torch.max(encoded_imgs)
        encoded_image = get_image(
            encoded_imgs,
            nrow=int(args.batch ** 0.5),
            normalize=True,
            value_range=(minv, maxv)
        )

        # save real vs. encoded images
        minv = torch.min(sample_images)
        maxv = torch.max(sample_images)
        real_image = get_image(
            sample_images,
            nrow=int(args.batch ** 0.5),
            normalize=True,
            value_range=(minv, maxv)
        )
        filename_img = os.path.join(args.output_path, f"sample/{args.dataset_name}/real_vs_encoded_{str(i).zfill(6)}.png")
        combined = concat_image_by_height(encoded_image, real_image)
        combined.save(filename_img)