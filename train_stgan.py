import random
from collections import abc
import torch
from models import *
from ops import *
from non_leaking import *
from sgan_utils import *
from dataset import *
from distributed import *
from training import *
import stylegan2_args as st_args
import dataset as dsets
import config as cfg
from torch import optim


def main():
    device = cfg.device
    args = st_args.Args()
    dataset = dsets.get_dataset(args.dataset_name, args.dataset_path, args.size, cfg.mean, cfg.std)
    dataloader = dsets.get_dataloader(args, dataset)
        
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device(cfg.device)

    if args.ckpt is not None:
        print(f"Loading checkpoint from {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        args.start_iter = checkpoint['iter']
        generator = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
            conditional_gan=args.cgan, nof_classes=args.classifier_nof_classes, embedding_size=args.embedding_size
        ).to(device)
        generator.load_state_dict(checkpoint['generator'])
    else:
        print("No checkpoint provided, initializing generator from scratch.")
        args.start_iter = 0
        generator = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
            conditional_gan=args.cgan, nof_classes=args.classifier_nof_classes, embedding_size=args.embedding_size
        ).to(device)

        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
            conditional_gan=args.cgan, nof_classes=args.classifier_nof_classes, embedding_size=args.embedding_size
        ).to(device)

        g_ema.eval()
        accumulate(g_ema, generator, 0)
        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)

        g_optim = optim.Adam(
            generator.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )

        discriminator = Discriminator(
                    args.size, channel_multiplier=args.channel_multiplier, conditional_gan=args.cgan
                ).to(device)

        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
        d_optim = optim.Adam(
            discriminator.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

    train_stylegan2(args, dataloader, generator, discriminator, g_optim, d_optim, g_ema, device, args.ckpt)


            

