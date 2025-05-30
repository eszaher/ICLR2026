import os
import torch
from models import Generator
import stylegan2_args as st_args
import dataset as dsets
import config as cfg
from training import train_encoder

def main():
    args = st_args.Args()  # includes args.ckpt, args.size, args.latent, args.dataset_name, etc.
    device = torch.device(cfg.device)

    # Load dataset and dataloader
    dataset = dsets.get_dataset(args.dataset_name, args.dataset_path, args.size, cfg.mean, cfg.std)
    dataloader = dsets.get_dataloader(args, dataset)

    # Check if checkpoint is provided
    if args.ckpt is not None and os.path.isfile(args.ckpt):
        print(f"[✓] Loading pretrained StyleGAN2 generator from {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)

        g_ema = Generator(
            args.size,
            args.latent,
            args.n_mlp,
            channel_multiplier=args.channel_multiplier,
            conditional_gan=args.cgan,
            nof_classes=args.classifier_nof_classes,
            embedding_size=args.embedding_size
        ).to(device)
        g_ema.load_state_dict(checkpoint['generator'])

        # Optional: print finetuning mode
        if getattr(args, 'finetune_generator', False):
            print("[*] Finetuning the generator during encoder training.")
        else:
            print("[*] Freezing the generator. Only training encoder.")

        # Train encoder with or without generator finetuning
        train_encoder(args, dataloader, g_ema, device, finetune_generator=args.finetune_generator)

    else:
        print("[✗] No valid StyleGAN2 checkpoint provided.")
        print("→ Please train StyleGAN2 first or provide --ckpt path to a pretrained generator.")
        return


if __name__ == "__main__":
    main()