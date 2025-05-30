
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from sgan_utils import *
from non_leaking import *
from ops import *
import random

def train_stylegan2(
    args, 
    loader, 
    generator, 
    discriminator, 
    g_optim, 
    d_optim, 
    g_ema, 
    device="cuda", 
    ckpt=None
):
    """
    Training loop for the StyleGAN2 architecture (unconditional).
    Omits any distributed training or autoencoder/classifier logic.
    """

    # Convert loader into an infinite iterator
    loader_iter = sample_data(loader)

    # Progress bar
    pbar = tqdm(range(args.start_iter, args.iter), dynamic_ncols=True, smoothing=0.01)

    # Initialize some variables
    mean_path_length = 0
    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    weighted_path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0

    # Exponential moving average decay factor
    accum = 0.5 ** (32 / (10 * 1000))

    # Augmentation probability (if using ADA)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0

    # If using adaptive augmentation:
    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    # Fixed noise for sampling
    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    # ----------------------------
    #  Optionally load checkpoint
    # ----------------------------
    # (Already assumed done outside or from `ckpt`. If so, you'd load states here.)

    # -----------------------
    #      Training Loop
    # -----------------------
    for idx in pbar:
        i = idx

        # If we've exceeded total iterations, break.
        if i > args.iter:
            print("Done!")
            break

        # ------------------
        # 1) Sample a batch
        # ------------------
        real_img = next(loader_iter)
        real_img = real_img.to(device)

        # ----------------------------
        # 2) Update the Discriminator
        # ----------------------------
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # Generate fake images from random noise
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        # Optionally apply ADA augmentations
        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img_aug, _ = augment(fake_img, ada_aug_p)
        else:
            real_img_aug = real_img
            fake_img_aug = fake_img

        # Discriminator prediction on real vs fake
        real_pred = discriminator(real_img_aug)
        fake_pred = discriminator(fake_img_aug)

        # Logistic loss
        d_loss = d_logistic_loss(real_pred, fake_pred)

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        # If using adaptive augment and p == 0, adjust p
        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)

        # R1 regularization
        d_regularize = (i % args.d_reg_every == 0)
        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)
            discriminator.zero_grad()
            r1_final = (args.r1 / 2) * r1_loss * args.d_reg_every
            r1_final.backward()
            d_optim.step()
            real_img.requires_grad = False

        # -------------------------
        # 3) Update the Generator
        # -------------------------
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            fake_img_aug, _ = augment(fake_img, ada_aug_p)
        else:
            fake_img_aug = fake_img

        fake_pred = discriminator(fake_img_aug)
        g_loss = g_nonsaturating_loss(fake_pred)

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # Path length regularization
        g_regularize = (i % args.g_reg_every == 0)
        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img_path, latents = generator(noise, return_latents=True)
            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img_path, latents, mean_path_length
            )
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            generator.zero_grad()
            weighted_path_loss.backward()
            g_optim.step()

            mean_path_length_avg = float(mean_path_length.item())

        # -----------------------------
        # 4) Update EMA (moving avg)
        # -----------------------------
        accumulate(g_ema, generator, accum)

        # -----------------------------------
        # 5) Logging & Periodic Checkpoints
        # -----------------------------------
        pbar.set_description(
            (
                f"d: {d_loss.item():.4f}; "
                f"g: {g_loss.item():.4f}; "
                f"r1: {r1_loss.item():.4f}; "
                f"path: {path_loss.item():.4f}; "
                f"mean path: {mean_path_length_avg:.4f}; "
                f"augment: {ada_aug_p:.4f}"
            )
        )

        # Save samples
        if i > 0 and i % args.save_samples_every == 0:
            save_sample_images(g_ema, args, i, sample_z)

        # Save checkpoint
        if i > 0 and i % args.save_checkpoint_every == 0:
            filename = os.path.join(args.output_path, f"checkpoint/{args.dataset_name}/{str(i).zfill(6)}.pt")
            torch.save(
                {
                    "g":       (generator.module.state_dict() if hasattr(generator, "module") else generator.state_dict()),
                    "d":       (discriminator.module.state_dict() if hasattr(discriminator, "module") else discriminator.state_dict()),
                    "g_ema":   g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "args":    args,
                    "ada_aug_p": ada_aug_p,
                },
                filename,
            )
def train_stylechexplain(
    args, 
    loader, 
    generator, 
    discriminator, 
    g_optim, 
    d_optim, 
    g_ema, 
    device, 
    classifier, 
    encoder, 
    e_optim, 
    ckpt=None
):
    from tqdm import tqdm
    import torch.nn.functional as F

    loader_iter = sample_data(loader)
    pbar = tqdm(range(args.start_iter, args.iter), dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0
    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    weighted_path_loss = torch.tensor(0.0, device=device)
    class_loss = torch.tensor(0.0, device=device)
    reconstruct_loss = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0

    # Exponential moving average decay factor
    accum = 0.5 ** (32 / (10 * 1000))

    # Augmentation probability (if you are using ADA)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    # For sampling
    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    sample_z_label = torch.randint(
        0, args.classifier_nof_classes, (args.n_sample, 1), device=device
    )
    sample_z_label = F.one_hot(
        sample_z_label, num_classes=args.classifier_nof_classes
    )

    # Grab one batch for "real vs encoded" image saving
    sample_images, sample_labels = next(iter(loader))
    sample_images = sample_images.to(device)
    sample_labels = sample_labels.to(device)

    for idx in pbar:
        i = idx

        ########################################################################
        # 1. Fetch real batch from loader
        ########################################################################
        real_img, real_labels = next(loader_iter)
        real_img = real_img.to(device)
        real_labels = real_labels.to(device)

        ########################################################################
        # 2. Update Discriminator
        ########################################################################
        # Freeze G and E
        requires_grad(generator, False)
        requires_grad(encoder,   False)
        requires_grad(discriminator, True)

        # Encode real images
        encoded_real = encoder(real_img)
        # Classify real images
        real_logits = classifier(real_img)
        _, encoded_labels_int = torch.max(real_logits, 1)
        encoded_labels = F.one_hot(
            encoded_labels_int, num_classes=args.classifier_nof_classes
        )
#here
        # Generate fake images from encoded real
        fake_img_by_encoded, _ = generator(
            [encoded_real], 
            labels=encoded_labels, 
            input_is_latent=True
        )

        # Generate fake images from random noise/label
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        random_label_int = torch.randint(
            0, args.classifier_nof_classes, (args.batch, 1), device=device
        )
        random_label = F.one_hot(
            random_label_int, num_classes=args.classifier_nof_classes
        )
#here
        fake_img_by_z, _ = generator(
            noise, 
            labels=random_label, 
            input_is_latent=False
        )
#here
        # Get D outputs
        fake_pred_by_encoded = discriminator(
            fake_img_by_encoded.detach(), 
            encoded_labels.detach()
        )
#here
        fake_pred_by_z = discriminator(
            fake_img_by_z.detach(), 
            random_label
        )
#here
        real_labels = F.one_hot(real_labels, num_classes=args.classifier_nof_classes)
    
        real_pred = discriminator(real_img, real_labels)

        # D losses
        d_loss_encoded = d_logistic_loss(real_pred, fake_pred_by_encoded)
        d_loss_z = d_logistic_loss(real_pred, fake_pred_by_z)
        d_loss = d_loss_encoded + d_loss_z

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        # R1 regularization
        if i % args.d_reg_every == 0:
            real_img.requires_grad = True
            real_pred = discriminator(real_img, real_labels)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            r1_final = (args.r1 / 2 * r1_loss * args.d_reg_every)
            r1_final.backward()
            d_optim.step()
            real_img.requires_grad = False

        ########################################################################
        # 3. Update Generator (two ways of generating fakes)
        ########################################################################
        # Unfreeze G, freeze D
        requires_grad(generator, True)
        requires_grad(encoder,   False)
        requires_grad(discriminator, False)

        # Generate fakes from random noise/labels
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        random_label_int = torch.randint(
            0, args.classifier_nof_classes, (args.batch, 1), device=device
        )
        random_label = F.one_hot(
            random_label_int, 
            num_classes=args.classifier_nof_classes
        )
#here
        fake_img_by_z, _ = generator(noise, labels=random_label, input_is_latent=False)
        fake_pred_by_z = discriminator(fake_img_by_z, random_label)

        # Generate fakes from encoder-latents
        encoded_real = encoder(real_img)
        real_logits = classifier(real_img)
        _, encoded_labels_int = torch.max(real_logits, 1)
        encoded_labels = F.one_hot(
            encoded_labels_int, 
            num_classes=args.classifier_nof_classes
        )
#here         
        fake_img_by_encoded, _ = generator(
            [encoded_real.detach()], 
            labels=encoded_labels.detach(), 
            input_is_latent=True
        )
#here
        fake_pred_by_encoded = discriminator(fake_img_by_encoded, encoded_labels)

        # G loss
        g_loss_z = g_nonsaturating_loss(fake_pred_by_z)
        g_loss_encoded = g_nonsaturating_loss(fake_pred_by_encoded)
        g_loss = (g_loss_z + g_loss_encoded) / 2.0

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # Path length regularization
        if i % args.g_reg_every == 0:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)

            # Another random label
            random_label_int = torch.randint(
                0, args.classifier_nof_classes, (path_batch_size, 1), device=device
            )
            random_label = F.one_hot(
                random_label_int, 
                num_classes=args.classifier_nof_classes
            )
#here 
            fake_img_path_reg, latents = generator(
                noise, 
                labels=random_label, 
                input_is_latent=False, 
                return_latents=True
            )
            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img_path_reg, latents, mean_path_length
            )

            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss
            generator.zero_grad()
            weighted_path_loss.backward()
            g_optim.step()

            mean_path_length_avg = float(mean_path_length.detach().item())

        ########################################################################
        # 4. Update Encoder
        #    Also update Generator to better reconstruct real images
        ########################################################################
        requires_grad(generator, True)
        requires_grad(encoder,   True)
        requires_grad(discriminator, False)

        # Encode real -> G -> reconstructed
        real_encoded = encoder(real_img)
        real_logits = classifier(real_img)
        _, encoded_labels_int = torch.max(real_logits, 1)
        encoded_labels = F.one_hot(
            encoded_labels_int, num_classes=args.classifier_nof_classes
        )
#here
        fake_img, _ = generator(
            [real_encoded], 
            labels=encoded_labels, 
            input_is_latent=True
        )
        fake_logits = classifier(fake_img)

        # Classification alignment loss
        logsoft = torch.nn.LogSoftmax(dim=1)
        class_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(
            logsoft(fake_logits),
            logsoft(real_logits)
        )

        # Reconstruction losses
        fake_encoded = encoder(fake_img)
        reconstruct_loss_x = F.l1_loss(fake_img, real_img)
        reconstruct_loss_w = F.l1_loss(fake_encoded, real_encoded)
        # We assume the function "lpips" is defined or imported
        reconstruct_loss_lpips = lpips(
            fake_img / fake_img.max(), 
            real_img / real_img.max(), 
            net_type='alex', 
            version='0.1'
        ).flatten()

        reconstruct_loss = reconstruct_loss_x + reconstruct_loss_w #+ reconstruct_loss_lpips

        total_loss = class_loss + reconstruct_loss

        encoder.zero_grad()
        generator.zero_grad()
        total_loss.backward()
        e_optim.step()
        g_optim.step()

        # Update EMA
        accumulate(g_ema, generator, accum)

        ########################################################################
        # 5. Logging & Saving
        ########################################################################
        d_loss_val = d_loss.item()
        g_loss_val = g_loss.item()
        r1_val = r1_loss.item()
        path_loss_val = path_loss.item()
        class_loss_val = class_loss.item()
        reconstruct_loss_val = reconstruct_loss.mean().item()

        pbar.set_description(
            (
                f"d: {d_loss_val:.2f}; g: {g_loss_val:.2f}; "
                f"r1: {r1_val:.2f}; path: {path_loss_val:.2f}; "
                f"mean path: {mean_path_length_avg:.2f}; class: {class_loss_val:.2f}; "
                f"recon: {reconstruct_loss_val:.2f}"
            )
        )

        # Example of saving sample images
        if i % args.save_samples_every == 0 and i > 0:
            save_sample_images(g_ema, args, i, sample_z, sample_z_label)
            save_real_vs_encoded(generator, encoder, args, i, sample_images, sample_labels)

        # Example of saving checkpoints
        if i % args.save_checkpoint_every == 0 and i > 0:
            filename = os.path.join(args.output_path, f"checkpoint/{args.dataset_name}/{str(i).zfill(6)}.pt")
            torch.save(
                {
                    "g":       generator.state_dict() if not hasattr(generator, 'module') else generator.module.state_dict(),
                    "d":       discriminator.state_dict() if not hasattr(discriminator, 'module') else discriminator.module.state_dict(),
                    "e":       encoder.state_dict() if not hasattr(encoder, 'module') else encoder.module.state_dict(),
                    "g_ema":   g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "e_optim": e_optim.state_dict(),
                    "args":    args,
                    "ada_aug_p": ada_aug_p,
                },
                filename,
            )


