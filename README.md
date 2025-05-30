# PCG: Perceptual Counterfactual Geodesics


A PyTorch implementation of **Perceptual Counterfactual Geodesics (PCG)** for generating semantically meaningful counterfactual explanations on high-dimensional image data. PCG traces geodesics in the latent space of a GAN under a robust perceptual metric, producing on-manifold, perceptually-aligned perturbations.

---

## 💻 Hardware & Compute

- All experiments were executed on a Slurm-based cluster using a single NVIDIA H100 GPU.
- Training each StyleGAN2 generator on AFHQ and PlantVillage required approximately **140 H100 GPU-hours**.
- The `stylegan2-ada-pytorch/` folder includes the official StyleGAN2-ADA implementation from NVlabs (https://github.com/NVlabs/stylegan2-ada-pytorch), which is used for the FFHQ experiments.
