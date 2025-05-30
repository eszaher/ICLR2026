import config as cfg 
class Args:
    def __init__(self, dataset_path=cfg.dataset_path, dataset_name=cfg.dataset, arch=cfg.arch, iter=cfg.iter, batch=cfg.batch, output_path=cfg.output_path,
                 size=cfg.size, latent=cfg.latent, mlp=cfg.mlp, ckpt=cfg.ckpt, lr=cfg.lr, finetune_generator=cfg.finetune_generator):
        
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.arch = arch  # (stylegan2 | autoencoder | StyleEx)
        self.iter = iter
        self.batch = batch
        self.output_path = output_path

        self.n_sample = 16
        self.size = size
        self.r1 = 10
        self.path_regularize = 2
        self.path_batch_shrink = 2
        self.d_reg_every = 16
        self.g_reg_every = 4
        self.mixing = 0.9
        self.ckpt = ckpt
        self.lr = lr
        self.channel_multiplier = 2
        self.wandb = False
        self.augment = False
        self.augment_p = 0.0
        self.ada_target = 0.6
        self.ada_length = 100000
        self.ada_every = 256

        # -- Conditional parameters if using a classifier
        self.classifier_nof_classes = 2
        self.classifier_ckpt = None
        self.cgan = False
        self.encoder_ckpt = None
        self.compare_to_healthy = False
        self.filter_label = None

        # -- Additional attributes used for building/training
        self.latent = latent
        self.n_mlp = mlp
        self.embedding_size = 10

        # -- Control how often samples and checkpoints are saved
        self.save_samples_every = 500
        self.save_checkpoint_every = 500

        # -- For continuing training from checkpoint
        self.start_iter = 0

        # -- Single-GPU (no distributed)
        self.distributed = False
        self.local_rank = 0  

        # If finetuning the generator during encoder training
        self.finetune_generator = finetune_generator
        
        
