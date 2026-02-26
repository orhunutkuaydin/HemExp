from denoising_diffusion_pytorch import MultiHeadAEKL, Unet, GaussianDiffusion, Trainer
import os
import json
import time
import random
import numpy as np
import torch
import argparse

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the JSON config file",
    )
    return parser.parse_args()


args = parse_args()
CONFIG_PATH = args.config_path

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

cfg_run = cfg["run"]
cfg_data = cfg["data"]
cfg_ae = cfg["ae"]
cfg_model = cfg["model"]
cfg_diff = cfg["diffusion"]
cfg_train = cfg["train"]

set_seed(int(cfg_run.get("seed", 0)))

run_name = cfg_run["name"]
results_root = cfg_run["results_root"]
stamp = time.strftime("%Y%m%d_%H%M%S")
results_folder = os.path.join(results_root, f"{stamp}__{run_name}")
os.makedirs(results_folder, exist_ok=True)

with open(os.path.join(results_folder, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_latent = bool(cfg_ae["use_latent_diffusion"])

if use_latent:
    ae = MultiHeadAEKL(
        spatial_dims=2,
        in_channels=3,
        channels=(64, 128, 256),
        latent_channels=int(cfg_ae["latent_channels"]),
        num_res_blocks=2,
        norm_num_groups=32,
        attention_levels=(False, False, False),
    ).to(device)

    ckpt = torch.load(cfg_ae["ae_ckpt"], map_location=device)
    ae.load_state_dict(ckpt["model_state_dict"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
else:
    ae = None

model = Unet(
    dim=int(cfg_model["dim"]),
    dim_mults=tuple(cfg_model["dim_mults"]),
    flash_attn=bool(cfg_model.get("flash_attn", False)),
    channels=int(cfg_model["channels"]),
    use_clinical=bool(cfg_model["use_clinical"]),
    use_expansion_flag_input=bool(cfg_model["use_expansion_flag_input"]),
    predict_segmentation=False,
    use_latent_diffusion=use_latent,
    dropout=float(cfg_model.get("dropout", 0.0)),
    clinical_drop_prob=float(cfg_model.get("clinical_drop_prob", 0.0)),
    expansion_flag_drop_prob=float(cfg_model.get("expansion_flag_drop_prob", 0.0)),
    context_slices=cfg_model.get("context_slices"),
)

image_size = int(cfg_ae["latent_size"] if use_latent else cfg_ae["ae_input_size"])

diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps=int(cfg_diff["timesteps"]),
    sampling_timesteps=int(cfg_diff["sampling_timesteps"]),
    objective=str(cfg_diff.get("objective", "pred_v")),
    beta_schedule=str(cfg_diff.get("beta_schedule", "sigmoid")),
    auto_normalize=not use_latent,
    use_latent_diffusion=use_latent,
    ddim_sampling_eta=float(cfg_diff.get("ddim_eta", 0.0)),
    segmentation_loss_weight=float(cfg_diff.get("segmentation_loss_weight", 0.0)),
    min_snr_loss_weight=bool(cfg_diff.get("min_snr_loss_weight", False)),
    min_snr_gamma=float(cfg_diff.get("min_snr_gamma", 5.0)),
    autoencoder=ae,
)

trainer = Trainer(
    diffusion,
    clinical_csv=cfg_data["clinical_csv"],
    train_batch_size=int(cfg_train["batch_size"]),
    train_lr=float(cfg_train["lr"]),
    train_num_steps=int(cfg_train["steps"]),
    gradient_accumulate_every=int(cfg_train["grad_accum"]),
    latent_predict_delta=bool(cfg_diff.get("latent_predict_delta", True)),
    save_and_sample_every=int(cfg_train["save_every"]),
    ema_decay=float(cfg_train["ema_decay"]),
    amp=bool(cfg_train["amp"]),
    augment_flip=bool(cfg_data["augment"]["flip"]),
    augment_rotation=bool(cfg_data["augment"]["rotation"]),
    rotation_range=float(cfg_data["augment"]["rotation_range_deg"]),
    use_clinical=bool(cfg_model["use_clinical"]),
    use_expansion_flag_input=bool(cfg_model["use_expansion_flag_input"]),
    use_latent_diffusion=use_latent,
    ae_input_size=int(cfg_ae["ae_input_size"]),
    context_slices=int(cfg_model.get("context_slices")),
    context_slice_mm=float(cfg_data["context_mm_step"]),
    slice_thickness_csv=cfg_data.get("slice_thickness_csv", None),
    default_slice_thickness_mm=float(cfg_data.get("default_slice_thickness_mm", 5.0)),
    results_folder=results_folder,
    resume_checkpoint=cfg_train.get("resume_checkpoint", None),
    latent_mean=cfg_ae["mean"],
    latent_std=cfg_ae["std"],
    slices_root=cfg_data["slices_root"],
    train_hospitals=cfg_data["train_hospitals"],
    val_hospitals=cfg_data["val_hospitals"],
    ct_do_window=cfg_data["ct_do_window"],
    ct_window_min=cfg_data["ct_window_min"],
    ct_window_max=cfg_data["ct_window_max"],
)

trainer.train()