#!/usr/bin/env python3
"""
Train a 2D AutoencoderKL (with 3-channel output heads: CT, IPH, IVH) from per-slice NIfTI files.

Expected input:
- THREE folders containing matching per-slice NIfTI filenames (e.g. Case_slice_012.nii or .nii.gz)
    1) CT_DIR:      CT slice volumes shaped (H, W, 1)
    2) SEG_IPH_DIR: IPH slice volumes shaped (H, W, 1)
    3) SEG_IVH_DIR: IVH slice volumes shaped (H, W, 1)

Assumptions:
- Seg slices are either {0,1} or {0,255} or soft masks in [0,1].
- CT intensities are scaled from [CT_A_MIN, CT_A_MAX] -> [0, 1] at training time.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn

IMAGE_SIZE = 384

_CT_BASE = os.environ.get("CT_BASE_DIR", os.path.join("data", f"nifti_slices_{IMAGE_SIZE}"))
CT_DIR = os.environ.get("CT_DIR", os.path.join(_CT_BASE, "ct_train"))
SEG_IPH_DIR = os.environ.get("SEG_IPH_DIR", os.path.join(_CT_BASE, "seg_iph_train"))
SEG_IVH_DIR = os.environ.get("SEG_IVH_DIR", os.path.join(_CT_BASE, "seg_ivh_train"))

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join("output", f"autoencoder_{IMAGE_SIZE}"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 4
ACCUMULATE_GRAD_BATCHES = 16
MIXED_PRECISION = "no"  # "fp16" or "no"
TOTAL_STEPS = 6500
LOG_INTERVAL = 50
SAVE_INTERVAL = 500
NUM_WORKERS = 8
PERSISTENT_WORKERS = True

CT_A_MIN = 0
CT_A_MAX = 100

SEG_THRESHOLD = 0.5

AFFINE_PROB = 0.5
ROT_DEG = 5.0
TRANSLATE_PX = 1
SCALE_FRAC = 0.05

KL_WEIGHT = 1e-8
SEG_LOSS_WEIGHT = 0.002
ADVERSARIAL_WEIGHT = 0.005

PERCEPTUAL_WEIGHT = 0.001
WARMUP_STEPS_PERCEPT = 2500

WARMUP_STEPS_SEG = 0
WARMUP_STEPS_ADV = 5000

W_IPH, W_IVH = 1.0, 2.0
FOREGROUND_WEIGHT = 1.0
BACKGROUND_WEIGHT = 1.0

WANDB_PROJECT = "autoencoder-training-nifti-slices"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "public-entity")
WANDB_ENABLED = True

USE_PRETRAINED_MODELS = True
LOAD_OPTIMIZERS = False
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", os.path.join("checkpoints", "checkpoint_step_6500.pth"))
START_GLOBAL_STEP = 0

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
torch.backends.cudnn.benchmark = True

try:
    from monai import transforms
    from monai.data import DataLoader, Dataset
    from monai.networks.layers import Act
    from monai.losses import PatchAdversarialLoss, DiceLoss
    from monai.networks.nets import AutoencoderKL, PatchDiscriminator
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "MONAI is not installed in this Python environment.\n"
        "Make sure you run this script from the same conda env where MONAI is installed."
    ) from e

PERCEPTUAL_AVAILABLE = True
try:
    from monai.losses import PerceptualLoss
except Exception:
    PERCEPTUAL_AVAILABLE = False

import wandb
from torchvision.utils import make_grid, save_image
from accelerate import Accelerator
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss


class MultiHeadAEKL(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 3,
        feature_channels: int = 256,
        latent_channels: int = 4,
        num_res_blocks: int = 2,
        channels: tuple = (64, 128, 128),
        norm_num_groups: int = 32,
        attention_levels: tuple = (False, False, False),
    ):
        super().__init__()
        self.backbone = AutoencoderKL(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_channels,
            channels=channels,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            attention_levels=attention_levels,
        )
        self.ct_head = nn.Conv2d(feature_channels, 1, 1)
        self.iph_head = nn.Conv2d(feature_channels, 1, 1)
        self.ivh_head = nn.Conv2d(feature_channels, 1, 1)

    def forward(self, x):
        y, z_mu, z_sigma = self.backbone(x)
        ct = self.ct_head(y)
        iph = self.iph_head(y)
        ivh = self.ivh_head(y)
        recon = torch.cat([ct, iph, ivh], dim=1)
        return recon, z_mu, z_sigma


def _is_nifti(fn: str) -> bool:
    return fn.endswith(".nii") or fn.endswith(".nii.gz")


def _list_common_cases(ct_dir, seg_iph_dir, seg_ivh_dir):
    ct_files = {f for f in os.listdir(ct_dir) if _is_nifti(f)}
    iph_files = {f for f in os.listdir(seg_iph_dir) if _is_nifti(f)}
    ivh_files = {f for f in os.listdir(seg_ivh_dir) if _is_nifti(f)}
    common = sorted(list(ct_files & iph_files & ivh_files))
    if len(common) == 0:
        raise RuntimeError(
            "No matching NIfTI filenames across CT_DIR / SEG_IPH_DIR / SEG_IVH_DIR.\n"
            "Make sure the SAME filenames exist in all 3 folders."
        )
    return common


def _make_datalist(ct_dir, seg_iph_dir, seg_ivh_dir, filenames):
    return [
        {
            "image": os.path.join(ct_dir, fn),
            "seg1": os.path.join(seg_iph_dir, fn),
            "seg2": os.path.join(seg_ivh_dir, fn),
            "filename": fn,
        }
        for fn in filenames
    ]


def _normalize_and_binarize_seg(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mx = float(x.max()) if x.size > 0 else 0.0
    if mx > 1.5:
        x = x / 255.0
    x = (x > SEG_THRESHOLD).astype(np.float32)
    return x


def log_reconstructions(model, loader, device, step, output_dir, tag="train"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        ct = batch["image"].to(device)
        seg1 = batch["seg1"].to(device)
        seg2 = batch["seg2"].to(device)

        inputs = torch.cat([ct, seg1, seg2], dim=1)
        recon, _, _ = model(inputs)

        n = min(ct.size(0), 8)
        grid_ct_orig = make_grid(ct[:n], nrow=n, normalize=False)
        grid_ct_recon = make_grid(recon[:, 0:1][:n], nrow=n, normalize=False)

        grid_seg1_orig = make_grid(seg1[:n], nrow=n, normalize=False)
        grid_seg1_recon = make_grid(torch.sigmoid(recon[:, 1:2])[:n], nrow=n, normalize=False)

        grid_seg2_orig = make_grid(seg2[:n], nrow=n, normalize=False)
        grid_seg2_recon = make_grid(torch.sigmoid(recon[:, 2:3])[:n], nrow=n, normalize=False)

        combined = torch.cat(
            [
                grid_ct_orig,
                grid_ct_recon,
                grid_seg1_orig,
                grid_seg1_recon,
                grid_seg2_orig,
                grid_seg2_recon,
            ],
            dim=1,
        )
        save_image(combined, os.path.join(output_dir, f"{tag}_step_{step}.png"))
    model.train()


def compute_kl(z_mu: torch.Tensor, z_sigma_or_logvar: torch.Tensor) -> torch.Tensor:
    x = z_sigma_or_logvar
    if x.min().item() < 0:
        logvar = x
        kl = 0.5 * torch.sum(z_mu.pow(2) + torch.exp(logvar) - logvar - 1.0, dim=[1, 2, 3])
    else:
        sigma = x
        kl = 0.5 * torch.sum(
            z_mu.pow(2) + sigma.pow(2) - torch.log(sigma.pow(2) + 1e-8) - 1.0, dim=[1, 2, 3]
        )
    return kl.mean()


def maybe_load_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    discriminator: nn.Module,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
):
    global_step = START_GLOBAL_STEP

    if not USE_PRETRAINED_MODELS:
        return global_step

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {os.path.basename(CHECKPOINT_PATH)}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    discriminator.load_state_dict(ckpt["discriminator_state_dict"], strict=True)

    if LOAD_OPTIMIZERS:
        if "optimizer_g_state_dict" in ckpt:
            optimizer_g.load_state_dict(ckpt["optimizer_g_state_dict"])
        if "optimizer_d_state_dict" in ckpt:
            optimizer_d.load_state_dict(ckpt["optimizer_d_state_dict"])
        if "scheduler_g_state_dict" in ckpt:
            scheduler_g.load_state_dict(ckpt["scheduler_g_state_dict"])
        if "scheduler_d_state_dict" in ckpt:
            scheduler_d.load_state_dict(ckpt["scheduler_d_state_dict"])
        if "global_step" in ckpt:
            global_step = int(ckpt["global_step"])

    if accelerator.is_main_process:
        msg = f"Loaded weights from {os.path.basename(CHECKPOINT_PATH)}"
        if LOAD_OPTIMIZERS:
            msg += f" (resumed global_step={global_step})"
        else:
            msg += " (weights-only; optimizer/scheduler reset)"
        print(msg)

    return global_step


def main():
    accelerator = Accelerator(mixed_precision=MIXED_PRECISION, gradient_accumulation_steps=ACCUMULATE_GRAD_BATCHES)
    device = accelerator.device
    print(f"Using device: {device}")

    if WANDB_ENABLED and accelerator.is_main_process:
        wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

    common = _list_common_cases(CT_DIR, SEG_IPH_DIR, SEG_IVH_DIR)
    print(f"Found {len(common)} matching SLICE NIfTIs (by filename)")

    rot_rad = float(np.deg2rad(ROT_DEG))

    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "seg1", "seg2"]),
            transforms.EnsureChannelFirstd(keys=["image", "seg1", "seg2"]),
            transforms.SqueezeDimd(keys=["image", "seg1", "seg2"], dim=-1),
            transforms.ScaleIntensityRanged(
                keys=["image"],
                a_min=CT_A_MIN,
                a_max=CT_A_MAX,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            transforms.Lambdad(keys=["seg1", "seg2"], func=_normalize_and_binarize_seg),
            transforms.SpatialPadd(
                keys=["image", "seg1", "seg2"],
                spatial_size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="constant",
            ),
            transforms.CenterSpatialCropd(keys=["image", "seg1", "seg2"], roi_size=(IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandAffined(
                keys=["image", "seg1", "seg2"],
                prob=AFFINE_PROB,
                rotate_range=[(-rot_rad, rot_rad), (-rot_rad, rot_rad)],
                translate_range=[(-TRANSLATE_PX, TRANSLATE_PX), (-TRANSLATE_PX, TRANSLATE_PX)],
                scale_range=[(-SCALE_FRAC, SCALE_FRAC), (-SCALE_FRAC, SCALE_FRAC)],
                spatial_size=[IMAGE_SIZE, IMAGE_SIZE],
                padding_mode="zeros",
                mode=("bilinear", "nearest", "nearest"),
            ),
            transforms.EnsureTyped(keys=["image", "seg1", "seg2"], dtype=torch.float32),
        ]
    )

    datalist = _make_datalist(CT_DIR, SEG_IPH_DIR, SEG_IVH_DIR, common)
    train_ds = Dataset(data=datalist, transform=train_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=True,
    )

    model = MultiHeadAEKL(
        spatial_dims=2,
        in_channels=3,
        channels=(64, 128, 256),
        latent_channels=4,
        num_res_blocks=2,
        norm_num_groups=32,
        attention_levels=(False, False, False),
    )

    discriminator = PatchDiscriminator(
        spatial_dims=2,
        num_layers_d=3,
        channels=128,
        in_channels=1,
        out_channels=1,
        kernel_size=4,
        activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
        norm="BATCH",
        bias=False,
        padding=1,
    )

    perceptual_loss = None
    effective_perceptual_weight = PERCEPTUAL_WEIGHT
    if PERCEPTUAL_AVAILABLE:
        try:
            perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="squeeze")
        except Exception as e:
            perceptual_loss = None
            effective_perceptual_weight = 0.0
            if accelerator.is_main_process:
                print(f"[WARN] PerceptualLoss disabled: {e}")
    else:
        perceptual_loss = None
        effective_perceptual_weight = 0.0
        if accelerator.is_main_process:
            print("[WARN] PerceptualLoss import failed. Disabling perceptual loss.")

    model = model.to(device)
    discriminator = discriminator.to(device)
    if perceptual_loss is not None:
        perceptual_loss = perceptual_loss.to(device)

    optimizer_g = torch.optim.Adam(model.parameters(), lr=5e-5)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-6)

    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=TOTAL_STEPS, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=TOTAL_STEPS, last_epoch=-1)

    global_step = maybe_load_checkpoint(
        accelerator=accelerator,
        model=model,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
    )

    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    dice_loss_fn = DiceLoss(sigmoid=True)
    bce_loss_fn = BCEWithLogitsLoss()

    model, discriminator, optimizer_g, optimizer_d, train_loader = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_loader
    )

    pbar = tqdm(
        total=TOTAL_STEPS,
        desc="Training",
        unit="step",
        disable=not accelerator.is_main_process,
        initial=min(global_step, TOTAL_STEPS),
    )

    total_start = time.time()

    accum_ct_loss = 0.0
    accum_seg_loss = 0.0
    accum_kl_loss = 0.0
    accum_gen_adv = 0.0
    accum_disc = 0.0
    accum_percept = 0.0
    accum_steps = 0

    while global_step < TOTAL_STEPS:
        model.train()
        discriminator.train()

        for batch in train_loader:
            current_seg_weight = 0.0 if global_step < WARMUP_STEPS_SEG else SEG_LOSS_WEIGHT
            train_discriminator = global_step >= WARMUP_STEPS_ADV
            current_adv_weight = 0.0 if not train_discriminator else ADVERSARIAL_WEIGHT

            if perceptual_loss is None:
                current_perceptual_weight = 0.0
            else:
                current_perceptual_weight = 0.0 if global_step < WARMUP_STEPS_PERCEPT else effective_perceptual_weight

            ct = batch["image"].to(device)
            s1 = batch["seg1"].to(device)
            s2 = batch["seg2"].to(device)
            inputs = torch.cat([ct, s1, s2], dim=1)

            with accelerator.accumulate(model):
                optimizer_g.zero_grad(set_to_none=True)
                with accelerator.autocast():
                    recon, z_mu, z_sigma = model(inputs)

                    ct_recons_loss = (((recon[:, 0:1] - ct) ** 2)).mean()

                    seg1_dice = dice_loss_fn(recon[:, 1:2], s1)
                    seg1_bce = bce_loss_fn(recon[:, 1:2], s1)
                    seg2_dice = dice_loss_fn(recon[:, 2:3], s2)
                    seg2_bce = bce_loss_fn(recon[:, 2:3], s2)
                    seg_loss = (W_IPH * (seg1_dice + seg1_bce) + W_IVH * (seg2_dice + seg2_bce))

                    recons_loss = ct_recons_loss + current_seg_weight * seg_loss
                    kl_loss = compute_kl(z_mu, z_sigma)

                    if perceptual_loss is not None and current_perceptual_weight > 0:
                        p_loss = perceptual_loss(recon[:, 0:1].float(), ct.float())
                    else:
                        p_loss = torch.tensor(0.0, device=device)

                    if train_discriminator:
                        logits_fake = discriminator(recon[:, 0:1].contiguous().float())[-1]
                        gen_adv_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    else:
                        gen_adv_loss = torch.tensor(0.0, device=device)

                    loss_g = (
                        recons_loss
                        + KL_WEIGHT * kl_loss
                        + current_perceptual_weight * p_loss
                        + current_adv_weight * gen_adv_loss
                    )

                accelerator.backward(loss_g)
                optimizer_g.step()

            if train_discriminator:
                with accelerator.accumulate(discriminator):
                    optimizer_d.zero_grad(set_to_none=True)
                    with accelerator.autocast():
                        logits_fake = discriminator(recon[:, 0:1].contiguous().detach())[-1]
                        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                        logits_real = discriminator(ct.contiguous().detach())[-1]
                        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                        disc_loss = 0.5 * (loss_d_fake + loss_d_real)
                        loss_d = current_adv_weight * disc_loss
                    accelerator.backward(loss_d)
                    optimizer_d.step()
            else:
                disc_loss = torch.tensor(0.0, device=device)

            if accelerator.sync_gradients:
                global_step += 1
                scheduler_g.step()
                scheduler_d.step()
                pbar.update(1)

                accum_ct_loss += float(ct_recons_loss.item())
                accum_seg_loss += float(seg_loss.item())
                accum_kl_loss += float(kl_loss.item())
                accum_percept += float(p_loss.item())
                accum_gen_adv += float(gen_adv_loss.item()) if train_discriminator else 0.0
                accum_disc += float(disc_loss.item()) if train_discriminator else 0.0
                accum_steps += 1

                if accelerator.is_main_process and (global_step % LOG_INTERVAL == 0):
                    denom = max(1, accum_steps)
                    log_dict = {
                        "Step": global_step,
                        "Train/CT_Recon_Loss": accum_ct_loss / denom,
                        "Train/Seg_Loss_raw": accum_seg_loss / denom,
                        "Train/KL": accum_kl_loss / denom,
                        "Train/Perceptual": accum_percept / denom,
                        "Train/GenAdv": accum_gen_adv / denom,
                        "Train/Disc": accum_disc / denom,
                        "lr_g": optimizer_g.param_groups[0]["lr"],
                        "lr_d": optimizer_d.param_groups[0]["lr"],
                        "w_seg": current_seg_weight,
                        "w_percept": current_perceptual_weight,
                        "w_adv": current_adv_weight,
                        "image_size": IMAGE_SIZE,
                        "loaded_ckpt": int(USE_PRETRAINED_MODELS),
                        "loaded_optim": int(LOAD_OPTIMIZERS),
                    }
                    if WANDB_ENABLED:
                        wandb.log(log_dict)

                    log_reconstructions(
                        accelerator.unwrap_model(model),
                        train_loader,
                        device,
                        global_step,
                        output_dir=os.path.join(OUTPUT_DIR, "output/train"),
                        tag="train",
                    )

                    accum_ct_loss = accum_seg_loss = accum_kl_loss = 0.0
                    accum_percept = accum_gen_adv = accum_disc = 0.0
                    accum_steps = 0

                if accelerator.is_main_process and (global_step % SAVE_INTERVAL == 0):
                    ckpt_dir = os.path.join(OUTPUT_DIR, "checkpoints")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt = {
                        "global_step": global_step,
                        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                        "discriminator_state_dict": accelerator.unwrap_model(discriminator).state_dict(),
                        "optimizer_g_state_dict": optimizer_g.state_dict(),
                        "optimizer_d_state_dict": optimizer_d.state_dict(),
                        "scheduler_g_state_dict": scheduler_g.state_dict(),
                        "scheduler_d_state_dict": scheduler_d.state_dict(),
                        "image_size": IMAGE_SIZE,
                    }
                    path = os.path.join(ckpt_dir, f"checkpoint_step_{global_step}.pth")
                    torch.save(ckpt, path)
                    print(f"Saved checkpoint: {os.path.basename(path)}")

                if global_step >= TOTAL_STEPS:
                    break

        if global_step >= TOTAL_STEPS:
            break

    pbar.close()
    total_time = time.time() - total_start
    print(f"Training completed in {total_time:.2f} seconds.")

    if WANDB_ENABLED and accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()