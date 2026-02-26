import os

CUDA_VISIBLE_DEVICES = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

import json
import math
import re
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn.functional as F
from ema_pytorch import EMA
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, MultiHeadAEKL
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import (
    norm_latent,
    unnorm_latent,
    extract_center_latent_from_stack,)

TEST_DIR = os.environ.get("TEST_DIR", str(Path("data") / "imagesTs"))
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", str(Path("output")))
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "experiment")

CONFIG_PATH = os.environ.get("CONFIG_PATH", str(Path("configs") / "config.json"))
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", str(Path("checkpoints") / "model.pt"))

OUTPUT_FOLDER = os.path.join(OUTPUT_ROOT, EXPERIMENT_NAME)
OUTPUT_FOLDER_SEG = os.path.join(OUTPUT_ROOT, f"{EXPERIMENT_NAME}_SEG")

USE_25D_CONTEXT = True
PREDICT_ONLY_IPH_SLICES = False

with open(CONFIG_PATH, "r") as f:
    cfg = json.load(f)

cfg_data = cfg["data"]
cfg_ae = cfg["ae"]
cfg_model = cfg["model"]
cfg_diff = cfg["diffusion"]

CLINICAL_CSV_PATH = os.environ.get("CLINICAL_CSV_PATH", str(Path("data") / "clinical_data.csv"))
TOTAL_SAMPLING_BUDGET = 5
INFER_BATCH_SIZE = 2
CFG_SCALE = 1.0
EXPANSION_FLAG = "ground_truth"  # "ground_truth" / "predicted" / 0 / 1
SAMPLING_TIMESTEPS_OVERRIDE = 100
SAMPLING_ETA_OVERRIDE = 0.5

SLICE_MARGIN_CM = 1.0

SEG_THRESHOLD = 0.5
MIN_LESION_PIXELS_2D = 1
SLICE_AXIS = 2
PREDICTED_EXPANSION_PROB_COL = "Predicted_prob"
FORCE_AT_LEAST_ONE_EACH_IF_INTERMEDIATE = True

WRITE_CT_AS_WINDOWED_HU = True
RESCALED_PHI = 0.0

DEBUG_SHAPE_ASSERTS = False


def _anonymize_identifier(x: str) -> str:
    s = str(x)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()[:10]
    return f"anon_{h}"


def _safe_path_for_log(p: str) -> str:
    try:
        return Path(str(p)).name
    except Exception:
        return "<path>"


def _context_indices(z: int, k: int, zmax: int):
    assert k % 2 == 1, "CONTEXT_SLICES must be odd (e.g., 1,3,5)"
    half = k // 2
    idxs = []
    for off in range(-half, half + 1):
        zz = z + off
        if zz < 0:
            zz = 0
        elif zz >= zmax:
            zz = zmax - 1
        idxs.append(zz)
    return idxs


def build_cond_stack_for_z(
    ct_hwz: np.ndarray,
    iph_hwz: np.ndarray,
    ivh_hwz: np.ndarray,
    z: int,
    k: int,
    out_size: int,
    use_25d: bool = True,
    thickness_mm: float = 5.0,
    mm_step: float = 5.0,
):
    H, W, Z = ct_hwz.shape

    if use_25d:
        idxs = _context_indices_mm(z, k, Z, thickness_mm=thickness_mm, mm_step=mm_step)
    else:
        idxs = [z] * k

    ct_slices, iph_slices, ivh_slices = [], [], []
    for zz in idxs:
        ct2d01 = _normalize_ct(ct_hwz[:, :, zz])
        iph2d01 = _normalize_seg(iph_hwz[:, :, zz])
        ivh2d01 = _normalize_seg(ivh_hwz[:, :, zz])

        ct_t = torch.from_numpy(ct2d01)[None, None, ...].float()
        iph_t = torch.from_numpy(iph2d01)[None, None, ...].float()
        ivh_t = torch.from_numpy(ivh2d01)[None, None, ...].float()

        ct_t = _resize_bchw(ct_t, out_size, is_mask=False)
        iph_t = _resize_bchw(iph_t, out_size, is_mask=True)
        ivh_t = _resize_bchw(ivh_t, out_size, is_mask=True)

        ct_slices.append(ct_t)
        iph_slices.append(iph_t)
        ivh_slices.append(ivh_t)

    ct_stack = torch.cat(ct_slices, dim=0)
    iph_stack = torch.cat(iph_slices, dim=0)
    ivh_stack = torch.cat(ivh_slices, dim=0)
    return ct_stack, iph_stack, ivh_stack


def _safe_float(x, default=0.0):
    if x is None:
        return float(default)
    try:
        if pd.isna(x):
            return float(default)
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return float(default)


def load_expansion_flag_from_csv(
    clinical_csv_path: str,
    patient_id: str,
    expansion_definition: str,
) -> int:
    df = pd.read_csv(clinical_csv_path)

    pid_col = CLIN_COLS["patient_id"]
    if pid_col not in df.columns:
        raise ValueError(f"CSV missing column {pid_col!r}. Available: {list(df.columns)}")

    row = df.loc[df[pid_col].astype(str) == str(patient_id)]
    if len(row) == 0:
        raise ValueError("Patient not found in clinical CSV.")
    row = row.iloc[0]

    exp = row.get(expansion_definition)
    if exp is None or (isinstance(exp, float) and np.isnan(exp)) or pd.isna(exp):
        return 0

    try:
        exp_int = int(exp)
    except Exception:
        return 0

    return 1 if exp_int == 1 else 0


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def load_clinical_vars_from_csv(clinical_csv_path: str, patient_id: str) -> torch.Tensor:
    df = pd.read_csv(clinical_csv_path)
    row = df.loc[df[CLIN_COLS["patient_id"]].astype(str) == str(patient_id)].iloc[0]

    onset = float(row.get(CLIN_COLS["onset"], 0.0))
    ich_vol = float(row.get(CLIN_COLS["ich_vol"], 0.0))
    anticoag = float(row.get(CLIN_COLS["anticoag"], 0.0))
    antiplate = float(row.get(CLIN_COLS["antiplate"], 0.0))
    age = float(row.get(CLIN_COLS["age"], 0.0))
    gcs = float(row.get(CLIN_COLS["gcs"], 3.0))
    sbp = float(row.get(CLIN_COLS["sbp"], 0.0))

    anticoag = 1.0 if anticoag == 1.0 else 0.0
    antiplate = 1.0 if antiplate == 1.0 else 0.0

    onset = _clamp(onset, 0.0, 24.0) / 24.0
    ich_vol = _clamp(ich_vol, 0.0, 250.0) / 250.0
    age = _clamp(age, 0.0, 100.0) / 100.0
    gcs = (_clamp(gcs, 3.0, 15.0) - 3.0) / 12.0
    sbp = _clamp(sbp, 0.0, 250.0) / 250.0

    return torch.tensor(
        [onset, ich_vol, anticoag, antiplate, age, gcs, sbp],
        dtype=torch.float32,
    ).unsqueeze(0)


NNUNET_CH_CT = 0
NNUNET_CH_IPH = 1
NNUNET_CH_IVH = 2


def collect_nnunet_flat_cases(images_dir: str, required_channels=(0, 1, 2)) -> dict:
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {_safe_path_for_log(str(images_dir))}")

    pat = re.compile(r"^(?P<pid>.+)_(?P<ch>\d{4})\.nii(?:\.gz)?$")

    tmp = {}
    for f in images_dir.iterdir():
        if not (f.name.endswith(".nii") or f.name.endswith(".nii.gz")):
            continue
        m = pat.match(f.name)
        if m is None:
            continue
        pid = m.group("pid")
        ch = int(m.group("ch"))
        tmp.setdefault(pid, {})[ch] = str(f)

    cases = {}
    for pid, ch_map in tmp.items():
        missing = [ch for ch in required_channels if ch not in ch_map]
        if missing:
            print(f"[SKIP] case: missing channels {missing}")
            continue
        cases[pid] = ch_map

    if len(cases) == 0:
        raise RuntimeError(
            f"No valid nnU-Net cases found in {_safe_path_for_log(str(images_dir))}. "
            f"Expected files like <case>_0000.nii.gz etc."
        )

    return cases


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SLICE_THICKNESS_CSV = cfg_data.get("slice_thickness_csv", None)

CONTEXT_MM_STEP = float(cfg_data.get("context_mm_step", 5.0))
DEFAULT_SLICE_THICKNESS_MM = float(cfg_data.get("default_slice_thickness_mm", 5.0))

CT_DO_WINDOW = bool(cfg_data.get("ct_do_window", True))
CT_WINDOW_MIN = float(cfg_data.get("ct_window_min", 0.0))
CT_WINDOW_MAX = float(cfg_data.get("ct_window_max", 100.0))

CLIN_COLS = cfg_data.get(
    "clin_cols",
    {
        "patient_id": "Patient_ID",
        "onset": "time_from_onset_to_CT",
        "ich_vol": "IPH_baseline_ml",
        "anticoag": "Anticoagulant",
        "antiplate": "Antiplatelet",
        "age": "Age",
        "gcs": "GCS",
        "sbp": "Systolic",
    },
)

USE_LATENT_DIFFUSION = bool(cfg_ae["use_latent_diffusion"])
AUTOENCODER_WEIGHTS_PATH = cfg_ae["ae_ckpt"]

AE_INPUT_SIZE = int(cfg_ae["ae_input_size"])
LATENT_SIZE = int(cfg_ae["latent_size"])
MODEL_IMAGE_SIZE = AE_INPUT_SIZE
CONTEXT_SLICES = int(cfg_model.get("context_slices", 1))

USE_CLINICAL = bool(cfg_model.get("use_clinical", False))
USE_EXPANSION_FLAG_INPUT = bool(cfg_model.get("use_expansion_flag_input", False))

LATENT_MEAN = list(cfg_ae["mean"])
LATENT_STD = list(cfg_ae["std"])

SAMPLING_TIMESTEPS = (
    int(SAMPLING_TIMESTEPS_OVERRIDE)
    if SAMPLING_TIMESTEPS_OVERRIDE is not None
    else int(cfg_diff.get("sampling_timesteps", 100))
)
SAMPLING_ETA = (
    float(SAMPLING_ETA_OVERRIDE)
    if SAMPLING_ETA_OVERRIDE is not None
    else float(cfg_diff.get("ddim_eta", 0.0))
)

PREDICT_DELTA = bool(cfg_diff.get("latent_predict_delta", True))
SEGMENTATION_LOSS_WEIGHT = float(cfg_diff.get("segmentation_loss_weight", 0.0))


def build_diffusion_and_autoencoder():
    ae = None
    if USE_LATENT_DIFFUSION:
        ae = MultiHeadAEKL(
            spatial_dims=2,
            in_channels=3,
            channels=(64, 128, 256),
            latent_channels=int(cfg_ae["latent_channels"]),
            num_res_blocks=2,
            norm_num_groups=32,
            attention_levels=(False, False, False),
        ).to(DEVICE)

        ckpt = torch.load(AUTOENCODER_WEIGHTS_PATH, map_location=DEVICE)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        ae.load_state_dict(state, strict=True)

        ae.eval()
        for p in ae.parameters():
            p.requires_grad = False

    model = Unet(
        dim=int(cfg_model["dim"]),
        dim_mults=tuple(cfg_model["dim_mults"]),
        flash_attn=bool(cfg_model.get("flash_attn", False)),
        channels=int(cfg_model["channels"]),
        use_clinical=USE_CLINICAL,
        use_expansion_flag_input=USE_EXPANSION_FLAG_INPUT,
        predict_segmentation=False,
        use_latent_diffusion=USE_LATENT_DIFFUSION,
        dropout=float(cfg_model.get("dropout", 0.0)),
        clinical_drop_prob=float(cfg_model.get("clinical_drop_prob", 0.0)),
        expansion_flag_drop_prob=float(cfg_model.get("expansion_flag_drop_prob", 0.0)),
        context_slices=CONTEXT_SLICES,
    ).to(DEVICE)

    diffusion = GaussianDiffusion(
        model,
        image_size=LATENT_SIZE if USE_LATENT_DIFFUSION else AE_INPUT_SIZE,
        timesteps=int(cfg_diff.get("timesteps", 1000)),
        sampling_timesteps=SAMPLING_TIMESTEPS,
        auto_normalize=not USE_LATENT_DIFFUSION,
        use_latent_diffusion=USE_LATENT_DIFFUSION,
        ddim_sampling_eta=SAMPLING_ETA,
        segmentation_loss_weight=SEGMENTATION_LOSS_WEIGHT,
        autoencoder=ae,
    ).to(DEVICE)

    return diffusion, ae, USE_LATENT_DIFFUSION


def load_expansion_prob_from_csv(clinical_csv_path: str, patient_id: str, prob_col: str) -> float:
    df = pd.read_csv(clinical_csv_path)

    pid_col = CLIN_COLS["patient_id"]
    if pid_col not in df.columns:
        raise ValueError(f"CSV missing column {pid_col!r}. Available: {list(df.columns)}")

    if prob_col not in df.columns:
        raise ValueError(f"CSV missing predicted prob column {prob_col!r}. Available: {list(df.columns)}")

    row = df.loc[df[pid_col].astype(str) == str(patient_id)]
    if len(row) == 0:
        raise ValueError("Patient not found in clinical CSV.")
    row = row.iloc[0]

    p = row.get(prob_col)
    if p is None or pd.isna(p):
        raise ValueError(f"[{_anonymize_identifier(patient_id)}] {prob_col} is missing/NaN (no fallback allowed).")

    p = float(p)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"[{_anonymize_identifier(patient_id)}] {prob_col}={p} is not in [0,1].")

    return p


def make_flag_schedule(p_exp: float, budget: int) -> list[int]:
    if p_exp <= 0.0:
        return [0] * budget
    if p_exp >= 1.0:
        return [1] * budget

    n1 = int(round(p_exp * budget))
    n1 = max(0, min(budget, n1))
    n0 = budget - n1

    if FORCE_AT_LEAST_ONE_EACH_IF_INTERMEDIATE and budget >= 2:
        if n1 == 0:
            n1, n0 = 1, budget - 1
        if n0 == 0:
            n0, n1 = 1, budget - 1

    return [1] * n1 + [0] * n0


def _load_seg_as_iph_ivh_masks(seg_path: str):
    seg = np.asarray(nib.load(seg_path).dataobj)
    if seg.ndim == 4 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    iph = (seg == 1).astype(np.float32)
    ivh = (seg == 2).astype(np.float32)
    return iph, ivh


def save_prob_maps_from_samples(out_patient_dir: str, flag_schedule: list[int], p_exp: float, ref_ct_path: str):
    ref = nib.load(ref_ct_path)
    ref_shape = np.asarray(ref.dataobj).shape
    if len(ref_shape) == 4 and ref_shape[-1] == 1:
        ref_shape = ref_shape[:-1]

    exp1_idxs = [i for i, f in enumerate(flag_schedule) if f == 1]
    exp0_idxs = [i for i, f in enumerate(flag_schedule) if f == 0]

    def accumulate(idxs):
        sum_iph = np.zeros(ref_shape, dtype=np.float32)
        sum_ivh = np.zeros(ref_shape, dtype=np.float32)
        for s in idxs:
            seg_path = os.path.join(out_patient_dir, f"sample_{s:02d}", "seg.nii.gz")
            iph, ivh = _load_seg_as_iph_ivh_masks(seg_path)
            sum_iph += iph
            sum_ivh += ivh
        n = max(1, len(idxs))
        return sum_iph / n, sum_ivh / n, len(idxs)

    iph1, ivh1, n1 = (
        accumulate(exp1_idxs)
        if len(exp1_idxs) > 0
        else (np.zeros(ref_shape, np.float32), np.zeros(ref_shape, np.float32), 0)
    )
    iph0, ivh0, n0 = (
        accumulate(exp0_idxs)
        if len(exp0_idxs) > 0
        else (np.zeros(ref_shape, np.float32), np.zeros(ref_shape, np.float32), 0)
    )

    iph_final = p_exp * iph1 + (1.0 - p_exp) * iph0
    ivh_final = p_exp * ivh1 + (1.0 - p_exp) * ivh0

    prob_dir = os.path.join(out_patient_dir, "prob_maps")
    os.makedirs(prob_dir, exist_ok=True)

    def _save(arr, name):
        hdr = ref.header.copy()
        img = nib.Nifti1Image(arr.astype(np.float32), ref.affine, hdr)
        nib.save(img, os.path.join(prob_dir, name))

    _save(iph1, "iph_prob_if_expand.nii.gz")
    _save(ivh1, "ivh_prob_if_expand.nii.gz")
    _save(iph0, "iph_prob_if_noexpand.nii.gz")
    _save(ivh0, "ivh_prob_if_noexpand.nii.gz")
    _save(iph_final, "iph_prob_final.nii.gz")
    _save(ivh_final, "ivh_prob_final.nii.gz")

    with open(os.path.join(prob_dir, "meta.txt"), "w") as f:
        f.write(f"p_exp={p_exp}\n")
        f.write(f"n_expand1={n1}\n")
        f.write(f"n_expand0={n0}\n")


def _normalize_seg(seg_2d: np.ndarray) -> np.ndarray:
    seg_2d = seg_2d.astype(np.float32)
    mx = float(seg_2d.max()) if seg_2d.size else 0.0
    if mx > 1.5:
        seg_2d = seg_2d / 255.0
    return seg_2d


def _normalize_ct(ct_2d: np.ndarray) -> np.ndarray:
    ct_2d = ct_2d.astype(np.float32)
    if CT_DO_WINDOW:
        ct_2d = np.clip(ct_2d, CT_WINDOW_MIN, CT_WINDOW_MAX)
        ct_2d = (ct_2d - CT_WINDOW_MIN) / (CT_WINDOW_MAX - CT_WINDOW_MIN + 1e-8)
    else:
        mn, mx = float(ct_2d.min()), float(ct_2d.max())
        ct_2d = (ct_2d - mn) / (mx - mn + 1e-8)
    return ct_2d


def _denormalize_ct(ct01: np.ndarray) -> np.ndarray:
    ct01 = np.clip(ct01, 0.0, 1.0)
    if WRITE_CT_AS_WINDOWED_HU:
        return ct01 * (CT_WINDOW_MAX - CT_WINDOW_MIN) + CT_WINDOW_MIN
    return ct01


def _iph_present(mask_2d01: np.ndarray) -> bool:
    return int((mask_2d01 > SEG_THRESHOLD).sum()) >= MIN_LESION_PIXELS_2D


def _resize_bchw(x: torch.Tensor, out_size: int, is_mask: bool) -> torch.Tensor:
    mode = "nearest" if is_mask else "bilinear"
    return F.interpolate(
        x,
        size=(out_size, out_size),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )


def _move_axis_to_last(vol: np.ndarray, slice_axis: int) -> np.ndarray:
    if slice_axis == 2:
        return vol
    return np.moveaxis(vol, slice_axis, 2)


def _move_axis_from_last(vol_hwz: np.ndarray, slice_axis: int) -> np.ndarray:
    if slice_axis == 2:
        return vol_hwz
    return np.moveaxis(vol_hwz, 2, slice_axis)


def load_checkpoint_into_ema(
    diffusion_model,
    checkpoint_path: str,
    ema_decay: float = 0.995,
    ema_update_every: int = 10,
):
    data = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)

    diffusion_model.load_state_dict(data["model"], strict=False)

    ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
    ema.load_state_dict(data["ema"])
    ema.copy_params_from_ema_to_model()
    ema.ema_model.eval()

    return ema


def _clamp_idx(i: int, n: int) -> int:
    return 0 if i < 0 else (n - 1 if i >= n else i)


def load_slice_thickness_map(csv_path: str):
    d = {}
    if csv_path is None or (not os.path.exists(csv_path)):
        return d
    df = pd.read_csv(csv_path)
    if "patient_id" not in df.columns or "slice_thickness" not in df.columns:
        raise ValueError("slice_thickness_csv must have columns: patient_id, slice_thickness")
    for pid, th in zip(df["patient_id"], df["slice_thickness"]):
        if pd.isna(pid) or pd.isna(th):
            continue
        d[str(pid)] = float(th)
    return d


def expand_slices_by_margin_cm(
    base_slices: list[int],
    zmax: int,
    thickness_mm: float,
    margin_cm: float,
) -> list[int]:
    if base_slices is None or len(base_slices) == 0:
        return []
    margin_mm = float(margin_cm) * 10.0
    if margin_mm <= 0:
        return sorted(set(base_slices))

    th = max(float(thickness_mm), 1e-6)
    n = int(math.ceil(margin_mm / th))

    out = set()
    for z in base_slices:
        for zz in range(z - n, z + n + 1):
            out.add(_clamp_idx(zz, zmax))
    return sorted(out)


def get_thickness_mm(patient_id: str, ct_nii, slice_axis: int) -> float:
    pid = str(patient_id)

    if pid in SLICE_THICKNESS_MAP:
        th = float(SLICE_THICKNESS_MAP[pid])
        if th > 0:
            return th

    if "_" in pid:
        local = pid.split("_", 1)[1]
        if local in SLICE_THICKNESS_MAP:
            th = float(SLICE_THICKNESS_MAP[local])
            if th > 0:
                return th

    try:
        zooms = ct_nii.header.get_zooms()
        th = float(zooms[slice_axis])
        if th > 0:
            return th
    except Exception:
        pass

    return float(DEFAULT_SLICE_THICKNESS_MM)


def _context_indices_mm(center_z: int, k: int, zmax: int, thickness_mm: float, mm_step: float) -> list[int]:
    assert k % 2 == 1, "CONTEXT_SLICES must be odd (1,3,5,...)"
    half = k // 2
    thickness_mm = max(float(thickness_mm), 1e-6)

    mm_offsets = [i * mm_step for i in range(-half, half + 1)]
    idxs = []
    for mm in mm_offsets:
        if mm == 0:
            off = 0
        else:
            steps = int(round(abs(mm) / thickness_mm))
            if steps < 1:
                steps = 1
            off = steps if mm > 0 else -steps

        idxs.append(_clamp_idx(center_z + off, zmax))
    return idxs


DEBUG_PRINT_ONCE = True


def _q(x: torch.Tensor, qs=(0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)):
    x = x.detach().float().flatten()
    if x.numel() == 0:
        return {f"q{int(q*100):02d}": float("nan") for q in qs}
    return {f"q{int(q*100):02d}": float(torch.quantile(x, q).item()) for q in qs}


def print_tensor_stats(name: str, x: torch.Tensor, per_channel: bool = False, max_channels: int = 12):
    x = x.detach()
    msg = (
        f"{name:>18} | shape={tuple(x.shape)} | min={x.min().item():.4f} "
        f"max={x.max().item():.4f} mean={x.mean().item():.4f} std={x.std().item():.4f}"
    )
    print(msg)
    q = _q(x)
    print(f"{'':>18} | " + " ".join([f"{k}={v:.4f}" for k, v in q.items()]))

    if per_channel and x.ndim >= 4:
        C = x.shape[1]
        for c in range(min(C, max_channels)):
            xc = x[:, c : c + 1]
            msgc = (
                f"{name}[c={c:02d}]".rjust(18)
                + f" | min={xc.min().item():.4f} max={xc.max().item():.4f} "
                + f"mean={xc.mean().item():.4f} std={xc.std().item():.4f}"
            )
            print(msgc)
        if C > max_channels:
            print(f"{'':>18} | ... ({C-max_channels} more channels)")


def predict_followup_3d(
    PATIENT_ID,
    baseline_ct_path,
    baseline_iph_segmentation_path,
    baseline_ivh_segmentation_path,
    expansion_flag=None,
    cfg_scale=1.0,
    predict_only_iph_slices: bool = False,
):
    allowed = (None, "predicted", "ground_truth", 0, 1)

    if USE_EXPANSION_FLAG_INPUT:
        if expansion_flag not in allowed:
            raise ValueError("expansion_flag must be one of: None, 'predicted', 'ground_truth', 0, 1")
        if expansion_flag == "ground_truth":
            EXPANSION_FLAG = load_expansion_flag_from_csv(CLINICAL_CSV_PATH, PATIENT_ID, "Expansion_classic")
        elif expansion_flag in (0, 1):
            EXPANSION_FLAG = int(expansion_flag)
        else:
            raise ValueError("expansion_flag must be one of: None, 'predicted', 'ground_truth', 0, 1")
        exp_flag_value = int(EXPANSION_FLAG)
    else:
        expansion_flag = None
        exp_flag_value = None

    CLINICAL_VARS_7 = load_clinical_vars_from_csv(CLINICAL_CSV_PATH, PATIENT_ID)

    ct_nii = nib.load(baseline_ct_path)
    iph_nii = nib.load(baseline_iph_segmentation_path)
    ivh_nii = nib.load(baseline_ivh_segmentation_path)

    ct_vol = np.asarray(ct_nii.dataobj)
    iph_vol = np.asarray(iph_nii.dataobj)
    ivh_vol = np.asarray(ivh_nii.dataobj)

    if ct_vol.shape != iph_vol.shape or ct_vol.shape != ivh_vol.shape:
        raise ValueError(f"Shape mismatch: ct={ct_vol.shape} iph={iph_vol.shape} ivh={ivh_vol.shape}")

    ct_hwz = _move_axis_to_last(ct_vol, SLICE_AXIS)
    iph_hwz = _move_axis_to_last(iph_vol, SLICE_AXIS)
    ivh_hwz = _move_axis_to_last(ivh_vol, SLICE_AXIS)

    H, W, Z = ct_hwz.shape

    out_ct01 = np.zeros((H, W, Z), dtype=np.float32)
    out_iph01 = np.zeros((H, W, Z), dtype=np.float32)
    out_ivh01 = np.zeros((H, W, Z), dtype=np.float32)

    for z in range(Z):
        out_ct01[:, :, z] = _normalize_ct(ct_hwz[:, :, z])
        out_iph01[:, :, z] = _normalize_seg(iph_hwz[:, :, z])
        out_ivh01[:, :, z] = _normalize_seg(ivh_hwz[:, :, z])

    iph_present_slices = []
    for z in range(Z):
        iph2d = _normalize_seg(iph_hwz[:, :, z])
        if _iph_present(iph2d):
            iph_present_slices.append(z)

    if len(iph_present_slices) == 0:
        raise ValueError(
            "No IPH slices found -> outputs are copied baseline volumes, this is not intended use."
        )

    clin = CLINICAL_VARS_7.to(DEVICE).float()
    thickness_mm = get_thickness_mm(PATIENT_ID, ct_nii, SLICE_AXIS)

    if predict_only_iph_slices:
        predict_slices = sorted(set(iph_present_slices))
        used_margin = 0.0
    else:
        used_margin = SLICE_MARGIN_CM
        predict_slices = expand_slices_by_margin_cm(
            base_slices=iph_present_slices,
            zmax=Z,
            thickness_mm=thickness_mm,
            margin_cm=used_margin,
        )

    print(
        f"IPH present slices: {iph_present_slices[0]}..{iph_present_slices[-1]} | "
        f"predicting: {predict_slices[0]}..{predict_slices[-1]} | "
        f"mode={'iph_only' if predict_only_iph_slices else f'iph_plus_margin({used_margin}cm)'} | "
        f"thickness_mm={thickness_mm}"
    )

    for i in range(0, len(predict_slices), INFER_BATCH_SIZE):
        batch_z = predict_slices[i : i + INFER_BATCH_SIZE]
        B = len(batch_z)
        K = CONTEXT_SLICES
        S = MODEL_IMAGE_SIZE

        ct_bK = torch.empty((B, K, 1, S, S), dtype=torch.float32)
        iph_bK = torch.empty((B, K, 1, S, S), dtype=torch.float32)
        ivh_bK = torch.empty((B, K, 1, S, S), dtype=torch.float32)

        expected = (K, 1, S, S)

        for bi, z in enumerate(batch_z):
            ct_stack, iph_stack, ivh_stack = build_cond_stack_for_z(
                ct_hwz,
                iph_hwz,
                ivh_hwz,
                z=int(z),
                k=K,
                out_size=S,
                use_25d=USE_25D_CONTEXT,
                thickness_mm=thickness_mm,
                mm_step=CONTEXT_MM_STEP,
            )

            if tuple(ct_stack.shape) != expected:
                raise RuntimeError(
                    f"[{_anonymize_identifier(PATIENT_ID)}] z={z} ct_stack shape {tuple(ct_stack.shape)} != {expected}"
                )
            if tuple(iph_stack.shape) != expected:
                raise RuntimeError(
                    f"[{_anonymize_identifier(PATIENT_ID)}] z={z} iph_stack shape {tuple(iph_stack.shape)} != {expected}"
                )
            if tuple(ivh_stack.shape) != expected:
                raise RuntimeError(
                    f"[{_anonymize_identifier(PATIENT_ID)}] z={z} ivh_stack shape {tuple(ivh_stack.shape)} != {expected}"
                )

            ct_bK[bi].copy_(ct_stack)
            iph_bK[bi].copy_(iph_stack)
            ivh_bK[bi].copy_(ivh_stack)

        ct_bK = ct_bK.to(DEVICE)
        iph_bK = iph_bK.to(DEVICE)
        ivh_bK = ivh_bK.to(DEVICE)

        cond_pix_stack = torch.cat([ct_bK, iph_bK, ivh_bK], dim=2)

        if USE_LATENT_DIFFUSION:
            with torch.no_grad():
                B, K, C, S, _ = cond_pix_stack.shape
                cond_flat = cond_pix_stack.view(B * K, C, S, S)
                z_flat = autoencoder.encode_stage_2_inputs(cond_flat)
                C_lat, h_lat, w_lat = z_flat.shape[1], z_flat.shape[2], z_flat.shape[3]

                cond_in = z_flat.view(B, K * C_lat, h_lat, w_lat)
                cond_in = norm_latent(cond_in, LATENT_MEAN, LATENT_STD, repeat_k=CONTEXT_SLICES)

                if PREDICT_DELTA:
                    base_center = extract_center_latent_from_stack(cond_in, CONTEXT_SLICES)

        else:
            ct_ch = ct_bK.squeeze(2)
            iph_ch = iph_bK.squeeze(2)
            ivh_ch = ivh_bK.squeeze(2)

            cond_in = torch.cat([ct_ch, iph_ch, ivh_ch], dim=1).float()

        if clin.shape[0] == 1:
            clin_b = clin.expand(cond_in.shape[0], -1).contiguous()
        else:
            clin_b = clin

        if expansion_flag is not None:
            exp_flag = torch.full((cond_in.shape[0],), exp_flag_value, device=DEVICE, dtype=torch.long)
        else:
            exp_flag = None

        with torch.no_grad():
            gen = ema_model.sample_conditioned(
                cond_img=cond_in,
                seg=None,
                clinical_vars=clin_b,
                expansion_flag=exp_flag,
                cond_scale=cfg_scale,
                rescaled_phi=RESCALED_PHI,
            )

        if PREDICT_DELTA:
            gen = gen + base_center

        if use_latent_diffusion:
            with torch.no_grad():
                gen_native = unnorm_latent(gen, LATENT_MEAN, LATENT_STD, repeat_k=1)
                gen_pix = autoencoder.decode(gen_native)
        else:
            gen_pix = gen

        gen_ct01 = gen_pix[:, 0:1]

        if use_latent_diffusion:
            gen_iph_prob = torch.sigmoid(gen_pix[:, 1:2])
            gen_ivh_prob = torch.sigmoid(gen_pix[:, 2:3])
            gen_iph01 = (gen_iph_prob > 0.5).float()
            gen_ivh01 = (gen_ivh_prob > 0.5).float()
        else:
            gen_iph01 = (gen_pix[:, 1:2] > 0.5).float()
            gen_ivh01 = (gen_pix[:, 2:3] > 0.5).float()

        gen_ct01 = F.interpolate(gen_ct01, size=(H, W), mode="bilinear", align_corners=False)
        gen_iph01 = F.interpolate(gen_iph01, size=(H, W), mode="nearest")
        gen_ivh01 = F.interpolate(gen_ivh01, size=(H, W), mode="nearest")

        gen_ct_np = gen_ct01.squeeze(1).detach().cpu().numpy()
        gen_iph_np = gen_iph01.squeeze(1).detach().cpu().numpy()
        gen_ivh_np = gen_ivh01.squeeze(1).detach().cpu().numpy()

        for b_idx, z in enumerate(batch_z):
            out_ct01[:, :, z] = gen_ct_np[b_idx]
            out_iph01[:, :, z] = gen_iph_np[b_idx]
            out_ivh01[:, :, z] = gen_ivh_np[b_idx]

    out_segmentation_hwz = np.zeros_like(out_iph01, dtype=np.uint8)
    out_segmentation_hwz[out_iph01 > 0.5] = 1
    out_segmentation_hwz[out_ivh01 > 0.5] = 2

    out_ct = _denormalize_ct(out_ct01)
    out_ct = _move_axis_from_last(out_ct, SLICE_AXIS)
    out_segmentation = _move_axis_from_last(out_segmentation_hwz, SLICE_AXIS)

    nib.save(nib.Nifti1Image(out_ct.astype(np.float32), ct_nii.affine, ct_nii.header), OUT_FU_CT_PATH)
    nib.save(nib.Nifti1Image(out_segmentation.astype(np.uint8), ct_nii.affine, ct_nii.header), OUT_FU_SEG_PATH)
    if OUT_FU_SEG_SECOND_PATH is not None:
        nib.save(
            nib.Nifti1Image(out_segmentation.astype(np.uint8), ct_nii.affine, ct_nii.header),
            OUT_FU_SEG_SECOND_PATH,
        )

    print("Saved:")
    print(" ", _safe_path_for_log(OUT_FU_CT_PATH))
    print(" ", _safe_path_for_log(OUT_FU_SEG_PATH))


def predict_all_patients_in_dir(
    test_root: str,
    expansion_flag="ground_truth",
    cfg_scale: float = 1.0,
):
    global PATIENT_DIR, PATIENT_ID
    global BASELINE_CT_PATH, BASELINE_IPH_PATH, BASELINE_IVH_PATH
    global OUT_FU_CT_PATH, OUT_FU_SEG_PATH, OUT_FU_SEG_SECOND_PATH

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER_SEG, exist_ok=True)

    cases = collect_nnunet_flat_cases(
        test_root,
        required_channels=(NNUNET_CH_CT, NNUNET_CH_IPH, NNUNET_CH_IVH),
    )

    skipped = 0
    case_counter = 0
    for PATIENT_ID in sorted(cases.keys()):
        case_counter += 1
        log_case = f"case_{case_counter:04d}"

        PATIENT_DIR = str(test_root)

        BASELINE_CT_PATH = cases[PATIENT_ID][NNUNET_CH_CT]
        BASELINE_IPH_PATH = cases[PATIENT_ID][NNUNET_CH_IPH]
        BASELINE_IVH_PATH = cases[PATIENT_ID][NNUNET_CH_IVH]

        if not (
            os.path.exists(BASELINE_CT_PATH)
            and os.path.exists(BASELINE_IPH_PATH)
            and os.path.exists(BASELINE_IVH_PATH)
        ):
            print(f"[SKIP] {log_case}: missing one of required channels")
            skipped += 1
            continue

        out_patient_dir = os.path.join(OUTPUT_FOLDER, PATIENT_ID)
        os.makedirs(out_patient_dir, exist_ok=True)

        p_exp = None
        flag_schedule = None
        if USE_EXPANSION_FLAG_INPUT:
            if expansion_flag == "predicted":
                p_exp = load_expansion_prob_from_csv(CLINICAL_CSV_PATH, PATIENT_ID, PREDICTED_EXPANSION_PROB_COL)
                flag_schedule = make_flag_schedule(p_exp, TOTAL_SAMPLING_BUDGET)
            elif expansion_flag == "ground_truth":
                gt = load_expansion_flag_from_csv(CLINICAL_CSV_PATH, PATIENT_ID, "Expansion_classic")
                p_exp = float(gt)
                flag_schedule = [gt] * TOTAL_SAMPLING_BUDGET
            elif expansion_flag in (0, 1):
                p_exp = float(expansion_flag)
                flag_schedule = [int(expansion_flag)] * TOTAL_SAMPLING_BUDGET
            else:
                flag_schedule = [None] * TOTAL_SAMPLING_BUDGET
        else:
            run_flag = None

        print("Expansion flag schedule: ", flag_schedule)

        for s in range(TOTAL_SAMPLING_BUDGET):
            sample_dir = os.path.join(out_patient_dir, f"sample_{s:02d}")
            os.makedirs(sample_dir, exist_ok=True)

            seg_flat_dir = os.path.join(OUTPUT_FOLDER_SEG, f"sample_{s:02d}")
            os.makedirs(seg_flat_dir, exist_ok=True)

            OUT_FU_CT_PATH = os.path.join(sample_dir, "ct.nii.gz")
            OUT_FU_SEG_PATH = os.path.join(sample_dir, "seg.nii.gz")
            OUT_FU_SEG_SECOND_PATH = os.path.join(seg_flat_dir, f"{PATIENT_ID}.nii.gz")

            if (
                os.path.exists(OUT_FU_CT_PATH)
                and os.path.getsize(OUT_FU_CT_PATH) > 0
                and os.path.exists(OUT_FU_SEG_PATH)
                and os.path.getsize(OUT_FU_SEG_PATH) > 0
                and os.path.exists(OUT_FU_SEG_SECOND_PATH)
                and os.path.getsize(OUT_FU_SEG_SECOND_PATH) > 0
            ):
                print(f"[RESUME-SKIP] {log_case} sample_{s:02d} already exists")
                continue

            if USE_EXPANSION_FLAG_INPUT:
                run_flag = flag_schedule[s]
            else:
                run_flag = None

            predict_followup_3d(
                PATIENT_ID,
                BASELINE_CT_PATH,
                BASELINE_IPH_PATH,
                BASELINE_IVH_PATH,
                expansion_flag=(run_flag if run_flag is not None else expansion_flag),
                cfg_scale=cfg_scale,
                predict_only_iph_slices=PREDICT_ONLY_IPH_SLICES,
            )

            print(f"[OK] {log_case} sample_{s:02d} -> {_safe_path_for_log(sample_dir)}")

        if (flag_schedule is not None) and (p_exp is not None):
            save_prob_maps_from_samples(out_patient_dir, flag_schedule, p_exp, BASELINE_CT_PATH)

    print(f"\nDone. Outputs under: {_safe_path_for_log(OUTPUT_FOLDER)}")
    print(f"Also flat nnU-Net segs under: {_safe_path_for_log(OUTPUT_FOLDER_SEG)}")
    print(f"Skipped patients: {skipped}")


SLICE_THICKNESS_MAP = load_slice_thickness_map(SLICE_THICKNESS_CSV)

diffusion_model, autoencoder, use_latent_diffusion = build_diffusion_and_autoencoder()
diffusion_model = diffusion_model.to(DEVICE).eval()

if use_latent_diffusion:
    if autoencoder is None:
        raise ValueError("use_latent_diffusion=True but autoencoder is None")
    autoencoder = autoencoder.to(DEVICE).eval()

ema = load_checkpoint_into_ema(diffusion_model, CHECKPOINT_PATH)
ema_model = ema.ema_model.to(DEVICE).eval()

predict_all_patients_in_dir(TEST_DIR, expansion_flag=EXPANSION_FLAG, cfg_scale=CFG_SCALE)