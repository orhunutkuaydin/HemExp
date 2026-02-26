import math
import os
import random
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as FU
import torchvision.transforms.functional as TF
import wandb
from accelerate import Accelerator
from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.version import __version__
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from monai.metrics import DiceMetric
from monai.networks.nets import AutoencoderKL
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import einsum, nn
from torch.amp import autocast
from torch.nn import Module, ModuleList
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Sampler
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchvision import transforms as T, utils
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True

worker_count = 12


def _stats_tensor(stats_list, device, dtype, repeat_k: int = 1):
    t = torch.tensor(stats_list, device=device, dtype=dtype)
    if repeat_k != 1:
        t = t.repeat(repeat_k)
    return t.view(1, -1, 1, 1)


def norm_latent(z: torch.Tensor, mean_list, std_list, repeat_k: int = 1):
    mean = _stats_tensor(mean_list, z.device, z.dtype, repeat_k)
    std = _stats_tensor(std_list, z.device, z.dtype, repeat_k)
    return (z - mean) / (std + 1e-8)


def unnorm_latent(z: torch.Tensor, mean_list, std_list, repeat_k: int = 1):
    mean = _stats_tensor(mean_list, z.device, z.dtype, repeat_k)
    std = _stats_tensor(std_list, z.device, z.dtype, repeat_k)
    return z * (std + 1e-8) + mean


def _load_nifti_2d(path: str) -> np.ndarray:
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D slice at {path}, got shape {arr.shape}")
    return arr.astype(np.float32)


def _normalize_seg(seg: np.ndarray) -> np.ndarray:
    seg = seg.astype(np.float32)
    mx = float(seg.max()) if seg.size else 0.0
    if mx > 1.5:
        seg = seg / 255.0
    return seg


def _normalize_ct(ct: np.ndarray, do_window: bool, wmin: float, wmax: float) -> np.ndarray:
    ct = ct.astype(np.float32)
    if do_window:
        ct = np.clip(ct, wmin, wmax)
        ct = (ct - wmin) / (wmax - wmin + 1e-8)
    else:
        mn, mx = float(ct.min()), float(ct.max())
        ct = (ct - mn) / (mx - mn + 1e-8)
    return ct


def _clamp_index(i: int, n: int) -> int:
    return 0 if i < 0 else (n - 1 if i >= n else i)


def load_stack_by_indices(
    mod_dir: Path,
    slice_names: list[str],
    indices: list[int],
    norm_fn,
    fallback_idx: int,
) -> torch.Tensor:
    planes = []
    for j in indices:
        j = _clamp_index(j, len(slice_names))
        name = slice_names[j]
        path = mod_dir / name
        if not path.exists():
            fb_name = slice_names[_clamp_index(fallback_idx, len(slice_names))]
            path = mod_dir / fb_name
        arr = norm_fn(_load_nifti_2d(str(path)))
        planes.append(torch.from_numpy(arr).float())
    return torch.stack(planes, dim=0)


class ExpansionBalancedSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = drop_last

        if self.batch_size % 2 != 0:
            raise ValueError("ExpansionBalancedSampler requires an EVEN batch_size.")

        self.exp_idx = []
        self.stab_idx = []

        for i, s in enumerate(dataset.samples):
            patient_key = s["patient_key"]
            exp = dataset.metadata_expansion.get(patient_key, 0)
            flag = 0 if pd.isna(exp) else int(exp)
            (self.exp_idx if flag == 1 else self.stab_idx).append(i)

        if len(self.exp_idx) == 0 or len(self.stab_idx) == 0:
            raise RuntimeError("Sampler found only one class in the dataset.")

    def __len__(self):
        max_class = max(len(self.exp_idx), len(self.stab_idx))
        n_samples = 2 * max_class
        n_batches = n_samples // self.batch_size
        if (n_samples % self.batch_size) != 0 and not self.drop_last:
            n_batches += 1
        return n_batches

    def __iter__(self):
        exp = self.exp_idx[:]
        stab = self.stab_idx[:]
        random.shuffle(exp)
        random.shuffle(stab)

        exp_ptr = 0
        stab_ptr = 0

        for _ in range(len(self)):
            batch = []
            half = self.batch_size // 2
            for _k in range(half):
                if exp_ptr >= len(exp):
                    random.shuffle(exp)
                    exp_ptr = 0
                if stab_ptr >= len(stab):
                    random.shuffle(stab)
                    stab_ptr = 0
                batch.append(exp[exp_ptr])
                exp_ptr += 1
                batch.append(stab[stab_ptr])
                stab_ptr += 1

            random.shuffle(batch)
            yield batch


import json
from einops import rearrange


def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def _latent_name_from_slice(name: str) -> str:
    return _strip_nii_suffix(name) + ".pt"


def _load_latent_pt(path: Path) -> torch.Tensor:
    z = torch.load(str(path), map_location="cpu")
    if not torch.is_tensor(z):
        raise ValueError(f"Expected tensor in {path}, got {type(z)}")
    return z.float()


def load_latent_stack_by_indices(
    lat_dir: Path,
    latent_names: list[str],
    indices: list[int],
    fallback_idx: int,
) -> torch.Tensor:
    planes = []
    for j in indices:
        j = _clamp_index(j, len(latent_names))
        name = latent_names[j]
        path = lat_dir / name
        if not path.exists():
            fb_name = latent_names[_clamp_index(fallback_idx, len(latent_names))]
            path = lat_dir / fb_name
        planes.append(_load_latent_pt(path))
    return torch.stack(planes, dim=0)


class PairedLatentSliceDataset(Dataset):
    def __init__(
        self,
        latents_root: str,
        clinical_csv: Optional[str],
        include_hospitals: Optional[List[str]] = None,
        use_clinical: bool = True,
        context_slices: int = 1,
        slice_thickness_csv: Optional[str] = None,
        context_mm_step: float = 5.0,
        default_slice_thickness_mm: float = 5.0,
    ):
        super().__init__()
        self.slices_root = Path(latents_root)
        self.context_slices = int(context_slices)
        self.half_ctx = self.context_slices // 2
        self.context_mm_step = float(context_mm_step)
        self.default_slice_thickness_mm = float(default_slice_thickness_mm)

        self.use_clinical = bool(use_clinical)
        self._load_clinical(clinical_csv) if (clinical_csv is not None) else self._init_empty_clinical()

        self._load_slice_thickness(slice_thickness_csv)

        self.patient_to_slice_names = {}
        self.samples = self._index_samples(include_hospitals)

    def _init_empty_clinical(self):
        self.metadata_onset = {}
        self.metadata_ICH_vol = {}
        self.metadata_anticoag = {}
        self.metadata_anti_plate = {}
        self.metadata_age = {}
        self.metadata_gcs = {}
        self.metadata_mmhg = {}
        self.metadata_expansion = {}

    def _load_clinical(self, csv_file: str):
        df = pd.read_csv(csv_file)
        self.metadata_onset = dict(zip(df["Patient_ID"], df["time_from_onset_to_CT"]))
        self.metadata_ICH_vol = dict(zip(df["Patient_ID"], df["IPH_baseline_ml"]))
        self.metadata_anticoag = dict(zip(df["Patient_ID"], df["Anticoagulant"]))
        self.metadata_anti_plate = dict(zip(df["Patient_ID"], df["Antiplatelet"]))
        self.metadata_age = dict(zip(df["Patient_ID"], df["Age"]))
        self.metadata_gcs = dict(zip(df["Patient_ID"], df["GCS"]))
        self.metadata_mmhg = dict(zip(df["Patient_ID"], df["Systolic"]))
        self.metadata_expansion = dict(zip(df["Patient_ID"], df["Expansion_classic"]))

    def _patient_key(self, hospital: str, patient_folder_name: str) -> str:
        parts = patient_folder_name.split("_", 1)
        patient_local = parts[1] if len(parts) == 2 else patient_folder_name
        return f"{hospital}_{patient_local}"

    def _load_slice_thickness(self, csv_path: Optional[str]):
        self.metadata_slice_thickness = {}
        if csv_path is None:
            return
        df = pd.read_csv(csv_path)
        if "patient_id" not in df.columns or "slice_thickness" not in df.columns:
            raise ValueError("slice_thickness_csv must have columns: patient_id, slice_thickness")
        for pid, th in zip(df["patient_id"], df["slice_thickness"]):
            if pd.isna(pid) or pd.isna(th):
                continue
            self.metadata_slice_thickness[str(pid)] = float(th)

    def _get_slice_thickness_mm(self, patient_key: str, patient_dir_name: str) -> float:
        local = patient_key.split("_", 1)[1] if "_" in patient_key else patient_key
        for k in (patient_key, local, patient_dir_name):
            if k in self.metadata_slice_thickness:
                th = float(self.metadata_slice_thickness[k])
                return th if th > 0 else self.default_slice_thickness_mm
        return self.default_slice_thickness_mm

    def _context_indices_mm(self, center_idx: int, n_slices: int, thickness_mm: float) -> list[int]:
        if self.context_slices == 1:
            return [center_idx]
        thickness_mm = max(float(thickness_mm), 1e-6)
        mm_offsets = [i * self.context_mm_step for i in range(-self.half_ctx, self.half_ctx + 1)]
        idxs = []
        for mm in mm_offsets:
            if mm == 0:
                off = 0
            else:
                steps = int(round(abs(mm) / thickness_mm))
                if steps < 1:
                    steps = 1
                off = steps if mm > 0 else -steps
            idxs.append(_clamp_index(center_idx + off, n_slices))
        return idxs

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return lo if x < lo else (hi if x > hi else x)

    def _require_float(self, x, name: str, patient_key: str) -> float:
        if x is None or pd.isna(x):
            raise ValueError(f"[clinical] Missing '{name}' for patient_key='{patient_key}'")
        return float(x)

    def _require_binary01(self, x, name: str, patient_key: str) -> float:
        v = self._require_float(x, name, patient_key)
        if v == 0.0 or v == 1.0:
            return v
        raise ValueError(f"[clinical] Expected '{name}' in {{0,1}} but got {v} for patient_key='{patient_key}'")

    def _get_clinical_vars(self, patient_key: str) -> torch.Tensor:
        if not self.use_clinical:
            return torch.zeros(7, dtype=torch.float32)
        onset = self._require_float(self.metadata_onset.get(patient_key), "time_from_onset_to_CT", patient_key)
        ICH_vol = self._require_float(self.metadata_ICH_vol.get(patient_key), "IPH_baseline_ml", patient_key)
        ac = self._require_binary01(self.metadata_anticoag.get(patient_key), "Anticoagulant", patient_key)
        anti_plate = self._require_binary01(self.metadata_anti_plate.get(patient_key), "Antiplatelet", patient_key)
        age = self._require_float(self.metadata_age.get(patient_key), "Age", patient_key)
        gcs = self._require_float(self.metadata_gcs.get(patient_key), "GCS", patient_key)
        SBP = self._require_float(self.metadata_mmhg.get(patient_key), "Systolic", patient_key)

        onset = self._clamp(onset, 0.0, 24.0) / 24.0
        ICH_vol = self._clamp(ICH_vol, 0.0, 250.0) / 250.0
        age = self._clamp(age, 0.0, 100.0) / 100.0
        gcs = (self._clamp(gcs, 3.0, 15.0) - 3) / 12
        SBP = self._clamp(SBP, 0.0, 250.0) / 250.0

        return torch.tensor([onset, ICH_vol, ac, anti_plate, age, gcs, SBP], dtype=torch.float32)

    def _get_expansion_flag(self, patient_key: str) -> torch.Tensor:
        exp = self.metadata_expansion.get(patient_key, 0)
        exp_int = int(exp) if not (pd.isna(exp)) else 0
        exp_int = 1 if exp_int == 1 else 0
        return torch.tensor([exp_int], dtype=torch.long)

    def _index_samples(self, include_hospitals: Optional[List[str]]) -> List[Dict]:
        samples = []
        root = self.slices_root

        hospitals = [p.name for p in root.iterdir() if p.is_dir()]
        if include_hospitals is not None:
            hospitals = [h for h in hospitals if h in set(include_hospitals)]

        for hosp in sorted(hospitals):
            hosp_dir = root / hosp
            for patient_dir in sorted([p for p in hosp_dir.iterdir() if p.is_dir()]):
                bl_lat = patient_dir / "baseline" / "latent"
                fu_lat = patient_dir / "followup" / "latent"
                if not (bl_lat.is_dir() and fu_lat.is_dir()):
                    continue

                bl_lat_slices = sorted([p for p in bl_lat.iterdir() if p.is_file() and p.name.endswith(".pt")])
                if len(bl_lat_slices) == 0:
                    continue

                patient_key = self._patient_key(hosp, patient_dir.name)

                latent_names = [p.name for p in bl_lat_slices]
                self.patient_to_slice_names[str(patient_dir)] = latent_names

                for slice_idx, bl_lat_path in enumerate(bl_lat_slices):
                    name = bl_lat_path.name
                    fu_lat_path = fu_lat / name
                    if not fu_lat_path.exists():
                        continue
                    samples.append(
                        {
                            "hospital": hosp,
                            "patient_dir": str(patient_dir),
                            "patient_key": patient_key,
                            "slice_name": name,
                            "slice_idx": slice_idx,
                        }
                    )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        patient_dir = Path(s["patient_dir"])
        patient_key = s["patient_key"]

        latent_names = self.patient_to_slice_names[str(patient_dir)]
        center_idx = int(s["slice_idx"])

        thickness_mm = self._get_slice_thickness_mm(patient_key, patient_dir.name)
        indices = self._context_indices_mm(center_idx, n_slices=len(latent_names), thickness_mm=thickness_mm)

        bl_lat_dir = patient_dir / "baseline" / "latent"
        fu_lat_dir = patient_dir / "followup" / "latent"

        bl_stack = load_latent_stack_by_indices(bl_lat_dir, latent_names, indices, fallback_idx=center_idx)
        cond_z = rearrange(bl_stack, "k c h w -> (k c) h w")

        target_z = _load_latent_pt(fu_lat_dir / latent_names[center_idx])

        clinical_vars = self._get_clinical_vars(patient_key)
        expansion_flag = self._get_expansion_flag(patient_key)

        return cond_z, target_z, clinical_vars, expansion_flag


class PairedNiftiSliceDataset(Dataset):
    def __init__(
        self,
        slices_root: str,
        clinical_csv: Optional[str],
        image_size: Optional[int],
        include_hospitals: Optional[List[str]] = None,
        keep_only_baseline_iph_slices: bool = True,
        seg_threshold: float = 0.5,
        min_lesion_pixels_2d: int = 1,
        augment_flip: bool = False,
        augment_rotation: bool = False,
        rotation_range_deg: float = 0.0,
        use_clinical: bool = True,
        context_slices: int = 1,
        slice_thickness_csv: Optional[str] = None,
        context_mm_step: float = 5.0,
        default_slice_thickness_mm: float = 5.0,
        ct_do_window: bool = True,
        ct_window_min: float = 0.0,
        ct_window_max: float = 100.0,
    ):
        super().__init__()
        self.slices_root = Path(slices_root)
        self.image_size = image_size

        self.context_slices = context_slices
        self.half_ctx = context_slices // 2

        self.context_mm_step = float(context_mm_step)
        self.default_slice_thickness_mm = float(default_slice_thickness_mm)
        self._load_slice_thickness(slice_thickness_csv)

        self.keep_only_baseline_iph_slices = keep_only_baseline_iph_slices
        self.seg_threshold = float(seg_threshold)
        self.min_lesion_pixels_2d = int(min_lesion_pixels_2d)

        self.augment_flip = augment_flip
        self.augment_rotation = augment_rotation
        self.rotation_range_deg = float(rotation_range_deg)

        self.use_clinical = use_clinical
        self._load_clinical(clinical_csv) if (clinical_csv is not None) else self._init_empty_clinical()

        self.patient_to_slice_names = {}
        self.samples = self._index_samples(include_hospitals)

        self.ct_do_window = ct_do_window
        self.ct_window_min = ct_window_min
        self.ct_window_max = ct_window_max

    def _load_slice_thickness(self, csv_path: Optional[str]):
        self.metadata_slice_thickness = {}
        if csv_path is None:
            return
        df = pd.read_csv(csv_path)
        if "patient_id" not in df.columns or "slice_thickness" not in df.columns:
            raise ValueError("slice_thickness_csv must have columns: patient_id, slice_thickness")
        for pid, th in zip(df["patient_id"], df["slice_thickness"]):
            if pd.isna(pid) or pd.isna(th):
                continue
            self.metadata_slice_thickness[str(pid)] = float(th)

    def _get_slice_thickness_mm(self, patient_key: str, patient_dir_name: str) -> float:
        if not hasattr(self, "metadata_slice_thickness"):
            return self.default_slice_thickness_mm

        local = patient_key.split("_", 1)[1] if "_" in patient_key else patient_key

        for k in (patient_key, local, patient_dir_name):
            if k in self.metadata_slice_thickness:
                th = float(self.metadata_slice_thickness[k])
                return th if th > 0 else self.default_slice_thickness_mm

        return self.default_slice_thickness_mm

    def _context_indices_mm(self, center_idx: int, n_slices: int, thickness_mm: float) -> list[int]:
        if self.context_slices == 1:
            return [center_idx]

        thickness_mm = max(float(thickness_mm), 1e-6)

        mm_offsets = [i * self.context_mm_step for i in range(-self.half_ctx, self.half_ctx + 1)]
        idxs = []
        for mm in mm_offsets:
            if mm == 0:
                off = 0
            else:
                steps = int(round(abs(mm) / thickness_mm))
                if steps < 1:
                    steps = 1
                off = steps if mm > 0 else -steps

            idxs.append(_clamp_index(center_idx + off, n_slices))

        return idxs

    def _init_empty_clinical(self):
        self.metadata_onset = {}
        self.metadata_ICH_vol = {}
        self.metadata_anticoag = {}
        self.metadata_anti_plate = {}
        self.metadata_age = {}
        self.metadata_gcs = {}
        self.metadata_mmhg = {}
        self.metadata_expansion = {}

    def _load_clinical(self, csv_file: str):
        df = pd.read_csv(csv_file)

        self.metadata_onset = dict(zip(df["Patient_ID"], df["time_from_onset_to_CT"]))
        self.metadata_ICH_vol = dict(zip(df["Patient_ID"], df["IPH_baseline_ml"]))
        self.metadata_anticoag = dict(zip(df["Patient_ID"], df["Anticoagulant"]))
        self.metadata_anti_plate = dict(zip(df["Patient_ID"], df["Antiplatelet"]))
        self.metadata_age = dict(zip(df["Patient_ID"], df["Age"]))
        self.metadata_gcs = dict(zip(df["Patient_ID"], df["GCS"]))
        self.metadata_mmhg = dict(zip(df["Patient_ID"], df["Systolic"]))
        self.metadata_expansion = dict(zip(df["Patient_ID"], df["Expansion_classic"]))

    def _patient_key(self, hospital: str, patient_folder_name: str) -> str:
        parts = patient_folder_name.split("_", 1)
        if len(parts) == 2:
            patient_local = parts[1]
        else:
            patient_local = patient_folder_name
        return f"{hospital}_{patient_local}"

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return lo if x < lo else (hi if x > hi else x)

    def _require_float(self, x, name: str, patient_key: str) -> float:
        if x is None or pd.isna(x):
            raise ValueError(f"[clinical] Missing '{name}' for patient_key='{patient_key}'")
        try:
            return float(x)
        except Exception as e:
            raise ValueError(f"[clinical] Non-numeric '{name}'='{x}' for patient_key='{patient_key}'") from e

    def _require_binary01(self, x, name: str, patient_key: str) -> float:
        v = self._require_float(x, name, patient_key)
        if v == 0.0 or v == 1.0:
            return v
        raise ValueError(f"[clinical] Expected '{name}' in {{0,1}} but got {v} for patient_key='{patient_key}'")

    def _get_clinical_vars(self, patient_key: str) -> torch.Tensor:
        if not self.use_clinical:
            return torch.zeros(7, dtype=torch.float32)

        onset = self._require_float(self.metadata_onset.get(patient_key), "time_from_onset_to_CT", patient_key)
        ICH_vol = self._require_float(self.metadata_ICH_vol.get(patient_key), "IPH_baseline_ml", patient_key)
        ac = self._require_binary01(self.metadata_anticoag.get(patient_key), "Anticoagulant", patient_key)
        anti_plate = self._require_binary01(self.metadata_anti_plate.get(patient_key), "Antiplatelet", patient_key)
        age = self._require_float(self.metadata_age.get(patient_key), "Age", patient_key)
        gcs = self._require_float(self.metadata_gcs.get(patient_key), "GCS", patient_key)
        SBP = self._require_float(self.metadata_mmhg.get(patient_key), "Systolic", patient_key)

        onset = self._clamp(onset, 0.0, 24.0) / 24.0
        ICH_vol = self._clamp(ICH_vol, 0.0, 250.0) / 250.0
        age = self._clamp(age, 0.0, 100.0) / 100.0
        gcs = (self._clamp(gcs, 3.0, 15.0) - 3) / 12
        SBP = self._clamp(SBP, 0.0, 250.0) / 250.0

        return torch.tensor([onset, ICH_vol, ac, anti_plate, age, gcs, SBP], dtype=torch.float32)

    def _get_expansion_flag(self, patient_key: str) -> torch.Tensor:
        exp = self.metadata_expansion.get(patient_key, 0)
        exp_int = int(exp) if not (pd.isna(exp)) else 0
        exp_int = 1 if exp_int == 1 else 0
        return torch.tensor([exp_int], dtype=torch.long)

    def _iph_present_2d(self, iph_slice_path: str) -> bool:
        seg = _load_nifti_2d(iph_slice_path)
        seg = _normalize_seg(seg)
        return int((seg > self.seg_threshold).sum()) >= self.min_lesion_pixels_2d

    def _index_samples(self, include_hospitals: Optional[List[str]]) -> List[Dict]:
        samples = []
        root = self.slices_root

        hospitals = [p.name for p in root.iterdir() if p.is_dir()]
        if include_hospitals is not None:
            hospitals = [h for h in hospitals if h in set(include_hospitals)]

        for hosp in sorted(hospitals):
            hosp_dir = root / hosp
            for patient_dir in sorted([p for p in hosp_dir.iterdir() if p.is_dir()]):
                bl_ct = patient_dir / "baseline" / "ct"
                bl_iph = patient_dir / "baseline" / "iph"
                bl_ivh = patient_dir / "baseline" / "ivh"

                fu_ct = patient_dir / "followup" / "ct"
                fu_iph = patient_dir / "followup" / "iph"
                fu_ivh = patient_dir / "followup" / "ivh"

                if not (
                    bl_ct.is_dir()
                    and bl_iph.is_dir()
                    and bl_ivh.is_dir()
                    and fu_ct.is_dir()
                    and fu_iph.is_dir()
                    and fu_ivh.is_dir()
                ):
                    continue

                bl_ct_slices = sorted(
                    [p for p in bl_ct.iterdir() if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))]
                )
                if len(bl_ct_slices) == 0:
                    continue

                patient_key = self._patient_key(hosp, patient_dir.name)

                slice_names = [p.name for p in bl_ct_slices]
                self.patient_to_slice_names[str(patient_dir)] = slice_names

                for slice_idx, bl_ct_path in enumerate(bl_ct_slices):
                    name = bl_ct_path.name
                    fu_ct_path = fu_ct / name
                    if not fu_ct_path.exists():
                        continue

                    bl_iph_path = bl_iph / name
                    bl_ivh_path = bl_ivh / name
                    fu_iph_path = fu_iph / name
                    fu_ivh_path = fu_ivh / name

                    if not (bl_iph_path.exists() and bl_ivh_path.exists() and fu_iph_path.exists() and fu_ivh_path.exists()):
                        continue

                    if self.keep_only_baseline_iph_slices:
                        if not self._iph_present_2d(str(bl_iph_path)):
                            continue

                    samples.append(
                        {
                            "hospital": hosp,
                            "patient_dir": str(patient_dir),
                            "patient_key": patient_key,
                            "slice_name": name,
                            "slice_idx": slice_idx,
                        }
                    )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        patient_dir = Path(s["patient_dir"])
        patient_key = s["patient_key"]

        slice_names = self.patient_to_slice_names[str(patient_dir)]
        center_idx = int(s["slice_idx"])

        thickness_mm = self._get_slice_thickness_mm(patient_key, patient_dir.name)
        indices = self._context_indices_mm(center_idx, n_slices=len(slice_names), thickness_mm=thickness_mm)

        bl_ct_dir = patient_dir / "baseline" / "ct"
        bl_iph_dir = patient_dir / "baseline" / "iph"
        bl_ivh_dir = patient_dir / "baseline" / "ivh"

        fu_ct_dir = patient_dir / "followup" / "ct"
        fu_iph_dir = patient_dir / "followup" / "iph"
        fu_ivh_dir = patient_dir / "followup" / "ivh"

        norm_ct = partial(_normalize_ct, do_window=self.ct_do_window, wmin=self.ct_window_min, wmax=self.ct_window_max)

        bl_ct_t = load_stack_by_indices(bl_ct_dir, slice_names, indices, norm_ct, fallback_idx=center_idx)
        fu_ct_t = load_stack_by_indices(fu_ct_dir, slice_names, indices, norm_ct, fallback_idx=center_idx)
        bl_iph_t = load_stack_by_indices(bl_iph_dir, slice_names, indices, _normalize_seg, fallback_idx=center_idx)
        bl_ivh_t = load_stack_by_indices(bl_ivh_dir, slice_names, indices, _normalize_seg, fallback_idx=center_idx)
        fu_iph_t = load_stack_by_indices(fu_iph_dir, slice_names, indices, _normalize_seg, fallback_idx=center_idx)
        fu_ivh_t = load_stack_by_indices(fu_ivh_dir, slice_names, indices, _normalize_seg, fallback_idx=center_idx)

        if self.augment_flip:
            if torch.rand(()) < 0.5:
                bl_ct_t = TF.vflip(bl_ct_t)
                fu_ct_t = TF.vflip(fu_ct_t)
                bl_iph_t = TF.vflip(bl_iph_t)
                bl_ivh_t = TF.vflip(bl_ivh_t)
                fu_iph_t = TF.vflip(fu_iph_t)
                fu_ivh_t = TF.vflip(fu_ivh_t)

        if self.augment_rotation:
            angle = (torch.rand(()) * 2 - 1).item() * self.rotation_range_deg

            bl_ct_t = TF.rotate(bl_ct_t, angle, interpolation=InterpolationMode.BILINEAR)
            fu_ct_t = TF.rotate(fu_ct_t, angle, interpolation=InterpolationMode.BILINEAR)

            bl_iph_t = TF.rotate(bl_iph_t, angle, interpolation=InterpolationMode.NEAREST)
            bl_ivh_t = TF.rotate(bl_ivh_t, angle, interpolation=InterpolationMode.NEAREST)
            fu_iph_t = TF.rotate(fu_iph_t, angle, interpolation=InterpolationMode.NEAREST)
            fu_ivh_t = TF.rotate(fu_ivh_t, angle, interpolation=InterpolationMode.NEAREST)

        clinical_vars = self._get_clinical_vars(s["patient_key"])
        expansion_flag = self._get_expansion_flag(s["patient_key"])

        return bl_ct_t, fu_ct_t, bl_iph_t, bl_ivh_t, fu_iph_t, fu_ivh_t, clinical_vars, expansion_flag


@torch.inference_mode()
def encode_triplet_stack_chunked(ae, ct, iph, ivh, batch_chunk=2):
    b, k, h, w = ct.shape
    zs = []
    for j in range(k):
        xj = torch.cat([ct[:, j : j + 1], iph[:, j : j + 1], ivh[:, j : j + 1]], dim=1)
        z_parts = []
        for s in range(0, b, batch_chunk):
            z_parts.append(ae.encode_stage_2_inputs(xj[s : s + batch_chunk]))
        zj = torch.cat(z_parts, dim=0)
        zs.append(zj)
    return torch.cat(zs, dim=1)


def extract_center_latent_from_stack(z_stack: torch.Tensor, k: int) -> torch.Tensor:
    b, ck, h, w = z_stack.shape
    c_lat = ck // k
    mid = k // 2
    return z_stack[:, mid * c_lat : (mid + 1) * c_lat]


def encode_center_triplet(ae, ct, iph, ivh, mid):
    x_mid = torch.cat([ct[:, mid : mid + 1], iph[:, mid : mid + 1], ivh[:, mid : mid + 1]], dim=1)
    return ae.encode_stage_2_inputs(x_mid)


def decode_center_from_stack(ae, z_stack, k):
    b, ck, h, w = z_stack.shape
    c_lat = ck // k
    mid = k // 2
    z_mid = z_stack[:, mid * c_lat : (mid + 1) * c_lat]
    return ae.decode(z_mid)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)


def divisible_by(numer, denom):
    return (numer % denom) == 0


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def pack_one_with_inverse(x, pattern):
    packed, packed_shape = pack([x], pattern)

    def inverse(x, inverse_pattern=None):
        inverse_pattern = default(inverse_pattern, pattern)
        return unpack(x, packed_shape, inverse_pattern)[0]

    return packed, inverse


def project(x, y):
    x, inverse = pack_one_with_inverse(x, "b *")
    y, _ = pack_one_with_inverse(y, "b *")

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim=-1)

    parallel = (x * unit).sum(dim=-1, keepdim=True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def Upsample(dim, dim_out=None):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(dim, default(dim_out, dim), 3, padding=1))


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2), nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class Block(Module):
    def __init__(self, dim, dim_out, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, "h c n -> b h c n", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4, flash=False):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class MultiHeadAEKL(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 3,
        feature_channels: int = 256,
        latent_channels: int = 3,
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
        y, mu, logvar = self.backbone(x)
        ct = self.ct_head(y)
        iph = self.iph_head(y)
        ivh = self.ivh_head(y)
        recon = torch.cat([ct, iph, ivh], dim=1)
        return recon, mu, logvar

    @torch.no_grad()
    def encode_stage_2_inputs(self, x):
        z_mu, z_sigma = self.backbone.encode(x)
        z = self.backbone.sampling(z_mu, z_sigma)
        return z

    @torch.no_grad()
    def decode(self, z):
        y = self.backbone.decode(z)
        ct = self.ct_head(y)
        iph = self.iph_head(y)
        ivh = self.ivh_head(y)
        return torch.cat([ct, iph, ivh], dim=1)


class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        context_slices=1,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        dropout=0.0,
        attn_dim_head=32,
        attn_heads=4,
        full_attn=None,
        flash_attn=False,
        use_clinical=True,
        use_expansion_flag_input=True,
        predict_segmentation=True,
        use_latent_diffusion=True,
        clinical_drop_prob=0,
        expansion_flag_drop_prob=0.1,
    ):
        super().__init__()

        self.use_latent_diffusion = use_latent_diffusion

        self.use_clinical = use_clinical
        self.use_expansion_flag_input = use_expansion_flag_input
        self.clinical_drop_prob = clinical_drop_prob
        self.expansion_flag_drop_prob = expansion_flag_drop_prob

        self.predict_segmentation = predict_segmentation

        self.channels = channels
        self.context_slices = context_slices
        self.self_condition = self_condition

        cond_channels = self.channels * self.context_slices
        input_channels = self.channels + cond_channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(sinu_pos_emb, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))

        if self.use_clinical:
            self.clinical_embedding_dim = time_dim
            self.clinical_vars_mlp = nn.Sequential(nn.Linear(7, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
            self.null_clin_emb = nn.Parameter(torch.randn(time_dim))
            self._pos_weight = None

        self.clin_drop_prob = clinical_drop_prob

        if self.use_expansion_flag_input:
            self.flag_embed = nn.Embedding(2, time_dim)

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)
        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash=flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
            zip(in_out, full_attn, attn_heads, attn_dim_head)
        ):
            is_last = ind >= (num_resolutions - 1)
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            self.downs.append(
                ModuleList(
                    [
                        resnet_block(dim_in, dim_in),
                        resnet_block(dim_in, dim_in),
                        attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
            zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))
        ):
            is_last = ind == (len(in_out) - 1)
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            self.ups.append(
                ModuleList(
                    [
                        resnet_block(dim_out + dim_in, dim_out),
                        resnet_block(dim_out + dim_in, dim_out),
                        attn_klass(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_image_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(
        self,
        x,
        time,
        cond_img=None,
        seg=None,
        x_self_cond=None,
        clinical_vars=None,
        clinical_drop_prob=None,
        expansion_flag=None,
        return_flag_loss=False,
    ):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), (
            f"your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet"
        )

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        if cond_img is not None:
            x = torch.cat((x, cond_img), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        drop_prob = self.clinical_drop_prob if clinical_drop_prob is None else clinical_drop_prob

        if self.use_clinical:
            clin_vec = clinical_vars
            B, device = clin_vec.size(0), clin_vec.device
            clin_emb_full = self.clinical_vars_mlp(clin_vec)

            if drop_prob == 1.0:
                keep_mask = torch.zeros(B, dtype=torch.bool, device=device)
            elif drop_prob == 0.0:
                keep_mask = torch.ones(B, dtype=torch.bool, device=device)
            else:
                keep_mask = torch.rand(B, device=device) > drop_prob

            clin_emb = torch.where(
                keep_mask[:, None],
                clin_emb_full,
                self.null_clin_emb[None, :].expand_as(clin_emb_full),
            )

            if self.use_expansion_flag_input:
                if expansion_flag is None:
                    raise ValueError(
                        "expansion_flag is required when use_expansion_flag_input=True. "
                        "Pass 0/1 (or disable use_expansion_flag_input)."
                    )

                flag_emb = self.flag_embed(expansion_flag.long())
                t = t + clin_emb + flag_emb
            else:
                t = t + clin_emb

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        out_image = self.final_image_conv(x)
        return out_image

    def forward_with_cond_scale(
        self,
        x,
        time,
        x_self_cond=None,
        *,
        cond_img=None,
        seg=None,
        clinical_vars=None,
        expansion_flag=None,
        cond_scale=1.0,
        rescaled_phi=0.0,
        remove_parallel_component=True,
        keep_parallel_frac=0.0,
    ):
        logits = self.forward(
            x,
            time,
            cond_img=cond_img,
            seg=seg,
            x_self_cond=x_self_cond,
            clinical_vars=clinical_vars,
            clinical_drop_prob=0.0,
            expansion_flag=expansion_flag,
        )

        if cond_scale == 1:
            return logits, None

        null_logits = self.forward(
            x,
            time,
            cond_img=cond_img,
            seg=seg,
            x_self_cond=x_self_cond,
            clinical_vars=clinical_vars,
            clinical_drop_prob=1.0,
            expansion_flag=expansion_flag,
        )

        update = logits - null_logits

        if remove_parallel_component:
            parallel, orthog = project(update, logits)
            update = orthog + parallel * keep_parallel_frac

        scaled = logits + update * (cond_scale - 1.0)

        if rescaled_phi == 0.0:
            return scaled, null_logits

        std_fn = partial(torch.std, dim=tuple(range(1, scaled.ndim)), keepdim=True)
        rescaled = scaled * (std_fn(logits) / std_fn(scaled))
        blended = rescaled * rescaled_phi + scaled * (1.0 - rescaled_phi)
        return blended, null_logits


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_v",
        beta_schedule="sigmoid",
        schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0.0,
        auto_normalize=True,
        offset_noise_strength=0.0,
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        immiscible=False,
        use_latent_diffusion=False,
        autoencoder=None,
        segmentation_loss_weight: float = 0.0,
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, "random_or_learned_sinusoidal_cond") or not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.autoencoder = autoencoder

        self.use_latent_diffusion = use_latent_diffusion
        self.segmentation_loss_weight = segmentation_loss_weight

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2
        self.image_size = image_size

        self.objective = objective
        assert objective in {"pred_noise", "pred_x0", "pred_v"}

        if beta_schedule == "linear":
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == "cosine":
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == "sigmoid":
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f"unknown beta schedule {beta_schedule}")

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register_buffer("posterior_variance", posterior_variance)
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        self.immiscible = immiscible
        self.offset_noise_strength = offset_noise_strength

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            register_buffer("loss_weight", maybe_clipped_snr / snr)
        elif objective == "pred_x0":
            register_buffer("loss_weight", maybe_clipped_snr)
        elif objective == "pred_v":
            register_buffer("loss_weight", maybe_clipped_snr / (snr + 1))

        if use_latent_diffusion:
            self.normalize = identity
            self.unnormalize = identity
        else:
            self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
            self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_v(self, x_start, t, noise):
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start

    def predict_start_from_v(self, x_t, t, v):
        return extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
        self,
        x,
        t,
        x_self_cond=None,
        cond_img=None,
        seg=None,
        clinical_vars=None,
        expansion_flag=None,
        clip_x_start=False,
        rederive_pred_noise=False,
        cond_scale=1,
        rescaled_phi=0.0,
    ):
        model_out, model_out_null = self.model.forward_with_cond_scale(
            x,
            t,
            cond_img=cond_img,
            seg=seg,
            clinical_vars=clinical_vars,
            expansion_flag=expansion_flag,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
        )

        pred_image = model_out
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = pred_image
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            x_start = pred_image
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = pred_image
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, cond_img=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond, cond_img=cond_img)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond=None, cond_img=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, x_self_cond=x_self_cond, cond_img=cond_img, clip_denoised=True
        )
        noise = torch.randn_like(x) if t > 0 else 0.0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond_img=None, return_all_timesteps=False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(
        self,
        shape,
        cond_img=None,
        seg=None,
        clinical_vars=None,
        expansion_flag=None,
        return_all_timesteps=False,
        cond_scale=1,
        rescaled_phi=0,
    ):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0],
            self.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            if self.use_latent_diffusion:
                model_pred = self.model_predictions(
                    img,
                    time_cond,
                    self_cond,
                    cond_img=cond_img,
                    seg=seg,
                    clinical_vars=clinical_vars,
                    expansion_flag=expansion_flag,
                    clip_x_start=False,
                    rederive_pred_noise=True,
                    cond_scale=cond_scale,
                    rescaled_phi=rescaled_phi,
                )
            else:
                clip_x0 = not self.use_latent_diffusion
                model_pred = self.model_predictions(
                    img,
                    time_cond,
                    self_cond,
                    cond_img=cond_img,
                    seg=seg,
                    clinical_vars=clinical_vars,
                    expansion_flag=expansion_flag,
                    clip_x_start=clip_x0,
                    rederive_pred_noise=True,
                    cond_scale=cond_scale,
                    rescaled_phi=rescaled_phi,
                )

            pred_noise = model_pred.pred_noise
            x_start = model_pred.pred_x_start

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timesteps=False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps=return_all_timesteps)

    @torch.inference_mode()
    def sample_conditioned(self, cond_img, seg, clinical_vars, expansion_flag, return_all_timesteps=False, cond_scale=1, rescaled_phi=0):
        (b, _, h, w), device = cond_img.shape, cond_img.device
        shape = (b, self.channels, h, w)

        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        img = sample_fn(
            shape,
            cond_img=cond_img,
            seg=seg,
            clinical_vars=clinical_vars,
            expansion_flag=expansion_flag,
            return_all_timesteps=return_all_timesteps,
            cond_scale=cond_scale,
            rescaled_phi=rescaled_phi,
        )

        return img

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc="interpolation sample time step", total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def noise_assignment(self, x_start, noise):
        x_start, noise = tuple(rearrange(t, "b ... -> b (...)") for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast("cuda", enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]

        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise

    def p_losses(
        self,
        x_start,
        t,
        noise=None,
        cond_img=None,
        seg=None,
        clinical_vars=None,
        target_seg=None,
        offset_noise_strength=None,
        expansion_flag=None,
        flag_alpha=0.2,
    ):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        pred_image = self.model(
            x,
            t,
            x_self_cond=None,
            cond_img=cond_img,
            seg=seg,
            clinical_vars=clinical_vars,
            expansion_flag=expansion_flag,
            return_flag_loss=False,
        )

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss_diffusion = F.mse_loss(pred_image, target, reduction="none")
        loss_diffusion = reduce(loss_diffusion, "b ... -> b", "mean")
        loss_diffusion = loss_diffusion * extract(self.loss_weight, t, loss_diffusion.shape)
        loss_diffusion = loss_diffusion.mean()

        loss = loss_diffusion

        if (not self.use_latent_diffusion) and (self.segmentation_loss_weight > 0):
            seg_gt = (x_start[:, 1:3] > 0).float()

            if self.objective == "pred_v":
                x0_pred = self.predict_start_from_v(x, t, pred_image)
            elif self.objective == "pred_noise":
                x0_pred = self.predict_start_from_noise(x, t, pred_image)
            elif self.objective == "pred_x0":
                x0_pred = pred_image

            seg_pred = x0_pred[:, 1:3].clamp(-1, 1)

            seg_logit_scale = 4.0
            seg_logits = seg_logit_scale * seg_pred

            loss_seg = F.binary_cross_entropy_with_logits(seg_logits, seg_gt, reduction="mean")
            loss = loss + self.segmentation_loss_weight * loss_seg

        return loss

    def forward(self, img, cond_img=None, seg=None, clinical_vars=None, target_seg=None, *args, **kwargs):
        b, c, h, w, device, img_size = *img.shape, img.device, self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = self.normalize(img)
        return self.p_losses(img, t, cond_img=cond_img, seg=seg, clinical_vars=clinical_vars, target_seg=target_seg, *args, **kwargs)


class Trainer:
    def __init__(
        self,
        diffusion_model,
        clinical_csv,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_flip=False,
        augment_rotation=False,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        convert_image_to=None,
        calculate_fid=True,
        inception_block_idx=2048,
        max_grad_norm=1.0,
        num_fid_samples=50000,
        save_best_and_latest_only=False,
        use_clinical=True,
        use_expansion_flag_input=True,
        rotation_range=0,
        use_latent_diffusion=False,
        latent_predict_delta: bool = True,
        ae_input_size=None,
        resume_checkpoint=None,
        context_slices=1,
        context_slice_mm=5,
        slice_thickness_csv=None,
        default_slice_thickness_mm=5.0,
        slices_root: str = None,
        train_hospitals=None,
        val_hospitals=None,
        ct_do_window: bool = True,
        ct_window_min: float = 0.0,
        ct_window_max: float = 100.0,
        keep_only_baseline_iph_slices: bool = False,
        seg_threshold: float = 0.5,
        min_lesion_pixels_2d: int = 1,
        latent_mean=None,
        latent_std=None,
    ):
        super().__init__()

        self.accelerator = Accelerator(split_batches=split_batches, mixed_precision=mixed_precision_type if amp else "no")

        self.use_latent_diffusion = use_latent_diffusion
        self.latent_predict_delta = bool(latent_predict_delta)

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", ignore_empty=True, get_not_nans=False)

        self._running = {
            "ssim": 0.0,
            "dice_iph": 0.0,
            "dice_ivh": 0.0,
            "vol_sim": 0.0,
            "abs_vol_diff": 0.0,
            "vol_sim_IPH": 0.0,
            "abs_vol_diff_IPH": 0.0,
            "vol_sim_IVH": 0.0,
            "abs_vol_diff_IVH": 0.0,
        }

        self._n_batches = 0
        self.use_clinical = use_clinical
        self.use_expansion_flag_input = bool(use_expansion_flag_input)

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        if not exists(convert_image_to):
            convert_image_to = {1: "L", 3: "RGB", 4: "RGBA"}.get(self.channels)

        assert has_int_squareroot(num_samples)
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        if self.use_latent_diffusion:
            assert ae_input_size is not None
            self.ae_input_size = int(ae_input_size)
            dataset_image_size = self.ae_input_size
        else:
            dataset_image_size = self.image_size[0] if isinstance(self.image_size, (tuple, list)) else self.image_size

        self.max_grad_norm = max_grad_norm
        self.context_slices = context_slices
        self.context_slice_mm = context_slice_mm
        self.slice_thickness_csv = slice_thickness_csv

        self.latent_mean = latent_mean
        self.latent_std = latent_std

        self.ds = PairedNiftiSliceDataset(
            slices_root=slices_root,
            clinical_csv=clinical_csv,
            image_size=dataset_image_size,
            include_hospitals=train_hospitals,
            keep_only_baseline_iph_slices=False,
            augment_flip=augment_flip,
            augment_rotation=augment_rotation,
            rotation_range_deg=rotation_range,
            use_clinical=self.use_clinical,
            context_slices=self.context_slices,
            context_mm_step=self.context_slice_mm,
            slice_thickness_csv=self.slice_thickness_csv,
            default_slice_thickness_mm=default_slice_thickness_mm,
            ct_do_window=ct_do_window,
            ct_window_min=ct_window_min,
            ct_window_max=ct_window_max,
        )

        val_ds = PairedNiftiSliceDataset(
            slices_root=slices_root,
            clinical_csv=clinical_csv,
            image_size=dataset_image_size,
            include_hospitals=val_hospitals,
            keep_only_baseline_iph_slices=False,
            augment_flip=False,
            augment_rotation=False,
            rotation_range_deg=0.0,
            use_clinical=self.use_clinical,
            context_slices=self.context_slices,
            context_mm_step=self.context_slice_mm,
            slice_thickness_csv=self.slice_thickness_csv,
            default_slice_thickness_mm=default_slice_thickness_mm,
            ct_do_window=ct_do_window,
            ct_window_min=ct_window_min,
            ct_window_max=ct_window_max,
        )

        self.val_ds = val_ds

        train_patients = sorted({s["patient_key"] for s in self.ds.samples})
        val_patients = sorted({s["patient_key"] for s in val_ds.samples})

        print(f"Train: {len(self.ds)} slices | {len(train_patients)} patients")
        print(f"Val:   {len(val_ds)} slices | {len(val_patients)} patients")

        val_sampler = ExpansionBalancedSampler(val_ds, batch_size=self.batch_size, drop_last=False)

        val_loader = DataLoader(val_ds, batch_sampler=val_sampler, pin_memory=True, num_workers=worker_count)
        self.val_dl = self.accelerator.prepare(val_loader)

        assert len(self.ds) >= 100

        sampler = ExpansionBalancedSampler(self.ds, batch_size=train_batch_size)

        dl = DataLoader(self.ds, batch_sampler=sampler, pin_memory=True, num_workers=worker_count)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            self.running_ssim = 0.0
            self.ssim_count = 0

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.step = 0

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        self.resume_checkpoint = resume_checkpoint
        if self.resume_checkpoint is not None:
            self.load(self.resume_checkpoint)


        if self.accelerator.is_main_process:
            wandb_project = os.environ.get("WANDB_PROJECT", "PUBLIC_PROJECT")
            wandb_entity = os.environ.get("WANDB_ENTITY")

            wandb_init_kwargs = dict(
                project=wandb_project,
                config={
                    "train_batch_size": train_batch_size,
                    "lr": train_lr,
                    "num_steps": train_num_steps,
                    "augment_flip": ema_update_every,
                    "augment_rotation": ema_decay,
                },
            )

            if wandb_entity:
                wandb_init_kwargs["entity"] = wandb_entity

            wandb.init(**wandb_init_kwargs)

            wandb.watch(self.model, log="all", log_freq=500)

    @torch.no_grad()
    def _compute_batch_metrics(self, real_ct: torch.Tensor, fake_ct: torch.Tensor, real_seg: torch.Tensor, fake_seg: torch.Tensor):
        self.dice_metric.reset()

        B = real_ct.size(0)
        eps = 1e-6

        ssim_val = float(self.ssim(fake_ct, real_ct).item())

        pred_bin = (fake_seg > 0.5).float()
        tgt_bin = (real_seg > 0.5).float()
        pred_onehot = torch.cat([1 - pred_bin, pred_bin], dim=1)
        tgt_onehot = torch.cat([1 - tgt_bin, tgt_bin], dim=1)

        self.dice_metric.reset()
        self.dice_metric(y_pred=pred_onehot, y=tgt_onehot)
        dice_val = float(self.dice_metric.aggregate().mean().item())
        self.dice_metric.reset()

        vol_p = pred_bin.sum(dim=(1, 2, 3))
        vol_t = tgt_bin.sum(dim=(1, 2, 3))
        abs_diff = (vol_p - vol_t).abs()
        vs = 1 - abs_diff / (vol_p + vol_t + eps)
        vol_sim_val = float(vs.mean().item())
        abs_vol_diff_val = float(abs_diff.mean().item())

        return ssim_val, dice_val, vol_sim_val, abs_vol_diff_val

    @torch.no_grad()
    def run_train_metrics(self, bl_ct, fu_ct, bl_iph, bl_ivh, fu_iph, fu_ivh, clinical_variables, expansion_flag, milestone):
        device = self.device
        K = self.context_slices
        mid = K // 2

        bl_ct = bl_ct.to(device)
        fu_ct = fu_ct.to(device)
        bl_iph = bl_iph.to(device)
        bl_ivh = bl_ivh.to(device)
        fu_iph = fu_iph.to(device)
        fu_ivh = fu_ivh.to(device)

        clinical_variables = clinical_variables.to(device)
        expansion_flag = expansion_flag.to(device)

        if self.use_latent_diffusion:
            cond_z = encode_triplet_stack_chunked(self.model.autoencoder, bl_ct, bl_iph, bl_ivh)
            target_z = encode_center_triplet(self.model.autoencoder, fu_ct, fu_iph, fu_ivh, mid=mid)

            cond_z = norm_latent(cond_z, self.latent_mean, self.latent_std, repeat_k=K)
            target_z = norm_latent(target_z, self.latent_mean, self.latent_std, repeat_k=1)

            gen_pred = self.ema.ema_model.sample_conditioned(cond_img=cond_z, seg=None, clinical_vars=clinical_variables, expansion_flag=expansion_flag)

            base_z = extract_center_latent_from_stack(cond_z, K)

            gen_z = gen_pred + base_z if self.latent_predict_delta else gen_pred

            gen_native = unnorm_latent(gen_z, self.latent_mean, self.latent_std, repeat_k=1)
            real_native = unnorm_latent(target_z, self.latent_mean, self.latent_std, repeat_k=1)
            cond_native = unnorm_latent(cond_z, self.latent_mean, self.latent_std, repeat_k=K)

            gen = self.model.autoencoder.decode(gen_native)
            real = self.model.autoencoder.decode(real_native)
            cond = decode_center_from_stack(self.model.autoencoder, cond_native, K)

        else:
            cond_img_pix = torch.cat([bl_ct, bl_iph, bl_ivh], dim=1)
            real = torch.cat([fu_ct[:, mid : mid + 1], fu_iph[:, mid : mid + 1], fu_ivh[:, mid : mid + 1]], dim=1)
            cond = torch.cat([bl_ct[:, mid : mid + 1], bl_iph[:, mid : mid + 1], bl_ivh[:, mid : mid + 1]], dim=1)

            gen = self.ema.ema_model.sample_conditioned(
                cond_img=cond_img_pix, seg=None, clinical_vars=clinical_variables, expansion_flag=expansion_flag
            ).clamp(0, 1)

        real_ct, real_iph, real_ivh = real[:, 0:1], real[:, 1:2], real[:, 2:3]
        gen_ct, gen_iph, gen_ivh = gen[:, 0:1], gen[:, 1:2], gen[:, 2:3]

        ssim_b, dice_iph, vs_IPH, abs_vd_IPH = self._compute_batch_metrics(real_ct, gen_ct, real_iph, gen_iph)
        ssim_b, dice_ivh, vs_IVH, abs_vd_IVH = self._compute_batch_metrics(real_ct, gen_ct, real_ivh, gen_ivh)

        self._running["ssim"] += ssim_b
        self._running["dice_iph"] += dice_iph
        self._running["dice_ivh"] += dice_ivh
        self._running["vol_sim_IPH"] += vs_IPH
        self._running["abs_vol_diff_IPH"] += abs_vd_IPH
        self._running["vol_sim_IVH"] += vs_IVH
        self._running["abs_vol_diff_IVH"] += abs_vd_IVH
        self._n_batches += 1

        avg = {k: self._running[k] / self._n_batches for k in self._running}

        wandb.log(
            {
                "train/ssim": avg["ssim"],
                "train/dice_iph": avg["dice_iph"],
                "train/dice_ivh": avg["dice_ivh"],
                "train/vol_sim_IPH": avg["vol_sim_IPH"],
                "train/vol_sim_IVH": avg["vol_sim_IVH"],
                "train/abs_vol_diff_IPH": avg["abs_vol_diff_IPH"],
                "train/abs_vol_diff_IVH": avg["abs_vol_diff_IVH"],
            },
            step=self.step,
        )

        utils.save_image(cond[:, 0:1], self.results_folder / f"{milestone}_sample-img-ct_cond.png")
        utils.save_image(cond[:, 1:2], self.results_folder / f"{milestone}_sample-img-seg1_cond.png")
        utils.save_image(cond[:, 2:3], self.results_folder / f"{milestone}_sample-img-seg2_cond.png")

        utils.save_image(gen_ct, self.results_folder / f"{milestone}_sample-img-ct_pred.png")
        utils.save_image((gen_iph > 0.5).float(), self.results_folder / f"{milestone}_sample-img-seg1_pred.png")
        utils.save_image((gen_ivh > 0.5).float(), self.results_folder / f"{milestone}_sample-img-seg2_pred.png")

        utils.save_image(real_ct, self.results_folder / f"{milestone}_sample-img-ct_target.png")
        utils.save_image(real_iph, self.results_folder / f"{milestone}_sample-img-seg1_target.png")
        utils.save_image(real_ivh, self.results_folder / f"{milestone}_sample-img-seg2_target.png")

    @torch.no_grad()
    def run_validation(self, milestone):
        USE_EMA = True

        val_ds = getattr(self, "val_ds", None) or getattr(self.val_dl, "dataset", None)
        if val_ds is None:
            raise RuntimeError("Could not access validation dataset (self.val_ds missing).")

        sampler_model = self.ema.ema_model if (USE_EMA and hasattr(self, "ema")) else self.model
        sampler_model.eval()

        exp_idx = []
        stab_by_pk = {}

        for i, s in enumerate(val_ds.samples):
            pk = s["patient_key"]
            exp = val_ds.metadata_expansion.get(pk, 0)
            exp_int = 0 if pd.isna(exp) else int(exp)
            exp_int = 1 if exp_int == 1 else 0

            if exp_int == 1:
                exp_idx.append(i)
            else:
                stab_by_pk.setdefault(pk, []).append(i)

        exp_idx = sorted(exp_idx)
        for pk in stab_by_pk:
            stab_by_pk[pk] = sorted(stab_by_pk[pk])

        if len(exp_idx) == 0:
            raise RuntimeError("Validation set has 0 expansion slices.")

        N = len(exp_idx)
        total_stab = sum(len(v) for v in stab_by_pk.values())
        if total_stab < N:
            raise RuntimeError(f"Not enough non-expansion slices to match expansion slices: {total_stab} < {N}")

        stab_pks = sorted(stab_by_pk.keys())
        ptr = {pk: 0 for pk in stab_pks}
        stab_sel = []

        while len(stab_sel) < N:
            picked_any = False
            for pk in stab_pks:
                j = ptr[pk]
                if j < len(stab_by_pk[pk]):
                    stab_sel.append(stab_by_pk[pk][j])
                    ptr[pk] = j + 1
                    picked_any = True
                    if len(stab_sel) == N:
                        break
            if not picked_any:
                raise RuntimeError("Round-robin selection got stuck (unexpected).")

        eval_idx = exp_idx + stab_sel

        bs = int(self.batch_size)
        batches = [eval_idx[i : i + bs] for i in range(0, len(eval_idx), bs)]

        val_loader = DataLoader(val_ds, batch_sampler=batches, pin_memory=True, num_workers=worker_count)
        val_loader = self.accelerator.prepare(val_loader)

        total = 0
        sum_ssim = 0.0
        sum_dice_iph = sum_vs_iph = sum_abs_iph = 0.0
        sum_dice_ivh = sum_vs_ivh = sum_abs_ivh = 0.0

        K = self.context_slices
        mid = K // 2
        device = self.device

        for bl_ct, fu_ct, bl_iph, bl_ivh, fu_iph, fu_ivh, clinical_vars, expansion_flag in val_loader:
            bl_ct, fu_ct, bl_iph, bl_ivh, fu_iph, fu_ivh, clinical_vars, expansion_flag = (
                t.to(device) for t in (bl_ct, fu_ct, bl_iph, bl_ivh, fu_iph, fu_ivh, clinical_vars, expansion_flag)
            )
            expansion_flag = expansion_flag.squeeze(1)

            if self.use_latent_diffusion:
                cond_z = encode_triplet_stack_chunked(self.model.autoencoder, bl_ct, bl_iph, bl_ivh)
                target_z = encode_center_triplet(self.model.autoencoder, fu_ct, fu_iph, fu_ivh, mid=mid)

                cond_z = norm_latent(cond_z, self.latent_mean, self.latent_std, repeat_k=K)
                target_z = norm_latent(target_z, self.latent_mean, self.latent_std, repeat_k=1)

                gen_pred = sampler_model.sample_conditioned(cond_img=cond_z, seg=None, clinical_vars=clinical_vars, expansion_flag=expansion_flag)

                base_z = extract_center_latent_from_stack(cond_z, K)
                gen_z = gen_pred + base_z if self.latent_predict_delta else gen_pred

                gen_native = unnorm_latent(gen_z, self.latent_mean, self.latent_std, repeat_k=1)
                real_native = unnorm_latent(target_z, self.latent_mean, self.latent_std, repeat_k=1)

                gen = self.model.autoencoder.decode(gen_native)
                real = self.model.autoencoder.decode(real_native)
            else:
                cond_img_pix = torch.cat([bl_ct, bl_iph, bl_ivh], dim=1)
                real = torch.cat([fu_ct[:, mid : mid + 1], fu_iph[:, mid : mid + 1], fu_ivh[:, mid : mid + 1]], dim=1)

                gen = sampler_model.sample_conditioned(cond_img=cond_img_pix, seg=None, clinical_vars=clinical_vars, expansion_flag=expansion_flag)

            gen = gen.clamp(0, 1)
            real = real.clamp(0, 1)

            B = int(gen.shape[0])
            total += B

            real_ct, real_iph, real_ivh = real[:, 0:1], real[:, 1:2], real[:, 2:3]
            gen_ct, gen_iph, gen_ivh = gen[:, 0:1], gen[:, 1:2], gen[:, 2:3]

            ssim_iph, dice_iph, vs_iph, abs_iph = self._compute_batch_metrics(real_ct, gen_ct, real_iph, gen_iph)
            ssim_ivh, dice_ivh, vs_ivh, abs_ivh = self._compute_batch_metrics(real_ct, gen_ct, real_ivh, gen_ivh)

            sum_ssim += ssim_iph * B
            sum_dice_iph += dice_iph * B
            sum_vs_iph += vs_iph * B
            sum_abs_iph += abs_iph * B
            sum_dice_ivh += dice_ivh * B
            sum_vs_ivh += vs_ivh * B
            sum_abs_ivh += abs_ivh * B

        wandb.log(
            {
                "val/ssim": sum_ssim / total,
                "val/dice_iph": sum_dice_iph / total,
                "val/dice_ivh": sum_dice_ivh / total,
                "val/vol_sim_iph": sum_vs_iph / total,
                "val/vol_sim_ivh": sum_vs_ivh / total,
                "val/abs_vol_diff_iph": sum_abs_iph / total,
                "val/abs_vol_diff_ivh": sum_abs_ivh / total,
            },
            step=self.step,
        )

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            "version": __version__,
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f"model-{milestone}.pt"), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)

        model.load_state_dict(data["model"], strict=False)

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    bl_ct, fu_ct, bl_iph, bl_ivh, fu_iph, fu_ivh, clinical_variables, expansion_flag = next(self.dl)

                    bl_ct, fu_ct, bl_iph, bl_ivh, fu_iph, fu_ivh = (t.to(device) for t in (bl_ct, fu_ct, bl_iph, bl_ivh, fu_iph, fu_ivh))

                    clinical_variables = clinical_variables.to(device)
                    expansion_flag = expansion_flag.to(device).squeeze(1)

                    K = self.context_slices
                    mid = K // 2

                    with self.accelerator.autocast():
                        if self.use_latent_diffusion:
                            with torch.no_grad():
                                cond_z = encode_triplet_stack_chunked(self.model.autoencoder, bl_ct, bl_iph, bl_ivh)
                                target_z = encode_center_triplet(self.model.autoencoder, fu_ct, fu_iph, fu_ivh, mid=mid)

                                cond_z = norm_latent(cond_z, self.latent_mean, self.latent_std, repeat_k=K)
                                target_z = norm_latent(target_z, self.latent_mean, self.latent_std, repeat_k=1)

                            base_z = extract_center_latent_from_stack(cond_z, K)

                            if self.latent_predict_delta:
                                diffusion_target = target_z - base_z
                            else:
                                diffusion_target = target_z

                            loss = self.model(diffusion_target, cond_img=cond_z, clinical_vars=clinical_variables, expansion_flag=expansion_flag)

                        else:
                            cond_img_pix = torch.cat([bl_ct, bl_iph, bl_ivh], dim=1)

                            target_center = torch.cat(
                                [fu_ct[:, mid : mid + 1], fu_iph[:, mid : mid + 1], fu_ivh[:, mid : mid + 1]], dim=1
                            )

                            loss = self.model(target_center, cond_img=cond_img_pix, clinical_vars=clinical_variables, expansion_flag=expansion_flag)

                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f"loss: {total_loss:.4f}")

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1

                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every

                            self.run_train_metrics(
                                bl_ct, fu_ct, bl_iph, bl_ivh, fu_iph, fu_ivh, clinical_variables, expansion_flag, milestone
                            )
                            self.run_validation(milestone)
                            self.save(milestone)

                pbar.update(1)
        accelerator.print("training complete")