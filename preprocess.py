#!/usr/bin/env python3
import math
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from scipy.ndimage import (
    binary_dilation,
    binary_propagation,
    generate_binary_structure,
    label,
)

DATA_ROOT = Path("data") #Input data root
OUTPUT_ROOT = Path("output") #Output data root
FREESURFER_HOME = os.environ.get("FREESURFER_HOME", "") # set freesurfer path

NCCT_DIR = DATA_ROOT / "images"                     # Folder with NCCT NIfTI volumes (.nii or .nii.gz); must include paired baseline_* and follow-up_* files
IPH_SEG_DIR = DATA_ROOT / "IPH_segs"                # Folder with IPH segmentation NIfTIs (binary/label masks) matching NCCT filenames 1:1 (same stem + extension)
IVH_SEG_DIR = DATA_ROOT / "IVH_segs"                # Folder with IVH segmentation NIfTIs matching NCCT filenames 1:1 (same stem + extension)
TEMPLATE_DIR = DATA_ROOT / "templates"              # Folder with template CT volumes: template_1mm.nii.gz ... template_5mm.nii.gz (chosen by slice thickness)
TEMPLATE_MASK_DIR = DATA_ROOT / "template_masks"    # Folder with corresponding brain masks: template_1mm.nii.gz ... template_5mm.nii.gz (same spacing as templates)

PATIENTS_OUT_DIR = OUTPUT_ROOT / "processed_patients"
SLICES_NIFTI_DIR = OUTPUT_ROOT / "ncct_niftis"
SLICES_PNG_DIR = OUTPUT_ROOT / "ncct_png"
SLICES_IPH_PNG_DIR = OUTPUT_ROOT / "segmentation_png_iph"
SLICES_IVH_PNG_DIR = OUTPUT_ROOT / "segmentation_png_ivh"
SLICES_SEG_NIFTI_DIR = OUTPUT_ROOT / "segmentation_niftis"

ANTSREG_CMD = os.environ.get("ANTSREG_CMD", "antsRegistration")
ANTSAPPLY_CMD = os.environ.get("ANTSAPPLY_CMD", "antsApplyTransforms")


USE_GPU_FOR_SYNTHSTRIP = False

NUM_THREADS = 12
TARGET_SLICE_SIZE = 384
FINAL_CROP_SIZE = (384, 384)
HU_WINDOW = (0, 100)

for d in [
    PATIENTS_OUT_DIR,
    SLICES_NIFTI_DIR,
    SLICES_PNG_DIR,
    SLICES_IPH_PNG_DIR,
    SLICES_IVH_PNG_DIR,
    SLICES_SEG_NIFTI_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


def get_file_stem(filepath: str) -> str:
    base = os.path.basename(filepath)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]


def window_image(input_path: str, output_path: str, lower: float, upper: float) -> None:
    img = nib.load(input_path)
    data = img.get_fdata()
    data_windowed = np.clip(data, lower, upper)
    nib.save(nib.Nifti1Image(data_windowed, affine=img.affine, header=img.header), output_path)


def fix_affine_to_orthonormal(input_path: str, output_path: str) -> None:
    img = nib.load(input_path)
    canonical_img = nib.as_closest_canonical(img)
    data = canonical_img.get_fdata()
    voxel_sizes = nib.affines.voxel_sizes(canonical_img.affine)
    new_affine = np.diag(list(voxel_sizes) + [1.0])
    nib.save(nib.Nifti1Image(data, new_affine, header=canonical_img.header), output_path)


def remove_surface_artifacts_outside_in(
    data: np.ndarray,
    intensity_threshold: float = 75,
    max_iters: int = 20
) -> np.ndarray:
    structure = generate_binary_structure(rank=3, connectivity=1)

    def get_outside_background_mask(data_3d: np.ndarray) -> np.ndarray:
        bg_mask = (data_3d == 0)
        seeds = np.zeros_like(bg_mask, dtype=bool)
        seeds[0, :, :] = bg_mask[0, :, :]
        seeds[-1, :, :] = bg_mask[-1, :, :]
        seeds[:, 0, :] = bg_mask[:, 0, :]
        seeds[:, -1, :] = bg_mask[:, -1, :]
        seeds[:, :, 0] = bg_mask[:, :, 0]
        seeds[:, :, -1] = bg_mask[:, :, -1]
        return binary_propagation(seeds, structure=structure, mask=bg_mask)

    iteration = 0
    while iteration < max_iters:
        brain_mask = (data != 0)
        outside_bg_mask = get_outside_background_mask(data)
        outside_bg_dilated = binary_dilation(outside_bg_mask, structure=structure)
        external_boundary = brain_mask & outside_bg_dilated
        boundary_to_remove = (data > intensity_threshold) & external_boundary

        if not np.any(boundary_to_remove):
            break

        data[boundary_to_remove] = 0
        iteration += 1

    return data


def keep_largest_component(data: np.ndarray) -> np.ndarray:
    structure = generate_binary_structure(rank=3, connectivity=1)
    mask = (data != 0)
    labeled, num_labels = label(mask, structure=structure)

    if num_labels < 2:
        return data

    label_sizes = np.bincount(labeled.ravel())
    label_sizes[0] = 0
    largest_label = int(label_sizes.argmax())
    largest_mask = (labeled == largest_label)
    data[~largest_mask] = 0
    return data


def custom_shaving_skullstripped_image(
    skullstripped_nifti_path: str,
    segmentation_nifti_path: str,
    original_nifti_path: str,
    output_nifti_path: str
) -> None:
    skull_img = nib.load(skullstripped_nifti_path)
    skull_data = skull_img.get_fdata()

    seg_data = nib.load(segmentation_nifti_path).get_fdata()
    orig_data = nib.load(original_nifti_path).get_fdata()

    cleaned_data = remove_surface_artifacts_outside_in(
        skull_data.copy(),
        intensity_threshold=75,
        max_iters=20
    )
    cleaned_data = keep_largest_component(cleaned_data)
    cleaned_data[seg_data > 0] = orig_data[seg_data > 0]

    nib.save(nib.Nifti1Image(cleaned_data, skull_img.affine, skull_img.header), output_nifti_path)


def skullstrip_ncct(input_path: str, output_path: str, output_mask: str, use_gpu: bool) -> None:
    if not FREESURFER_HOME:
        raise RuntimeError("FREESURFER_HOME is not set.")

    env = os.environ.copy()
    env["FREESURFER_HOME"] = FREESURFER_HOME

    fs_bin = str(Path(FREESURFER_HOME) / "bin")
    env["PATH"] = env.get("PATH", "") + os.pathsep + fs_bin

    fs_lib = str(Path(FREESURFER_HOME) / "lib")
    env["LD_LIBRARY_PATH"] = fs_lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        "mri_synthstrip",
        "-i", input_path,
        "-o", output_path,
        "-m", output_mask,
        "-b", "0",
    ]
    if use_gpu:
        cmd.append("--gpu")

    subprocess.run(cmd, check=True, env=env)

    if not os.path.exists(output_path):
        raise RuntimeError(f"Skullstripped output not found: {output_path}")


def register_to_template(
    template_ct: str,
    moving_ct: str,
    fixed_mask: str,
    moving_mask: str,
    transform_prefix: str,
    warped_image: str
) -> None:
    cmd = [
        ANTSREG_CMD, "-d", "3",
        "-r", f"[{template_ct},{moving_ct},1]",
        "-m", f"MI[{template_ct},{moving_ct},1,128,Regular,1]",
        "-t", "Rigid[0.1]",
        "-c", "[1000x500x250x100,1e-6,10]",
        "-s", "3x2x1x0vox",
        "-f", "8x4x2x1",
        "-x", f"[{fixed_mask},{moving_mask}]",
        "-n", "Linear",
        "-z", "1",
        "-o", f"[{transform_prefix},{warped_image}]",
    ]
    subprocess.run(cmd, check=True)


def run_affine_registration(fixed: str, moving: str, transform_prefix: str, warped_image: str) -> None:
    cmd = [
        ANTSREG_CMD, "-d", "3",
        "-r", f"[{fixed},{moving},1]",
        "-m", f"MI[{fixed},{moving},1,32,Regular,0.25]",
        "-t", "Affine[0.1]",
        "-c", "[1000x500x250x100,1e-6,10]",
        "-s", "4x3x2x1",
        "-f", "12x8x4x2",
        "-n", "Linear",
        "-z", "1",
        "-o", f"[{transform_prefix},{warped_image}]",
    ]
    subprocess.run(cmd, check=True)


def apply_transforms(
    moving_image: str,
    fixed_image: str,
    transform: str,
    output_image: str,
    interpolation: str,
    invert_transform: bool = False
) -> None:
    t_arg = f"[{transform},1]" if invert_transform else transform
    cmd = [
        ANTSAPPLY_CMD, "-d", "3",
        "-i", moving_image,
        "-r", fixed_image,
        "-o", output_image,
        "-t", t_arg,
        "-n", interpolation,
    ]
    subprocess.run(cmd, check=True)


def center_crop_nifti_inplace(fn: str, crop_size=(384, 384)) -> None:
    img = nib.load(fn)
    data = img.get_fdata()
    affine = img.affine
    hdr = img.header.copy()

    nx, ny, nz = data.shape
    cx, cy = nx // 2, ny // 2
    hx, hy = crop_size[0] // 2, crop_size[1] // 2

    x0 = max(cx - hx, 0)
    x1 = min(cx + hx, nx)
    y0 = max(cy - hy, 0)
    y1 = min(cy + hy, ny)

    cropped = data[x0:x1, y0:y1, :]
    hdr.set_data_shape(cropped.shape)
    nib.save(nib.Nifti1Image(cropped, affine, hdr), fn)


def center_crop_multiple_niftis(filenames, crop_size=(384, 384)) -> None:
    for fn in filenames:
        center_crop_nifti_inplace(fn, crop_size)


def crop_pad_2d(image: np.ndarray, target_size=(512, 512)) -> np.ndarray:
    h, w = image.shape
    th, tw = target_size

    if h > th:
        start_h = (h - th) // 2
        image = image[start_h:start_h + th, :]
    elif h < th:
        pad_h = th - h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        image = np.pad(image, ((pad_top, pad_bottom), (0, 0)), mode="constant")

    if w > tw:
        start_w = (w - tw) // 2
        image = image[:, start_w:start_w + tw]
    elif w < tw:
        pad_w = tw - w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image = np.pad(image, ((0, 0), (pad_left, pad_right)), mode="constant")

    return image


def get_slice_affine(full_affine: np.ndarray, slice_index: int) -> np.ndarray:
    new_affine = full_affine.copy()
    new_affine[:3, 3] = full_affine[:3, 3] + slice_index * full_affine[:3, 2]
    new_affine[:3, 2] = 0
    new_affine[2, 2] = 1
    return new_affine


def save_slice(slice_array: np.ndarray, output_nii_path: str, output_png_path: str, affine: np.ndarray) -> None:
    slice_3d = slice_array[..., np.newaxis]
    nib.save(nib.Nifti1Image(slice_3d, affine=affine), output_nii_path)

    lower, upper = HU_WINDOW
    slice_clipped = np.clip(slice_array, lower, upper)
    denom = max((upper - lower), 1e-6)
    norm_slice = ((slice_clipped - lower) / denom * 255.0).astype(np.uint8)
    Image.fromarray(norm_slice).save(output_png_path)


def save_segmentation_slice(seg_slice: np.ndarray, output_png_path: str) -> None:
    seg_bin = (seg_slice > 0).astype(np.uint8) * 255
    Image.fromarray(seg_bin).save(output_png_path)


def save_segmentation_slice_nii(seg_slice: np.ndarray, output_nii_path: str, affine: np.ndarray) -> None:
    seg_slice_bin = (seg_slice > 0).astype(np.uint8)
    seg_slice_3d = seg_slice_bin[..., np.newaxis]
    nib.save(nib.Nifti1Image(seg_slice_3d, affine=affine), output_nii_path)


def extract_slices(img_path: str, seg_path: str, case_id: str, seg_png_dir: Path, target_size: int) -> None:
    img = nib.load(img_path)
    data = img.get_fdata()

    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()

    num_slices = data.shape[-1]
    for slice_idx in range(num_slices):
        seg_slice = seg_data[:, :, slice_idx]
        if not np.any(seg_slice):
            continue

        ct_slice = data[:, :, slice_idx]
        ct_slice_processed = crop_pad_2d(ct_slice, target_size=(target_size, target_size))
        seg_slice_processed = crop_pad_2d(seg_slice, target_size=(target_size, target_size))
        slice_affine = get_slice_affine(img.affine, slice_idx)

        slice_base = f"6_slice_{slice_idx}"
        nii_path = str(SLICES_NIFTI_DIR / f"{case_id}_{slice_base}.nii.gz")
        png_path = str(SLICES_PNG_DIR / f"{case_id}_{slice_base}.png")
        save_slice(ct_slice_processed, nii_path, png_path, slice_affine)

        seg_png_path = str(seg_png_dir / f"{case_id}_{slice_base}.png")
        save_segmentation_slice(seg_slice_processed, seg_png_path)

        seg_nii_path = str(SLICES_SEG_NIFTI_DIR / f"{case_id}_{slice_base}.nii.gz")
        save_segmentation_slice_nii(seg_slice_processed, seg_nii_path, slice_affine)


def is_case_fully_processed(case_dir: Path) -> bool:
    required = [
        "baseline_ct_final.nii.gz",
        "baseline_IPH_seg_final.nii.gz",
        "baseline_IVH_seg_final.nii.gz",
        "followup_ct_final.nii.gz",
        "followup_IPH_seg_final.nii.gz",
        "followup_IVH_seg_final.nii.gz",
    ]

    paths = [case_dir / f for f in required]
    if not all(p.exists() for p in paths):
        return False

    shapes = [nib.load(str(p)).shape for p in paths]
    return all(s == shapes[0] for s in shapes[1:])


def process_case(ncct_followup_file: str, iph_followup_file: str, ivh_followup_file: str, use_gpu: bool) -> None:
    followup_id = get_file_stem(ncct_followup_file)
    baseline_id = followup_id.replace("follow-up", "baseline")

    case_dir = PATIENTS_OUT_DIR / followup_id
    case_dir.mkdir(parents=True, exist_ok=True)

    followup_raw_ct = str(case_dir / "0_raw_followup_ct.nii.gz")
    followup_raw_iph = str(case_dir / "0_raw_followup_IPH_seg.nii.gz")
    followup_raw_ivh = str(case_dir / "0_raw_followup_IVH_seg.nii.gz")

    baseline_raw_ct = str(case_dir / "0_raw_baseline_ct.nii.gz")
    baseline_raw_iph = str(case_dir / "0_raw_baseline_IPH_seg.nii.gz")
    baseline_raw_ivh = str(case_dir / "0_raw_baseline_IVH_seg.nii.gz")

    followup_windowed_ct = str(case_dir / "1_windowed_followup_ct.nii.gz")
    baseline_windowed_ct = str(case_dir / "1_windowed_baseline_ct.nii.gz")

    baseline_ss_ct = str(case_dir / "2_skullstripped_baseline_ct.nii.gz")
    baseline_ss_mask = str(case_dir / "2_skullstrip_mask_baseline.nii.gz")
    followup_ss_ct = str(case_dir / "2_skullstripped_followup_ct.nii.gz")
    followup_ss_mask = str(case_dir / "2_skullstrip_mask_followup.nii.gz")

    baseline_tpl_ct = str(case_dir / "3_template_registered_baseline_ct.nii.gz")
    followup_tpl_ct = str(case_dir / "3_template_registered_followup_ct.nii.gz")
    baseline_tpl_prefix = str(case_dir / f"{baseline_id}_template_baseline_")
    followup_tpl_prefix = str(case_dir / f"{followup_id}_template_followup_")
    baseline_tpl_mat = baseline_tpl_prefix + "0GenericAffine.mat"
    followup_tpl_mat = followup_tpl_prefix + "0GenericAffine.mat"

    followup_tpl_iph = str(case_dir / "3_template_registered_followup_IPH_seg.nii.gz")
    followup_tpl_ivh = str(case_dir / "3_template_registered_followup_IVH_seg.nii.gz")

    baseline_tpl_with_skull = str(case_dir / "4_template_registered_baseline_ct_with_skull.nii.gz")
    followup_tpl_with_skull = str(case_dir / "4_template_registered_followup_ct_with_skull.nii.gz")

    baseline_tpl_ss2 = str(case_dir / "4_baseline_template_registered_ct_ss2.nii.gz")
    baseline_tpl_ss2_mask = str(case_dir / "4_baseline_template_registered_ct_ss2_mask.nii.gz")
    followup_tpl_ss2 = str(case_dir / "4_followup_template_registered_ct_ss2.nii.gz")
    followup_tpl_ss2_mask = str(case_dir / "4_followup_template_registered_ct_ss2_mask.nii.gz")

    registered_ct = str(case_dir / "5_registered_followup_ct.nii.gz")
    registered_iph = str(case_dir / "5_registered_IPH_seg_followup_ct.nii.gz")
    registered_ivh = str(case_dir / "5_registered_IVH_seg_followup_ct.nii.gz")

    coreg_prefix = str(case_dir / f"{followup_id}_coreg_")
    coreg_mat = coreg_prefix + "0GenericAffine.mat"

    final_baseline_ct = str(case_dir / "baseline_ct_final.nii.gz")
    final_baseline_iph = str(case_dir / "baseline_IPH_seg_final.nii.gz")
    final_baseline_ivh = str(case_dir / "baseline_IVH_seg_final.nii.gz")

    final_followup_ct = str(case_dir / "followup_ct_final.nii.gz")
    final_followup_iph = str(case_dir / "followup_IPH_seg_final.nii.gz")
    final_followup_ivh = str(case_dir / "followup_IVH_seg_final.nii.gz")

    # Step 0: Copy raw files, select template, fix affines
    baseline_ct_src = str(NCCT_DIR / f"{baseline_id}.nii.gz")
    baseline_iph_src = str(IPH_SEG_DIR / f"{baseline_id}.nii.gz")
    baseline_ivh_src = str(IVH_SEG_DIR / f"{baseline_id}.nii.gz")

    shutil.copy2(baseline_ct_src, baseline_raw_ct)
    shutil.copy2(baseline_iph_src, baseline_raw_iph)
    shutil.copy2(baseline_ivh_src, baseline_raw_ivh)

    shutil.copy2(ncct_followup_file, followup_raw_ct)
    shutil.copy2(iph_followup_file, followup_raw_iph)
    shutil.copy2(ivh_followup_file, followup_raw_ivh)

    baseline_img = nib.load(baseline_raw_ct)
    followup_img = nib.load(followup_raw_ct)

    z_baseline = float(baseline_img.header.get_zooms()[2])
    z_followup = float(followup_img.header.get_zooms()[2])

    max_thick = math.ceil(max(z_baseline, z_followup))
    template_mm = int(min(max(max_thick, 1), 5))

    template_ct = str(TEMPLATE_DIR / f"template_{template_mm}mm.nii.gz")
    template_mask = str(TEMPLATE_MASK_DIR / f"template_{template_mm}mm.nii.gz")

    for p in [baseline_raw_iph, baseline_raw_ivh, followup_raw_ct, followup_raw_iph, followup_raw_ivh, baseline_raw_ct]:
        fix_affine_to_orthonormal(p, p)

    # Step 1: Windowing
    lower, upper = HU_WINDOW
    window_image(followup_raw_ct, followup_windowed_ct, lower=lower, upper=upper)
    window_image(baseline_raw_ct, baseline_windowed_ct, lower=lower, upper=upper)

    # Step 2: Skullstripping
    skullstrip_ncct(baseline_windowed_ct, baseline_ss_ct, baseline_ss_mask, use_gpu=use_gpu)
    skullstrip_ncct(followup_windowed_ct, followup_ss_ct, followup_ss_mask, use_gpu=use_gpu)

    custom_shaving_skullstripped_image(baseline_ss_ct, baseline_raw_iph, baseline_ss_ct, baseline_ss_ct)
    custom_shaving_skullstripped_image(followup_ss_ct, followup_raw_iph, followup_ss_ct, followup_ss_ct)

    # Step 3: Template Registration
    register_to_template(template_ct, baseline_ss_ct, template_mask, baseline_ss_mask, baseline_tpl_prefix, baseline_tpl_ct)
    register_to_template(template_ct, followup_ss_ct, template_mask, followup_ss_mask, followup_tpl_prefix, followup_tpl_ct)

    apply_transforms(baseline_raw_iph, baseline_tpl_ct, baseline_tpl_mat, final_baseline_iph, interpolation="NearestNeighbor")
    apply_transforms(baseline_raw_ivh, baseline_tpl_ct, baseline_tpl_mat, final_baseline_ivh, interpolation="NearestNeighbor")

    apply_transforms(followup_raw_iph, followup_tpl_ct, followup_tpl_mat, followup_tpl_iph, interpolation="NearestNeighbor")
    apply_transforms(followup_raw_ivh, followup_tpl_ct, followup_tpl_mat, followup_tpl_ivh, interpolation="NearestNeighbor")

    # Step 4: Second skullstrip iteration on registered CTs
    apply_transforms(baseline_windowed_ct, baseline_tpl_ct, baseline_tpl_mat, baseline_tpl_with_skull, interpolation="Linear")
    apply_transforms(followup_windowed_ct, baseline_tpl_ct, followup_tpl_mat, followup_tpl_with_skull, interpolation="Linear")

    skullstrip_ncct(baseline_tpl_with_skull, baseline_tpl_ss2, baseline_tpl_ss2_mask, use_gpu=use_gpu)
    skullstrip_ncct(followup_tpl_with_skull, followup_tpl_ss2, followup_tpl_ss2_mask, use_gpu=use_gpu)

    custom_shaving_skullstripped_image(baseline_tpl_ss2, final_baseline_iph, baseline_tpl_with_skull, baseline_tpl_ss2)
    custom_shaving_skullstripped_image(followup_tpl_ss2, followup_tpl_iph, followup_tpl_with_skull, followup_tpl_ss2)

    # Step 5: Co-Registration.
    run_affine_registration(baseline_tpl_ss2, followup_tpl_ss2, coreg_prefix, registered_ct)
    apply_transforms(followup_tpl_iph, registered_ct, coreg_mat, registered_iph, interpolation="NearestNeighbor")
    apply_transforms(followup_tpl_ivh, registered_ct, coreg_mat, registered_ivh, interpolation="NearestNeighbor")

    # Step 6: Final processing, windowing and center cropping
    shutil.copy2(registered_iph, final_followup_iph)
    shutil.copy2(registered_ivh, final_followup_ivh)

    window_image(baseline_tpl_ss2, final_baseline_ct, lower=lower, upper=upper)
    window_image(registered_ct, final_followup_ct, lower=lower, upper=upper)

    center_crop_multiple_niftis(
        [
            final_baseline_ct, final_baseline_iph, final_baseline_ivh,
            final_followup_ct, final_followup_iph, final_followup_ivh,
        ],
        crop_size=FINAL_CROP_SIZE,
    )

    extract_slices(final_baseline_ct, final_baseline_iph, baseline_id, SLICES_IPH_PNG_DIR, TARGET_SLICE_SIZE)
    extract_slices(final_baseline_ct, final_baseline_ivh, baseline_id, SLICES_IVH_PNG_DIR, TARGET_SLICE_SIZE)
    extract_slices(final_followup_ct, final_followup_iph, followup_id, SLICES_IPH_PNG_DIR, TARGET_SLICE_SIZE)
    extract_slices(final_followup_ct, final_followup_ivh, followup_id, SLICES_IVH_PNG_DIR, TARGET_SLICE_SIZE)


def main() -> None:
    if not NCCT_DIR.exists():
        raise RuntimeError(f"Missing directory: {NCCT_DIR}")
    if not IPH_SEG_DIR.exists():
        raise RuntimeError(f"Missing directory: {IPH_SEG_DIR}")
    if not IVH_SEG_DIR.exists():
        raise RuntimeError(f"Missing directory: {IVH_SEG_DIR}")
    if not TEMPLATE_DIR.exists():
        raise RuntimeError(f"Missing directory: {TEMPLATE_DIR}")
    if not TEMPLATE_MASK_DIR.exists():
        raise RuntimeError(f"Missing directory: {TEMPLATE_MASK_DIR}")

    ncct_files = [
        str(NCCT_DIR / f)
        for f in os.listdir(NCCT_DIR)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ]

    tasks = []
    for ncct_file in ncct_files:
        base_filename = os.path.basename(ncct_file)
        if "follow-up" not in base_filename:
            continue

        iph_file = str(IPH_SEG_DIR / base_filename)
        ivh_file = str(IVH_SEG_DIR / base_filename)

        if not os.path.exists(iph_file) or not os.path.exists(ivh_file):
            continue

        followup_id = get_file_stem(ncct_file)
        case_dir = PATIENTS_OUT_DIR / followup_id

        if is_case_fully_processed(case_dir):
            continue

        tasks.append((ncct_file, iph_file, ivh_file, USE_GPU_FOR_SYNTHSTRIP))

    if not tasks:
        return

    Parallel(n_jobs=NUM_THREADS)(delayed(process_case)(*t) for t in tasks)


if __name__ == "__main__":
    main()