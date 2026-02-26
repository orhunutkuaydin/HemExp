# HemExp:  Clinically-Guided Latent Diffusion for Hematoma Expansion Modeling

This repository provides a conditional diffusion model that synthesizes **follow-up non-contrast CT** (and corresponding **IPH/IVH masks**) from baseline imaging. The model can be conditioned on **clinical variables** using classifier free guidance and a binary **expansion flag** indicating the expansion status.

**Paper** : submitted to MICCAI 2026

### Model weights
- **Autoencoder (VAE)**: *(will be released after preprint/journal publication)*
- **Diffusion model**: *(will be released after preprint/journal publication)*
## Followup generation diffusion example

![diffusion_steps_baseline_mid_real_followup.gif](visualization/diffusion_steps_baseline_mid_real_followup.gif)

## Sampling
See: **[`./sampling.py`](sampling.py)**
-  Set paths in your config [json](CONFIGS/config_25D_3context.json)
### Clinical and imaging data
Clinical data is an optional input but highly recommended See [Conditioning options](#conditioning-options) for details.
Voxel z spacing is used to select input slices for the 2.5 D models.

---
## Conditioning options

### Clinical variables (CFG)
This model was trained **with clinical conditioning enabled**. At inference time you can control the impact using classifier-free guidance (CFG):

- **Disable clinical conditioning:** set the clinical CFG scale to `0.0` this will replace the clinical input with a learned null representation.
- **Default behavior:** CFG scale `1.0`
- **Stronger clinical effect:** CFG scale `> 1.0`

The clinical conditioning vector contains the following variables:

| # | Variable | Meaning | Allowed raw range |
|---:|---|---|---|
| 1 | `time_from_onset_to_CT` | Hours from symptom onset to baseline CT | 0–24 h| 
| 2 | `IPH_baseline_ml` | Baseline IPH volume in mL | 0–250 mL  |
| 3 | `Anticoagulant` | Anticoagulant use (binary) | 0 or 1 |
| 4 | `Antiplatelet` | Antiplatelet use (binary) | 0 or 1 
| 5 | `Age` | Age in years | 0–100 years |
| 6 | `GCS` | Glasgow Coma Scale | 3–15 |
| 7 | `Systolic` | Systolic blood pressure in mmHg | 0–250 mmHg |

### Expansion flag 
Training uses an **oracle expansion flag** (`0` stable / `1` expander). 

For inference you have two options:
- **Expansion status available:** pass `expansion_flag ∈ {0,1}` directly, this can be defined by the user for simulation. 

- **Expansion status  not available:** use any (ideally calibrated) binary hematoma expansion classifiers probabilities `p(expansion=1)` as input. Optionally thresholded to simulate a decision. 

---

## Intraparenchymal and intraventricular hemorrhage segmentations
The variational autoencoder needs IPH and IVH segmentations as 2 additional channels.  

Ideally the segmentations need to be manually verified by experts.   

---

## Installation
```bash
# create env
conda env create -f environment.yml
conda activate hemexp_demo
```


## Preprocessing
See: **[`./preprocess.py`](./preprocess.py)**

Expected output format:

```
OUTPUT_DATA_ROOT/
└── slices/
    ├── HospitalA/
    │   └── HospitalA_<patient_id>/         
    │       ├── baseline/
    │       │   ├── ct/
    │       │   │   ├── slice_001.nii.gz
    │       │   │   ├── slice_002.nii.gz
    │       │   │   └── ...
    │       │   ├── iph/
    │       │   │   ├── slice_001.nii.gz
    │       │   │   ├── slice_002.nii.gz
    │       │   │   └── ...
    │       │   └── ivh/
    │       │       ├── slice_001.nii.gz
    │       │       ├── slice_002.nii.gz
    │       │       └── ...
    │       └── followup/
    │           ├── ct/
    │           │   ├── slice_001.nii.gz
    │           │   ├── slice_002.nii.gz
    │           │   └── ...
    │           ├── iph/
    │           │   ├── slice_001.nii.gz
    │           │   ├── slice_002.nii.gz
    │           │   └── ...
    │           └── ivh/
    │               ├── slice_001.nii.gz
    │               ├── slice_002.nii.gz
    │               └── ...
    │
    ├── HospitalB/
    │   └── HospitalB_<patient_id>/
```

## Training
See: **[`./train_diffusion.py`](./train_diffusion.py)**
```
python train_diffusion.py ./CONFIGS/config_25D_3context.json
```

