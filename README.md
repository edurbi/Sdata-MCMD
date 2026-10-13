# Scientific Data - MCMD


This repository provides an end-to-end pipeline for medical image preprocessing, multimodal classification (PET/CT), clinical metadata fusion, evaluation, and statistical plotting.

---

## Table of Contents
1. [Pipeline Overview](#pipeline-overview)
2. [Prerequisites & Installation](#prerequisites--installation)
3. [Step 1: Data Preprocessing to `.npy` (`process.py`)](#step-1-data-preprocessing-to-npy-processpy)
4. [Dataset & Directory Structure](#dataset--directory-structure)
5. [Step 2: Training Pipelines](#step-2-training-pipelines)
   - [Pipeline A: `train_supervised.py` (Transformer / LLM / MLP Fusion)](#pipeline-a-train_supervisedpy)
   - [Pipeline B: `train_resnet_n.py` (CNN / ResNet & ViT Fusion)](#pipeline-b-train_resnet_npy)
6. [Step 3: Evaluation & Testing (`Testing_resnet.py`)](#step-3-evaluation--testing-testing_resnetpy)
7. [Step 4: Metric Plotting & Analysis (`utils/print_results.py`)](#step-4-metric-plotting--analysis-utilsprint_resultspy)
8. [Command-Line Arguments Reference](#command-line-arguments-reference)

---

## Pipeline Overview

The complete end-to-end workflow consists of:
```
Raw Scans (CT.nii.gz, PET.nii.gz)
            │
            ▼  (Run process.py)
Preprocessed Volumes (CT_Processed.npy, PET_Processed.npy)
            │
            ▼  (Indexed via train/val/test CSVs)
Model Training (train_supervised.py OR train_resnet_n.py)
            │
            ▼  (Outputs checkpoints and logging.txt)
Evaluation (Testing_resnet.py) & Plotting (utils/print_results.py)
```

---

## Prerequisites & Installation

### Requirements
- **Python**: 3.9+
- **Hardware**: NVIDIA GPU with CUDA drivers (the training pipelines utilize CUDA and `bfloat16`/`half` precision).

### Dependencies
Install the required packages:
```bash
pip install torch torchvision torcheval torchio monai transformers peft bitsandbytes scikit-learn scikit-plot pandas numpy matplotlib seaborn nibabel joblib
```

---

## Step 1: Data Preprocessing to `.npy` (`process.py`)

> [!IMPORTANT]
> Raw medical image volumes (NIfTI `.nii.gz`) **must be preprocessed into `.npy` files before training**. The dataset loaders ([`Supervised_Dataset.py`](file:///Users/edurb/Downloads/PETCT_code_only/PETCT/Code/Supervised_Dataset.py) and [`fold_dataset.py`](file:///Users/edurb/Downloads/PETCT_code_only/PETCT/Code/fold_dataset.py)) specifically load `.npy` or `.npz` arrays.

### What `process.py` does:
1. Iterates over all patient subfolders in your raw image folder (`subjects_path`).
2. Reads `CT.nii.gz` and `PET.nii.gz` using `nibabel`.
3. Standardizes slice depth (truncating if slices exceed 1000).
4. Applies TorchIO spatial transforms and intensity normalization:
   - **CT**:
     - `tio.CropOrPad((512, 512, 512))`
     - `tio.Resize((256, 256, 256))`
     - `tio.ZNormalization()`
   - **PET**:
     - `tio.CropOrPad((360, 256, 256))`
     - `tio.Resize((256, 256, 256))`
     - `tio.ZNormalization()`
5. Exports preprocessed arrays directly into each subject directory as:
   - `CT_Processed.npy`
   - `PET_Processed.npy`
6. Utilizes `joblib.Parallel` for multi-threaded batch processing.

### How to Run:
Ensure `subjects_path` in `process.py` points to your folder of raw patient scans:
```python
subjects_path = "/path/to/raw/ImagesTr/"
```
Then execute:
```bash
python process.py
```

---

## Dataset & Directory Structure

Once preprocessing has finished, organize your directories as follows:

```
/path/to/datapath/                     <-- Pass to --datapath
├── train_final.csv                   (or train_miss.csv if --missing True)
├── val_final.csv                     (or val_miss.csv if --missing True)
└── test_final.csv                    (or test_miss.csv if --missing True)

/path/to/modality_files/               <-- Pass to --data_prefix (defaults to /data/edu/)
├── patient_001/
│   ├── CT_Processed.npy
│   └── PET_Processed.npy
├── patient_002/
│   ├── CT_Processed.npy
│   └── PET_Processed.npy
└── ...
```

### CSV Annotation Format
Each row corresponds to one subject:
- `Mod0`, `Mod1`, ...: Path to the generated `.npy` file relative to `--data_prefix`.
- `Dim`: Modality dimensionality string, e.g. `"33"` (both modalities are 3D).
- `Type`: Dataset or cohort identifier (`0`, `1`, etc.).
- `Result`: Binary or categorical target label (`0`, `1`, etc.).
- `Meta0`, `Meta1`, ...: Clinical/tabular metadata variables.

---

## Step 2: Training Pipelines

The framework provides two training pipelines:

### Pipeline A: `train_supervised.py`
This script trains the multimodal architecture in [`model_semi.py`](file:///Users/edurb/Downloads/PETCT_code_only/PETCT/Code/model_semi.py). It encodes 2D and 3D modalities using MONAI Vision Transformers (`ViT`), projects features through an unfolding adapter, merges clinical metadata, and processes them through a sequence/language backbone.

#### Supported Backends (`--mode`):
- `MLP`: Multi-Layer Perceptron projection head.
- `Trans`: Multi-head Transformer encoder block.
- `PreLF`: Pretrained Longformer classification model.
- `Roberta`: RoBERTa model with LoRA fine-tuning (`DBR`).
- `DBLora`: LLaMA-based sequence classification with LoRA.
- `Med`: Medical domain language model backend.

#### Example Commands:

**1. 5-Fold Stratified Cross-Validation:**
```bash
python train_supervised.py \
  --datapath /path/to/csv_folder \
  --data_prefix /path/to/modality_files/ \
  --savepath /path/to/checkpoints/ \
  --mode Trans \
  --gpu 0 \
  --kfold 0 \
  --batch_size 16 \
  --num_epochs 100 \
  --lr 3e-4 \
  --meta 5
```

**2. Standard Single Train/Val Split:**
```bash
python train_supervised.py \
  --datapath /path/to/csv_folder \
  --data_prefix /path/to/modality_files/ \
  --savepath /path/to/checkpoints/ \
  --mode MLP \
  --gpu 0 \
  --batch_size 32 \
  --num_epochs 50 \
  --lr 3e-4 \
  --meta 5
```

**3. With Longformer Backbone:**
```bash
python train_supervised.py \
  --datapath /path/to/csv_folder \
  --data_prefix /path/to/modality_files/ \
  --savepath /path/to/checkpoints/ \
  --model_path /path/to/Longformer_weights/ \
  --mode PreLF \
  --gpu 0 \
  --batch_size 8 \
  --meta 5
```

---

### Pipeline B: `train_resnet_n.py`
This script trains a CNN/ViT baseline architecture combining ResNet (torchvision ResNet-18/101 and MONAI 3D ResNet-18) with Vision Transformers.

#### Supported Modes (`--mode`):
- `Resnet`: Combines 2D ResNet-18 and 3D ResNet-18 image encoders.
- `Vit`: Vision Transformer backbones for both 2D and 3D inputs.
- `Trans`: Hybrid Transformer fusion.
- `Multi`: Iteratively trains over multiple subsets/modalities.

#### Example Command:
```bash
python train_resnet_n.py \
  --datapath /path/to/csv_folder \
  --data_prefix /path/to/modality_files/ \
  --savepath /path/to/checkpoints/ \
  --mode Resnet \
  --gpu 0 \
  --kfold 0 \
  --batch_size 8 \
  --num_epochs 50 \
  --lr 5e-3 \
  --meta 5
```

---

## Step 3: Evaluation & Testing (`Testing_resnet.py`)

To evaluate saved checkpoints on test splits:

```bash
python Testing_resnet.py \
  --datapath /path/to/csv_folder \
  --data_prefix /path/to/modality_files/ \
  --savepath /path/to/checkpoints/ \
  --mode Resnet \
  --gpu 0 \
  --n_cross 5 \
  --meta 5
```

---

## Step 4: Metric Plotting & Analysis (`utils/print_results.py`)

Parses `logging.txt` outputs across training rounds, calculates accuracy, AUC, precision, recall, and F1 score, and plots train vs. validation trajectories:

```bash
python utils/print_results.py \
  --input /path/to/checkpoints/logging.txt \
  --n 5 \
  --save run_summary \
  --out_dir /path/to/output_plots/
```

- `--input`: Path to the generated `logging.txt`.
- `--n`: Number of folds/runs to aggregate.
- `--save`: Subfolder name for generated plots.
- `--out_dir`: Target directory for saved visualizations.

---

## Command-Line Arguments Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--datapath` | `str` | `None` | Path to the directory containing annotation CSVs. |
| `--data_prefix` | `str` | `"/data/edu/"` | Prefix path prepended to `.npy` files specified in CSVs. |
| `--savepath` | `str` | `None` | Directory where logs and model weights will be saved. |
| `--gpu` | `str` | `"0"` / `"1"` | CUDA GPU ID (configures `CUDA_VISIBLE_DEVICES`). |
| `--mode` | `str` | `"MLP"` / `"Resnet"` | Model architecture / backbone to instantiate. |
| `--batch_size` | `int` | `50` / `8` | Batch size for training and evaluation. |
| `--lr` | `float` | `3e-4` / `5e-3` | Initial learning rate. |
| `--weight_decay` | `float` | `0.01` | Adam / AdamW weight decay. |
| `--num_epochs` | `int` | `100` / `50` | Number of epochs to train. |
| `--seed` | `int` | `1024` | Random seed for data splits and weights initialization. |
| `--kfold` | `int` | `-1` | Cross-validation fold index (`0` runs all 5 folds; `-1` runs single split). |
| `--meta` | `int` | `0` | Number of clinical metadata features to concatenate. |
| `--missing` | `bool` | `False` | Whether to load missing modality CSV splits. |
| `--model_path` | `str` | `"/data/edu/PETCT/models/Longformer"` | (*`train_supervised.py` only*) Path to pretrained LLM/Longformer weights. |
| `--resume` | `bool` | `False` | Resume training from existing checkpoints in `--savepath`. |
