# DogFit: Domain-Guided Fine-Tuning for Diffusion Models (AAAI 2026)
```
Bahram, Yara, Mohammadhadi Shateri, and Eric Granger. "DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models." arXiv preprint arXiv:2508.05685 (2025).
```

This repository contains the official implementation of **DogFit**, an efficient domain-guided fine-tuning method for transfer learning of diffusion models.

We demonstrate our method using **SiT-XL/2** and **DiT-XL/2** on the **Food-101** dataset and provide support for evaluating key metrics such as **FID**, **FD_DINOV2**, **Precision**, and **Recall**.


## Setup

We recommend using `conda` for environment management.  
To use a pure Python environment, modify the `create_environment()` function in `scripts/config.sh`.

Change the following paths in `scripts/config.sh`:

```
CODE_PRE_DIR="/path/to/main/directory"
DATASETS_DIR="$CODE_PRE_DIR/datasets" # "/path/to/datasets/" # Can keep this the way it is
RESULTS_PRE_DIR="$CODE_PRE_DIR/results" # "/path/to/results/directory" # Can keep this the way it is
ENV_PATH="path/to/python/environment" # Where you want the environment to be created 
```

### One-Line Execution

This script automates the entire pipeline on Food-101, applying DogFit on DiT with Control:

- Creates a conda environment and downloads packages
- Downloads and preprocesses the Food-101 dataset
- Fine-tunes a DiT model pre-trained on ImageNet with DogFit+Control and saves checkpoints
- Generates 10,000 samples for a variety of guidance values using the fine-tuned model
- Evaluates results and logs them in a `.log` file

> Example script provided in: `scripts/DogFit_DiT_SiT_noControl.sh`

```bash
bash scripts/run_DogFit.sh \
    --dataset "food-101_processed" \
    --server "bool" \
    --cuda_devices "0,1" \
    --experiment_prename "DiT-XL_FD_DINOV2_control/" \
    --latestart "12000" \
    --mghigh "1" \
    --model_name "DiT-XL/2" \
    --guidance_control "1" \
    --sample_guidance "0" \
    --control_distribution "95in1to2"
```

Choices for model: DiT-XL/2  SiT-XL/2

## 📊 Results on Food-101 (DiT with Control)

| w   | FD_DINOV2 ↓ | Precision ↑ | Recall ↑ | Density ↑ | Coverage ↑ | FID ↓     | Precision ↑ | Recall ↑ | Density ↑ | Coverage ↑ |
|-----|-------------|-------------|----------|-----------|------------|-----------|-------------|----------|-----------|------------|
| 1.0 | 459.59      | 0.4735      | 0.6207   | 0.206     | 0.2555     | 12.98     | 0.8264      | 0.5198   | 1.3725    | 0.9291     |
| 1.5 | 302.84      | 0.5860      | 0.6350   | 0.3224    | 0.3983     | 10.94     | 0.8802      | 0.4878   | 1.6792    | 0.9586     |
| 2.0 | 228.32      | 0.6655      | 0.6209   | 0.4398    | 0.5011     | 13.05     | 0.9008      | 0.4377   | 1.7896    | 0.9524     |
| 3.0 | 199.91      | 0.7441      | 0.5445   | 0.5752    | 0.5631     | 19.81     | 0.8971      | 0.3433   | 1.6148    | 0.8862     |
| 4.0 | 219.13      | 0.7500      | 0.4904   | 0.5884    | 0.5506     | 24.84     | 0.8789      | 0.2845   | 1.4062    | 0.8240     |
| 5.0 | 238.87      | 0.7450      | 0.4632   | 0.5729    | 0.5330     | 27.95     | 0.8625      | 0.2594   | 1.2588    | 0.7814     |


## Guidance without Control

This script applies DogFit on DiT without Control with a focus on optimizing FD_DINOV2:

```bash
bash scripts/run_DogFit.sh \
    --dataset "food-101_processed" \
    --server "bool" \
    --cuda_devices "0,1" \
    --experiment_prename "DiT-XL_FD_DINOV2_control/" \
    --latestart "12000" \
    --mghigh "1" \
    --model_name "DiT-XL/2" \
```

Choices for model: DiT-XL/2  SiT-XL/2

## Baselines

> To run the baselines, refer to the example scripts provided in: `scripts/Baselines_DiT_SiT_noControl.sh`


## Repository Structure

This is the repository structure. 
We further provide a code for running the baselines, normal fine-tuning, CFG, DoG, and MG. 

```
.
├── scripts/                       # Main execution scripts
│   ├── config.sh                  # Global configurations
│   ├── DogFit_DiT_SiT_control.sh  # Pipeline script: DogFit with guidance control
│   ├── DogFit_DiT_SiT_nocontrol.sh# Pipeline script: DogFit without control
│   ├── Baselines_DiT_SiT.sh       # Pipeline script: Baseline comparison script
│   ├── run_baseline_MG.sh         # Run MG
│   ├── run_baselines_finetune.sh  # Run Fine-tune, CFG, DoG
│   └── run_DogFit.sh              # Run DogFit
├── models/                        # DiT and SiT model architectures
├── datasets/                      # Target domain datasets (e.g., Food-101)
├── dgm-eval/                      # Evaluation metrics (from https://github.com/layer6ai-labs/dgm-eval)
├── diffusion/                     # Diffusion code for DiT
├── transport/                     # Diffusion code for SiT
├── train.py                       # Training for fine-tune, CFG, DoG
├── train_MG.py                    # Training for MG
├── train_DogFit.py                # Training for DogFit 
├── sample.py                      # Sampling for fine-tune, CFG, DoG, DogFit
└── sample_DoG.py                  # Sampling for DoG
```

### Cite
```
@article{bahram2025dogfit,
  title={DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models},
  author={Bahram, Yara and Shateri, Mohammadhadi and Granger, Eric},
  journal={arXiv preprint arXiv:2508.05685},
  year={2025}
}
```
