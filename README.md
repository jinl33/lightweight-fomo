# FOMO-60k Brain Diffusion Pipeline (Test Implementation)

This is a test implementation combining the [FOMO-60k dataset](https://github.com/fomo25/baseline-codebase) preprocessing pipeline with the [lightweight brain diffusion model](https://github.com/wilmsm/lightweightbraindiff). It's configured for single-subject testing (sub_1) to demonstrate:

- Basic preprocessing of FOMO-60k brain MRI data
- Wavelet transformation for data preprocessing
- Training of the lightweight diffusion model

Note: This is a minimal implementation intended for testing and learning purposes. For full dataset processing, please refer to the original repositories.

## Repository Structure
```
.
├── scripts/                    # Standalone scripts for data processing
│   ├── download_one_subject.py # Downloads single subject from FOMO-60k
│   ├── preprocess_sub1.py     # FOMO preprocessing script
│   └── wavelet_preprocess.py  # Wavelet transformation script
├── src/                       # Source code
│   ├── train_diffusion.py    # Main training script
│   └── utils/                # Utility functions
│       └── simple_utils.py   # Helper utilities
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/jinl33/lightweight-fomo.git
cd lightweight-fomo
```

2. Create and activate Python virtual environment (Python 3.11 recommended):
```bash
python3.11 -m venv fomo_env
source fomo_env/bin/activate  # On Windows use: fomo_env\Scripts\activate
pip install --upgrade pip setuptools wheel
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Clone required external repositories:
```bash
# Clone in a separate directory outside this repository
cd ..
git clone https://github.com/fomo25/baseline-codebase.git
git clone https://github.com/wilmsm/lightweightbraindiff.git
cd baseline-codebase
pip install -e ".[dev,test]"
cd ../lightweight-fomo  # Return to main repository
```

## Data Download and Processing Pipeline

1. Download single subject (sub_1) data:
```bash
python download_one_subject.py
```

2. Run FOMO preprocessing:
```bash
python preprocess_sub1.py \
    --in_path=data/fomo-60k/sub_1 \
    --out_path=data/preprocessed/fomo_60k/sub_1
```

3. Run wavelet preprocessing:
```bash
python wavelet_preprocess.py \
    --input_dir=data/preprocessed/fomo_60k/sub_1 \
    --output_dir=data/wavelet/fomo_60k/sub_1 \
    --levels=2
```

4. Train diffusion model:
```bash
cd lightweightbraindiff
python train_diffusion.py \
    --data_dir ../data/wavelet/fomo_60k/sub_1 \
    --device cpu \
    --batch_size_train 1 \
    --num_workers 1 \
    --train_epochs 5 \
    --num_train_timesteps 100 \
    --save_path ../models/diffusion_model.pth
```

## Directory Structure
```
.
├── baseline-codebase/
├── lightweightbraindiff/
├── data/
│   ├── fomo-60k/
│   ├── preprocessed/
│   └── wavelet/
├── models/
├── download_one_subject.py
├── preprocess_sub1.py
├── wavelet_preprocess.py
└── requirements.txt
```

## GPU Support

To use GPU, simply change the `--device` parameter to `cuda` in the training command if a CUDA-compatible GPU is available.

## Notes

- The pipeline is designed to work with a single subject (sub_1) from the FOMO-60k dataset
- All preprocessing steps create their output in separate directories to maintain the original data
- The model checkpoint will be saved in the `models` directory
