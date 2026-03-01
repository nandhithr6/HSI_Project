# Hyperspectral Image (HSI) Segmentation for Placental Tissue

A deep learning project for segmenting hyperspectral images of human placentas using a dual-stream architecture that processes spatial and spectral information in parallel.

## Project Description

This project implements a state-of-the-art deep learning model (`HSIModel`) for accurate pixel-wise segmentation of hyperspectral placental images. The model identifies different tissue types including arteries, veins, stroma, umbilical cord, and specular reflections.

### Key Features

- **Dual-Stream Architecture**: Simultaneously processes spatial and spectral information
  - **Spatial Stream**: Combines local features (3D/2D convolutions) with global context (Mamba-based)
  - **Spectral Stream**: Multi-scale windowed Mamba architecture for spectral token generation
- **Advanced Fusion Mechanisms**:
  - Token Cross-Modal Enhancer (TCME) for cross-attention fusion
  - Hierarchical Cross-Modality Frequency Fusion (HCMFF) for frequency domain fusion
- **Robust Training**:
  - 5-fold stratified cross-validation
  - Mixed-precision training (AMP) for efficiency
  - Multi-GPU support with NCCL communication
  - Checkpointing of best models per fold

### Dataset

The project uses the **Hyperspectral Placenta Dataset** (Puustinen et al., 2023) with:

- **101 hyperspectral image cubes** from 4 fresh human placentas
- **37-38 spectral bands** spanning ~515–700 nm
- **1024 × 1024 spatial resolution** per band
- **Multiple imaging conditions**: baseline, food dyes (red/blue), and ICG fluorescent dye
- **Pixel-wise annotations** for multiple tissue classes

**Note**: Preprocessing details, dataset overview, and splitting strategy are documented in [docs/Placenta_preprocessing.md](docs/Placenta_preprocessing.md).

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- pip

### Setup Steps

1. Clone the repository:

```bash
git clone <repository-url>
cd HSI_Project
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

Key packages include:

- PyTorch (torch, torchvision, torchaudio)
- Scientific computing: numpy, scikit-learn, scikit-image, pandas
- Visualization: matplotlib
- Model components: einops, mamba-ssm
- Utilities: tqdm, thop, tabulate

## Usage

### Training

Run the training script with your dataset:

```bash
python src/training/train.py --data-dir /path/to/npz_dataset --epochs 100 --batch-size 4
```

**Required Arguments:**

- `--data-dir`: Path to directory containing `.npz` files with `image` and `mask` fields

**Optional Arguments:**

- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 4)
- `--lr`: Learning rate (default: 1e-4)

**Features:**

- Performs 5-fold stratified cross-validation automatically
- Saves best model checkpoint for each fold
- Logs training metrics to CSV
- Optional TensorBoard integration
- Deterministic training with fixed random seed
- Multi-GPU support with environment variable optimization

### Data Format

The training script expects data in NumPy `.npz` files:

```python
# Each .npz file should contain:
{
    'image': array of shape (C, H, W),  # C = spectral bands, H, W = spatial dims
    'mask': array of shape (H, W)        # pixel-wise class labels
}
```

### Model Architecture

Key modules:

- **Spatial Stream**: `src/models/spatial_stream/` (local + global feature extraction)
- **Spectral Stream**: `src/models/spectral_stream/` (spectral token generation)
- **Cross-Modal Fusion**: `src/models/TCME/` and `src/models/HCMFF/`
- **Decoder**: `src/models/decoder/` (segmentation head)
- **Main Model**: `src/models/full_model.py` (HSIModel class)

### Evaluation

Use the evaluation script to test trained models:

```bash
python tests/evaluate.py --model /path/to/checkpoint --data-dir /path/to/test_data
```

## Project Structure

```
HSI_Project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── docs/                        # Documentation
│   ├── architecture.md          # Model architecture details
│   ├── Placenta_preprocessing.md # Dataset preprocessing guide
│   └── *.pdf                    # Component design documents
├── src/
│   ├── models/
│   │   ├── full_model.py        # Main HSIModel class
│   │   ├── spatial_stream/      # Spatial processing components
│   │   ├── spectral_stream/     # Spectral processing components
│   │   ├── TCME/                # Token Cross-Modal Enhancer
│   │   ├── HCMFF/               # Hierarchical Cross-Modality Frequency Fusion
│   │   └── decoder/             # Segmentation decoder
│   ├── training/
│   │   └── train.py             # Main training script
│   ├── data/                    # Data loading utilities
│   └── utils/                   # Helper utilities
├── tests/
│   └── evaluate.py              # Evaluation script
├── DATA/
│   └── Placenta/                # Dataset directory
│       ├── preprocessed_cubes/  # Preprocessed HSI cubes
│       └── p**/                 # Raw data by patient group
└── scripts/                      # Additional utility scripts
```

## Contributors

The project is being developed by a collaborative team:

- Nandhitha
- Abiram M. Sree
- Siddhi Mohanty
- And other contributors

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

**MIT License Summary**: You are free to use, modify, and distribute this software for any purpose, provided that you include the license and copyright notice.

## Citation

If you use this project in your research, please cite:

- **Dataset**: Puustinen et al., 2023 - Hyperspectral Placenta Dataset
- **Architecture**: Refer to the model architecture documentation in [docs/architecture.md](docs/architecture.md)

## References

- Original dataset publication: Puustinen et al., 2023
- Mamba architecture: Related state-space models literature
- For preprocessing details: See [docs/Placenta_preprocessing.md](docs/Placenta_preprocessing.md)

## Getting Help

For issues, questions, or contributions:

1. Check existing documentation in `docs/` folder
2. Review code comments in `src/models/full_model.py` for architecture details
3. Consult `docs/Placenta_preprocessing.md` for dataset-related questions
4. Create an issue on the repository if you encounter problems

---

**Last Updated**: March 2026
