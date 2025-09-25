# HSI Project Copilot Instructions

This document provides guidance for AI coding agents working on the Hyperspectral Image (HSI) segmentation project.

## Project Overview

This project implements a deep learning model for segmenting hyperspectral images, specifically focusing on medical imaging of placentas. The goal is to accurately identify different tissue types (arteries, veins, stroma, etc.) from HSI cubes.

The core of the project is a dual-stream architecture that processes spatial and spectral information in parallel before fusing them for final segmentation.

## Architecture

The main model is defined in `src/models/full_model.py` as `HSIModel`. It follows this high-level data flow:

1.  **Input**: An HSI cube of shape `(B, C, H, W)`, where `C` is the number of spectral bands.

2.  **Dual Streams**:
    *   **Spatial Stream**: Processes the spatial features of the HSI cube. It consists of:
        *   `LocalFeatureStream`: Extracts local features using 3D and 2D convolutions.
        *   `GlobalFeatureStream`: Captures global context using a Mamba-based architecture.
        *   `SpatialFusionModule`: Fuses local and global spatial features.
        *   `SpatialTokenizer`: Converts the fused feature map into a sequence of tokens.
    *   **Spectral Stream (`SpectralStream`)**: Processes the spectral information for each pixel independently. It uses a multi-scale windowed Mamba architecture to generate spectral tokens.

3.  **Cross-Modal Fusion**:
    *   **`TokenCrossModalEnhancer` (TCME)**: Fuses the spatial and spectral tokens using a cross-attention mechanism.
    *   **`HierarchicalCrossModalityFrequencyFusion` (HCMFF)**: Further fuses the features in the frequency domain.

4.  **Decoder (`MSTDHSHDecoder`)**: Takes the fused tokens and upsamples them to produce the final segmentation mask.

## Data

-   **Dataset**: The project uses the Hyperspectral Placenta Dataset. Preprocessing details are documented in `docs/Placenta_preprocessing.md`.
-   **Data Format**: The training script expects data to be in `.npz` files, each containing an `image` and a `mask`.
-   **Data Loading**: The `NPZDataset` class in `src/training/train.py` handles loading the data.
-   **Data Splitting**: The training script uses 5-fold stratified cross-validation to ensure a balanced distribution of data across folds.

## Development Workflow

### Training

The main training script is `src/training/train.py`. To run a training session, use a command like this:

```bash
python src/training/train.py --data-dir /path/to/your/npz_dataset --epochs 100 --batch-size 4
```

-   `--data-dir`: **Required**. Path to the directory containing the `.npz` dataset.
-   `--epochs`: Number of training epochs.
-   `--batch-size`: Batch size.
-   `--lr`: Learning rate.

The script handles:
-   5-fold cross-validation.
-   Mixed-precision training (AMP).
-   Logging to CSV and optional TensorBoard.
-   Checkpointing the best model for each fold.

### Environment and Reproducibility

The training script sets several environment variables and PyTorch options for stability and reproducibility, especially in a multi-GPU environment. Key settings include:
-   `NCCL` environment variables for reliable multi-GPU communication.
-   `torch.backends.cudnn.deterministic = True` for reproducible results.
-   A fixed random seed.

## Key Files and Directories

-   `src/models/full_model.py`: Contains the main `HSIModel` class, integrating all components.
-   `src/training/train.py`: The primary script for training and evaluation.
-   `src/models/spatial_stream/`: Contains the components of the spatial processing stream.
-   `src/models/spectral_stream/`: Contains the spectral stream implementation.
-   `src/models/TCME/`, `src/models/HCMFF/`, `src/models/decoder/`: Contain the fusion and decoder modules.
-   `docs/Placenta_preprocessing.md`: Essential reading for understanding the dataset and preprocessing steps.
-   `DATA/Placenta/preprocessed_visualisations/`: Contains visualizations of the preprocessed data.
