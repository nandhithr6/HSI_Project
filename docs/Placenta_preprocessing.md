# Hyperspectral Placenta Dataset: Preprocessing and Data Splitting Report

## Dataset Overview

The Hyperspectral Placenta Dataset (Puustinen et al., 2023) comprises:

- **101 hyperspectral images** acquired from four fresh human placentas  
- **Spectral range**: 515–900 nm (visible to near-infrared)  
- **Spatial resolution**: 1 024 × 1 024 pixels per band  
- **Annotations**: Pixel-wise segmentation masks delineating arteries, veins, stroma and umbilical cord  

Imaging conditions by group:
1. **Patient 1** (6 images): No dye (baseline tissue)  
2. **Patient 2** (24 images): Red/blue food dye (enhanced vessel contrast)  
3. **Patient 3** (23 images): Red/blue food dye (enhanced vessel contrast)  
4. **Patient 4** (48 images): ICG fluorescent dye (near-infrared fluorescence)

---

## Rationale for Preprocessing

Hyperspectral raw data contain several systematic distortions:

1. **Sensor Noise & Dark Current**  
   - Electronic readout noise and thermal charge accumulate in each pixel even with no illumination.  
2. **Illumination Non-Uniformity**  
   - Vignetting, spatial variation in light source intensity and optical path irregularities create shading artifacts.  
3. **Spectral Response Variability**  
   - Detector sensitivity and optical components differ across wavelengths; raw values lack physical units.  
4. **Dye-Induced Spectral Shifts**  
   - Contrast agents (food dyes, ICG) introduce absorption/emission peaks that must be isolated from instrument artifacts.

Without correction, machine learning models will learn instrument-specific patterns rather than true tissue reflectance, leading to poor generalization and unreliable biological interpretation.

---

## Preprocessing Pipeline (Following Original Authors)

The published methodology specifies the following steps. Each step is applied exactly as defined in the dataset’s reference materials:

1. **Flat-Field and Dark-Current Correction**  
   - **Formula**
      Corrected image = (Raw image − Dark reference) / (White reference − Dark reference),
  
      where 
      Raw image → what the sensor actually captured
      Dark reference (D) → sensor’s own noise (measured with no light)
      White reference (W) → image of a uniform bright surface
 
   - **Purpose**  
     - Removes pixel-level electronic noise  
     - Compensates for spatial illumination non-uniformity  
     - Yields relative reflectance free of shading artifacts  

2. **Absolute Reflectance Calibration**  
   - **Data Source**  
     - `white reference reflectance.txt`: Spectrophotometer-measured reflectance values for the white standard at each wavelength.  
   - **Procedure**  
     1. Interpolate the measured reflectance curve to the camera’s spectral band centers.  
     2. Multiply each band of the flat-field–corrected cube by its corresponding interpolated reflectance value.  
   - **Outcome**  
     - Converts relative units to true reflectance (0–1), enabling quantitative comparison across datasets and literature.  

3. **Band Selection, Clipping & Normalization**  
   - **Band Range**  
     - Retain only 515–700 nm as per experimental focus and dataset definitions.  
   - **Clipping**  
     - Apply [0,1] clamp to eliminate outliers and enforce physical reflectance bounds.  
   - **Normalization**  
     - Scale pixel values linearly to [0,1] if any custom clamp range is specified in `definitions.json`.  
   - **Reason**  
     - Standardizes input range for neural networks and removes residual extreme values.  

4. **Segmentation Mask Handling**  
   - **Provided Masks**  
     - Pre-rasterized TIFF masks aligned with hyperspectral cubes.  
   - **Usage**  
     - Employed directly as ground truth for pixel-wise classification.  
   - **Justification**  
     - Avoids any interpolation or misalignment; retains expert annotations exactly.

---

## Data Splitting Strategies

Effective model evaluation requires balancing spectral diversity, patient variability, and preventing data leakage. The following approaches were evaluated:

### 1. Patient-Level or Dye-Level Split

- **Description**  
  - Assign entire patient groups or dye conditions exclusively to train, validation or test.  
- **Limitations**  
  - **Imbalanced samples**: Patient 1’s 6 images vs. Patient 4’s 48 images.  
  - **Domain shift**: Test set may contain dyes (e.g., ICG) unseen during training.  
  - **Unrealistic evaluation**: In practice, models must handle mixtures of protocols.  

### 2. Random Image-Level Split

- **Description**  
  - Random 70/15/15% assignment of images, ignoring patient or dye labels.  
- **Limitations**  
  - **Data leakage risk**: Adjacently acquired or spatially similar regions may appear in both train and test.  
  - **Uncontrolled dye distribution**: No guarantee of balanced dye representation.  

### 3. Stratified Split Across All Conditions  **← Selected**

- **Principle**  
  - Each split (train/validation/test) must contain proportional representation from all four patient groups and all three dye conditions.  
- **Implementation**  
  - **Training**: 70% (~71 images)  
  - **Validation**: 15% (~15 images)  
  - **Test**: 15% (~15 images)  
  - **Stratification targets**: ~6% no dye, ~46% red/blue dye, ~48% ICG in each split.  
- **Advantages**  
  1. **Spectral Robustness**: Model learns features from every dye-induced spectral variation.  
  2. **Clinical Relevance**: Reflects real-world variability in dye protocols.  
  3. **Data Integrity**: Prevents leakage by stratifying at patient and dye level simultaneously.  
  4. **Balanced Evaluation**: Ensures fair performance assessment across all conditions.

---

### Why Dye Stratification Matters

- **Food Dyes**: Produce absorption changes in visible bands (450–700 nm).  
- **ICG Fluorescence**: Alters near-infrared reflectance around 780–900 nm.  
- **No Dye**: Baseline tissue spectra devoid of contrast agent effects.  

A split that omits any dye in training will leave the model blind to those spectral effects, causing catastrophic failure when encountering them in validation or test. The stratified approach ensures that training, validation and test datasets each contain examples of every spectral condition, fostering generalization and reliable performance in downstream tissue classification tasks. 
