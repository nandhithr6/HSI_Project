# Hyperspectral Placenta Dataset: Preprocessing and Data Splitting Report

## Dataset Overview

The Hyperspectral Placenta Dataset (Puustinen et al., 2023) comprises:

- **101 hyperspectral image cubes** from four fresh human placentas  
- **Spectral range (after preprocessing)**: 37–38 bands, spanning ~515–700 nm  
- **Spatial resolution**: 1024 × 1024 pixels per band  
- **Annotations**: Pixel-wise segmentation masks for arteries, veins, stroma, umbilical cord, specular reflection, and (for some subsets) suture, red/blue dyes, and ICG-marked tissue  

Imaging conditions by group:
1. **Patient 1** (6 images): No dye (baseline tissue)  
2. **Patient 2** (24 images): Red/blue food dye (enhanced vessel contrast)  
3. **Patient 3** (23 images): Red/blue food dye (enhanced vessel contrast)  
4. **Patient 4** (48 images): ICG fluorescent dye (near-infrared fluorescence, truncated to visible bands in preprocessed cubes)

---

## Preprocessing Notes

### Raw vs. Preprocessed Data
- The original release includes **raw `.dat` + `.hdr` files** along with white/dark references.  
- The authors also provide **preprocessed `.tif` cubes**:  
  - Already flat-field & dark corrected  
  - Calibrated with white-reference reflectance  
  - Cropped to the relevant 37–38 visible bands (~515–700 nm)  
  - Intensity scaled to reflectance [0, 1]  
- Masks are distributed as aligned `.tif` files.  

👉 In this project, we use the **authors’ preprocessed `.tif` cubes and masks** directly.  
This avoids redundant calibration and guarantees reproducibility with published work.  

### Why Understanding the Raw Pipeline Still Matters
Even though we rely on `.tif` cubes, we studied the raw preprocessing procedure because:  
1. It clarifies **how reflectance values were derived**.  
2. It helps when extending methods to **new datasets** (where only raw data may be available).  
3. It ensures transparency when explaining **what preprocessing steps were applied**.  

---

## Data Splitting Strategies

Effective evaluation requires balancing dye conditions and patient variability. Three strategies were considered:

### 1. Patient- or Dye-Level Splits  
- Assign whole patient groups or dye conditions to train/val/test.  
- ❌ Leads to imbalance (6 vs. 48 images), domain shift, and unrealistic scenarios.  

### 2. Random Image-Level Split  
- Random 70/15/15% assignment without stratification.  
- ❌ Risks data leakage and uncontrolled dye distribution.  

### 3. **Stratified Split Across All Conditions (Chosen)**  
- Each split contains proportional representation from all patient groups and dye conditions.  
- **Training**: ~70% (≈71 images)  
- **Validation**: ~15% (≈15 images)  
- **Test**: ~15% (≈15 images)  
- Stratification ensures:  
  - Spectral robustness  
  - Balanced dye effects  
  - No leakage  
  - Clinically realistic evaluation  

---

### Why Dye Stratification Matters

- **Food Dyes**: Alter visible absorption (450–700 nm).  
- **ICG**: Alters reflectance in near-infrared (~780–900 nm), though truncated in `.tif` cubes.  
- **No Dye**: Baseline tissue for comparison.  

👉 Including all dye conditions in every split is essential for generalization. Otherwise, models would catastrophically fail when tested on unseen dye protocols.  

---

## Summary

- We **use the authors’ preprocessed `.tif` cubes** and corresponding masks.  
- We understand and can reimplement the **raw preprocessing pipeline** if required for other datasets.  
- For model training and evaluation, we employ a **stratified split** across all conditions (no dye, food dye, ICG).  
- This ensures **balanced, robust, and clinically meaningful evaluation**.
