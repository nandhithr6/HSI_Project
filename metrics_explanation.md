# HSI Training Analysis & Metrics Explanation

## 🎯 Metrics That Show NaN and Why

### **NaN Values Explained:**

1. **MCC (Matthews Correlation Coefficient)**: 
   - Shows `nan` when denominator is zero (no true/false positives)
   - Common in early epochs or for rare classes
   - **Normal behavior** - will stabilize as training progresses

2. **AUC (Area Under Curve)**:
   - Shows `nan` when:
     - Only one class present in validation batch  
     - No positive samples for ROC calculation
   - **Solution**: Your dataset has only 15 validation samples, so some classes might be missing in validation batches

3. **HD/HD95 (Hausdorff Distance)**:
   - Shows `nan` when:
     - No predicted pixels for a class
     - Segmentation masks are empty
   - **Normal in early epochs** - model hasn't learned to predict all classes yet

### **Large Values Explained:**

4. **HD/HD95 Values (like 100+ pixels)**:
   - Hausdorff distance measures maximum distance between predicted and ground truth boundaries
   - Large values = poor boundary prediction (normal early in training)
   - Will decrease as model learns better segmentation

5. **High Loss Values**:
   - Focal loss + Dice loss combination can start high (0.5-0.6)
   - **Your current 0.5984 train loss is actually good** for a complex HSI model

## 📈 Expected Training Progression

**Epoch 1-10**: High loss, many NaNs, large HD values (learning basic features)
**Epoch 10-50**: Loss decreases, fewer NaNs, HD values stabilize  
**Epoch 50-150**: Fine-tuning, stable metrics, best performance
**Epoch 150+**: Potential overfitting (watch validation loss)

## 🎯 What to Monitor

✅ **Good Signs:**
- Training loss decreasing steadily
- Validation loss following training loss (not diverging)
- Dice/IoU increasing over epochs
- Fewer NaN metrics as training progresses

⚠️ **Warning Signs:**
- Validation loss increasing while training decreases (overfitting)
- All metrics stuck at same values for many epochs
- Consistent OOM errors despite retries

## 💡 Your Current Status

From your test run showing:
- Train Loss: 0.5984 (good starting point)
- Val Loss: 0.5955 (very close to train - healthy)
- Some NaN metrics (expected early in training)

**This is completely normal and healthy training behavior!**