
import os
import argparse
import time
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# To import from parent directory src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.full_model import HSIModel
from src.training.train import NPZHSIDataset, set_seed, discover_npz_files

def calculate_metrics_from_cm(cm):
    """Calculate metrics from a confusion matrix."""
    num_classes = cm.shape[0]
    metrics = {}

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        # Dice Similarity Coefficient (DSC)
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        # Intersection over Union (IoU)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        # Recall (Sensitivity)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        # F1-Score
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[i] = {
            'Dice': dice,
            'IoU': iou,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
        }
    return metrics

def plot_predictions(images, true_masks, pred_masks, save_dir, num_samples=5):
    """Save visualizations of model predictions."""
    os.makedirs(save_dir, exist_ok=True)
    num_samples = min(num_samples, len(images))
    
    for i in range(num_samples):
        # Use a simple pseudo-color representation for the HSI image (e.g., first 3 bands)
        img_slice = images[i][:3, :, :].numpy().transpose(1, 2, 0)
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) # Normalize for visualization

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_slice)
        axes[0].set_title('Image (Pseudo-color)')
        axes[0].axis('off')

        axes[1].imshow(true_masks[i], cmap='jet', vmin=0, vmax=pred_masks.max())
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')

        axes[2].imshow(pred_masks[i], cmap='jet', vmin=0, vmax=pred_masks.max())
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_sample_{i}.png'))
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Evaluate HSI Segmentation Model")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model checkpoint (.pt file)')
    parser.add_argument('--data-dir', type=str, default='/ssd_scratch/placenta/Placenta', help='Root directory of the dataset')
    parser.add_argument('--save-dir', type=str, default='saved/evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Model ---
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")
    
    ckpt = torch.load(args.model_path, map_location=device)
    
    # Re-create model architecture. We need to know the original model's params.
    # This assumes the model params are consistent with the training script defaults.
    # A more robust way is to save args in the checkpoint.
    
    # Get num_bands and num_classes from the dataset
    test_dir = os.path.join(args.data_dir, 'test')
    test_files = discover_npz_files(test_dir)
    
    temp_ds = NPZHSIDataset([test_files[0]], verbose=False)
    sample_img, _ = temp_ds[0]
    num_bands = sample_img.shape[0]
    num_classes = 1 + len(getattr(temp_ds, 'class_map', {}))
    class_names = ['background'] + [k.replace('mask__', '') for k in temp_ds.class_keys]

    print(f"Detected {num_bands} bands and {num_classes} classes.")

    model = HSIModel(num_bands=num_bands, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # --- Computational Metrics ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params / 1e6:.2f}M")

    # --- Load Data ---
    test_ds = NPZHSIDataset(test_files, augment=False, crop_size=512, verbose=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Evaluation Loop ---
    all_preds = []
    all_targets = []
    all_images = []
    inference_times = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.to(device)

            start_time = time.time()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                logits = outputs['final_logits']
            inference_times.append(time.time() - start_time)

            # Resize logits to match target size if necessary
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            if i < (5 // args.batch_size) + 1: # Save some images for visualization
                 all_images.append(images.cpu())


    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_images = torch.cat(all_images, dim=0)

    # --- Calculate and Display Metrics ---
    cm = confusion_matrix(all_targets.flatten(), all_preds.flatten(), labels=range(num_classes))
    class_metrics = calculate_metrics_from_cm(cm)

    headers = ["Class", "Dice", "IoU", "Precision", "Recall", "F1-Score"]
    table_data = []
    for i in range(num_classes):
        metrics = class_metrics[i]
        table_data.append([
            class_names[i],
            f"{metrics['Dice']:.4f}",
            f"{metrics['IoU']:.4f}",
            f"{metrics['Precision']:.4f}",
            f"{metrics['Recall']:.4f}",
            f"{metrics['F1-Score']:.4f}",
        ])

    print("\n--- Per-Class Segmentation Metrics ---")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # --- Performance Metrics ---
    avg_inference_time = np.mean(inference_times)
    fps = 1 / avg_inference_time if avg_inference_time > 0 else 0
    print("\n--- Performance Metrics ---")
    print(f"Average Inference Time per Batch: {avg_inference_time:.4f}s")
    print(f"Inference FPS (approx): {fps * args.batch_size:.2f}")
    
    # --- Save Visualizations ---
    vis_save_dir = os.path.join(args.save_dir, 'visualizations')
    print(f"\nSaving prediction visualizations to: {vis_save_dir}")
    plot_predictions(all_images, all_targets, all_preds, vis_save_dir)

if __name__ == '__main__':
    main()
