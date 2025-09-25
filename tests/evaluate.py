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
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
try:
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import binary_erosion
except Exception:
    distance_transform_edt = None
    binary_erosion = None

# To import from parent directory src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.full_model import HSIModel
from src.training.train import NPZDataset, set_seed, discover_npz_files, discover_mask_keys_in_npz

def _per_class_stats_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        specificity = np.where(tn + fp > 0, tn / (tn + fp), 0.0)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
        dice = np.where(2 * tp + fp + fn > 0, (2 * tp) / (2 * tp + fp + fn), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'iou': iou,
        'dice': dice,
        'f1': f1,
    }

def _mcc_from_cm(cm: np.ndarray) -> float:
    cm = cm.astype(np.float64)
    t_k = cm.sum(axis=1)
    p_k = cm.sum(axis=0)
    c = np.trace(cm)
    s = cm.sum()
    sum_pk_tk = (p_k * t_k).sum()
    sum_pk_sq = (p_k ** 2).sum()
    sum_tk_sq = (t_k ** 2).sum()
    denom = np.sqrt((s**2 - sum_pk_sq) * (s**2 - sum_tk_sq))
    if denom == 0:
        return float('nan')
    return float((c * s - sum_pk_tk) / denom)

def calculate_metrics_from_cm(cm):
    """Calculate per-class metrics (including specificity)."""
    num_classes = cm.shape[0]
    metrics = {}
    stats = _per_class_stats_from_cm(cm)
    for i in range(num_classes):
        metrics[i] = {
            'Dice': float(stats['dice'][i]),
            'IoU': float(stats['iou'][i]),
            'Precision': float(stats['precision'][i]),
            'Recall': float(stats['recall'][i]),
            'Specificity': float(stats['specificity'][i]),
            'F1-Score': float(stats['f1'][i]),
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
    # Dataset/masks
    parser.add_argument('--mask-keys', type=str, default='mask__artery,mask__vein,mask__suture,mask__stroma,mask__umbilical_cord', help='Comma-separated mask keys; order defines class ids 1..N (0=background)')
    parser.add_argument('--auto-discover-mask-keys', action='store_true', help='Scan dataset to auto-discover mask_* keys and override --mask-keys')
    parser.add_argument('--crop-size', type=int, default=512)
    # Optional inference-time filtering of absent classes
    parser.add_argument('--filter-absent-classes', action='store_true', help='Post-process predictions to drop classes predicted absent')
    parser.add_argument('--presence-prob-thresh', type=float, default=0.5, help='Probability threshold for presence check')
    parser.add_argument('--presence-min-area', type=float, default=0.0005, help='Min area ratio above prob threshold to consider class present')
    # Metrics controls
    parser.add_argument('--compute-hd', action='store_true', help='Compute Hausdorff/HD95 metrics (can be slow)')
    parser.add_argument('--auc-max-pixels', type=int, default=200000, help='Max pixels sampled for AUC computation; set 0 to disable AUC')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Model ---
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")
    
    ckpt = torch.load(args.model_path, map_location=device)
    
    test_dir = os.path.join(args.data_dir, 'test')
    test_files = discover_npz_files(test_dir)
    
    # Resolve mask keys
    if isinstance(args.mask_keys, str) and args.mask_keys.strip():
        mask_keys = [k.strip() for k in args.mask_keys.split(',') if k.strip()]
    else:
        mask_keys = []
    if args.auto_discover_mask_keys:
        discovered = discover_mask_keys_in_npz(test_files)
        if discovered:
            mask_keys = discovered
            print(f"[Auto] Discovered mask keys in test set: {mask_keys}")
        else:
            print("[Auto] No mask_* keys discovered in test set; using provided --mask-keys")

    temp_ds = NPZDataset([test_files[0]], mask_keys=mask_keys or None, crop_size=args.crop_size, return_path=False)
    sample = temp_ds[0]
    sample_img = sample[0]
    num_bands = sample_img.shape[0]
    num_classes = 1 + (len(mask_keys) if mask_keys else 0)
    class_names = ['background'] + [k.replace('mask__', '') for k in mask_keys]

    print(f"Detected {num_bands} bands and {num_classes} classes.")

    model = HSIModel(num_bands=num_bands, num_classes=num_classes).to(device)
    missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
    if missing:
        print(f"[Load] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Load] Unexpected keys: {len(unexpected)}")
    model.eval()

    # --- Computational Metrics ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params / 1e6:.2f}M")

    # --- Load Data ---
    test_ds = NPZDataset(test_files, mask_keys=mask_keys or None, crop_size=args.crop_size, return_path=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    def filter_absent_classes(logits: torch.Tensor, prob_thresh: float, min_area_ratio: float) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        B, C, H, W = probs.shape
        area = H * W
        keep_mask = torch.ones((B, C), dtype=torch.bool, device=probs.device)
        keep_mask[:, 0] = True
        if C > 1:
            cls_probs = probs[:, 1:, :, :]
            max_prob = cls_probs.amax(dim=(2, 3))
            area_ratio = (cls_probs > prob_thresh).sum(dim=(2, 3)) / float(area)
            present = (max_prob >= prob_thresh) | (area_ratio >= min_area_ratio)
            keep_mask[:, 1:] = present
        keep = keep_mask[:, :, None, None]
        filtered = torch.where(keep, logits, torch.full_like(logits, -1e6))
        return filtered

    # --- Evaluation Loop ---
    all_preds = []
    all_targets = []
    all_images = []
    inference_times = []
    # AUC accumulators
    auc_labels = []
    auc_scores = []
    auc_cap = args.auc_max_pixels
    # HD lists
    hd_list = []
    hd95_list = []

    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.to(device)

            start_time = time.time()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                logits = outputs['final_logits']
            inference_times.append(time.time() - start_time)

            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            if args.filter_absent_classes:
                logits = filter_absent_classes(logits, args.presence_prob_thresh, args.presence_min_area)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            if i < (5 // args.batch_size) + 1:
                all_images.append(images.cpu())
            # AUC sampling
            if auc_cap > 0:
                probs = F.softmax(logits, dim=1).detach().cpu()
                B, C, Hh, Ww = probs.shape
                take = min(auc_cap - sum(len(a) for a in auc_labels), B * Hh * Ww)
                if take > 0:
                    idx = torch.randperm(B * Hh * Ww)[:take]
                    labels_b = targets.view(-1).cpu()[idx]
                    scores_b = probs.permute(0,2,3,1).reshape(-1, C)[idx]
                    auc_labels.append(labels_b.numpy())
                    auc_scores.append(scores_b.numpy())
            # HD per image per class
            if args.compute_hd:
                pr_np = preds.cpu().numpy()
                gt_np = targets.cpu().numpy()
                for bi in range(pr_np.shape[0]):
                    for c in range(1, num_classes):
                        gt_c = (gt_np[bi] == c).astype(np.uint8)
                        pr_c = (pr_np[bi] == c).astype(np.uint8)
                        if gt_c.max() == 0 and pr_c.max() == 0:
                            continue
                        if distance_transform_edt is None:
                            continue
                        gt_b = gt_c.astype(bool)
                        pr_b = pr_c.astype(bool)
                        if gt_b.sum() == 0 or pr_b.sum() == 0:
                            hd = float('inf'); hd95 = float('inf')
                        else:
                            gt_edge = gt_b ^ binary_erosion(gt_b)
                            pr_edge = pr_b ^ binary_erosion(pr_b)
                            dt_gt = distance_transform_edt(~gt_edge)
                            dt_pr = distance_transform_edt(~pr_edge)
                            d_gt_pr = dt_pr[gt_edge]
                            d_pr_gt = dt_gt[pr_edge]
                            if d_gt_pr.size == 0 or d_pr_gt.size == 0:
                                hd = float('nan'); hd95 = float('nan')
                            else:
                                hd = float(max(d_gt_pr.max(), d_pr_gt.max()))
                                hd95 = float(max(np.percentile(d_gt_pr, 95), np.percentile(d_pr_gt, 95)))
                        if not np.isnan(hd): hd_list.append(hd)
                        if not np.isnan(hd95): hd95_list.append(hd95)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_images = torch.cat(all_images, dim=0)

    # --- Calculate and Display Metrics ---
    cm = confusion_matrix(all_targets.flatten(), all_preds.flatten(), labels=range(num_classes))
    class_metrics = calculate_metrics_from_cm(cm)
    # AUC macro
    if auc_labels:
        y_auc = np.concatenate(auc_labels)
        s_auc = np.concatenate(auc_scores)
        try:
            auc_macro = float(roc_auc_score(y_auc, s_auc, multi_class='ovr', average='macro'))
        except Exception:
            auc_macro = float('nan')
    else:
        auc_macro = float('nan')

    headers = ["Class", "Dice", "IoU", "Precision", "Recall", "Specificity", "F1-Score"]
    table_data = []
    for i in range(num_classes):
        metrics = class_metrics[i]
        table_data.append([
            class_names[i],
            f"{metrics['Dice']:.4f}",
            f"{metrics['IoU']:.4f}",
            f"{metrics['Precision']:.4f}",
            f"{metrics['Recall']:.4f}",
            f"{metrics['Specificity']:.4f}",
            f"{metrics['F1-Score']:.4f}",
        ])

    print("\n--- Per-Class Segmentation Metrics ---")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Macro aggregates and extras
    tp = np.diag(cm).astype(np.float64)
    total = cm.sum()
    acc = float(tp.sum() / total) if total > 0 else 0.0
    # Macro aggregates from stats helper
    stats = _per_class_stats_from_cm(cm)
    dice_macro = float(np.nanmean(stats['dice']))
    iou_macro = float(np.nanmean(stats['iou']))
    prec_macro = float(np.nanmean(stats['precision']))
    rec_macro = float(np.nanmean(stats['recall']))
    spec_macro = float(np.nanmean(stats['specificity']))
    bal_acc = rec_macro  # balanced accuracy = macro recall (multiclass)
    # Kappa
    po = (tp.sum() / total) if total > 0 else 0.0
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total * total) if total > 0 else 0.0
    kappa = float((po - pe) / (1 - pe)) if (1 - pe) > 0 else 0.0
    mcc = _mcc_from_cm(cm)
    hd_mean = float(np.nanmean(hd_list)) if hd_list else float('nan')
    hd95_mean = float(np.nanmean(hd95_list)) if hd95_list else float('nan')
    print("\n--- Summary Metrics ---")
    print(tabulate([
        ['Acc', f"{acc:.4f}"],
        ['Dice(macro)', f"{dice_macro:.4f}"],
        ['IoU(macro)', f"{iou_macro:.4f}"],
        ['Prec(macro)', f"{prec_macro:.4f}"],
        ['Rec(macro)', f"{rec_macro:.4f}"],
        ['Spec(macro)', f"{spec_macro:.4f}"],
        ['BalAcc', f"{bal_acc:.4f}"],
        ['Kappa', f"{kappa:.4f}"],
        ['MCC', f"{mcc:.4f}"],
        ['AUC(macro)', f"{auc_macro:.4f}"],
        ['HD', f"{hd_mean:.2f}"],
        ['HD95', f"{hd95_mean:.2f}"],
    ], headers=['Metric','Value'], tablefmt='github'))

    # Save CSV of summary
    os.makedirs(args.save_dir, exist_ok=True)
    import csv
    with open(os.path.join(args.save_dir, 'metrics_test.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'acc','dice_macro','iou_macro','prec_macro','rec_macro','spec_macro','balanced_accuracy','kappa','mcc','auc_macro','hd_mean','hd95_mean'
        ])
        w.writerow([acc, dice_macro, iou_macro, prec_macro, rec_macro, spec_macro, bal_acc, kappa, mcc, auc_macro, hd_mean, hd95_mean])

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
