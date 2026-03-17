import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model.yoloe import build_yoloe, YOLOEWithLoss
from data.dataset import UltrasoundDataset
from utils.metrics import decode_predictions

def load_model(weights_path, use_pu_loss, device):
    base_model = build_yoloe(model_size='s', num_classes=1)
    model = YOLOEWithLoss(model=base_model, num_classes=1, use_pu_loss=use_pu_loss)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    base_dir = '/h/dhaneshr/code/obj-det-test'
    data_dir = '/projects/tenomix/ml-share/training/07/data'
    baseline_weights = os.path.join(base_dir, 'checkpoints', 'best_model_20260315_101545.pth')
    pu_weights = os.path.join(base_dir, 'checkpoints', 'best_model_20260314_170531.pth')
    
    # Load models
    print("Loading models...")
    model_baseline = load_model(baseline_weights, use_pu_loss=False, device=device)
    model_pu = load_model(pu_weights, use_pu_loss=True, device=device)
    
    # Dataset
    dataset = UltrasoundDataset(root_dir=data_dir, split='val')
    
    # Select 6 random images
    indices = random.sample(range(len(dataset)), 6)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        image, target = dataset[idx]
        img_tensor = image.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            cls_scores_b, reg_dists_b = model_baseline(img_tensor)
            feat_sizes = [(img_tensor.shape[2] // s, img_tensor.shape[3] // s) for s in model_baseline.model.strides]
            anchor_points, stride_tensor = model_baseline.model.get_anchor_points(feat_sizes, device, img_tensor.dtype)
            preds_b = decode_predictions(cls_scores_b, reg_dists_b, anchor_points, stride_tensor)[0]
            
            cls_scores_p, reg_dists_p = model_pu(img_tensor)
            preds_p = decode_predictions(cls_scores_p, reg_dists_p, anchor_points, stride_tensor)[0]
            
        ax = axes[i]
        
        # image is (3, H, W). Permute to (H, W, 3) for plotting
        img_np = image.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img_np)
        
        # GT
        for box in target['boxes']:
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none', label='GT')
            ax.add_patch(rect)
            
        # Baseline
        if len(preds_b['boxes']) > 0:
            box = preds_b['boxes'][0] # Best prediction
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none', label='Baseline')
            ax.add_patch(rect)
            
        # PU
        if len(preds_p['boxes']) > 0:
            box = preds_p['boxes'][0] # Best prediction
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='y', linestyle='--', facecolor='none', label='PU Loss')
            ax.add_patch(rect)
            
        ax.axis('off')
        
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=3)
    
    out_path = os.path.join(base_dir, 'inference', 'results_grid.png')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == '__main__':
    main()
