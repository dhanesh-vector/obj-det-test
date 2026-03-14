"""
YOLOE: End-to-End Object Detection Model
Combines CSPResNet backbone, CustomCSPPAN neck, and PPYOLOEHead
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CSPResNet
from .pan import CustomCSPPAN
from .head import PPYOLOEHead


class YOLOE(nn.Module):
    """
    YOLOE Object Detection Model
    
    Args:
        num_classes (int): Number of classes to detect
        width_mult (float): Width multiplier for model scaling
        depth_mult (float): Depth multiplier for model scaling
        reg_max (int): Maximum value for DFL regression
    """
    
    def __init__(self, num_classes=80, width_mult=1.0, depth_mult=1.0, reg_max=16):
        super(YOLOE, self).__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # Backbone: CSPResNet
        self.backbone = CSPResNet(
            width_mult=width_mult,
            depth_mult=depth_mult,
            return_idx=[1, 2, 3]  # Return features from stages 1, 2, 3
        )
        
        # Get backbone output channels
        backbone_out_channels = [256, 512, 1024]
        backbone_out_channels = [max(round(c * width_mult), 1) for c in backbone_out_channels]
        
        # Neck: CustomCSPPAN
        self.neck = CustomCSPPAN(
            in_channels=backbone_out_channels,
            out_channels=[1024, 512, 256],
            width_mult=width_mult,
            depth_mult=depth_mult
        )
        
        # Head: PPYOLOEHead
        head_in_channels = [1024, 512, 256]
        head_in_channels = [max(round(c * width_mult), 1) for c in head_in_channels]
        self.head = PPYOLOEHead(
            in_channels=head_in_channels,
            num_classes=num_classes,
            reg_max=reg_max
        )
        
        # Feature map strides
        self.strides = [32, 16, 8]
        
        # Initialize projection weights for DFL
        self._init_proj_weights()
        
    def _init_proj_weights(self):
        """Initialize projection layer weights for DFL"""
        proj = torch.linspace(0, self.reg_max, self.reg_max + 1)
        self.head.proj_conv.weight.data = proj.view(1, self.reg_max + 1, 1, 1)
        self.head.proj_conv.weight.requires_grad = False
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (Tensor): Input image tensor of shape (B, 3, H, W)
            
        Returns:
            Training: (cls_scores, reg_distri) - classification scores and regression distribution
            Inference: (cls_scores, reg_dists) - classification scores and decoded regression
        """
        # Backbone features
        features = self.backbone(x)
        
        # Neck features (FPN + PAN)
        neck_features = self.neck(features)
        
        # Detection head
        outputs = self.head(neck_features)
        
        return outputs
    
    def get_anchor_points(self, feat_sizes, device, dtype=torch.float32):
        """
        Generate anchor points for all feature levels
        
        Args:
            feat_sizes: List of (H, W) tuples for each feature level
            device: Device to create tensors on
            dtype: Data type for tensors
            
        Returns:
            anchor_points: Tensor of shape (total_anchors, 2)
            stride_tensor: Tensor of shape (total_anchors, 1)
        """
        anchor_points = []
        stride_tensor = []
        
        for feat_size, stride in zip(feat_sizes, self.strides):
            h, w = feat_size
            # Generate grid
            shift_x = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride
            shift_y = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).reshape(-1, 2)
            anchor_points.append(anchor_point)
            stride_tensor.append(torch.full((h * w, 1), stride, device=device, dtype=dtype))
            
        anchor_points = torch.cat(anchor_points, dim=0)
        stride_tensor = torch.cat(stride_tensor, dim=0)
        
        return anchor_points, stride_tensor


class YOLOEWithLoss(nn.Module):
    """
    YOLOE with integrated loss computation for training
    """
    
    def __init__(self, model, num_classes=80, reg_max=16, use_pu_loss=False, gamma=2.0, beta=1.0, label_smooth=0.0):
        super(YOLOEWithLoss, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.use_pu_loss = use_pu_loss

        if self.use_pu_loss:
            from .pu_loss import YOLOEPUFocalLoss
            self.loss_fn = YOLOEPUFocalLoss(num_classes=num_classes, reg_max=reg_max, strides=model.strides, gamma=gamma, beta=beta, label_smooth=label_smooth)
        else:
            from .loss import YOLOELoss
            self.loss_fn = YOLOELoss(num_classes=num_classes, reg_max=reg_max, strides=model.strides, label_smooth=label_smooth)
        
    def forward(self, x, targets=None):
        """
        Forward pass with optional loss computation
        
        Args:
            x: Input images
            targets: Optional targets for training
            
        Returns:
            Training: loss dict
            Inference: predictions
        """
        outputs = self.model(x)
        
        if self.training and targets is not None:
            return self.compute_loss(outputs, targets, x)
        
        return outputs
    
    def compute_loss(self, outputs, targets, x):
        """
        Compute training losses
        
        Args:
            outputs: Model outputs (cls_scores, reg_distri)
            targets: Ground truth targets
            x: Input images (for getting feature map sizes)
            
        Returns:
            Dictionary of losses
        """
        h, w = x.shape[2:]
        feat_sizes = [(h // s, w // s) for s in self.model.strides]
        
        cls_scores = outputs[0]
        anchor_points, stride_tensor = self.model.get_anchor_points(feat_sizes, cls_scores.device, cls_scores.dtype)
        
        return self.loss_fn(outputs, targets, anchor_points, stride_tensor)


def build_yoloe(model_size='s', num_classes=80):
    """
    Build YOLOE model with predefined configurations
    
    Args:
        model_size: One of 's', 'm', 'l', 'x'
        num_classes: Number of detection classes
        
    Returns:
        YOLOE model
    """
    configs = {
        's': {'width_mult': 0.50, 'depth_mult': 0.33},
        'm': {'width_mult': 0.75, 'depth_mult': 0.67},
        'l': {'width_mult': 1.00, 'depth_mult': 1.00},
        'x': {'width_mult': 1.25, 'depth_mult': 1.33},
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    
    return YOLOE(
        num_classes=num_classes,
        width_mult=config['width_mult'],
        depth_mult=config['depth_mult']
    )


if __name__ == '__main__':
    # Test the model
    model = build_yoloe('s', num_classes=80)
    model.eval()
    
    # Test input
    x = torch.randn(1, 3, 640, 640)
    
    with torch.no_grad():
        cls_scores, reg_dists = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Cls scores shape: {cls_scores.shape}")
    print(f"Reg dists shape: {reg_dists.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

