"""
Explainable AI (XAI) Module for Cotton Disease Classification
Provides visualization of what features the model uses to make predictions
Includes: Grad-CAM, Grad-CAM++, Attention Rollout (for ViTs), and attention visualization
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
import json
import argparse
from train import get_model, DinoV2Classifier
import config
 
 
class GradCAM:
    """
    Generates Grad-CAM (Gradient-weighted Class Activation Map) for visualization
    Shows which regions of the image contributed most to the model's prediction
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
 
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
 
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
 
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
 
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            self.model.zero_grad()
            target_score = output[0, target_class]
            target_score.backward()
 
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_np = cam.squeeze().cpu().numpy()
        cam_min = np.percentile(cam_np, 2)
        cam_max = np.percentile(cam_np, 98)
 
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            cam_np = np.clip(cam_np, 0, 1)
 
        cam_np = np.power(cam_np, 0.8)
        return cam_np
 
 
class GradCAMPlusPlus:
    """
    Generates Grad-CAM++ (improved version of Grad-CAM)
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
 
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
 
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        with torch.enable_grad():
            input_tensor.requires_grad_(True)
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            self.model.zero_grad()
            target_score = output[0, target_class]
            target_score.backward(retain_graph=True)
 
        gradients = self.gradients
        activations = self.activations

        # GradCAM++ math expects spatial dims (H,W) so it can do mean/sum over
        # dim=(2,3). For some backbones (e.g., Swin internals) hooks can end up
        # producing token-shaped activations (B, N, C) or other rank-3 tensors.
        # In those cases, fall back to a Grad-CAM-like CAM that works with
        # 3D tensors by reducing over the token dimension.
        if activations.dim() < 4 or gradients.dim() < 4:
            # Expected token activations: (B, N, C)
            # Use gradients-weighted sum over tokens => (B, N)
            w = gradients.mean(dim=1)  # (B, C)
            cam = (activations * w.unsqueeze(1)).sum(dim=-1, keepdim=True)  # (B, N, 1)
            cam = cam.squeeze(-1)  # (B, N)
            cam = F.relu(cam)
            cam_np = cam.squeeze(0).cpu().numpy()
            # normalize + power transform to keep behavior consistent downstream
            cam_min = np.percentile(cam_np, 2)
            cam_max = np.percentile(cam_np, 98)
            if cam_max > cam_min:
                cam_np = (cam_np - cam_min) / (cam_max - cam_min)
                cam_np = np.clip(cam_np, 0, 1)
            cam_np = np.power(cam_np, 0.8)
            return cam_np

        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        spatial_sum = (activations * grad_3).sum(dim=(2, 3), keepdim=True)
        alpha_denom = 2 * grad_2 + spatial_sum
        alpha = grad_2 / (alpha_denom + 1e-10)
        relu_grad = F.relu(gradients)
        weights = (alpha * relu_grad).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        cam_min = np.percentile(cam_np, 2)
        cam_max = np.percentile(cam_np, 98)
 
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            cam_np = np.clip(cam_np, 0, 1)
 
        cam_np = np.power(cam_np, 0.8)
        return cam_np


# =============================================================================
# ATTENTION ROLLOUT (for DINOv2 / pure ViT backbones)
# =============================================================================
# Grad-CAM and Grad-CAM++ rely on spatial conv feature maps. DINOv2 has none —
# its backbone (transformers.Dinov2Model) only produces token sequences
# (B, N, C). Forcing transformer attention maps through the existing 3D
# fallback in GradCAMPlusPlus (designed for Swin's hybrid token-conv hooks)
# is not the right tool for a pure ViT: it weights tokens by their gradient
# contribution to a single class logit, but ignores how information actually
# flows between tokens across layers via attention.
#
# Attention Rollout (Abnar & Zuidema, 2020) is the standard alternative for
# pure ViTs: it multiplies the attention matrices of all transformer layers
# together (with a residual correction for the skip connections), producing
# a single map of how much each patch token contributes to the final CLS
# token representation. This requires no backward pass and is the technique
# most commonly used for DINOv2 visualization in practice.
class AttentionRollout:
    """
    Computes Attention Rollout for a DINOv2 (or other ViT) backbone exposed
    via DinoV2Classifier.backbone.

    Usage mirrors GradCAM/GradCAMPlusPlus: construct once, call .generate().
    """
    def __init__(self, model, discard_ratio=0.0):
        self.model = model
        self.discard_ratio = discard_ratio  # fraction of lowest attentions to zero out (0 = keep all)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        with torch.no_grad():
            # output_attentions=True returns one (B, heads, N, N) tensor per
            # transformer layer. Requires attn_implementation='eager' on the
            # backbone config (already set in DinoV2Classifier) — the default
            # 'sdpa' implementation does not populate attentions (verified).
            backbone_out = self.model.backbone(input_tensor, output_attentions=True)
            logits = self.model.head(backbone_out.pooler_output)
            if target_class is None:
                target_class = logits.argmax(dim=1).item()

            attentions = backbone_out.attentions  # tuple of (B, heads, N, N), one per layer
            num_tokens = attentions[0].shape[-1]

            # Start from identity — rollout accumulates layer-by-layer
            rollout = torch.eye(num_tokens, device=input_tensor.device).unsqueeze(0)

            for attn in attentions:
                # Average over attention heads -> (B, N, N)
                attn_heads_avg = attn.mean(dim=1)

                # Optionally discard the lowest attention values per row to
                # reduce noise from near-uniform attention (off by default).
                if self.discard_ratio > 0:
                    flat = attn_heads_avg.view(attn_heads_avg.size(0), -1)
                    k = int(flat.size(-1) * self.discard_ratio)
                    if k > 0:
                        threshold = flat.kthvalue(k, dim=-1, keepdim=True).values
                        attn_heads_avg = torch.where(
                            attn_heads_avg < threshold.unsqueeze(-1),
                            torch.zeros_like(attn_heads_avg),
                            attn_heads_avg,
                        )

                # Add identity to account for the residual/skip connection
                # around each attention block, then renormalise rows to sum
                # to 1 so the rollout remains a valid attention-like
                # distribution at every layer (standard Attention Rollout step).
                identity = torch.eye(num_tokens, device=attn.device).unsqueeze(0)
                attn_with_residual = (attn_heads_avg + identity) / 2
                attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)

                rollout = torch.matmul(attn_with_residual, rollout)

        # Row 0 of the rollout matrix = how much the CLS token (index 0)
        # attends to every other token, accumulated through all layers.
        # Drop the CLS->CLS entry (index 0) to keep only the 256 patch tokens.
        cls_attention = rollout[0, 0, 1:]  # (num_patches,)

        # Reshape the flat patch sequence back into its spatial grid.
        # For a 224x224 image with patch_size=14 (DINOv2's native patch
        # size), this is always a 16x16 grid (224/14 = 16) regardless of
        # which DINOv2 variant (vitb14 / vits14) is used, since patch_size
        # is constant across variants.
        grid_size = int(np.sqrt(cls_attention.shape[0]))
        cam_np = cls_attention.reshape(grid_size, grid_size).cpu().numpy()

        # Normalise to [0, 1] the same way GradCAM/GradCAM++ do, so
        # overlay_heatmap() and the downstream report code work unchanged.
        cam_min = np.percentile(cam_np, 2)
        cam_max = np.percentile(cam_np, 98)
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
            cam_np = np.clip(cam_np, 0, 1)
        cam_np = np.power(cam_np, 0.8)
        return cam_np
 
 
def get_target_layer(model, model_name):
    if model_name in ['resnet34', 'resnet50']:
        return model.layer4[-1]
    elif model_name == 'mobilenet_v2':
        return model.features[-1]
    elif model_name == "efficientnet_b7":
        return model.features[-1]
    elif model_name == "densenet201":
        return model.features
    # ===== NEW MODEL: ConvNeXt-Base =====
    # model.features(x) outputs a (B, 1024, 7, 7) conv feature map (verified),
    # so model.features[-1] (the final CNBlock stage) is a valid Grad-CAM
    # target layer — same pattern as efficientnet_b7 above.
    elif model_name == "convnext_base":
        return model.features[-1]
    # ===== NEW MODEL: CSPDarkNet53 =====
    # model.stages[-1] (the last CrossStage block) outputs (B, 1024, 7, 7)
    # for 224x224 inputs — verified via forward pass. This is the last
    # convolutional stage before global average pooling in the head,
    # making it the correct Grad-CAM target layer.
    elif model_name == "cspdarknet53":
        return model.stages[-1]
    # ===== NEW MODEL: VGG16 / VGG19 =====
    # VGG backbone is model.features; last conv layer is the last nn.Conv2d in features.
    elif model_name in ['vgg16', 'vgg19']:
        for layer in reversed(model.features):
            if isinstance(layer, nn.Conv2d):
                return layer
        raise ValueError(f"Could not find a Conv2d layer in VGG features for {model_name}")
    # ===== NEW MODEL: Swin Transformer =====
    # The existing GradCAM++ implementation expects gradients/activations
    # shaped like (B, C, H, W) so it can do mean(dim=(2,3)).
    # timm's Swin transformer returns token features (B, N, C) in many of
    # its intermediate modules; to keep GradCAM++ working without touching
    # the GradCAM++ code, we hook the last *attn* block output inside the
    # final stage which produces spatial feature maps.
    elif model_name == 'swin_transformer':
        # timm Swin has: model.layers[0..3] each containing blocks.
        # Each block has an attention module (submodule name: attn).
        # We hook that attention module as a stable target for gradients.
        if hasattr(model, 'layers') and len(model.layers) > 0:
            last_stage = model.layers[-1]
            if hasattr(last_stage, 'blocks') and len(last_stage.blocks) > 0:
                return last_stage.blocks[-1].attn
            # Fallback: return last stage module.
            return last_stage
        raise ValueError("Swin model does not expose `layers` for Grad-CAM")
    # ===== NEW MODEL: DINOv2 =====
    # DINOv2 has no convolutional feature maps at all — get_target_layer()
    # is not used for this model. is_vit_model() (below) routes DINOv2
    # through AttentionRollout instead, which operates on the model's
    # attention weights directly rather than a single hooked layer.
    # This branch exists only so an explicit, informative error is raised
    # if get_target_layer() is ever called for 'dinov2' by mistake (e.g.
    # a future code path that doesn't check is_vit_model() first).
    elif model_name == 'dinov2':
        raise ValueError(
            "get_target_layer() does not apply to 'dinov2' — DINOv2 is a "
            "pure ViT with no spatial conv feature maps. Use "
            "AttentionRollout via is_vit_model() routing instead."
        )

    else:
        raise ValueError(f"Model {model_name} not supported")


def is_vit_model(model_name):
    """
    Returns True if model_name is a pure Vision Transformer that has no
    convolutional feature maps and therefore cannot use Grad-CAM/Grad-CAM++
    via get_target_layer(). Currently only 'dinov2' — Swin is a hybrid
    architecture that still exposes spatially-shaped attention outputs
    compatible with the existing GradCAM++ 3D fallback, so it is NOT
    included here and continues to use get_target_layer() as before.
    """
    return model_name == 'dinov2'


 
 
def overlay_heatmap(image, heatmap, alpha=0.5):
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def get_cam_generator(model, model_name):
    """
    Returns a ready-to-use CAM/attention generator object for the given
    model, automatically choosing between Grad-CAM++ (CNNs, Swin) and
    Attention Rollout (DINOv2). Both returned objects expose the same
    .generate(input_tensor, target_class=None) -> np.ndarray interface,
    so calling code (explain_single_prediction, batch_explain,
    predict_batch.py) does not need to know which technique is used.
    """
    if is_vit_model(model_name):
        return AttentionRollout(model)
    else:
        target_layer = get_target_layer(model, model_name)
        return GradCAMPlusPlus(model, target_layer)
 
 
def explain_single_prediction(image_path, model_path, model_name=None, output_dir=None, true_label=None):
    """
    Generate complete explainability report for a single image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
   
    class_to_idx_path = os.path.splitext(model_path)[0] + '_class_to_idx.json'
    if not os.path.exists(class_to_idx_path):
        print(f"Error: Class mapping not found at {class_to_idx_path}")
        return
    
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)
   
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)
   
    if model_name is None:
        model_name = os.path.basename(os.path.splitext(model_path)[0]).split('_')[0]
        if model_name not in config.TrainingConfig.AVAILABLE_MODELS:
            model_name = config.TrainingConfig.MODEL_NAME

   
    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])
   
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"Error: Could not read image at {image_path}")
        return
   
    image_cv = cv2.resize(image_cv, config.IMAGE_SIZE)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)
   
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_idx = outputs.argmax(dim=1).item()
        predicted_class = idx_to_class[predicted_idx]
        confidence = probabilities[0, predicted_idx].item()
   
    print(f"\n{'='*70}")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.3f})")
    print(f"{'='*70}\n")
   
    print("All class probabilities:")
    for idx in sorted(idx_to_class.keys()):
        print(f"  {idx_to_class[idx]:20s}: {probabilities[0, idx].item():.4f}")
   
    # ===== ViT-AWARE XAI ROUTING =====
    # get_cam_generator() automatically picks AttentionRollout for DINOv2
    # and GradCAM++ (with the correct hooked layer) for every other model,
    # so this call site is identical regardless of architecture.
    technique_name = "Attention Rollout" if is_vit_model(model_name) else "Grad-CAM++"
    print(f"\nGenerating {technique_name} heatmap...")
    grad_cam = get_cam_generator(model, model_name)
    cam = grad_cam.generate(image_tensor, target_class=predicted_idx)
   
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.1)
    fig.suptitle(f'Explainability Report: {predicted_class} (Confidence: {confidence:.1%})',
                 fontsize=18, fontweight='bold', y=0.95)
   
    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.imshow(image_rgb)
    actual_label_text = f"Original Image = {true_label if true_label else 'Unknown'}"
    ax1.set_title(actual_label_text, fontsize=13, fontweight='bold', pad=10)
    ax1.axis('off')
   
    ax2 = fig.add_subplot(gs[0:2, 1])
    im = ax2.imshow(cam, cmap='jet')
    ax2.set_title(f'{technique_name} Heatmap', fontsize=13, fontweight='bold', color='red', pad=10)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
   
    ax3 = fig.add_subplot(gs[0:2, 2])
    overlay_img = overlay_heatmap(image_rgb, cam, alpha=0.5)
    ax3.imshow(overlay_img)
    pred_label_text = f"Pred: {predicted_class}"
    ax3.set_title(pred_label_text, fontsize=13, fontweight='bold', color='darkred', pad=15)
    ax3.axis('off')
   
    ax4 = fig.add_subplot(gs[2, 0:2])
    classes = sorted([idx_to_class[i] for i in range(num_classes)])
    confs = [probabilities[0, class_to_idx[c]].item() for c in classes]
    colors_list = ['#2ecc71' if c == predicted_class else '#3498db' for c in classes]

    bars = ax4.barh(classes, confs, color=colors_list, edgecolor='black', linewidth=2, height=0.6)
    ax4.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
    ax4.set_title('Model Confidence per Class', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, 1.1)
    ax4.grid(axis='x', alpha=0.2, linestyle='--')

    for bar, conf in zip(bars, confs):
        ax4.text(conf + 0.01, bar.get_y() + bar.get_height()/2, f'{conf:.1%}', va='center', fontweight='bold')
   
    ax5 = fig.add_subplot(gs[2, 2])
    ax5.axis('off')
   
    cam_std = np.std(cam)
    high_importance_region = np.sum(cam > 0.7) # Using a fixed threshold for analysis
    importance_pct = (high_importance_region / cam.size) * 100
   
    stats_text = (
        f"ANALYSIS SUMMARY\n"
        f"{'-'*20}\n"
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.1%}\n\n"
        f"Focus Metrics:\n"
        f"• Feature Spread (Std): {cam_std:.3f}\n"
        f"• Heat Coverage: {importance_pct:.1f}%\n\n"
        f"Top Alternative:\n"
        f"• {classes[np.argsort(confs)[-2]]}: {confs[np.argsort(confs)[-2]]*100:.1f}%"
    )

    ax5.text(0.0, 0.8, stats_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))
    
    plt.subplots_adjust(top=0.88, bottom=0.1, left=0.05, right=0.95)
    
    if output_dir is None: output_dir = os.path.dirname(model_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'xai_{os.path.basename(image_path)}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved report: {os.path.basename(image_path)}")
   
    # Generate detailed attention map
    print("\nGenerating detailed attention map...")
    fig_attn = plt.figure(figsize=(20, 12))
    gs = fig_attn.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    fig_attn.suptitle(f'Detailed Attention Analysis for {predicted_class}',
                     fontsize=16, fontweight='bold')
   
    thresholds = [0.25, 0.5, 0.75]
    for idx, threshold in enumerate(thresholds):
        ax1 = fig_attn.add_subplot(gs[0, idx])
        cam_thresholded = np.where(cam > threshold, cam, 0)
        im1 = ax1.imshow(cam_thresholded, cmap='hot')
        ax1.set_title(f'Attention (Threshold > {threshold})', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
       
        ax2 = fig_attn.add_subplot(gs[1, idx])
        overlay = overlay_heatmap(image_rgb, cam_thresholded, alpha=0.6)
        ax2.imshow(overlay)
        ax2.set_title(f'Overlay (Threshold > {threshold})', fontsize=12, fontweight='bold')
        ax2.axis('off')
   
    attn_output_path = os.path.join(output_dir, f'attention_map_{os.path.basename(image_path)}.png')
    fig_attn.savefig(attn_output_path, dpi=150, bbox_inches='tight')
    plt.close()
 
 
def batch_explain(test_dir, model_path, model_name=None, output_dir=None, num_samples_per_class=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_to_idx_path = os.path.splitext(model_path)[0] + '_class_to_idx.json'
    if not os.path.exists(class_to_idx_path):
        print(f"Error: Class mapping not found at {class_to_idx_path}")
        return
    
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)
   
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)
   
    if model_name is None:
        model_name = os.path.basename(os.path.splitext(model_path)[0]).split('_')[0]
        if model_name not in config.TrainingConfig.AVAILABLE_MODELS:
            model_name = config.TrainingConfig.MODEL_NAME
   
    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
   
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
   
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating batch explainability reports from: {test_dir}")
   
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
       
        print(f"Processing class: {class_name}")
        image_files = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
       
        image_files = image_files[:num_samples_per_class]
       
        for image_file in image_files:
            image_path = os.path.join(class_dir, image_file)
            if "Actual_" in image_file:
                extracted_label = image_file.split('Actual_')[1].split('_Pred')[0]
            else:
                extracted_label = class_name
            
            try:
                explain_single_prediction(image_path, model_path, model_name, output_dir, true_label=extracted_label)
            except Exception as e:
                print(f"  ✗ {image_file}: {str(e)}")
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explainable AI (XAI) visualization')
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None, choices=config.TrainingConfig.AVAILABLE_MODELS)

    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=10)
   
    args = parser.parse_args()
    if not args.model_path:
        args.model_path = os.path.join(os.getcwd(), 'models', 'trained_model.pth')
   
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        exit(1)
   
    if args.image_path:
        inferred_label = os.path.basename(os.path.dirname(args.image_path))
        explain_single_prediction(args.image_path, args.model_path, args.model_name, args.output_dir, true_label=inferred_label)
    elif args.test_dir:
        batch_explain(args.test_dir, args.model_path, args.model_name, args.output_dir, args.num_samples)
    else:
        print("Error: Provide either --image_path or --test_dir")
        exit(1)