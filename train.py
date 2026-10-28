import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json
import config
import argparse
import shutil
from PIL import Image
from tqdm import tqdm
import timm  # required for cspdarknet53, swin_transformer
from transformers import Dinov2Model, Dinov2Config  # required for dinov2

# Respect a user override for debugging / compatibility.
# Example:
#   CUDA_VISIBLE_DEVICES=0 USE_CUDA=0 python train.py ...
# or
#   set USE_CUDA=0
use_cuda_env = os.environ.get('USE_CUDA', '').strip().lower()
# If USE_CUDA is set to 0/false/off/cpu => force CPU.
force_cpu = use_cuda_env in ('0', 'false', 'no', 'off', 'cpu')

cuda_available = torch.cuda.is_available()
DEVICE = torch.device("cuda") if (cuda_available and not force_cpu) else torch.device("cpu")
print(f"Using device: {DEVICE} (torch.cuda.is_available()={cuda_available}, USE_CUDA={use_cuda_env!r})")



class PlantDiseaseDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            target_dir = os.path.join(self.data_dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target_class])
                    samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = cv2.imread(path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if self.transform:
            sample = self.transform(sample)
        return sample, target


# =============================================================================
# CLASS WEIGHTS
# =============================================================================
def build_class_weights_tensor(class_to_idx, device):
    """
    Builds a class-weight tensor for CrossEntropyLoss, resolved by class
    NAME against config.TrainingConfig.CLASS_WEIGHTS (a name -> weight
    dict), then mapped to the correct index order via class_to_idx.

    This indirection matters because class index order is determined by
    sorted folder names at dataset-load time (see PlantDiseaseDataset
    above), which varies across datasets. Resolving by name instead of
    position means CLASS_WEIGHTS in config.py stays correct even if the
    dataset changes, or folders are added/removed/renamed.

    Any class name not present in CLASS_WEIGHTS defaults to weight 1.0.
    Returns None if USE_CLASS_WEIGHTS is False or CLASS_WEIGHTS is empty,
    in which case CrossEntropyLoss falls back to its default unweighted
    behaviour (weight=None) — identical to the original pipeline behaviour.
    """
    use_class_weights = getattr(config.TrainingConfig, 'USE_CLASS_WEIGHTS', False)
    weights_dict = getattr(config.TrainingConfig, 'CLASS_WEIGHTS', {})

    if not use_class_weights or not weights_dict:
        print("  [Class Weights] DISABLED — using unweighted CrossEntropyLoss")
        return None

    num_classes = len(class_to_idx)
    weights = [1.0] * num_classes

    unmatched = []
    for class_name, weight in weights_dict.items():
        if class_name in class_to_idx:
            weights[class_to_idx[class_name]] = float(weight)
        else:
            unmatched.append(class_name)

    if unmatched:
        print(f"  [Class Weights] WARNING: these names in CLASS_WEIGHTS "
              f"don't match any dataset folder and were ignored: {unmatched}")

    # Print weights aligned with class names for easy verification that
    # weights landed on the intended classes.
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    weight_summary = ", ".join(
        f"{idx_to_class[i]}={weights[i]}" for i in range(num_classes)
    )
    print(f"  [Class Weights] ENABLED — {weight_summary}")

    return torch.tensor(weights, dtype=torch.float32).to(device)


# =============================================================================
# DINOv2 WRAPPER
# =============================================================================
# DINOv2 (via Hugging Face transformers) is a pure backbone — it has no
# built-in classification head, unlike every CNN model above. This thin
# wrapper adds a Linear head on top of the pooled CLS token output, and
# exposes the backbone under `self.backbone` so freeze logic and Grad-CAM
# can address its internals using dotted parameter names
# (e.g. 'backbone.encoder.layer.9'), consistent with how other models in
# this pipeline expose their backbone (e.g. ConvNeXt's `features`).
#
# attn_implementation='eager' is set on the config so that
# output_attentions=True actually returns per-layer attention weight
# tensors (the default 'sdpa' implementation does not populate these,
# verified empirically) — needed for the Attention Rollout XAI method.
class DinoV2Classifier(nn.Module):
    def __init__(self, variant, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            hf_config = Dinov2Config.from_pretrained(variant, attn_implementation='eager')
            self.backbone = Dinov2Model.from_pretrained(variant, config=hf_config)
        else:
            hf_config = Dinov2Config(attn_implementation='eager')
            self.backbone = Dinov2Model(hf_config)
        hidden_dim = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        # pooler_output = CLS token after the final layernorm — (B, hidden_dim).
        # This is the standard DINOv2 image-level embedding for classification.
        pooled = out.pooler_output
        return self.head(pooled)


def get_model(model_name, num_classes, base_weights_path=None, freeze_until=None):
    if model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    elif model_name == 'efficientnet_b7':
        model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
    elif model_name == 'densenet201':
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    # ===== NEW MODEL: VGG16 / VGG19 =====
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == 'vgg19':
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    # ===== NEW MODEL: ConvNeXt-Base =====
    elif model_name == 'convnext_base':
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

    # ===== NEW MODEL: CSPDarkNet53 (via timm) =====
    elif model_name == 'cspdarknet53':
        # timm handles ImageNet weight loading and the full backbone internally.
        # num_classes is set here so timm builds the correct Linear output size.
        model = timm.create_model('cspdarknet53', pretrained=True, num_classes=num_classes)
    # ===== NEW MODEL: Swin Transformer =====
    elif model_name == 'swin_transformer':
        model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            num_classes=num_classes,
        )
    # ===== NEW MODEL: DINOv2 (via Hugging Face transformers) =====
    # DINOv2 has no num_classes argument like timm models — instead it's a
    # pure backbone wrapped by DinoV2Classifier (defined above), which adds
    # the classification head. The variant (vitb14 vs vits14) is read from
    # config.TrainingConfig.DINOV2_VARIANT so it stays configurable without
    # touching this function.
    elif model_name == 'dinov2':
        variant = getattr(config.TrainingConfig, 'DINOV2_VARIANT', 'facebook/dinov2-base')
        model = DinoV2Classifier(variant, num_classes, pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not supported.")


    if base_weights_path:
        print(f"Loading custom base weights from: {base_weights_path}")
        model.load_state_dict(torch.load(base_weights_path), strict=False)

    if freeze_until:
        print(f"Freezing layers until (and including): {freeze_until}")
        freezing = True
        for name, param in model.named_parameters():
            if name.startswith(freeze_until):
                freezing = False
            if freezing:
                param.requires_grad = False
            else:
                param.requires_grad = True

    # Dropout(0.3) before the final FC layer to regularise the classifier head.
    if model_name in ['resnet34', 'resnet50']:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == 'efficientnet_b7':
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == 'densenet201':
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    # ===== NEW MODEL: VGG16 / VGG19 classifier head =====
    elif model_name in ['vgg16', 'vgg19']:

        # torchvision VGG classifier is: Sequential(..., Linear(4096, 4096), Linear(4096, 1000))
        # The last layer is at index 6.
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    # ===== NEW MODEL: ConvNeXt-Base classifier head =====

    # torchvision's convnext_base.classifier is:
    #   Sequential(LayerNorm2d(1024), Flatten(), Linear(1024, 1000))
    # Unlike the other models, the Linear layer is NOT at index 1 — it's at
    # index 2, after a LayerNorm2d and a Flatten. The LayerNorm2d normalises
    # the (B, 1024, 1, 1) conv output and Flatten reshapes it to (B, 1024)
    # before the Linear layer. Both must be preserved; only the final
    # Linear (classifier[2]) is replaced, following the same
    # Dropout(0.3) + Linear pattern used everywhere else in this function.
    elif model_name == 'convnext_base':
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    # ===== NEW MODEL: CSPDarkNet53 classifier head =====
    # timm's CspNet uses model.head.fc for its final Linear layer.
    

    # The head also contains global_pool and Dropout(0.0), which must be
    # preserved — only head.fc is replaced with our Dropout(0.3) + Linear.
    # in_features = 1024 (verified via timm.create_model + num_features).
    # Because timm already set num_classes correctly above, this
    # replacement purely adds the Dropout regulariser and ensures
    # the head structure matches the other models in the pipeline.
    elif model_name == 'cspdarknet53':
        in_features = model.num_features  # 1024 for cspdarknet53
        model.head.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
    # NOTE: 'swin_transformer' and 'dinov2' need no further head replacement
    # here — timm already set num_classes for Swin above, and
    # DinoV2Classifier.__init__ already builds the correct Linear(hidden_dim,
    # num_classes) head. Both are intentionally excluded from this block.

    return model


def get_backbone_freeze_point(model_name):
    if 'resnet' in model_name:
        return 'fc'
    elif 'mobilenet' in model_name:
        return 'classifier'
    elif 'efficientnet' in model_name:
        return 'classifier'
    elif 'densenet' in model_name:
        return 'classifier'
    # ===== NEW MODEL: VGG16 / VGG19 =====
    # VGG backbone is under 'features' and classifier head is under 'classifier'.
    # Stage 1 returns 'classifier' so only the classifier trains (backbone frozen).
    # Stage 2 uses FINE_TUNING_CONFIG['UNFREEZE_LAYERS']='features' to unfreeze backbone.
    elif 'vgg' in model_name:
        return 'classifier'

    # ===== NEW MODEL: Swin Transformer =====
    # Use the same freezing granularity as ConvNeXt via Stage 1:
    # - Stage 1 should freeze backbone and train the head only.
    # In timm Swin, the classifier lives under `head`.
    elif 'swin' in model_name:
        return 'head'

    # ===== NEW MODEL: ConvNeXt-Base =====

    # ConvNeXt-Base exposes its backbone under the 'features' prefix and its
    # head under 'classifier' (confirmed via named_parameters()), identical
    # in structure to MobileNetV2/EfficientNet/DenseNet. Returning
    # 'classifier' here freezes everything under 'features' during Stage 1
    # warmup. Stage 2 fine-tuning then unfreezes from
    # FINE_TUNING_CONFIG['UNFREEZE_LAYERS'] = 'features', which unfreezes
    # the entire backbone since 'features' is the first parameter prefix.
    elif 'convnext' in model_name:
        return 'classifier'
    # ===== NEW MODEL: Swin Transformer =====
    # timm Swin backbone lives under `layers.*` and classification head under `head`.
    elif 'swin' in model_name:
        return 'head'

    # ===== NEW MODEL: CSPDarkNet53 =====
    # timm's CspNet uses 'stem' and 'stages.0..4' for the backbone, and
    # 'head' for the classifier. Returning 'head' here ensures Stage 1 warmup
    # trains only the head while freezing the entire backbone.
    # Stage 2 uses UNFREEZE_LAYERS (e.g. 'stages.4') to partially unfreeze.
    # Supported FREEZE_LAYERS_UNTIL values:
    #   'head'     — freeze all backbone, train head only (Stage 1 default)
    #   'stages.4' — unfreeze last CSP stage + head
    #   'stages.3' — unfreeze last two CSP stages + head
    elif 'cspdarknet' in model_name:
        return 'head'
    # ===== NEW MODEL: DINOv2 =====
    # DinoV2Classifier exposes the backbone under 'backbone' and the
    # classification head under 'head' (see class definition above).
    # Returning 'head' here freezes the entire backbone — patch embeddings
    # ('backbone.embeddings.*') and all 12 transformer blocks
    # ('backbone.encoder.layer.0' through '.11') — and trains only the
    # final Linear head during Stage 1 warmup.
    #
    # Supported freeze/unfreeze values for DINOv2:
    #   'head'                          — freeze entire backbone (Stage 1 default)
    #   'backbone.encoder.layer.9'      — unfreeze last 3 transformer blocks
    #                                      (9, 10, 11) + final layernorm + head
    #   'backbone.encoder.layer.6'      — unfreeze last 6 transformer blocks
    #   'backbone.embeddings'           — unfreeze everything (full fine-tune,
    #                                      not recommended for small datasets)
    # This mirrors the recommended ViT fine-tuning strategy: keep early
    # layers (low-level, generic features) frozen, only adapt the deeper
    # semantic layers to the new task.
    elif 'dinov2' in model_name:
        return 'head'
    else:
        return None


# =============================================================================
# MIXUP UTILITIES
# =============================================================================

def mixup_data(images, labels, alpha):
    """
    Produces a MixUp-blended batch.

    Samples lambda from Beta(alpha, alpha). When alpha is small (e.g. 0.2),
    the Beta distribution is U-shaped, meaning lambda is usually close to 0
    or 1 — so most blended images are dominated by one of the two sources.
    This keeps the training signal realistic while still exposing the model
    to interpolated appearances.

    Args:
        images : float tensor  (B, C, H, W)  — already normalised
        labels : long tensor   (B,)
        alpha  : float         — Beta distribution concentration parameter

    Returns:
        mixed_images : (B, C, H, W)
        labels_a     : (B,)   original labels
        labels_b     : (B,)   shuffled labels
        lam          : float  mixing coefficient (same scalar for whole batch)
    """
    lam = np.random.beta(alpha, alpha)

    batch_size = images.size(0)
    # Random permutation within the batch to pair each image with another
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1.0 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, predictions, labels_a, labels_b, lam):
    """
    Computes the MixUp loss:
        loss = lam * CE(pred, y_a) + (1 - lam) * CE(pred, y_b)

    This is the correct way to compute loss for a mixed sample.
    It cannot simply be cross-entropy on a soft label because
    label_smoothing in PyTorch's CrossEntropyLoss expects integer targets.
    The two-term weighted sum is mathematically equivalent to CE with
    a mixed soft target but works with PyTorch's standard criterion.

    Args:
        criterion   : nn.CrossEntropyLoss instance
        predictions : model output logits  (B, num_classes)
        labels_a    : original labels      (B,)
        labels_b    : shuffled labels      (B,)
        lam         : float mixing coefficient

    Returns:
        scalar loss tensor
    """
    return lam * criterion(predictions, labels_a) + (1.0 - lam) * criterion(predictions, labels_b)


# =============================================================================
# TRAINING STAGE
# =============================================================================

def train_model_stage(model, train_loader, val_loader, epochs, lr,
                      model_path, model_dir, stage_name, class_weights=None):
    """

    Runs the training loop for a single stage.

    MixUp is applied during training only, controlled by:
        config.TrainingConfig.USE_MIXUP
        config.TrainingConfig.MIXUP_ALPHA

    Validation always runs on clean (un-mixed) images so the val accuracy
    metric is a fair measure of the model on real images.

    class_weights : optional torch.Tensor of shape (num_classes,), built by
        build_class_weights_tensor() and passed in from train(). When None
        (the default, and the behaviour when
        config.TrainingConfig.USE_CLASS_WEIGHTS is False), CrossEntropyLoss
        behaves exactly as before — fully unweighted.

    Mixed precision (torch.cuda.amp) is enabled automatically when CUDA is
    available. This is particularly important for transformer backbones
    like DINOv2 and Swin, which are far more memory-hungry than the CNN
    models in this pipeline — autocast + GradScaler roughly halves
    activation memory with negligible accuracy impact.
    """
    # CrossEntropyLoss with label smoothing, optionally weighted per-class
    # to counter class imbalance / ghost-class absorption. class_weights
    # is None (unweighted) unless config.TrainingConfig.USE_CLASS_WEIGHTS
    # is True — see build_class_weights_tensor() in train().
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # AdamW: weight decay regularisation helps reduce overfitting to
    # training-distribution-specific features.

    # If SAM is enabled, swap optimizer from AdamW -> SAM(AdamW).
    # When USE_SAM=False this behaves exactly like the original code.

    # ===== SAM MODIFICATION =====
    use_sam = getattr(config.TrainingConfig, 'USE_SAM', False)
    if use_sam:
        from sam import SAM
        print(f"  [SAM] ENABLED  rho={config.TrainingConfig.SAM_RHO} adaptive={config.TrainingConfig.SAM_ADAPTIVE}")

        base_optimizer_cls = optim.AdamW
        optimizer = SAM(
            model.parameters(),
            base_optimizer=base_optimizer_cls,
            rho=config.TrainingConfig.SAM_RHO,
            adaptive=config.TrainingConfig.SAM_ADAPTIVE,
            lr=lr,
            weight_decay=1e-4,
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # ===== SAM MODIFICATION =====


    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.TrainingConfig.LR_SCHEDULER_STEP_SIZE,
        gamma=config.TrainingConfig.LR_SCHEDULER_GAMMA
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_loss_history = []
    val_loss_history = []
    best_val_acc = 0.0

    # Read MixUp settings from config once so they are consistent per stage
    use_mixup = config.TrainingConfig.USE_MIXUP
    mixup_alpha = config.TrainingConfig.MIXUP_ALPHA

    if use_mixup:
        print(f"  [MixUp] ENABLED  alpha={mixup_alpha}")
    else:
        print(f"  [MixUp] DISABLED")

    print(f"Starting {stage_name} with lr={lr}...")

    use_amp = (DEVICE.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for epoch in range(epochs):
        # ------------------------------------------------------------------ #
        #  TRAINING PASS                                                      #
        # ------------------------------------------------------------------ #
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=True)
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_mixup:
                # Mix the batch and compute the two-term MixUp loss
                mixed_images, labels_a, labels_b, lam = mixup_data(
                    images, labels, mixup_alpha
                )
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(mixed_images)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            # ===== SAM MODIFICATION =====
            if not use_sam:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                # SAM two-step update:
                #   1) first_step(): ascent to worst-case neighborhood
                #   2) second_step(): restore + descent update

                # Step 1: maximize loss within rho-ball
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer.base_optimizer) if hasattr(optimizer, 'base_optimizer') else None
                else:
                    loss.backward()
                optimizer.first_step(zero_grad=True)

                # Step 2: re-compute loss at perturbed weights and minimize
                # NOTE: MixUp is re-applied exactly like your original code path
                # but after the first_step().
                if use_mixup:
                    mixed_images_2, labels_a_2, labels_b_2, lam_2 = mixup_data(
                        images, labels, mixup_alpha
                    )
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs_2 = model(mixed_images_2)
                        loss_2 = mixup_criterion(criterion, outputs_2, labels_a_2, labels_b_2, lam_2)
                else:
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs_2 = model(images)
                        loss_2 = criterion(outputs_2, labels)

                if use_amp:
                    scaler.scale(loss_2).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_2.backward()
                    optimizer.second_step(zero_grad=True)

                # SAM requires second_step even in non-amp case; in amp case we
                # already called scaler.step(optimizer) above.
                if use_amp:
                    optimizer.second_step(zero_grad=True)
                loss = loss_2
            # ===== SAM MODIFICATION =====

            running_loss += loss.item() * images.size(0)



        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)

        # ------------------------------------------------------------------ #
        #  VALIDATION PASS — always on clean images, no MixUp                #
        # ------------------------------------------------------------------ #
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                # Val: use autocast for speed/memory on CUDA.
                with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_loss_history.append(val_epoch_loss)
        val_acc = correct / total

        print(f"[{stage_name}] Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {epoch_loss:.3f}.. "
              f"Val loss: {val_epoch_loss:.3f}.. "
              f"Val accuracy: {val_acc:.3f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with accuracy: {best_val_acc:.3f}")



        if config.TrainingConfig.APPLY_EARLY_STOPPING:
            if val_epoch_loss < best_val_loss - config.TrainingConfig.EARLY_STOPPING_MIN_DELTA:

                best_val_loss = val_epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"Early stopping counter: {epochs_no_improve}/{config.TrainingConfig.EARLY_STOPPING_PATIENCE}")

            if epochs_no_improve >= config.TrainingConfig.EARLY_STOPPING_PATIENCE:
                print(f"Stopping early as validation loss did not improve for "
                      f"{config.TrainingConfig.EARLY_STOPPING_PATIENCE} epochs.")
                break

    # If no best accuracy was ever saved, still save the final model so
    # downstream evaluation can run.
    # (We check for file existence instead of relying on val_acc improvements.)
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"[Fallback] No improvement checkpoint saved; saved final model to {model_path}")


    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')

    plt.title(f'{stage_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(model_dir, f'loss_plot_{stage_name.lower().replace(" ", "_")}.png')
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    plt.close()


# =============================================================================
# MAIN TRAIN FUNCTION
# =============================================================================

def train(data_dir, model_name, epochs, lr, batch_size, model_path, model_dir):

    # -------------------------------------------------------------------------
    # TRAINING TRANSFORM — same for all classes (class-aware transforms removed)
    # -------------------------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.6, 1.0),
            ratio=(0.85, 1.15),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=25, fill=0),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.25,
            hue=0.08,
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0))],
            p=0.4,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
        transforms.RandomErasing(
            p=0.35,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value='random',
        ),
    ])

    # -------------------------------------------------------------------------
    # VALIDATION TRANSFORM — deterministic, no augmentation
    # -------------------------------------------------------------------------
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])

    train_dataset = PlantDiseaseDataset(
        data_dir=os.path.join(data_dir, 'train'),
        transform=train_transform,
    )
    val_dataset = PlantDiseaseDataset(
        data_dir=os.path.join(data_dir, 'val'),
        transform=val_transform,
    )

    # Performance knobs: disable pin_memory for vGPU compatibility
    # (vGPU with newer CUDA drivers has known issues with pinned allocator)
    dl_kwargs = {
        'batch_size': batch_size,
        'pin_memory': False,
        'num_workers': 0,
        'persistent_workers': False,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)

    num_classes = len(train_dataset.classes)

    # Build the class weight tensor once, resolved against this dataset's
    # actual class_to_idx mapping (see build_class_weights_tensor() docstring
    # above for why name-based resolution matters). Reused for both Stage 1
    # and Stage 2 so weighting is applied consistently across the whole run.
    print("\n[Class Weights] Resolving weights against dataset classes...")
    class_weights = build_class_weights_tensor(train_dataset.class_to_idx, DEVICE)

    # --- STAGE 1: WARMUP ---
    print("\n=== STAGE 1: Warmup Training ===")

    stage1_freeze = get_backbone_freeze_point(model_name)
    if stage1_freeze is None:
        stage1_freeze = config.TrainingConfig.FREEZE_LAYERS_UNTIL
        print(f"Warning: Could not determine backbone freeze point for {model_name}. "
              f"Using config default: {stage1_freeze}")

    model = get_model(
        model_name,
        num_classes,
        base_weights_path=config.TrainingConfig.BASE_MODEL_PATH,
        freeze_until=stage1_freeze
    )
    model.to(DEVICE)

    stage1_model_path = os.path.join(model_dir, 'best_model_stage1.pth')

    train_model_stage(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        model_path=stage1_model_path,
        model_dir=model_dir,
        stage_name="Stage 1 (Warmup)",
        class_weights=class_weights,
    )

    final_model_path = stage1_model_path

    # --- STAGE 2: FINE-TUNING ---
    if config.TrainingConfig.FINE_TUNING_CONFIG.get('ENABLED', False):
        print("\n=== STAGE 2: Fine-Tuning ===")

        ft_config = config.TrainingConfig.FINE_TUNING_CONFIG
        stage2_freeze = ft_config.get('UNFREEZE_LAYERS', 'layer4')

        model = get_model(
            model_name,
            num_classes,
            base_weights_path=None,
            freeze_until=stage2_freeze
        )

        if os.path.exists(stage1_model_path):
            print(f"Loading Stage 1 weights from {stage1_model_path}...")
            model.load_state_dict(torch.load(stage1_model_path))
        else:
            print(f"Error: Stage 1 model not found at {stage1_model_path}. Cannot proceed to Stage 2.")
            return

        model.to(DEVICE)

        stage2_model_path = os.path.join(model_dir, 'best_model_stage2.pth')

        train_model_stage(
            model,
            train_loader,
            val_loader,
            epochs=ft_config.get('EPOCHS', 10),
            lr=ft_config.get('LEARNING_RATE', lr * 0.1),
            model_path=stage2_model_path,
            model_dir=model_dir,
            stage_name="Stage 2 (Fine-Tuning)",
            class_weights=class_weights,
        )

        final_model_path = stage2_model_path

    if model_path != final_model_path:
        print(f"Copying best model from {final_model_path} to {model_path}")
        shutil.copy(final_model_path, model_path)

    class_to_idx_path = os.path.splitext(model_path)[0] + '_class_to_idx.json'
    with open(class_to_idx_path, 'w') as f:
        json.dump(train_dataset.class_to_idx, f)
    print(f"Class to index mapping saved to {class_to_idx_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--data_dir', type=str, required=False, default=None,
                        help='Directory with train and val data')
    parser.add_argument('--processed_dir', type=str, default=None,
                        help='Path to processed dataset containing train/val/test subfolders')
    parser.add_argument('--model_name', type=str, default=config.TrainingConfig.MODEL_NAME,
                        choices=config.TrainingConfig.AVAILABLE_MODELS, help='Model to train')
    parser.add_argument('--epochs', type=int, default=config.TrainingConfig.EPOCHS,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=config.TrainingConfig.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=config.TrainingConfig.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to save the trained model')
    args = parser.parse_args()

    if not args.data_dir and not args.processed_dir:
        default_path = r"C:\Users\USER\Codebase\processed_dataset"
        print(f"No data_dir provided. Defaulting to: {default_path}", flush=True)
        args.data_dir = default_path

    data_root = args.processed_dir if args.processed_dir else args.data_dir

    if not os.path.isdir(data_root):
        parser.error(f"Data directory does not exist: {data_root}")

    train_sub = os.path.join(data_root, 'train')
    val_sub = os.path.join(data_root, 'val')
    if not os.path.isdir(train_sub):
        parser.error(f"Required subdirectory missing: {train_sub}")
    if not os.path.isdir(val_sub):
        parser.error(f"Required subdirectory missing: {val_sub}")

    if not args.model_path:
        default_model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(default_model_dir, exist_ok=True)
        args.model_path = os.path.join(default_model_dir, 'trained_model.pth')
        print(f"No --model_path provided. Defaulting to: {args.model_path}", flush=True)

    model_dir = os.path.dirname(args.model_path) if os.path.dirname(args.model_path) else '.'
    train(data_root, args.model_name, args.epochs, args.lr, args.batch_size,
          args.model_path, model_dir)
