import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from train import PlantDiseaseDataset, get_model
import os
import json
import config
from collections import defaultdict
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= CLASSIFICATION REPORT IMAGE =================
def save_classification_report_image(report_dict, class_names, output_path, font_size=20):
    fig, ax = plt.subplots(figsize=(12, len(class_names) * 0.6 + 3))
    ax.axis("off")

    headers = ["precision", "recall", "f1-score", "support"]
    table_data = []

    for cls in class_names:
        row = report_dict[cls]
        table_data.append([
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1-score']:.3f}",
            int(row['support'])
        ])

    table = ax.table(
        cellText=table_data,
        rowLabels=class_names,
        colLabels=headers,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ================= EVALUATION =================
def evaluate(data_dir, model_name, model_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])
 
    dataset = PlantDiseaseDataset(data_dir=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=config.TrainingConfig.BATCH_SIZE, shuffle=False)

    split_name = os.path.basename(os.path.normpath(data_dir))

    model_root = os.path.splitext(model_path)[0]
    output_dir = os.path.join(model_root, "eval", split_name)
    os.makedirs(output_dir, exist_ok=True)

    class_to_idx_path = os.path.splitext(model_path)[0] + "_class_to_idx.json"
    if not os.path.exists(class_to_idx_path):
        print("Class mapping not found.")
        return

    with open(class_to_idx_path, "r") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = get_model(
        model_name,
        len(class_names),
        base_weights_path=config.TrainingConfig.BASE_MODEL_PATH
    )
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []
    misclassified_data = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            mask = preds != labels
            if mask.any():
                for img, gt, pr in zip(images[mask], labels[mask], preds[mask]):
                    misclassified_data.append((img.cpu(), gt.item(), pr.item()))

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ================= CLASSIFICATION REPORT =================
    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True
    )

    txt_path = os.path.join(output_dir, f"classification_report_{split_name}.txt")
    with open(txt_path, "w") as f:
        f.write(report_text)

    img_path = os.path.join(output_dir, f"classification_report_{split_name}.png")
    save_classification_report_image(
        report_dict,
        class_names,
        img_path,
        font_size=22
    )

    # ================= CONFUSION MATRIX =================
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(18, 12))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 14})
    plt.title(f"Confusion Matrix ({split_name})", fontsize=18)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    cm_path = os.path.join(output_dir, f"confusion_matrix_{split_name}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ================= ERROR METRICS =================
    error_metrics = []
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        error_metrics.append({
            "Class": cls,
            "False Positives": fp,
            "False Negatives": fn,
            "Precision": tp / (tp + fp) if tp + fp else 0,
            "Recall": tp / (tp + fn) if tp + fn else 0
        })

    error_df = pd.DataFrame(error_metrics)
    error_csv = os.path.join(output_dir, f"error_analysis_metrics_{split_name}.csv")
    error_df.to_csv(error_csv, index=False)

    # ================= MISCLASSIFIED IMAGES =================
    save_misclassified_samples(
        misclassified_data,
        idx_to_class,
        os.path.join(output_dir, "misclassified")
    )

    # ================= VISUALIZATION =================
    visualize_predictions(
        model,
        loader,
        idx_to_class,
        output_dir,
        DEVICE
    )


def save_misclassified_samples(misclassified_data, idx_to_class, output_dir, max_images=50):
    os.makedirs(output_dir, exist_ok=True)

    mean = torch.tensor(config.MEAN).view(3, 1, 1)
    std = torch.tensor(config.STD).view(3, 1, 1)

    for i, (img, gt, pr) in enumerate(misclassified_data[:max_images]):
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        fname = f"err_{i}_Actual_{idx_to_class[gt]}_Pred_{idx_to_class[pr]}.jpg"
        cv2.imwrite(os.path.join(output_dir, fname), img)


def visualize_predictions(model, loader, idx_to_class, output_dir, device, num_samples_per_class=4):
    images_by_class = defaultdict(list)

    for images, labels in loader:
        for i in range(len(labels)):
            images_by_class[labels[i].item()].append(images[i])

    fig, axes = plt.subplots(
        nrows=len(images_by_class),
        ncols=num_samples_per_class,
        figsize=(num_samples_per_class * 3, len(images_by_class) * 3.5)
    )

    for row, cls in enumerate(images_by_class):
        samples = images_by_class[cls][:num_samples_per_class]
        batch = torch.stack(samples).to(device)

        with torch.no_grad():
            preds = torch.argmax(model(batch), 1)

        for col in range(len(samples)):
            ax = axes[row, col]
            img = samples[col].cpu()
            img = img * torch.tensor(config.STD).view(3, 1, 1) + torch.tensor(config.MEAN).view(3, 1, 1)
            img = img.permute(1, 2, 0).clamp(0, 1)

            ax.imshow(img)
            ax.axis("off")

            color = "green" if preds[col].item() == cls else "red"
            ax.set_title(
                f"Pred: {idx_to_class[preds[col].item()]}\nActual: {idx_to_class[cls]}",
                color=color,
                fontsize=10
            )

    out_path = os.path.join(output_dir, f"{os.path.basename(loader.dataset.data_dir)}_class_predictions.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_name", default=config.TrainingConfig.MODEL_NAME)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    evaluate(args.data_dir, args.model_name, args.model_path)
