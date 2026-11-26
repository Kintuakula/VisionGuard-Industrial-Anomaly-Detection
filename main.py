import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ImageFolderDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path)


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def image_paths(folder):
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def build_model():
    # Uses ImageNet pretrained weights as a fixed feature extractor.
    weights = models.ResNet18_Weights.DEFAULT
    net = models.resnet18(weights=weights)
    net.eval().to(DEVICE)
    for p in net.parameters():
        p.requires_grad = False
    return net


@torch.no_grad()
def extract_features(net, batch):
    # Combine two intermediate feature levels to keep local texture/shape information.
    x = net.conv1(batch)
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)
    x = net.layer1(x)
    f2 = net.layer2(x)
    f3 = net.layer3(f2)
    f2 = F.interpolate(f2, size=f3.shape[-2:], mode="bilinear", align_corners=False)
    features = torch.cat([f2, f3], dim=1)
    features = F.normalize(features, p=2, dim=1)
    return features


def collect_normal_paths(data_dir):
    paths = image_paths(data_dir / "train" / "good")
    if not paths:
        raise FileNotFoundError(
            f"No normal training images found in: {data_dir / 'train' / 'good'}\n"
            "Check that the category folder follows the MVTec AD structure."
        )
    return paths


def train(args):
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    normal_paths = collect_normal_paths(data_dir)
    if args.max_train_images > 0:
        normal_paths = normal_paths[:args.max_train_images]

    ds = ImageFolderDataset(normal_paths, get_transform())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    net = build_model()
    all_features = []
    print(f"Device: {DEVICE}")
    print(f"Building memory from {len(ds)} normal training images...")

    for batch, _ in tqdm(loader, desc="Extracting features"):
        batch = batch.to(DEVICE)
        feats = extract_features(net, batch)
        # [B,C,H,W] -> [B*H*W,C]
        patches = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])
        all_features.append(patches.cpu())

    memory = torch.cat(all_features, dim=0)
    # Keep the bank manageable by uniform subsampling if it is very large.
    max_bank = args.max_memory_patches
    if max_bank > 0 and memory.shape[0] > max_bank:
        indices = torch.randperm(memory.shape[0])[:max_bank]
        memory = memory[indices]

    save_path = output_dir / "memory_bank.pt"
    torch.save({"memory": memory, "category_path": str(data_dir)}, save_path)
    print(f"Saved {memory.shape[0]} normal feature patches to: {save_path}")
    print("Training/memory-bank creation complete.")


def load_ground_truth(data_dir, image_path, image_size):
    # MVTec test image path: category/test/defect_type/file.png
    try:
        relative = image_path.relative_to(data_dir / "test")
        defect_type = relative.parts[0]
        if defect_type == "good":
            return np.zeros((image_size, image_size), dtype=np.uint8)
        mask_path = data_dir / "ground_truth" / defect_type / (image_path.stem + "_mask.png")
        if not mask_path.exists():
            return None
        mask = Image.open(mask_path).convert("L").resize(
            (image_size, image_size), Image.Resampling.NEAREST)
        return (np.asarray(mask) > 0).astype(np.uint8)
    except (ValueError, IndexError):
        return None


@torch.no_grad()
def score_batch(net, memory, batch):
    features = extract_features(net, batch)
    b, c, h, w = features.shape
    query = features.permute(0, 2, 3, 1).reshape(-1, c).cpu()
    # Chunk queries to avoid excessive RAM use.
    nearest = []
    chunk_size = 2048
    for start in range(0, query.shape[0], chunk_size):
        q = query[start:start + chunk_size]
        # torch.cdist is supported on CPU and avoids requiring a database.
        distances = torch.cdist(q, memory)
        nearest.append(distances.min(dim=1).values)
    distances = torch.cat(nearest).reshape(b, h, w)
    heatmaps = F.interpolate(distances.unsqueeze(1).to(DEVICE),
                             size=(IMAGE_SIZE, IMAGE_SIZE),
                             mode="bilinear", align_corners=False).squeeze(1)
    image_scores = heatmaps.flatten(1).topk(
        max(1, int(heatmaps.shape[-1] * heatmaps.shape[-2] * 0.01)),
        dim=1).values.mean(dim=1)
    return image_scores.cpu().numpy(), heatmaps.cpu().numpy()


def save_visualization(image_path, heatmap, score, prediction, out_path):
    original = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original image")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Anomaly heatmap")
    axes[2].imshow(original)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[2].set_title(f"{prediction}\\nScore: {score:.4f}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def test(args):
    data_dir = Path(args.data)
    output_dir = Path(args.output)
    memory_path = output_dir / "memory_bank.pt"
    if not memory_path.exists():
        raise FileNotFoundError(
            f"Memory bank not found: {memory_path}\nRun the train command first."
        )

    memory = torch.load(memory_path, map_location="cpu", weights_only=False)["memory"]
    memory = F.normalize(memory.float(), p=2, dim=1)
    net = build_model()

    test_root = data_dir / "test"
    paths = image_paths(test_root)
    if not paths:
        raise FileNotFoundError(f"No test images found under {test_root}")

    ds = ImageFolderDataset(paths, get_transform())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    image_scores, image_labels = [], []
    pixel_scores, pixel_labels = [], []
    threshold = args.threshold
    index = 0
    print(f"Device: {DEVICE}")
    print(f"Scoring {len(paths)} test images...")

    for batch, batch_paths in tqdm(loader, desc="Testing"):
        scores, maps = score_batch(net, memory, batch.to(DEVICE))
        for j, path_string in enumerate(batch_paths):
            path = Path(path_string)
            defect_type = path.relative_to(test_root).parts[0]
            label = 0 if defect_type == "good" else 1
            score = float(scores[j])
            prediction = "Anomalous" if score >= threshold else "Normal"
            out_path = result_dir / f"{index:04d}_{path.stem}_result.png"
            save_visualization(path, maps[j], score, prediction, out_path)
            image_scores.append(score)
            image_labels.append(label)

            mask = load_ground_truth(data_dir, path, IMAGE_SIZE)
            if mask is not None:
                pixel_scores.extend(maps[j].reshape(-1).tolist())
                pixel_labels.extend(mask.reshape(-1).tolist())
            index += 1

    if len(set(image_labels)) == 2:
        print(f"Image-level AUROC: {roc_auc_score(image_labels, image_scores):.4f}")
    else:
        print("Image-level AUROC skipped: test set needs both normal and anomalous images.")

    if pixel_labels and len(set(pixel_labels)) == 2:
        print(f"Pixel-level AUROC: {roc_auc_score(pixel_labels, pixel_scores):.4f}")
    else:
        print("Pixel-level AUROC skipped: usable ground-truth masks need both pixel classes.")

    print(f"Saved visual results to: {result_dir}")
    print("Note: the default threshold is a starter setting; calibrate it using a proper validation protocol.")


def main():
    parser = argparse.ArgumentParser(
        description="PatchCore-style industrial anomaly detection starter")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Build memory bank from normal images only")
    p_train.add_argument("--data", required=True, help="Path to one MVTec category, e.g. data/MVTec_AD/bottle")
    p_train.add_argument("--output", default="outputs/bottle")
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--max-train-images", type=int, default=0,
                         help="0 means use all normal training images")
    p_train.add_argument("--max-memory-patches", type=int, default=20000,
                         help="Maximum number of stored normal patches; 0 means all")
    p_train.set_defaults(func=train)

    p_test = sub.add_parser("test", help="Evaluate test images and save heatmaps")
    p_test.add_argument("--data", required=True, help="Path to one MVTec category")
    p_test.add_argument("--output", default="outputs/bottle")
    p_test.add_argument("--batch-size", type=int, default=2)
    p_test.add_argument("--threshold", type=float, default=0.5,
                        help="Starter anomaly-score threshold, not calibrated")
    p_test.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
