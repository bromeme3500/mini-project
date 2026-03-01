"""
Fine-Tuning Script for CSRNetCAN Crowd Counter

Fine-tune the CSRNet+CAN model on your own annotated crowd images
for significantly improved accuracy on your specific camera setup.

USAGE:
    python finetune.py --data_dir ./my_crowd_data --epochs 100

DATA FORMAT:
    your_data_dir/
    ├── images/          # Crowd images (jpg/png)
    │   ├── img_001.jpg
    │   ├── img_002.jpg
    │   └── ...
    └── annotations/     # JSON files with dot annotations
        ├── img_001.json
        ├── img_002.json
        └── ...

    Each JSON annotation file should contain a list of [x, y] head positions:
    {
        "points": [[x1, y1], [x2, y2], ...],
        "count": 42
    }

    The "count" field is optional (derived from len(points) if missing).
    Points represent the center of each person's head in pixel coordinates.

TIPS:
    - 100-200 annotated images is enough for good fine-tuning
    - Use images from your actual camera/setup for best results
    - Training takes ~30 min on CPU for 100 epoch, ~5 min on GPU
    - The best model is saved to weights/csrnet_can_finetuned.pth
"""

import os
import sys
import json
import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from cnn_counter import CSRNetCAN, CSRNet
from download_weights import get_weights_path


# ---------------------------------------------------------------------------
# Density Map Generation
# ---------------------------------------------------------------------------

def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    center = size // 2
    return np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))


def generate_density_map(image_shape, points, sigma=15, adaptive=True):
    """
    Generate a ground truth density map from dot annotations.

    Args:
        image_shape: (height, width) of the image
        points: List of [x, y] head positions
        sigma: Base Gaussian sigma (kernel spread)
        adaptive: If True, uses adaptive sigma based on nearest neighbor distance

    Returns:
        density_map: 2D numpy array where sum ≈ number of people
    """
    h, w = image_shape[:2]
    density = np.zeros((h, w), dtype=np.float32)

    if len(points) == 0:
        return density

    points = np.array(points, dtype=np.float32)

    if adaptive and len(points) > 3:
        # Adaptive sigma: use k-nearest-neighbor distance
        from scipy.spatial import KDTree
        tree = KDTree(points)
        distances, _ = tree.query(points, k=min(4, len(points)))
        # Average distance to 3 nearest neighbors
        if distances.ndim > 1:
            avg_distances = np.mean(distances[:, 1:], axis=1)
        else:
            avg_distances = np.full(len(points), sigma)
    else:
        avg_distances = np.full(len(points), sigma)

    for i, (x, y) in enumerate(points):
        x, y = int(round(x)), int(round(y))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        # Adaptive or fixed sigma
        if adaptive and len(points) > 3:
            s = max(1.0, avg_distances[i] * 0.3)  # Scale factor
        else:
            s = sigma

        # Kernel size (6 sigma to capture 99.7% of distribution)
        k_size = int(np.ceil(s * 6))
        if k_size % 2 == 0:
            k_size += 1
        k_size = max(3, k_size)

        kernel = gaussian_kernel(k_size, s)
        kernel = kernel / kernel.sum()  # Normalize so each point sums to 1

        # Place kernel centered at (x, y)
        half = k_size // 2
        y_start = max(0, y - half)
        y_end = min(h, y + half + 1)
        x_start = max(0, x - half)
        x_end = min(w, x + half + 1)

        ky_start = max(0, half - y)
        ky_end = k_size - max(0, (y + half + 1) - h)
        kx_start = max(0, half - x)
        kx_end = k_size - max(0, (x + half + 1) - w)

        density[y_start:y_end, x_start:x_end] += kernel[ky_start:ky_end, kx_start:kx_end]

    return density


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CrowdDataset(Dataset):
    """Dataset for crowd counting fine-tuning."""

    def __init__(self, data_dir, target_size=(512, 512)):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.annotations_dir = os.path.join(data_dir, "annotations")
        self.target_size = target_size

        # Find all image files
        self.image_files = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for f in sorted(os.listdir(self.images_dir)):
            if os.path.splitext(f)[1].lower() in valid_extensions:
                # Check matching annotation exists
                base = os.path.splitext(f)[0]
                ann_path = os.path.join(self.annotations_dir, f"{base}.json")
                if os.path.exists(ann_path):
                    self.image_files.append(f)
                else:
                    print(f"  Warning: No annotation for {f}, skipping")

        print(f"  Found {len(self.image_files)} annotated images")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        base = os.path.splitext(img_file)[0]

        # Load image
        img_path = os.path.join(self.images_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Load annotation
        ann_path = os.path.join(self.annotations_dir, f"{base}.json")
        with open(ann_path, "r") as f:
            annotation = json.load(f)
        points = annotation.get("points", [])

        # Resize image
        tw, th = self.target_size
        image_resized = image.resize((tw, th))

        # Scale points to resized dimensions
        scale_x = tw / orig_w
        scale_y = th / orig_h
        scaled_points = [[p[0] * scale_x, p[1] * scale_y] for p in points]

        # Generate density map at 1/8 resolution (after 3 pooling layers)
        density_h, density_w = th // 8, tw // 8
        density_points = [[p[0] / 8, p[1] / 8] for p in scaled_points]
        density_map = generate_density_map((density_h, density_w), density_points, sigma=2)

        # Scale density map so it sums to the original count
        # (Our density map is at 1/8 res, so the sum is already ~correct
        #  because we normalize each kernel to sum to 1)

        input_tensor = self.transform(image_resized)
        density_tensor = torch.from_numpy(density_map).unsqueeze(0).float()

        return input_tensor, density_tensor, len(points)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(args):
    print("=" * 60)
    print("  CSRNetCAN Fine-Tuning")
    print("=" * 60)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load dataset
    print(f"  Data directory: {args.data_dir}")
    dataset = CrowdDataset(args.data_dir, target_size=(args.img_size, args.img_size))

    if len(dataset) == 0:
        print("\n  ERROR: No annotated images found!")
        print("  Expected structure:")
        print("    data_dir/images/   - your crowd images")
        print("    data_dir/annotations/ - JSON dot annotations")
        sys.exit(1)

    # Split into train/val (80/20)
    n_val = max(1, len(dataset) // 5)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"  Train: {n_train} images, Val: {n_val} images")
    print(f"  Batch size: {args.batch_size}, Epochs: {args.epochs}")
    print()

    # Create model with pre-trained weights
    model = CSRNetCAN(load_weights=True)
    weights_path = get_weights_path()

    if weights_path and os.path.exists(weights_path):
        try:
            base_model = CSRNet(load_weights=True)
            try:
                checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(weights_path, map_location=device)

            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            base_model.load_state_dict(state_dict)

            model.frontend.load_state_dict(base_model.frontend.state_dict())
            model.backend.load_state_dict(base_model.backend.state_dict())
            print("  Loaded pre-trained CSRNet backbone")
            del base_model
        except Exception as e:
            print(f"  Warning: Could not load pre-trained weights: {e}")
            print("  Training from VGG-16 backbone only")

    # Check for existing fine-tuned weights
    finetuned_path = os.path.join(os.path.dirname(__file__), "weights", "csrnet_can_finetuned.pth")
    if os.path.exists(finetuned_path) and not args.from_scratch:
        try:
            try:
                state = torch.load(finetuned_path, map_location=device, weights_only=False)
            except TypeError:
                state = torch.load(finetuned_path, map_location=device)
            model.load_state_dict(state)
            print("  Resuming from existing fine-tuned weights")
        except Exception:
            print("  Could not load existing fine-tuned weights, starting fresh")

    model = model.to(device)

    # Optimizer: lower LR for pre-trained layers, higher for new CAN module
    pretrained_params = list(model.frontend.parameters()) + list(model.backend.parameters())
    new_params = list(model.context.parameters())

    optimizer = torch.optim.Adam([
        {"params": pretrained_params, "lr": args.lr * 0.1},  # Fine-tune slowly
        {"params": new_params, "lr": args.lr},                # Train CAN module faster
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()

    # Training
    best_mae = float('inf')
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    os.makedirs(weights_dir, exist_ok=True)

    print("  Training started...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0

        for batch_idx, (images, density_maps, counts) in enumerate(train_loader):
            images = images.to(device)
            density_maps = density_maps.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, density_maps)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for images, density_maps, counts in val_loader:
                images = images.to(device)
                output = model(images)
                predicted_count = output.sum().item()
                actual_count = counts[0].item()
                val_mae += abs(predicted_count - actual_count)

        val_mae /= len(val_loader)

        # Log
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.6f} | Val MAE: {val_mae:.1f}", end="")
            if val_mae < best_mae:
                print(" ★ Best!")
            else:
                print()

        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            save_path = os.path.join(weights_dir, "csrnet_can_finetuned.pth")
            torch.save(model.state_dict(), save_path)

    print("-" * 60)
    print(f"\n  Training complete!")
    print(f"  Best Validation MAE: {best_mae:.1f}")
    print(f"  Model saved to: {os.path.join(weights_dir, 'csrnet_can_finetuned.pth')}")
    print(f"\n  The server will automatically use these weights on next restart.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune CSRNetCAN for crowd counting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLE:
    python finetune.py --data_dir ./my_crowd_data --epochs 100

DATA FORMAT:
    data_dir/
    ├── images/          # Your crowd images
    │   ├── img_001.jpg
    │   └── ...
    └── annotations/     # JSON dot annotations
        ├── img_001.json  → {"points": [[x1,y1], [x2,y2], ...]}
        └── ...
"""
    )
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--img_size", type=int, default=512, help="Image resize dimension (default: 512)")
    parser.add_argument("--from_scratch", action="store_true", help="Train from scratch, ignore existing fine-tuned weights")

    args = parser.parse_args()
    train(args)
