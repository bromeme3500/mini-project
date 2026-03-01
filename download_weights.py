"""
Download pre-trained CSRNet weights for crowd counting.

Supports three datasets:
- ShanghaiTech Part A (dense crowds, concerts/protests)
- ShanghaiTech Part B (sparse crowds, surveillance-style)
- UCF-QNRF (diverse, high-resolution, most generalizable)

Source: https://github.com/leeyeehoo/CSRNet-pytorch
Paper: "CSRNet: Dilated Convolutional Neural Networks for Understanding
        the Highly Congested Scenes" (CVPR 2018)
"""

import os
import sys


# Weight configurations: dataset_key → (Google Drive file ID, filename, description)
WEIGHT_CONFIGS = {
    "sha": {
        "gdrive_id": "1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH",
        "filename": "csrnet_shanghai_a.pth.tar",
        "description": "ShanghaiTech Part A (dense crowds, MAE: 66.4)",
    },
    "shb": {
        "gdrive_id": "1fSMGDqAnU1GKYQ7ysFgF2Gdxd2LQ6B37",
        "filename": "csrnet_shanghai_b.pth.tar",
        "description": "ShanghaiTech Part B (sparse/surveillance, MAE: 10.6)",
    },
    "qnrf": {
        "gdrive_id": "1nnIHPaV9RGqo8wam1jcGl-tEpW6S1FaZ",
        "filename": "csrnet_ucf_qnrf.pth.tar",
        "description": "UCF-QNRF (diverse/high-res, most generalizable)",
    },
}

# Default dataset to use
DEFAULT_DATASET = "sha"

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")


def download_from_gdrive(file_id, dest_path):
    """Download a file from Google Drive using gdown (handles large files)."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown for Google Drive downloads...")
        os.system(f"{sys.executable} -m pip install gdown -q")
        import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    print(f"Downloading weights...")
    print(f"  Source: Google Drive ({file_id})")
    print(f"  Destination: {dest_path}")
    print()

    try:
        gdown.download(url, dest_path, quiet=False)
        if os.path.exists(dest_path):
            file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            if file_size_mb > 1:
                print(f"\nDownload complete! File size: {file_size_mb:.1f} MB")
                return True
            else:
                print(f"\nDownloaded file too small ({file_size_mb:.2f} MB) — likely a redirect page.")
                os.remove(dest_path)
                return False
        return False
    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print(f"\nPlease download manually:")
        print(f"  1. Go to: https://drive.google.com/uc?export=download&id={file_id}")
        print(f"  2. Save the file as: {dest_path}")
        return False


def get_weights_path(dataset=None):
    """
    Get the path to pre-trained weights. Downloads if not present.

    Args:
        dataset: One of "sha" (Part A), "shb" (Part B), "qnrf" (UCF-QNRF).
                 If None, uses DEFAULT_DATASET.
                 Can also be set via CROWD_WEIGHTS env var.

    Returns:
        Path to weights file, or None if download failed.
    """
    # Allow env var override
    if dataset is None:
        dataset = os.environ.get("CROWD_WEIGHTS", DEFAULT_DATASET).lower()

    if dataset not in WEIGHT_CONFIGS:
        print(f"[Weights] Unknown dataset '{dataset}'. Available: {list(WEIGHT_CONFIGS.keys())}")
        dataset = DEFAULT_DATASET

    config = WEIGHT_CONFIGS[dataset]
    weights_path = os.path.join(WEIGHTS_DIR, config["filename"])

    if os.path.exists(weights_path):
        file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        if file_size_mb > 1:  # Valid weight file should be > 1MB
            print(f"[Weights] Found: {config['description']} ({file_size_mb:.1f} MB)")
            return weights_path
        else:
            print(f"[Weights] File seems corrupted ({file_size_mb:.2f} MB). Re-downloading...")
            os.remove(weights_path)

    print(f"[Weights] Downloading: {config['description']}")
    success = download_from_gdrive(config["gdrive_id"], weights_path)
    return weights_path if success else None


def download_all():
    """Download all available weight files."""
    print("=" * 60)
    print("  Downloading all CSRNet weight files")
    print("=" * 60)
    for key, config in WEIGHT_CONFIGS.items():
        print(f"\n--- {config['description']} ---")
        get_weights_path(dataset=key)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download CSRNet pre-trained weights")
    parser.add_argument("--dataset", choices=list(WEIGHT_CONFIGS.keys()) + ["all"],
                        default=DEFAULT_DATASET,
                        help="Which weights to download (default: sha)")
    args = parser.parse_args()

    if args.dataset == "all":
        download_all()
    else:
        print("=" * 60)
        print(f"  CSRNet Weights: {WEIGHT_CONFIGS[args.dataset]['description']}")
        print("=" * 60)
        print()
        path = get_weights_path(dataset=args.dataset)
        if path:
            print(f"\nWeights ready at: {path}")
        else:
            print("\nFailed to download. Please download manually.")
