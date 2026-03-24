"""
CNN-based Crowd Density Estimator Module
Uses CSRNet with Context-Aware Module (CAN) for improved accuracy.

Base Architecture: CSRNet (CVPR 2018)
Enhancement: Context-Aware Module inspired by CAN (CVPR 2019)
Features: Multi-scale inference, higher resolution input

Paper: "CSRNet: Dilated Convolutional Neural Networks for Understanding
        the Highly Congested Scenes" (CVPR 2018)
Paper: "Context-Aware Crowd Counting" (CVPR 2019)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from download_weights import get_weights_path


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """Build VGG-style layers from config list, matching original CSRNet."""
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class ContextAwareModule(nn.Module):
    """
    Multi-scale context module inspired by CAN (CVPR 2019).

    Applies parallel dilated convolutions at different rates to capture
    context at multiple scales, then fuses them. This helps the model
    understand crowd density at various distances from the camera.
    """

    def __init__(self, in_channels=64):
        super().__init__()
        mid_channels = in_channels // 2

        # Three parallel branches with different dilation rates
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=3, dilation=3),
            nn.ReLU(inplace=True),
        )

        # Fusion: concatenate 3 branches → 1x1 conv → density output
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 3, in_channels, 1),
            nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(in_channels, 1, kernel_size=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        fused = self.fuse(torch.cat([b1, b2, b3], dim=1))
        return self.output_layer(fused)


class CSRNet(nn.Module):
    """
    CSRNet matching the original paper's architecture.

    Frontend: VGG-16 first 10 conv layers (features[:23], up through pool3)
    Backend: 6 dilated conv layers (dilation=2) producing density map
    Output: 1x1 conv producing single-channel density map
    """

    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0

        # Frontend: VGG-16 features up to conv4_3 + pool3 (first 23 layers)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            # Initialize with VGG-16 pre-trained weights for frontend
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self._initialize_weights()
            # Copy VGG frontend weights
            vgg_items = list(vgg.state_dict().items())
            frontend_items = list(self.frontend.state_dict().items())
            for i in range(len(frontend_items)):
                frontend_items[i] = (frontend_items[i][0], vgg_items[i][1])
            self.frontend.load_state_dict(dict(frontend_items))

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class CSRNetCAN(nn.Module):
    """
    CSRNet + Context-Aware Module for improved multi-scale accuracy.

    Uses the same CSRNet frontend + backend, but replaces the simple
    1x1 output conv with a multi-scale context-aware module that
    captures density patterns at different scales.
    """

    def __init__(self, load_weights=False):
        super().__init__()
        self.seen = 0

        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.context = ContextAwareModule(in_channels=64)

        if not load_weights:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self._initialize_weights()
            vgg_items = list(vgg.state_dict().items())
            frontend_items = list(self.frontend.state_dict().items())
            for i in range(len(frontend_items)):
                frontend_items[i] = (frontend_items[i][0], vgg_items[i][1])
            self.frontend.load_state_dict(dict(frontend_items))

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.context(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Counter Wrapper
# ---------------------------------------------------------------------------

class CNNCounter:
    """
    Wrapper class for CSRNet/CSRNetCAN crowd counting.

    Features:
    - Loads pre-trained CSRNet weights then bolts on CAN context module
    - Multi-scale inference for better accuracy (GPU) or fast single-scale (CPU)
    - Adaptive resolution based on device
    """

    # Maximum resolution for CNN input
    # 640 is plenty for density estimation and significantly faster
    MAX_INPUT_WIDTH = 640

    # Multi-scale inference settings (2 scales for speed)
    DEFAULT_SCALES = [0.9, 1.1]

    def __init__(self):
        """
        Initialize the CNN model.

        Uses standard CSRNet with pre-trained weights by default (reliable counts).
        Only upgrades to CSRNetCAN if fine-tuned CAN weights are available,
        since the CAN context module needs training to produce meaningful output.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_loaded = False

        # Try to load pre-trained CSRNet weights
        weights_path = get_weights_path()

        if weights_path and os.path.exists(weights_path):
            try:
                # Load standard CSRNet with pre-trained weights
                self.model = CSRNet(load_weights=True)
                try:
                    checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
                except TypeError:
                    checkpoint = torch.load(weights_path, map_location=self.device)

                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                self.model.load_state_dict(state_dict)

                self.pretrained_loaded = True
                print("[CNNCounter] Loaded pre-trained CSRNet weights")

            except Exception as e:
                print(f"[CNNCounter] Failed to load pre-trained weights: {e}")
                import traceback
                traceback.print_exc()
                print("[CNNCounter] Falling back to VGG-16 backbone only")
                self.model = CSRNet(load_weights=False)
        else:
            print("[CNNCounter] Pre-trained weights not available, using VGG-16 backbone only")
            self.model = CSRNet(load_weights=False)

        # Check if user has fine-tuned CAN weights — only then use CSRNetCAN
        # (The CAN module needs training; random weights produce garbage output)
        can_weights_path = os.path.join(os.path.dirname(__file__), "weights", "csrnet_can_finetuned.pth")
        if os.path.exists(can_weights_path):
            try:
                can_model = CSRNetCAN(load_weights=True)
                try:
                    can_state = torch.load(can_weights_path, map_location=self.device, weights_only=False)
                except TypeError:
                    can_state = torch.load(can_weights_path, map_location=self.device)
                can_model.load_state_dict(can_state)
                self.model = can_model  # Upgrade to CAN model
                self.pretrained_loaded = True
                print("[CNNCounter] Upgraded to fine-tuned CSRNetCAN model!")
            except Exception as e:
                print(f"[CNNCounter] Could not load fine-tuned CAN weights: {e}")
                print("[CNNCounter] Keeping standard CSRNet model")

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.device.type == "cuda":
            # Massive speedup for real-time execution by using half precision
            self.model = self.model.half()

        # Force single-scale inference even on GPU for real-time video speed
        self.use_multiscale = False
        
        if self.device.type == "cpu":
            print("[CNNCounter] CPU detected — using single-scale inference for speed")
            # Also reduce resolution further on CPU for real-time performance
            self.MAX_INPUT_WIDTH = 768
        else:
            print("[CNNCounter] GPU detected — using FP16 single-scale inference for real-time speed")


        # Image preprocessing (same as original CSRNet training)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _prepare_input(self, frame, scale=1.0):
        """
        Prepare a frame for model input at the given scale - OPTIMIZED version.
        """
        original_h, original_w = frame.shape[:2]
        target_w = min(original_w, self.MAX_INPUT_WIDTH)
        target_h = int(original_h * (target_w / original_w))

        # Apply scale factor
        target_w = int(target_w * scale)
        target_h = int(target_h * scale)

        # Make dimensions divisible by 8 for the 3 pooling layers
        target_w = max((target_w // 8) * 8, 8)
        target_h = max((target_h // 8) * 8, 8)

        # FAST PREPROCESSING: Use OpenCV instead of PIL
        resized = cv2.resize(frame, (target_w, target_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize (0-1)
        img_data = rgb.astype(np.float32) / 255.0
        
        # Mean/Std normalization (matching ImageNet stats manually for SPEED)
        img_data -= np.array([0.485, 0.456, 0.406])
        img_data /= np.array([0.229, 0.224, 0.225])
        
        # HWC to CHW
        img_data = img_data.transpose(2, 0, 1)
        
        # Create tensor directly
        input_tensor = torch.from_numpy(img_data).unsqueeze(0).to(self.device)
        
        if self.device.type == "cuda":
            input_tensor = input_tensor.half()

        return input_tensor, (target_w, target_h)

    def _run_inference(self, input_tensor):
        """Run model inference and return the density map."""
        with torch.no_grad():
            density_map = self.model(input_tensor)
        return density_map.squeeze().cpu().numpy()

    def count(self, frame):
        """
        Estimate crowd count from a frame using single-scale density estimation.

        Args:
            frame: BGR image (numpy array from OpenCV)

        Returns:
            tuple: (estimated_count, density_heatmap_overlay)
        """
        input_tensor, _ = self._prepare_input(frame, scale=1.0)
        density = self._run_inference(input_tensor)

        # Density map values sum to the estimated count
        density = np.maximum(density, 0)
        estimated_count = max(0, int(round(density.sum())))

        # Create density heatmap overlay
        density_overlay = self._create_heatmap_overlay(frame, density)

        return estimated_count, density_overlay

    def count_multiscale(self, frame, scales=None):
        """
        Estimate crowd count using multi-scale inference for better accuracy.

        Runs the model at multiple scales, normalizes each count by the
        scale factor squared (area), and returns the average. This reduces
        sensitivity to the scale at which people appear in the image.

        Args:
            frame: BGR image (numpy array from OpenCV)
            scales: List of scale factors (default: [0.8, 1.0, 1.2])

        Returns:
            tuple: (estimated_count, density_heatmap_overlay)
        """
        if scales is None:
            scales = self.DEFAULT_SCALES

        counts = []
        density_at_1x = None

        for scale in scales:
            input_tensor, _ = self._prepare_input(frame, scale=scale)
            density = self._run_inference(input_tensor)
            density = np.maximum(density, 0)

            # Normalize count by scale^2 (area ratio)
            raw_count = density.sum()
            normalized_count = raw_count / (scale * scale)
            counts.append(normalized_count)

            # Keep the 1.0 scale density map for visualization
            if abs(scale - 1.0) < 0.01:
                density_at_1x = density

        estimated_count = max(0, int(round(np.mean(counts))))

        # Use 1x density for heatmap (or fallback to last scale)
        if density_at_1x is None:
            density_at_1x = density

        density_overlay = self._create_heatmap_overlay(frame, density_at_1x)

        return estimated_count, density_overlay

    def _create_heatmap_overlay(self, frame, density_map):
        """
        Create a color heatmap overlay on the original frame.

        Args:
            frame: Original BGR frame
            density_map: 2D density map from the model

        Returns:
            frame with heatmap overlay
        """
        h, w = frame.shape[:2]

        # Normalize density map to 0-255
        abs_density = np.abs(density_map)
        if abs_density.max() > 0:
            density_normalized = (abs_density / abs_density.max() * 255).astype(np.uint8)
        else:
            density_normalized = np.zeros_like(density_map, dtype=np.uint8)

        # Resize density map to match frame size
        density_resized = cv2.resize(density_normalized, (w, h))

        # Apply color map (JET: blue=low, red=high density)
        heatmap = cv2.applyColorMap(density_resized, cv2.COLORMAP_JET)

        # Blend with original frame
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        return overlay
