import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
import cv2
import os

# -----------------------------
# Re-ID Model Configuration
# -----------------------------
PRIMARY_MODEL = 'osnet_x1_0_imagenet.pth'
FEATURE_DIM = 512       # Output feature embedding dimension
INPUT_SIZE = (224, 224) # Required input resolution for the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    import osnet
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        import osnet
    except ImportError:
        raise ImportError("osnet.py not found in the project directory.")

# -----------------------------
# Advanced Preprocessing
# -----------------------------
class LetterboxResize(object):
    """Resizes image while keeping aspect ratio to prevent distortion."""
    def __init__(self, size, fill_color=(128, 128, 128)):
        self.size = size
        self.fill_color = fill_color

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        target_h, target_w = self.size
        w, h = img.size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.BICUBIC)
        new_img = Image.new("RGB", (target_w, target_h), self.fill_color)
        top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
        new_img.paste(img_resized, (left, top))
        return new_img

# Standard ImageNet normalization
transform = T.Compose([
    LetterboxResize(INPUT_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Model Initialization
# -----------------------------
def load_reid_encoder():
    """
    Initializes OSNet and loads weights from the local directory.
    """
    try:
        logger_name = "ReID"
        print(f"{logger_name}: Loading OSNet with local weights...")

        # Build model architecture (classifier head is ignored during feature extraction)
        model = osnet.osnet_x1_0(num_classes=1000, pretrained=False)

        # Expected weight filename
        weight_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "osnet_x1_0_imagenet.pth"
        )

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        # Load weights partially to support different model variants if needed
        state_dict = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        model.to(DEVICE)

        print(f"{logger_name}: OSNet loaded successfully on {DEVICE}")
        return model

    except Exception as e:
        print(f"ReID ERROR: Failed to load encoder: {e}")
        return None

# Global instance for thread-safe access
reid_encoder = load_reid_encoder()

# -----------------------------
# Feature Extraction Logic
# -----------------------------
def get_embedding(img: np.ndarray) -> np.ndarray:
    """
    Extracts a normalized 512-dim feature vector from a vehicle crop.
    """
    if reid_encoder is None or img is None or img.size == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    try:
        # Preprocessing runs on CPU
        x = transform(img).unsqueeze(0).to(DEVICE)

        # Inference runs on designated device (GPU or CPU)
        with torch.no_grad():
            f = reid_encoder.forward(x)
            if isinstance(f, tuple):
                f = f[0]

        # Flatten and normalize
        f = f.cpu().numpy().reshape(-1).astype("float32")
        norm = np.linalg.norm(f)
        if norm > 1e-9:
            f /= norm

        return f

    except Exception as e:
        print(f"ReID: Extraction failed: {e}")
        return np.zeros(FEATURE_DIM, dtype=np.float32)
