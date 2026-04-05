import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torch.serialization
from sklearn.decomposition import PCA

from resnet_ibn import resnet50_ibn_a


# ========= 路径配置：改成你的本地路径 =========
PROJECT_ROOT = os.path.expanduser("~/Desktop/scene_node_project")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "raw_images", "image_train_scene")
CKPT_PATH = os.path.join(PROJECT_ROOT, "models", "best_resnet50_ibn.pth")
OUT_NPY = os.path.join(PROJECT_ROOT, "outputs", "scene_features.npy")

# ========= 参数 =========
IMG_SIZE = (256, 256)
PCA_DIM = 256
BATCH_PRINT = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model():
    model = resnet50_ibn_a(last_stride=1)

    # 兼容某些旧 checkpoint
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["state_dict"]

    # 去掉 module. 前缀
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)

    print("Model loaded.")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model = model.to(DEVICE)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def main():
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"Image dir not found: {IMAGE_DIR}")
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    os.makedirs(os.path.dirname(OUT_NPY), exist_ok=True)

    model = build_model()
    transform = get_transform()

    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"Found {len(image_files)} images in {IMAGE_DIR}")

    features = []

    for i, fname in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, fname)
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = model(x)   # [1, 2048]
            feat = feat.squeeze(0).cpu().numpy()

        features.append(feat)

        if (i + 1) % BATCH_PRINT == 0:
            print(f"Processed {i+1}/{len(image_files)} images")

    features = np.stack(features, axis=0)
    print("Original feature shape:", features.shape)

    n_samples, n_dims = features.shape
    if n_samples < PCA_DIM:
        raise ValueError(
            f"PCA requires at least {PCA_DIM} samples, but only {n_samples} were found."
        )

    print(f"Running PCA: {n_dims} -> {PCA_DIM}")
    pca = PCA(n_components=PCA_DIM, random_state=42)
    features_pca = pca.fit_transform(features)

    np.save(OUT_NPY, features_pca)

    print(f"Saved scene features to: {OUT_NPY}")
    print("Final scene_features shape:", features_pca.shape)
    print("Explained variance ratio sum:", pca.explained_variance_ratio_.sum())


if __name__ == "__main__":
    main()
    