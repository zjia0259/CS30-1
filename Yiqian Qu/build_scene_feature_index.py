import os
import pandas as pd

# 改成你的本机项目路径
BASE = os.path.expanduser("~/Desktop/scene_node_project")

IMAGE_DIR = os.path.join(BASE, "raw_images", "image_train_scene")
FEATURE_PATH = os.path.join(BASE, "outputs", "scene_features.npy")
OUT_CSV = os.path.join(BASE, "outputs", "scene_feature_index.csv")

def main():
    # 和提特征时保持完全一致的排序规则
    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    df = pd.DataFrame({
        "feature_idx": list(range(len(image_files))),
        "file_name": image_files,
        "image_path": [os.path.join(IMAGE_DIR, f) for f in image_files]
    })

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    main()