import pandas as pd
from pathlib import Path


def main():
    veri_root = Path("/Users/cuilinxing/Documents/capstone/VeRi")

    # 优先读取带路径版本；如果没有，就读取基础版本
    csv_with_paths = veri_root / "scene_nodes_metadata_with_paths.csv"
    csv_basic = veri_root / "scene_nodes_metadata.csv"

    if csv_with_paths.exists():
        input_csv = csv_with_paths
    elif csv_basic.exists():
        input_csv = csv_basic
    else:
        raise FileNotFoundError("没有找到 scene_nodes_metadata_with_paths.csv 或 scene_nodes_metadata.csv")

    print(f"Reading: {input_csv}")

    # 全部按字符串读，避免 0001 变成 1
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    # --------- 基础检查 ---------
    required_cols = [
        "scene_id", "image_name", "vehicle_id", "camera_id",
        "color_id", "color_name", "type_id", "type_name", "split"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少字段: {col}")

    # 如果路径列不存在，就补空列
    for col in ["scene_feature_path", "scene_image_path", "mask_path"]:
        if col not in df.columns:
            df[col] = ""

    # 检查 scene_id 是否唯一
    if df["scene_id"].duplicated().any():
        dup = df[df["scene_id"].duplicated(keep=False)]["scene_id"].tolist()
        raise ValueError(f"scene_id 不唯一，重复示例: {dup[:10]}")

    # 检查 image_name 是否唯一
    if df["image_name"].duplicated().any():
        dup = df[df["image_name"].duplicated(keep=False)]["image_name"].tolist()
        raise ValueError(f"image_name 不唯一，重复示例: {dup[:10]}")

    # --------- 生成 scene_idx 和 image_idx ---------
    # 当前第一版：1 image 对应 1 scene，所以直接按行分配索引
    df = df.reset_index(drop=True)
    df["scene_idx"] = df.index.astype(int)
    df["image_idx"] = df.index.astype(int)

    # --------- scene_info.csv ---------
    scene_info = df[[
        "scene_idx",
        "scene_id",
        "image_name",
        "camera_id",
        "split",
        "scene_image_path",
        "mask_path",
        "scene_feature_path",
    ]].copy()

    scene_info_path = veri_root / "scene_info.csv"
    scene_info.to_csv(scene_info_path, index=False, encoding="utf-8")

    # --------- image_info.csv ---------
    image_info = df[[
        "image_idx",
        "image_name",
        "vehicle_id",
        "camera_id",
        "color_id",
        "color_name",
        "type_id",
        "type_name",
        "split",
        "scene_id",
        "scene_idx",
    ]].copy()

    # 列名统一一下
    image_info = image_info.rename(columns={
        "image_name": "file_name"
    })

    image_info_path = veri_root / "image_info.csv"
    image_info.to_csv(image_info_path, index=False, encoding="utf-8")

    # --------- image_scene_edges.csv ---------
    image_scene_edges = df[["image_idx", "scene_idx"]].copy()
    image_scene_edges = image_scene_edges.rename(columns={
        "image_idx": "source_image_idx",
        "scene_idx": "target_scene_idx"
    })

    image_scene_edges_path = veri_root / "image_scene_edges.csv"
    image_scene_edges.to_csv(image_scene_edges_path, index=False, encoding="utf-8")

    # --------- 输出信息 ---------
    print(f"Saved: {scene_info_path}")
    print(f"Saved: {image_info_path}")
    print(f"Saved: {image_scene_edges_path}")
    print()
    print(f"Total rows: {len(df)}")
    print()
    print("scene_info preview:")
    print(scene_info.head(3))
    print()
    print("image_info preview:")
    print(image_info.head(3))
    print()
    print("image_scene_edges preview:")
    print(image_scene_edges.head(3))


if __name__ == "__main__":
    main()