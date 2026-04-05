import pandas as pd
from pathlib import Path


def load_camera_order(camera_id_txt_path):
    """
    camera_ID.txt 内容类似：
    7 5 8 4 9 3 10 6 11 2 1 13
    12 18 17 19 20 15 14 16

    读取后转成：
    ['c007', 'c005', 'c008', ...]
    """
    numbers = []
    with open(camera_id_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            for p in parts:
                if p.isdigit():
                    numbers.append(int(p))

    camera_ids = [f"c{n:03d}" for n in numbers]
    return camera_ids


def main():
    veri_root = Path("/Users/cuilinxing/Documents/capstone/VeRi")

    image_info_path = veri_root / "image_info.csv"
    camera_id_txt = veri_root / "camera_ID.txt"

    if not image_info_path.exists():
        raise FileNotFoundError(f"找不到 {image_info_path}")

    if not camera_id_txt.exists():
        raise FileNotFoundError(f"找不到 {camera_id_txt}")

    # 读 image_info
    image_info = pd.read_csv(image_info_path, dtype=str).fillna("")

    if "image_idx" not in image_info.columns or "camera_id" not in image_info.columns:
        raise ValueError("image_info.csv 里必须包含 image_idx 和 camera_id")

    # 按 camera_ID.txt 的顺序建立官方 camera 列表
    camera_order = load_camera_order(camera_id_txt)

    # 数据里实际用到的 camera_id
    used_cameras = set(image_info["camera_id"].unique())

    # 先保留官方顺序里实际出现的 camera
    ordered_used_cameras = [cid for cid in camera_order if cid in used_cameras]

    # 如果 image_info 里有 camera_ID.txt 里没出现的，也补到最后
    remaining_cameras = sorted(list(used_cameras - set(ordered_used_cameras)))
    final_cameras = ordered_used_cameras + remaining_cameras

    # 构建 camera_info
    camera_rows = []
    camera_id_to_idx = {}

    for idx, camera_id in enumerate(final_cameras):
        camera_id_to_idx[camera_id] = idx
        # dist_row_idx 尽量对应 camera_ID.txt 的原顺序
        if camera_id in camera_order:
            dist_row_idx = camera_order.index(camera_id)
        else:
            dist_row_idx = -1

        camera_rows.append({
            "camera_idx": idx,
            "camera_id": camera_id,
            "dist_row_idx": dist_row_idx
        })

    camera_info = pd.DataFrame(camera_rows)
    camera_info_path = veri_root / "camera_info.csv"
    camera_info.to_csv(camera_info_path, index=False, encoding="utf-8")

    # 构建 image_camera_edges
    image_camera_edges = image_info[["image_idx", "camera_id"]].copy()
    image_camera_edges["camera_idx"] = image_camera_edges["camera_id"].map(camera_id_to_idx)

    if image_camera_edges["camera_idx"].isnull().any():
        bad = image_camera_edges[image_camera_edges["camera_idx"].isnull()]
        raise ValueError(f"有 camera_id 没映射成功，例如: {bad.head(5)}")

    image_camera_edges = image_camera_edges.rename(columns={
        "image_idx": "source_image_idx",
        "camera_idx": "target_camera_idx"
    })[["source_image_idx", "target_camera_idx"]]

    image_camera_edges_path = veri_root / "image_camera_edges.csv"
    image_camera_edges.to_csv(image_camera_edges_path, index=False, encoding="utf-8")

    print(f"Saved: {camera_info_path}")
    print(f"Saved: {image_camera_edges_path}")
    print()
    print("camera_info preview:")
    print(camera_info.head(10))
    print()
    print("image_camera_edges preview:")
    print(image_camera_edges.head(10))


if __name__ == "__main__":
    main()