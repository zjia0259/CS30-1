import numpy as np
import pandas as pd
from pathlib import Path


def load_raw_camera_dist(txt_path):
    """
    读取 camera_Dist.txt
    文件看起来是一个“上三角/半完整矩阵”，下半部分用 0 占位。
    我们先按原样读成矩阵，再用 max(mat, mat.T) 补成对称矩阵。
    """
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            nums = [int(x) for x in line.split()]
            rows.append(nums)

    # 找最大列数
    max_len = max(len(r) for r in rows)
    n = len(rows)

    # 先按原样放进矩阵
    mat = np.zeros((n, max_len), dtype=np.int32)
    for i, row in enumerate(rows):
        mat[i, :len(row)] = row

    # 如果不是方阵，裁成 n x n
    mat = mat[:, :n]

    # 用上三角和下三角互补成对称矩阵
    sym_mat = np.maximum(mat, mat.T)

    # 对角线保持 0
    np.fill_diagonal(sym_mat, 0)

    return sym_mat


def main():
    veri_root = Path("/Users/cuilinxing/Documents/capstone/VeRi")

    camera_info_path = veri_root / "camera_info.csv"
    camera_dist_txt = veri_root / "camera_Dist.txt"

    if not camera_info_path.exists():
        raise FileNotFoundError(f"找不到 {camera_info_path}")
    if not camera_dist_txt.exists():
        raise FileNotFoundError(f"找不到 {camera_dist_txt}")

    camera_info = pd.read_csv(camera_info_path)
    raw_dist = load_raw_camera_dist(camera_dist_txt)

    # 按 camera_info 里的 dist_row_idx 顺序重排
    dist_row_indices = camera_info["dist_row_idx"].tolist()
    reordered = raw_dist[np.ix_(dist_row_indices, dist_row_indices)]

    # 保存矩阵
    matrix_path = veri_root / "camera_dist_matrix.npy"
    np.save(matrix_path, reordered)

    # 生成边表
    edges = []
    n = reordered.shape[0]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dist = int(reordered[i, j])

            # 跳过无效的 0 距离边
            if dist == 0:
                continue

            edges.append({
                "source_camera_idx": i,
                "target_camera_idx": j,
                "distance": dist
            })

    camera_camera_edges = pd.DataFrame(edges)
    edges_path = veri_root / "camera_camera_edges.csv"
    camera_camera_edges.to_csv(edges_path, index=False, encoding="utf-8")

    print(f"Saved: {matrix_path}")
    print(f"Saved: {edges_path}")
    print()
    print("camera_dist_matrix shape:", reordered.shape)
    print()
    print("camera_dist_matrix preview:")
    print(reordered[:5, :5])
    print()
    print("camera_camera_edges preview:")
    print(camera_camera_edges.head(10))


if __name__ == "__main__":
    main()