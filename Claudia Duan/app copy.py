# 文件名: app.py
import os
import cv2
import torch
import numpy as np
import io
from collections import Counter
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

# ========================================================
# 🚨 确保你的数据路径是正确的
DATA_ROOT = "/Users/liangshilin/Desktop/COMP5703/新思想/VeRi"
# ========================================================

from train_kg_gnn import (
    ResNet50IBN_ReID, VisualKGModule,
    build_distance_matrix, apply_st_penalty,
    build_batch_hetero_graph, ReIDDataset,
    get_test_meta_map, read_txt_lines, extract_features_kg
)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ================= 1. 全局初始化与模型加载 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 575
TOTAL_CAMERAS = 20

print("=> [1/4] 正在加载 Baseline 视觉模型...")
car_encoder = ResNet50IBN_ReID(num_classes).to(device)
baseline_ckpt = torch.load("/Users/liangshilin/Desktop/COMP5703/新思想/best_module/best_resnet50_ibn.pth",
                           map_location=device, weights_only=False)
car_encoder.load_state_dict(baseline_ckpt['state_dict'])
car_encoder.eval()

print("=> [2/4] 正在加载 GNN 图网络模型...")
kg_module = VisualKGModule(in_channels=2048, hidden_channels=2048, num_classes=num_classes,
                           num_cameras=TOTAL_CAMERAS).to(device)
kg_ckpt = torch.load("/Users/liangshilin/Desktop/COMP5703/新思想/best_module/best_kg_module.pth", map_location=device,
                     weights_only=False)
kg_module.load_state_dict(kg_ckpt['state_dict'])
kg_module.eval()
cam2idx = kg_ckpt['cam2idx']

print("=> [3/4] 正在加载时空矩阵...")
dist_mat, cam_idx_map = build_distance_matrix()

transform_test = transforms.Compose([
    transforms.Resize((256, 256)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================= 2. 提取并加载真实的底库 =================
print("=> [4/4] 准备真实底库特征 (Gallery)...")
cache_path = "gallery_cache.pt"

if os.path.exists(cache_path):
    print("   [+] 检测到本地特征缓存，秒速加载中...")
    cache_data = torch.load(cache_path, map_location='cpu')
    real_gallery_feats = cache_data['feats'].numpy()
    real_gallery_names = cache_data['names']
else:
    print("   [!] 首次启动，正在提取特征...")
    test_xml = os.path.join(DATA_ROOT, 'test_label.xml')
    test_img_to_vid, test_img_to_cam = get_test_meta_map(test_xml)
    test_names = read_txt_lines(os.path.join(DATA_ROOT, 'name_test.txt'))
    real_test_list = [(name, test_img_to_vid.get(name, "-1"), test_img_to_cam.get(name, "-1")) for name in test_names]

    val_g_loader = DataLoader(
        ReIDDataset(os.path.join(DATA_ROOT, 'image_test'), real_test_list, cam2idx, transform_test),
        batch_size=64, shuffle=False, num_workers=0
    )
    g_feats, _, _, g_names = extract_features_kg(car_encoder, kg_module, val_g_loader, device)
    real_gallery_feats = g_feats.numpy()
    real_gallery_names = g_names
    torch.save({'feats': g_feats, 'names': g_names}, cache_path)

print("=> 🚀 服务启动完成！等待前端请求...")


# ================= 3. API 接口路由 =================

@app.post("/search")
async def search_vehicle(file: UploadFile = File(...), mode: str = Form(...)):
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img_tensor = transform_test(img).unsqueeze(0).to(device)

    filename = file.filename
    try:
        vid = int(filename.split('_')[0])
        camid = int(filename.split('_')[1].replace('c', ''))
    except:
        vid, camid = 0, 0

    with torch.no_grad():
        res_car = car_encoder(img_tensor)
        vbase = res_car[1] if isinstance(res_car, tuple) else res_car

        if mode == "baseline":
            q_feat = torch.nn.functional.normalize(vbase, p=2, dim=1).cpu().numpy()
            top_k = 150
        else:
            vid_tensor = torch.tensor([vid], dtype=torch.long).to(device)
            cam_tensor = torch.tensor([camid], dtype=torch.long).to(device)
            hetero_data, unique_cams = build_batch_hetero_graph(vbase, vid_tensor, cam_tensor, device, is_train=False)

            v_kg = kg_module(hetero_data, unique_cams)
            q_feat = torch.nn.functional.normalize(v_kg, p=2, dim=1).cpu().numpy()
            top_k = 150 if mode == "gnn" else 10

        sim_matrix = np.dot(q_feat, real_gallery_feats.T)

        if mode == "st":
            sim_matrix = apply_st_penalty(sim_matrix, [filename], real_gallery_names, dist_mat, cam_idx_map, fps=25.0,
                                          max_speed=20.0)

        indices = np.argsort(sim_matrix[0])[::-1][:top_k]

        results = []
        for rank, idx in enumerate(indices):
            results.append({
                "rank": rank + 1,
                "name": real_gallery_names[idx],
                "score": float(sim_matrix[0][idx])
            })

        voted_uid = None
        if mode == "st" and len(results) > 0:
            top_10_names = [res["name"] for res in results[:10]]
            top_10_uids = [name.split('_')[0] for name in top_10_names]
            uid_counts = Counter(top_10_uids)
            voted_uid = uid_counts.most_common(1)[0][0]

    return {
        "status": "success",
        "mode": mode,
        "results": results,
        "voted_uid": voted_uid
    }


# 新增：专为前端交互图表提供纯净的序列数据
@app.get("/trajectory_data/{vehicle_id}")
async def get_trajectory_data(vehicle_id: str):
    gallery_dir = os.path.join(DATA_ROOT, "image_test")
    if not os.path.exists(gallery_dir):
        return {"error": "图片目录不存在"}

    # 取出所有该 ID 的照片并排序
    images = [img for img in os.listdir(gallery_dir) if img.startswith(vehicle_id)]
    images = sorted(images, key=lambda x: int(x.split('_')[2]) if len(x.split('_')) > 2 else 0)

    # 结构化返回数据
    trajectory = []
    for img in images:
        parts = img.split('_')
        cam = parts[1] if len(parts) > 1 else "Unknown"
        frame = parts[2] if len(parts) > 2 else "0"
        trajectory.append({
            "filename": img,
            "cam": cam,
            "frame": frame
        })

    return {"data": trajectory}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)