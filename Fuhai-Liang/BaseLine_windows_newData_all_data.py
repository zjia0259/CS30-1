import os
import random
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# ==========================================
# [NEW] 固定所有随机种子，保证实验绝对可复现
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"=> 已固定全局随机种子 Seed: {seed}")

# ==========================================
# 1. 数据解析工具 (兼容 GBK/GB2312)
# ==========================================
def parse_xml_compat(xml_path):
    with open(xml_path, 'r', encoding='gbk', errors='ignore') as f:
        xml_content = f.read()
    xml_content = xml_content.replace('encoding="gb2312"', 'encoding="utf-8"')
    xml_content = xml_content.replace('encoding="GB2312"', 'encoding="utf-8"')
    root = ET.fromstring(xml_content)
    return root.find('Items').findall('Item')

def get_train_data(train_xml):
    items = parse_xml_compat(train_xml)
    train_data, id_map = [], {}
    current_idx = 0
    for item in items:
        vid = item.get('vehicleID')
        if vid not in id_map:
            id_map[vid] = current_idx
            current_idx += 1
        train_data.append((item.get('imageName'), id_map[vid], item.get('cameraID')))
    return train_data, len(id_map)

def get_test_eval_data(test_xml, eval_k_queries='all', seed=42):
    items = parse_xml_compat(test_xml)
    id_dict = defaultdict(list)
    for item in items:
        id_dict[item.get('vehicleID')].append({
            'img_name': item.get('imageName'),
            'cam_id': item.get('cameraID')
        })
        
    unique_vids = sorted(list(id_dict.keys())) # 排序保证一致性
    random.seed(seed) # 保证每次抽样相同
    
    if eval_k_queries == 'all':
        selected_vids = set(unique_vids)
    else:
        k = int(eval_k_queries)
        selected_vids = set(random.sample(unique_vids, min(k, len(unique_vids))))
        
    test_q_list, test_g_list = [], []
    for vid, imgs in id_dict.items():
        if vid in selected_vids and len(imgs) > 1:
            q_idx = random.randint(0, len(imgs) - 1)
            for i, img_info in enumerate(imgs):
                if i == q_idx:
                    test_q_list.append((img_info['img_name'], vid, img_info['cam_id']))
                else:
                    test_g_list.append((img_info['img_name'], vid, img_info['cam_id']))
        else:
            for img_info in imgs:
                test_g_list.append((img_info['img_name'], vid, img_info['cam_id']))
                
    return test_q_list, test_g_list

# ===== [改动 1] 新增：解析 Test 元数据 (同时返回 vid 和 cam 字典) =====
def get_test_meta_map(test_xml):
    items = parse_xml_compat(test_xml)
    vid_map, cam_map = {}, {}
    for item in items:
        img_name = item.get('imageName')
        vid_map[img_name] = item.get('vehicleID')
        cam_map[img_name] = item.get('cameraID')
    return vid_map, cam_map

# ===== [改动 2] 新增：从 Query 文件名提取 vehicleID 和 cameraID =====
def parse_query_meta(name):
    """
    解析 query 图片名，例如 0002_c002_00030600_0.jpg
    返回 (vehicleID, cameraID)
    """
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "-1", "-1"

def read_txt_lines(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# ==========================================
# 2. Dataset 定义
# ==========================================
class ReIDDataset(Dataset):
    def __init__(self, img_dir, data_list, transform=None):
        self.img_dir = img_dir
        self.data_list = data_list 
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_name, label_or_vid, cam_id = self.data_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label_or_vid, cam_id, img_name

# ==========================================
# 3. ResNet50 Baseline 模型
# ==========================================
class ResNet50ReID(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50ReID, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        features = self.backbone(x)
        features = self.gap(features).view(features.size(0), -1)
        feat_bn = self.bottleneck(features)
        return (self.classifier(feat_bn), features) if self.training else feat_bn

# ==========================================
# 4. 特征提取与标准 ReID 评测
# ==========================================
def extract_features(model, dataloader, device):
    from tqdm import tqdm
    model.eval()
    feats_list, vids_list, cams_list, names_list = [], [], [], []
    with torch.no_grad():
        for imgs, vids, cams, names in tqdm(dataloader, desc="提取特征", leave=False):
            imgs = imgs.to(device)
            feats = model(imgs)
            feats = F.normalize(feats, p=2, dim=1)
            feats_list.append(feats.cpu())
            vids_list.extend(vids)
            cams_list.extend(cams)
            names_list.extend(names)
    return torch.cat(feats_list, dim=0), vids_list, cams_list, names_list

def evaluate_reid(q_feats, q_vids, q_cams, g_feats, g_vids, g_cams):
    sim_matrix = torch.mm(q_feats, g_feats.t()).numpy()
    q_vids, q_cams = np.array(q_vids), np.array(q_cams)
    g_vids, g_cams = np.array(g_vids), np.array(g_cams)
    num_q = sim_matrix.shape[0]
    indices = np.argsort(sim_matrix, axis=1)[:, ::-1]
    matches = (g_vids[indices] == q_vids[:, np.newaxis]).astype(np.int32)
    all_ap, all_cmc = [], []

    for q_idx in range(num_q):
        q_vid, q_cam = q_vids[q_idx], q_cams[q_idx]
        order = indices[q_idx]
        # 标准 ReID 过滤：同 ID 且 同摄像头 视为 junk，不计入评测
        remove = (g_vids[order] == q_vid) & (g_cams[order] == q_cam)
        keep = np.invert(remove)
        raw_match = matches[q_idx][keep]
        
        if not np.any(raw_match): continue
        
        num_rel = raw_match.sum()
        tmp_cmc = raw_match.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_match
        all_ap.append(tmp_cmc.sum() / num_rel)
        
        cmc = raw_match.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:100])

    mAP = np.mean(all_ap) if all_ap else 0.0
    CMC = np.mean(np.asarray(all_cmc).astype(np.float32), axis=0) if all_cmc else np.zeros(100)
    return mAP, CMC

# ==========================================
# 5. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help="数据集根目录")
    parser.add_argument('--epochs', type=int, default=10) 
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_k_queries', type=str, default='all')
    parser.add_argument('--save_dir', type=str, default='./model')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.save_dir, 'best_baseline.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=> Using device: {device}")

    # 路径配置
    train_dir = os.path.join(args.root, 'image_train')
    test_dir = os.path.join(args.root, 'image_test')
    query_dir = os.path.join(args.root, 'image_query')
    train_xml = os.path.join(args.root, 'train_label.xml')
    test_xml = os.path.join(args.root, 'test_label.xml')
    query_txt = os.path.join(args.root, 'name_query.txt')
    test_txt = os.path.join(args.root, 'name_test.txt')

    print("\n=> 正在解析数据字典...")
    train_list, num_classes = get_train_data(train_xml)
    test_eval_q, test_eval_g = get_test_eval_data(test_xml, args.eval_k_queries, args.seed)
    
    query_names = read_txt_lines(query_txt)
    test_names = read_txt_lines(test_txt)
    
    # ===== [改动 3] 应用解析函数构建包含真实 GT 的 List =====
    test_img_to_vid, test_img_to_cam = get_test_meta_map(test_xml)
    # 对于 query，使用 parse_query_meta 解析真实 vid 和 cam
    real_q_list = [(name, parse_query_meta(name)[0], parse_query_meta(name)[1]) for name in query_names] 
    # 对于 gallery，从 test_xml 映射真实 vid 和 cam
    real_test_list = [(name, test_img_to_vid.get(name, "-1"), test_img_to_cam.get(name, "-1")) for name in test_names]

    print(f"   [Train] 样本数: {len(train_list)} (ID数: {num_classes})")
    print(f"   [Test Eval] Query数: {len(test_eval_q)}, Gallery数: {len(test_eval_g)}")
    print(f"   [Real Query] 全量真实查询数: {len(real_q_list)}\n")

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(ReIDDataset(train_dir, train_list, transform_train), batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_q_loader = DataLoader(ReIDDataset(test_dir, test_eval_q, transform_test), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_g_loader = DataLoader(ReIDDataset(test_dir, test_eval_g, transform_test), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ResNet50ReID(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-4)
    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    best_mAP = 0.0
    best_epoch = -1

    # ==========================================
    # 训练循环
    # ==========================================
    from tqdm import tqdm
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=False)
        
        for i, (imgs, labels, _, _) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                outputs, _ = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- 每个 Epoch 结束后在 Test 集合上计算 mAP 和 CMC ---
        print(f"\n=> Epoch {epoch+1} 结束，正在进行 Test 评测...")
        q_feats, q_vids, q_cams, _ = extract_features(model, val_q_loader, device)
        g_feats, g_vids, g_cams, _ = extract_features(model, val_g_loader, device)
        mAP, CMC = evaluate_reid(q_feats, q_vids, q_cams, g_feats, g_vids, g_cams)
        
        print(f"   [Epoch {epoch+1} 指标] mAP: {mAP:.4f} | Rank-1: {CMC[0]:.4f} | Rank-5: {CMC[4]:.4f} | Rank-10: {CMC[9]:.4f}")

        if mAP > best_mAP:
            best_mAP = mAP
            best_epoch = epoch + 1
            state = {
                'epoch': best_epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mAP': best_mAP
            }
            torch.save(state, best_ckpt_path)
            print(f"   [✔] 发现新最佳 mAP ({best_mAP:.4f})，已保存到 {best_ckpt_path}\n")
        else:
            print(f"   [ ] 未超过历史最佳 mAP ({best_mAP:.4f} at Epoch {best_epoch})\n")

    # ==========================================
    # 最终检索预测与导出 Excel
    # ==========================================
    print("=" * 60)
    print(f"=> 训练全部结束！best epoch = {best_epoch}, best mAP = {best_mAP:.4f}")
    
    print(f"=> 正在加载最佳权重 {best_ckpt_path} ...")
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> 已成功加载 best_baseline.pth。准备生成真实 Query 的 Excel 和最终指标...")
    print("=" * 60)

    real_q_loader = DataLoader(ReIDDataset(query_dir, real_q_list, transform_test), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    real_g_loader = DataLoader(ReIDDataset(test_dir, real_test_list, transform_test), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 提取特征 (因为 real_q_list 已经包含了真实的 vid 和 cam，提取出来的直接就能用)
    rq_feats, rq_vids, rq_cams, rq_names = extract_features(model, real_q_loader, device)
    rg_feats, rg_vids, rg_cams, rg_names = extract_features(model, real_g_loader, device)

    print("\n=> 计算相似度并生成 Excel ...")
    sim_matrix_final = torch.mm(rq_feats, rg_feats.t())
    top5_indices = torch.topk(sim_matrix_final, k=5, dim=1, largest=True)[1]

    results = []
    for i in range(len(rq_names)):
        query_name = rq_names[i]
        row_data = {'query_image': query_name}
        for rank in range(5):
            gallery_idx = top5_indices[i, rank].item()
            gallery_name = rg_names[gallery_idx]
            vid = test_img_to_vid.get(gallery_name, "Unknown")
            row_data[f'rank{rank+1}_vehicleID'] = vid
        results.append(row_data)

    df = pd.DataFrame(results)
    output_excel = "query_top5_predictions.xlsx"
    df.to_excel(output_excel, index=False)
    print(f"=> [成功] Excel 预测结果已保存至：{output_excel}")

    # ===== [改动 4] 新增：对全量真实 Query 进行标准 mAP/CMC 评测 =====
    print("\n=> ==========================================================")
    print(f"=> [最终评测] 开始对全量真实 Query ({len(rq_names)}) vs Test ({len(rg_names)}) 计算指标")
    print("=> 评测标准：基于 vehicleID 匹配 + cameraID 过滤 (同 camera 视为 junk 不计分)")
    
    # 完美复用我们现成的 evaluate_reid 函数
    mAP_final, CMC_final = evaluate_reid(rq_feats, rq_vids, rq_cams, rg_feats, rg_vids, rg_cams)
    
    print(f"   [Real Query 最终指标] mAP: {mAP_final:.4f} | Rank-1: {CMC_final[0]:.4f} | Rank-5: {CMC_final[4]:.4f} | Rank-10: {CMC_final[9]:.4f}")
    print("=> ==========================================================\n")

if __name__ == '__main__':
    main()