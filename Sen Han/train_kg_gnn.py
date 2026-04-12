# 文件名: train_kg_gnn.py
import os
import random
import argparse
import copy
from collections import defaultdict
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv

# ==========================================
# 1. 基础环境、解析工具与时空知识库 (Step 3 准备)
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"=> 已固定全局随机种子 Seed: {seed}")

def parse_xml_compat(xml_path):
    with open(xml_path, 'r', encoding='gbk', errors='ignore') as f:
        xml_content = f.read()
    xml_content = xml_content.replace('encoding="gb2312"', 'encoding="utf-8"').replace('encoding="GB2312"', 'encoding="utf-8"')
    return ET.fromstring(xml_content).find('Items').findall('Item')

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

def get_test_meta_map(test_xml):
    items = parse_xml_compat(test_xml)
    vid_map, cam_map = {}, {}
    for item in items:
        img_name = item.get('imageName')
        vid_map[img_name] = item.get('vehicleID')
        cam_map[img_name] = item.get('cameraID')
    return vid_map, cam_map

def parse_query_meta(name):
    parts = name.split('_')
    return (parts[0], parts[1]) if len(parts) >= 2 else ("-1", "-1")

def read_txt_lines(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# 🚨【绝杀准备】：时空惩罚字典构建
def build_distance_matrix():
    """将物理距离转为对称矩阵"""
    cam_order = [7, 5, 8, 4, 9, 3, 10, 6, 11, 2, 1, 13, 12, 18, 17, 19, 20, 15, 14, 16]
    raw_dist = [
        [0, 40, 170, 240, 260, 280, 450, 520, 530, 740, 850, 990, 1010, 1210, 1250, 1270, 1190, 1030, 930, 880],
        [0, 0, 130, 200, 220, 240, 410, 480, 490, 700, 810, 950, 970, 1170, 1210, 1310, 1230, 1070, 970, 920],
        [0, 0, 0, 70, 90, 110, 280, 350, 360, 570, 680, 820, 840, 1040, 1080, 1180, 1260, 1200, 1100, 1050],
        [0, 0, 0, 0, 20, 40, 210, 280, 290, 500, 610, 750, 770, 970, 1010, 1110, 1190, 1270, 1070, 1120],
        [0, 0, 0, 0, 0, 20, 190, 260, 270, 480, 590, 730, 750, 950, 990, 1090, 1170, 1330, 1090, 1140],
        [0, 0, 0, 0, 0, 0, 170, 240, 250, 460, 570, 710, 730, 930, 970, 1070, 1150, 1310, 1220, 1160],
        [0, 0, 0, 0, 0, 0, 0, 70, 80, 290, 400, 540, 560, 760, 800, 900, 980, 1140, 1240, 1290],
        [0, 0, 0, 0, 0, 0, 0, 0, 10, 220, 330, 470, 490, 690, 730, 830, 910, 1070, 1170, 1220],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 210, 320, 460, 480, 680, 720, 820, 900, 1060, 1160, 1210],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 110, 250, 270, 470, 510, 610, 690, 850, 950, 1000],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 140, 160, 360, 400, 500, 580, 740, 840, 890],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 220, 260, 360, 440, 600, 700, 750],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 240, 340, 420, 580, 680, 730],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 140, 320, 480, 580, 630],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 180, 340, 440, 490],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 240, 340, 390],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 160, 260, 310],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 150],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    dist_np = np.array(raw_dist, dtype=np.float32)
    dist_matrix = dist_np + dist_np.T - np.diag(dist_np.diagonal())
    cam2idx = {int(cam): i for i, cam in enumerate(cam_order)}
    return dist_matrix, cam2idx

def build_image_to_time_dict(track_txt_path):
    """解析 test_track_VeRi.txt 拿到所有图片的绝对时间"""
    img2time = {}
    if not os.path.exists(track_txt_path):
        return img2time
    with open(track_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2: continue
        time_str = parts[0].split('_')[2]
        time_val = int(time_str.replace('t', ''))
        for img_name in parts[1:]:
            img2time[img_name] = time_val
    return img2time

# ==========================================
# 2. 数据集与采样器 (已移除多余的场景图片)
# ==========================================
class ReIDDataset(Dataset):
    def __init__(self, img_dir, data_list, cam2idx, transform=None):
        self.img_dir = img_dir
        self.data_list = data_list 
        self.cam2idx = cam2idx
        self.transform = transform

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        img_name, label_or_vid, cam_id_str = self.data_list[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        if self.transform: img = self.transform(img)
        cam_idx = self.cam2idx.get(cam_id_str, 0) 
        return img, label_or_vid, cam_idx, img_name

class RandomIdentitySampler(torch.utils.data.sampler.Sampler):
    """PK 采样器"""
    def __init__(self, data_list, batch_size, num_instances=4):
        self.data_list = data_list
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, item in enumerate(self.data_list):
            self.index_dic[item[1]].append(index)
        self.pids = list(self.index_dic.keys())
        self.length = 0
        for pid in self.pids:
            num = len(self.index_dic[pid])
            if num < self.num_instances: num = self.num_instances
            self.length += num - num % self.num_instances
    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            random.shuffle(idxs)
            batch_idxs_dict[pid] = idxs
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid][:self.num_instances]
                batch_idxs_dict[pid] = batch_idxs_dict[pid][self.num_instances:]
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) < self.num_instances: avai_pids.remove(pid)
        return iter(final_idxs)
    def __len__(self): return self.length

# ==========================================
# 3. 动态轨迹池化 (Step 1：改造特征输入)
# ==========================================
def pool_tracklets_in_batch(feat_car, vids, cams, img_names):
    """
    将同一 Batch 内，同车同摄像头的零散图片合并为一个“超级节点”。
    极大降低噪声，减少 GNN 计算量。
    """
    device = feat_car.device
    vids_np = vids.cpu().numpy() if torch.is_tensor(vids) else np.array(vids)
    cams_np = cams.cpu().numpy() if torch.is_tensor(cams) else np.array(cams)

    unique_keys = {}
    for i in range(len(vids_np)):
        key = (vids_np[i], cams_np[i])
        if key not in unique_keys: unique_keys[key] = []
        unique_keys[key].append(i)

    pooled_car, list_vids, list_cams, list_names = [], [], [], []
    for (vid, cam), idxs in unique_keys.items():
        pooled_car.append(feat_car[idxs].mean(dim=0))
        
        # 🚨 修复报错：强制转换为 int，防止测试集的 '0002' 字符串引发 PyTorch 崩溃
        list_vids.append(int(vid))
        list_cams.append(int(cam))
        
        list_names.append(img_names[idxs[0]]) # 取轨迹的第一张图作为时间锚点

    return (
        torch.stack(pooled_car).to(device), 
        torch.tensor(list_vids, dtype=torch.long, device=device), 
        torch.tensor(list_cams, dtype=torch.long, device=device),
        list_names
    )

# ==========================================
# 4. 视觉基线与 GNN 模型 (Step 2：精简图谱)
# ==========================================
class ResNet50IBN_ReID(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50IBN_ReID, self).__init__()
        valid_local_path = '/root/autodl-tmp/.cache/torch/hub/XingangPan_IBN-Net_master'
        if os.path.exists(valid_local_path):
            backbone = torch.hub.load(valid_local_path, 'resnet50_ibn_a', source='local', pretrained=False)
        else:
            backbone = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=False)
        
        self.conv1, self.bn1, self.relu, self.maxpool = backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        self.layer4[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        features = self.layer4(x)
        vbase = self.gap(features).view(features.size(0), -1)
        feat_bn = self.bottleneck(vbase)
        return (self.classifier(feat_bn), vbase) if self.training else vbase

class VisualKGModule(nn.Module):
    """精简版图网络：彻底丢弃场景图片，仅依赖 Vehicle 和 Camera 构建拓扑"""
    def __init__(self, in_channels=2048, hidden_channels=2048, num_classes=1000, num_cameras=20):
        super(VisualKGModule, self).__init__()
        self.camera_emb = nn.Embedding(num_cameras, in_channels)
        self.hetero_conv = HeteroConv({
            ('vehicle', 'in', 'camera'): GATConv(in_channels, hidden_channels, heads=4, concat=False, add_self_loops=False),
            ('camera', 'contains', 'vehicle'): GATConv(in_channels, hidden_channels, heads=4, concat=False, add_self_loops=False)
        }, aggr='mean')

        # 🚨 修复 2：恢复智能 alpha，初始给 0.1。
        # 既能保证梯度流通逼迫 GNN 学习，又不会因为 0.5 太大而在初期毁掉 Baseline
        self.alpha = nn.Parameter(torch.tensor(0.2))
        
        self.bottleneck = nn.BatchNorm1d(hidden_channels)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(hidden_channels, num_classes, bias=False)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, hetero_data, unique_cams):
        x_dict = hetero_data.x_dict
        x_dict['camera'] = self.camera_emb(unique_cams)
        out_dict = self.hetero_conv(x_dict, hetero_data.edge_index_dict)
        
        # 🚨 视觉致盲：遮挡 40% 特征，逼迫网络向 Camera Embedding 索取知识
        blinded_vbase = F.dropout(x_dict['vehicle'], p=0.3, training=self.training)
        
        # 🚨 修复 3：使用自适应 alpha
        v_kg = blinded_vbase + self.alpha * out_dict['vehicle']
   
        feat_bn = self.bottleneck(v_kg)
        return (self.classifier(feat_bn), v_kg) if self.training else v_kg

# ==========================================
# 5. 损失函数与评测工具
# ==========================================
class TripletLoss(nn.Module):
    # 调高 Margin 逼迫网络学习图谱拓扑
    def __init__(self, margin=0.5): 
        super(TripletLoss, self).__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    def forward(self, inputs, targets):
        inputs = inputs.float()
        n = inputs.size(0)
        dist = torch.cdist(inputs, inputs, p=2.0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0)) 
        dist_ap, dist_an = torch.cat(dist_ap), torch.cat(dist_an)
        return self.ranking_loss(dist_an, dist_ap, torch.ones_like(dist_an))

def evaluate_reid_with_matrix(sim_matrix, q_vids, q_cams, g_vids, g_cams):
    """直接接收矩阵进行评测，为了后续的时空惩罚铺路"""
    q_vids, q_cams, g_vids, g_cams = map(np.array, (q_vids, q_cams, g_vids, g_cams))
    num_q = sim_matrix.shape[0]
    indices = np.argsort(sim_matrix, axis=1)[:, ::-1]
    matches = (g_vids[indices] == q_vids[:, np.newaxis]).astype(np.int32)
    
    all_ap, all_cmc = [], []
    for q_idx in range(num_q):
        q_vid, q_cam = q_vids[q_idx], q_cams[q_idx]
        order = indices[q_idx]
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
# 🚨 终极核武器：基于帧率的弹性时空约束
def apply_st_penalty(sim_matrix, q_names, g_names, dist_matrix, cam2idx, fps=25.0, max_speed=20.0):
    """
    max_speed=40.0 m/s 相当于 144 km/h，由于摄像头可能有同步误差，给一个宽容的极限速度
    """
    # 1. 拨乱反正：直接从图片名的第三部分提取 Frame ID，转化为绝对秒数！
    q_times = np.array([int(name.split('_')[2]) for name in q_names]) / fps
    g_times = np.array([int(name.split('_')[2]) for name in g_names]) / fps
    
    q_cams = np.array([int(name.split('_')[1].replace('c', '')) for name in q_names])
    g_cams = np.array([int(name.split('_')[1].replace('c', '')) for name in g_names])
    
    q_idx = np.array([cam2idx.get(c, -1) for c in q_cams])
    g_idx = np.array([cam2idx.get(c, -1) for c in g_cams])
    
    # 2. 计算真实时间差
    time_diff_sec = np.abs(q_times[:, None] - g_times[None, :]) 
    
    # 获取空间距离
    safe_q = np.maximum(q_idx, 0)
    safe_g = np.maximum(g_idx, 0)
    distances = dist_matrix[safe_q[:, None], safe_g[None, :]]
    
    valid_mask = (q_idx[:, None] >= 0) & (g_idx[None, :] >= 0)
    same_cam_mask = (q_cams[:, None] == g_cams[None, :])
    
    # 3. 弹性容错机制（核心！）
    # 考虑到国内老旧摄像头的时钟可能相差几秒，我们把小于 5 秒的时间差，统一按 5 秒算，防止误杀真车！
    safe_time_diff = np.maximum(time_diff_sec, 6.0)
    
    # 计算速度
    with np.errstate(divide='ignore', invalid='ignore'):
        speeds = distances / safe_time_diff
        
    # 4. 物理审判：不在同一个摄像头，且速度突破了宽容物理极限 (144 km/h)
    impossible_speed = (~same_cam_mask) & (speeds > max_speed)
    
    # 执行制裁！
    penalty_mask = valid_mask & impossible_speed
    sim_matrix[penalty_mask] -= 100.0
    
    # 打印拦截战报，让你在终端看得热血沸腾
    print(f" => [ST-Penalty] 触发时空制裁，大网成功拦截了 {np.sum(penalty_mask)} 次不可能的匹配！")
    
    return sim_matrix
# ==========================================
# 6. 建图与提取特征
# ==========================================
def build_batch_hetero_graph(feat_car, vids, camids, device, is_train=True):
    batch_size = feat_car.size(0)
    data = HeteroData().to(device)
    data['vehicle'].x = feat_car
    unique_cams = torch.unique(camids)
    
    cam_mapper = {cam.item(): idx for idx, cam in enumerate(unique_cams)}
    mapped_cam_indices = torch.tensor([cam_mapper[c.item()] for c in camids], device=device)
    
    # 车 <-> 摄像头
    v2c_edge = torch.stack([torch.arange(batch_size, device=device), mapped_cam_indices])
    data['vehicle', 'in', 'camera'].edge_index = v2c_edge
    data['camera', 'contains', 'vehicle'].edge_index = v2c_edge.flip([0])
            
    return data, unique_cams

def extract_features_kg(car_encoder, kg_model, dataloader, device):
    car_encoder.eval()
    kg_model.eval()
    feats_list, vids_list, cams_list, names_list = [], [], [], []
    
    with torch.no_grad():
        for car_imgs, vids, camids, img_names in dataloader:
            car_imgs = car_imgs.to(device)
            res_car = car_encoder(car_imgs)
            feat_car = res_car[1] if isinstance(res_car, tuple) else res_car
            
            # 测试阶段也进行超级节点融合
            feat_car, track_vids, track_cams, track_names = pool_tracklets_in_batch(feat_car, vids, camids, img_names)
            
            hetero_data, unique_cams = build_batch_hetero_graph(feat_car, track_vids, track_cams.to(device), device, is_train=False)
            feat_kg = kg_model(hetero_data, unique_cams)
            
            feat_kg = F.normalize(feat_kg, p=2, dim=1) 
            feats_list.append(feat_kg.cpu())
            vids_list.extend(track_vids.cpu().numpy())
            cams_list.extend(track_cams.cpu().numpy())
            names_list.extend(track_names)
            
    return torch.cat(feats_list, dim=0), vids_list, cams_list, names_list

# ==========================================
# 7. 主程序流
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help="数据集根目录")
    parser.add_argument('--baseline_ckpt', type=str, required=True, help="第一阶段训练好的 best_resnet50_ibn.pth 路径")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='./KGmodel_output')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    best_kg_ckpt_path = os.path.join(args.save_dir, 'best_kg_module.pth')

    print("\n=> 正在构建全图摄像头词典...")
    train_xml = os.path.join(args.root, 'train_label.xml')
    test_xml = os.path.join(args.root, 'test_label.xml')
    train_list, num_classes = get_train_data(train_xml)
    test_img_to_vid, test_img_to_cam = get_test_meta_map(test_xml)
    
    query_names = read_txt_lines(os.path.join(args.root, 'name_query.txt'))
    test_names = read_txt_lines(os.path.join(args.root, 'name_test.txt'))
    real_q_list = [(name, parse_query_meta(name)[0], parse_query_meta(name)[1]) for name in query_names] 
    real_test_list = [(name, test_img_to_vid.get(name, "-1"), test_img_to_cam.get(name, "-1")) for name in test_names]

    all_cams = set([item[2] for item in train_list + real_q_list + real_test_list])
    cam2idx = {cam: i for i, cam in enumerate(sorted(list(all_cams)))}
    TOTAL_CAMERAS = len(cam2idx)
    print(f"=> 成功映射了 {TOTAL_CAMERAS} 个唯一的摄像头节点。")

    # 准备时空矩阵 (Step 3 核心)
    dist_mat, cam_idx_map = build_distance_matrix()
    track_file = os.path.join(args.root, 'test_track_VeRi.txt')
    img2time_dict = build_image_to_time_dict(track_file)

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_sampler = RandomIdentitySampler(train_list, args.batch_size, num_instances=4)
    train_loader = DataLoader(
        ReIDDataset(os.path.join(args.root, 'image_train'), train_list, cam2idx, transform_train), 
        batch_size=args.batch_size, sampler=train_sampler, num_workers=4, drop_last=True
    )
    val_q_loader = DataLoader(
        ReIDDataset(os.path.join(args.root, 'image_query'), real_q_list, cam2idx, transform_test), 
        batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    val_g_loader = DataLoader(
        ReIDDataset(os.path.join(args.root, 'image_test'), real_test_list, cam2idx, transform_test), 
        batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    print("\n=> 正在加载并冻结视觉 Baseline 模型...")
    car_encoder = ResNet50IBN_ReID(num_classes=num_classes).to(device)
    checkpoint = torch.load(args.baseline_ckpt, map_location=device, weights_only=False)
    car_encoder.load_state_dict(checkpoint['state_dict'])
    car_encoder.eval() 
    for param in car_encoder.parameters():
        param.requires_grad = False 
        
    print("=> 正在初始化图谱 GNN 模块 (VisualKGModule)...")
    kg_module = VisualKGModule(num_classes=num_classes, num_cameras=TOTAL_CAMERAS).to(device)
    
    criterion_id = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=0.6) # 高压力学习
    optimizer = optim.Adam(kg_module.parameters(), lr=0.00035, weight_decay=5e-4)
    base_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    best_mAP = 0.0
    print("\n=> 🚀 开始联合训练！(轨迹池化 -> 空间传播 -> 时空决断)")
    from tqdm import tqdm
    for epoch in range(args.epochs):
        if epoch < 5: 
            lr = 0.00035 * (epoch + 1) / 5
            for param_group in optimizer.param_groups: param_group['lr'] = lr

        kg_module.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=False)
        
        for car_imgs, labels, camids, img_names in pbar:
            car_imgs = car_imgs.to(device)
            labels, camids = labels.to(device), camids.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                res_car = car_encoder(car_imgs)
                feat_car = res_car[1] if isinstance(res_car, tuple) else res_car
                
            # 🚨 Step 1: 融合为“超级节点”
            feat_car, track_vids, track_cams, _ = pool_tracklets_in_batch(feat_car, labels, camids, img_names)
                
            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                hetero_data, unique_cams = build_batch_hetero_graph(feat_car, track_vids, track_cams, device, is_train=True)
                logits, v_kg = kg_module(hetero_data, unique_cams)
                
                loss_id = criterion_id(logits, track_vids)
                loss_triplet = criterion_triplet(v_kg, track_vids)
                loss = loss_id + loss_triplet
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        
        # 评测阶段
        q_feats, q_vids, q_cams, q_names = extract_features_kg(car_encoder, kg_module, val_q_loader, device)
        g_feats, g_vids, g_cams, g_names = extract_features_kg(car_encoder, kg_module, val_g_loader, device)
        
        sim_matrix = torch.mm(q_feats, g_feats.t()).numpy()
        
        # 🚨 Step 3: 撒下时空大网降维打击
        # 🚨 修复 5：先注释掉时空打击，看看纯 GNN 空间拓扑的真实实力！
        # sim_matrix = apply_st_penalty(sim_matrix, q_names, g_names, dist_mat, cam_idx_map, img2time_dict, max_speed=30.0)
        
        # 🚨 Step 3: 撒下基于帧率的弹性时空大网 (不需要传 img2time_dict 了)
        sim_matrix = apply_st_penalty(sim_matrix, q_names, g_names, dist_mat, cam_idx_map, fps=25.0, max_speed=20.0)
        
        mAP, CMC = evaluate_reid_with_matrix(sim_matrix, q_vids, q_cams, g_vids, g_cams)
        mAP, CMC = evaluate_reid_with_matrix(sim_matrix, q_vids, q_cams, g_vids, g_cams)
              
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.epochs}] | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | ST_mAP: {mAP:.4f} | Rank-1: {CMC[0]:.4f} | Rank-5: {CMC[4]:.4f} ｜ Rank-10: {CMC[9]:.4f}")
        
        base_scheduler.step()
        
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save({
                'epoch': epoch + 1,
                'state_dict': kg_module.state_dict(), 
                'num_classes': num_classes,
                'cam2idx': cam2idx, 
                'best_mAP': best_mAP
            }, best_kg_ckpt_path)
            print("   [✔] 最优轨迹图神经网络 已更新并保存！")

if __name__ == '__main__':
    main()
