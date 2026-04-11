# 文件名: train_resnet_ibn.py
import os
import random
import argparse
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==========================================
# 1. 硬件环境检测与随机种子设定
# ==========================================
def check_device():
    """
    智能检测硬件环境。
    如果是 Mac 电脑，会自动调用 MPS (Metal Performance Shaders) 进行 M1/M2/M3 芯片加速。
    如果是服务器，会自动调用 CUDA。
    """
    print("\n" + "=" * 60)
    print("=> 🖥️ 硬件环境检测报告")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"   [✔] 成功检测到 GPU 加速环境！({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"   [✔] 成功检测到 Apple Silicon (MPS) 加速环境！")
    else:
        device = torch.device("cpu")
        print(f"   [!] 警告: 未检测到 GPU/MPS，当前正在使用纯 CPU 运行！")
    print("=" * 60 + "\n")
    return device

def set_seed(seed=42):
    """固定随机种子，保证实验的可重复性，方便你在修改模型后能公平对比效果。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"=> 已固定全局随机种子 Seed: {seed}")


# ==========================================
# 2. 数据解析工具 (专门针对 XML 和属性提取)
# ==========================================
def parse_xml_compat(xml_path):
    """兼容各种诡异的中文编码（GBK/GB2312），安全读取 XML 内容"""
    with open(xml_path, 'r', encoding='gbk', errors='ignore') as f:
        xml_content = f.read()
    xml_content = xml_content.replace('encoding="gb2312"', 'encoding="utf-8"').replace('encoding="GB2312"', 'encoding="utf-8"')
    return ET.fromstring(xml_content).find('Items').findall('Item')

def get_train_data(train_xml):
    """
    解析训练集。
    【核心思路】：由于 XML 里的 ID 可能不是从 0 开始连续的（比如 1, 3, 7），
    而 PyTorch 的 CrossEntropyLoss 要求标签必须是 0 到 N-1 的连续整数，
    所以我们必须用字典 (id_map, color_map, type_map) 重新映射一次。
    """
    items = parse_xml_compat(train_xml)
    train_data, id_map, color_map, type_map = [], {}, {}, {}
    
    for item in items:
        vid = item.get('vehicleID')
        cid = item.get('colorID')
        tid = item.get('typeID')
        
        # 建立连续的映射字典
        if vid not in id_map: id_map[vid] = len(id_map)
        if cid not in color_map: color_map[cid] = len(color_map)
        if tid not in type_map: type_map[tid] = len(type_map)
        
        # 将映射后的安全整数索引加入列表
        train_data.append((
            item.get('imageName'), 
            id_map[vid], 
            item.get('cameraID'), 
            color_map[cid], 
            type_map[tid]
        ))
    
    # 返回训练数据，以及三种标签的总类别数（用于构建分类器的全连接层维度）
    return train_data, len(id_map), len(color_map), len(type_map), color_map, type_map

def get_test_meta_map(test_xml, color_map, type_map):
    """
    解析测试集的 XML。
    【知识图谱前置准备】：既然测试集 XML 也有颜色和车型，我们把它们也提取出来。
    如果在测试集中遇到了训练集没见过的颜色，我们将其标记为 -1，在计算准确率时忽略它们。
    """
    items = parse_xml_compat(test_xml)
    vid_map, cam_map, c_map, t_map = {}, {}, {}, {}
    for item in items:
        img_name = item.get('imageName')
        vid_map[img_name] = item.get('vehicleID')
        cam_map[img_name] = item.get('cameraID')
        
        cid = item.get('colorID')
        tid = item.get('typeID')
        # 复用训练集的映射，确保测试集标签和网络输出维度对齐
        c_map[img_name] = color_map.get(cid, -1)
        t_map[img_name] = type_map.get(tid, -1)
        
    return vid_map, cam_map, c_map, t_map

def read_txt_lines(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


# ==========================================
# 3. 数据集与采样器
# ==========================================
class ReIDDataset(Dataset):
    """标准的 ReID 数据集加载器"""
    def __init__(self, img_dir, data_list, transform=None):
        self.img_dir = img_dir
        self.data_list = data_list 
        self.transform = transform

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        # 统一解包 5 个元素：图片名，车辆ID，摄像头ID，颜色ID，车型ID
        img_name, label_or_vid, cam_id, color_id, type_id = self.data_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform: 
            img = self.transform(img)
            
        return img, label_or_vid, cam_id, color_id, type_id, img_name

class RandomIdentitySampler(torch.utils.data.sampler.Sampler):
    """
    【必须保留】：PK 采样器。保证每个 Batch 内包含 P 个人(车)，每辆车 K 张图。
    这是 Triplet Loss(三元组损失) 能正常工作的基础，如果不用它，Batch 里大概率找不到同一辆车的不同图片，
    Triplet Loss 就会因为找不到正样本而直接崩溃。
    """
    def __init__(self, data_list, batch_size, num_instances=4):
        self.data_list = data_list
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        
        # item[1] 是车辆 ID
        for index, item in enumerate(self.data_list):
            pid = item[1]  
            self.index_dic[pid].append(index)
            
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
            # 如果某辆车的图片少于 K 张，就复制补齐
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
                if len(batch_idxs_dict[pid]) < self.num_instances:
                    avai_pids.remove(pid)
                    
        return iter(final_idxs)

    def __len__(self):
        return self.length

class TripletLoss(nn.Module):
    """带有难例挖掘的三元组损失，利用 cdist 防止混合精度下的 NaN 报错"""
    def __init__(self, margin=0.4):
        super(TripletLoss, self).__init__()
        self.margin = margin
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

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


        


# ==========================================
# 4. 核心感知网络：多任务 ResNet50-IBN
# ==========================================
class ResNet50IBN_ReID_MultiTask(nn.Module):
    def __init__(self, num_classes, num_colors, num_types):
        super(ResNet50IBN_ReID_MultiTask, self).__init__()

        # ================== 修改这里 ==================
        print("=> 正在从本地完全离线加载 resnet50_ibn_a 代码...")
        local_repo_path = '/root/autodl-tmp/.cache/torch/hub/XingangPan_IBN-Net_master'
        
        # 核心改动：传入本地路径，并强制加上 source='local'
        backbone = torch.hub.load(local_repo_path, 'resnet50_ibn_a', pretrained=True, source='local')
        # ==============================================
        # # 加载官方 IBN，它在消除跨摄像头的光照、色偏差异上比普通 ResNet 强很多
        # print("=> 正在加载官方 resnet50_ibn_a 预训练模型...")
        # backbone = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # 剥离最后的下采样，保留 16x16 特征图，保留更多细节（对小目标、局部花纹很有用）
        self.layer4[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        
        # ==========================================
        # 🚀 知识图谱数据源：多分支分类器
        # ==========================================
        # 分支1：预测车辆专属身份 (用于 ReID 评测)
        self.classifier_id = nn.Linear(2048, num_classes, bias=False)
        # 分支2：预测颜色 (产生 KG 的 Node: Color)
        self.classifier_color = nn.Linear(2048, num_colors, bias=False)
        # 分支3：预测车型 (产生 KG 的 Node: VehicleType)
        self.classifier_type = nn.Linear(2048, num_types, bias=False)

        # 权重初始化
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)
        nn.init.normal_(self.classifier_id.weight, std=0.001)
        nn.init.normal_(self.classifier_color.weight, std=0.001)
        nn.init.normal_(self.classifier_type.weight, std=0.001)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x) 
        
        vbase = self.gap(features).view(features.size(0), -1)
        feat_bn = self.bottleneck(vbase)
        
        if self.training:
            # 训练阶段：给三个 Loss 提供预测结果
            return self.classifier_id(feat_bn), self.classifier_color(feat_bn), self.classifier_type(feat_bn), vbase
        else:
            # 推理阶段：返回计算距离用的特征，同时返回颜色和车型预测，用于统计准确率或后续图谱重排序！
            return feat_bn, self.classifier_color(feat_bn), self.classifier_type(feat_bn)


# ==========================================
# 5. 评测模块 (提取特征 + 统计属性准确率)
# ==========================================
def extract_features_and_eval_attr(model, dataloader, device):
    """
    不仅提取特征，还在遍历数据的同时测试颜色和车型的识别率
    """
    model.eval()
    feats_list, vids_list, cams_list = [], [], []
    
    correct_color, correct_type = 0, 0
    total_samples = 0
    
    with torch.no_grad():
        for imgs, vids, cams, colors, types, _ in dataloader:
            imgs = imgs.to(device)
            colors = colors.to(device)
            types = types.to(device)
            
            # 模型在 eval 模式下返回：特征，颜色 logits，车型 logits
            feats, pred_colors, pred_types = model(imgs)
            
            # ReID 的特征必须要 L2 归一化，这样计算余弦距离才准确
            feats = F.normalize(feats, p=2, dim=1) 
            
            feats_list.append(feats.cpu())
            vids_list.extend(vids)
            cams_list.extend(cams)
            
            # 【属性准确率统计逻辑】
            # 因为不是所有测试图都有真实颜色/车型标签，我们只统计真实标签 >= 0 的数据
            valid_mask = (colors >= 0) & (types >= 0)
            if valid_mask.sum() > 0:
                # torch.max 返回最大值和索引，索引即预测类别
                _, predicted_c = torch.max(pred_colors.data, 1)
                _, predicted_t = torch.max(pred_types.data, 1)
                
                # 累加正确数量
                correct_color += (predicted_c[valid_mask] == colors[valid_mask]).sum().item()
                correct_type += (predicted_t[valid_mask] == types[valid_mask]).sum().item()
                total_samples += valid_mask.sum().item()

    # 计算百分比
    acc_color = (correct_color / total_samples) * 100 if total_samples > 0 else 0
    acc_type = (correct_type / total_samples) * 100 if total_samples > 0 else 0
    
    return torch.cat(feats_list, dim=0), vids_list, cams_list, acc_color, acc_type

def evaluate_reid(q_feats, q_vids, q_cams, g_feats, g_vids, g_cams):
    """标准的 CMC 和 mAP 计算逻辑，纯矩阵运算，速度极快"""
    sim_matrix = torch.mm(q_feats, g_feats.t()).numpy()
    q_vids, q_cams, g_vids, g_cams = map(np.array, (q_vids, q_cams, g_vids, g_cams))
    num_q = sim_matrix.shape[0]
    indices = np.argsort(sim_matrix, axis=1)[:, ::-1]
    matches = (g_vids[indices] == q_vids[:, np.newaxis]).astype(np.int32)
    
    all_ap, all_cmc = [], []
    for q_idx in range(num_q):
        q_vid, q_cam = q_vids[q_idx], q_cams[q_idx]
        order = indices[q_idx]
        
        # 必须排除：同一辆车在同一个摄像头下拍的连续帧（这种算简单样本，不计入成绩）
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
# 6. 主流程架构
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help="数据集根目录")
    parser.add_argument('--epochs', type=int, default=30) 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='./IBNmodel_output')
    args = parser.parse_args()

    set_seed(42)
    device = check_device()
    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.save_dir, 'best_resnet50_ibn_mt.pth')
    
    # --- 1. 读取基础文件 ---
    print("\n=> 正在解析包含属性的多任务数据字典...")
    train_xml = os.path.join(args.root, 'train_label.xml')
    test_xml = os.path.join(args.root, 'test_label.xml')
    query_txt = os.path.join(args.root, 'name_query.txt')
    test_txt = os.path.join(args.root, 'name_test.txt')

    # 解析训练集，拿到类别数和映射表
    train_list, num_classes, num_colors, num_types, color_map, type_map = get_train_data(train_xml)
    
    # --- 2. 构建包含真实属性的 Query 和 Test 评测列表 ---
    # 利用测试集的 XML，提取出所有验证图的 车辆ID、摄像头ID、颜色ID、车型ID
    test_img_to_vid, test_img_to_cam, test_img_to_color, test_img_to_type = get_test_meta_map(test_xml, color_map, type_map)
    query_names = read_txt_lines(query_txt)
    test_names = read_txt_lines(test_txt)
    
    # 按照统一的 Tuple 格式组装 (img_name, vid, cam, cid, tid)
    # 遍历 query_names 中的每一个名字，去刚才建好的大字典里“查”它的颜色和车型
    real_q_list = [(
        name, 
        test_img_to_vid.get(name, "-1"), # 查真实车辆 ID
        test_img_to_cam.get(name, "-1"), # 查真实摄像头 ID
        test_img_to_color.get(name, -1), # 👈 查真实颜色 ID！查不到就给 -1
        test_img_to_type.get(name, -1)   # 👈 查真实车型 ID！查不到就给 -1
    ) for name in query_names] 
    
    real_test_list = [(
        name, 
        test_img_to_vid.get(name, "-1"), 
        test_img_to_cam.get(name, "-1"),
        test_img_to_color.get(name, -1),
        test_img_to_type.get(name, -1)
    ) for name in test_names]
    
    print(f"   [训练集] 样本数: {len(train_list)}")
    print(f"            身份类别数: {num_classes} | 颜色分类数: {num_colors} | 车型分类数: {num_types}")
    print(f"   [Query]  数量: {len(real_q_list)}")
    print(f"   [Test]   图库数量: {len(real_test_list)}\n")

    # --- 3. Dataloader 设定 ---
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

    train_sampler = RandomIdentitySampler(train_list, args.batch_size, num_instances=4)
    train_loader = DataLoader(
        ReIDDataset(os.path.join(args.root, 'image_train'), train_list, transform_train), 
        batch_size=args.batch_size, sampler=train_sampler, num_workers=4, drop_last=True
    )
    val_q_loader = DataLoader(ReIDDataset(os.path.join(args.root, 'image_query'), real_q_list, transform_test), batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_g_loader = DataLoader(ReIDDataset(os.path.join(args.root, 'image_test'), real_test_list, transform_test), batch_size=args.batch_size, shuffle=False, num_workers=4)

    # --- 4. 实例化模型与优化器 ---
    model = ResNet50IBN_ReID_MultiTask(
        num_classes=num_classes, 
        num_colors=num_colors, 
        num_types=num_types
    ).to(device)

    # 【四重损失集结】：主身份识别 + 特征距离约束 + 颜色感知 + 车型感知
    criterion_id = nn.CrossEntropyLoss()
    criterion_color = nn.CrossEntropyLoss()
    criterion_type = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=0.3)

    optimizer = optim.Adam(model.parameters(), lr=0.00035, weight_decay=5e-4)
    base_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=0.1)
    
    use_amp = (device.type == 'cuda')
    scaler = GradScaler('cuda') if use_amp else None

    best_mAP = 0.0
    epoch_losses = [] 

    # --- 5. 训练大循环 ---
    print("=> 🚀 启动多任务引擎训练 (ID + 颜色 + 车型)...")
    from tqdm import tqdm
    for epoch in range(args.epochs):
        # Warmup：前十轮慢慢加热学习率，防止大模型瞬间崩坏
        if epoch < 10:
            lr = 0.00035 * (epoch + 1) / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=False)
        
        for imgs, labels, cams, colors, types, _ in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            colors = colors.to(device)
            types = types.to(device)
            
            optimizer.zero_grad()
            
            # --- 核心前向与反向传播 ---
            if use_amp: # CUDA 环境用混合精度加速
                with autocast('cuda'):
                    logits_id, logits_color, logits_type, features = model(imgs) 
                    
                    loss_id = criterion_id(logits_id, labels)
                    loss_color = criterion_color(logits_color, colors)
                    loss_type = criterion_type(logits_type, types)
                    loss_triplet = criterion_triplet(features, labels)
                    
                    # 组合损失：知识图谱属性（颜色和车型）由于是副任务，给予 0.5 的惩罚权重
                    loss = loss_id + loss_triplet + loss_color + loss_type
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: # CPU 或 Mac MPS 走常规流程
                logits_id, logits_color, logits_type, features = model(imgs) 
                
                loss_id = criterion_id(logits_id, labels)
                loss_color = criterion_color(logits_color, colors)
                loss_type = criterion_type(logits_type, types)
                loss_triplet = criterion_triplet(features, labels)
                # 组合损失：知识图谱属性（颜色和车型）由于是副任务，给予 0.5 的惩罚权重
                loss = loss_id + loss_triplet + loss_color + loss_type
                
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()
            
            # 实时在控制台尾部刷新各项 Loss，方便你盯着看哪项没收敛
            pbar.set_postfix({
                'L_id': f"{loss_id.item():.2f}", 
                'L_tri': f"{loss_triplet.item():.2f}",
                'L_col': f"{loss_color.item():.2f}",
                'L_typ': f"{loss_type.item():.2f}"
            })

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss) 
        
        # --- 6. 每一个 Epoch 结束后的深度评测 ---
        # 现在，特征提取函数不仅能算特征，还能自动算出颜色和车型的正确率
        q_feats, q_vids, q_cams, q_col_acc, q_typ_acc = extract_features_and_eval_attr(model, val_q_loader, device)
        g_feats, g_vids, g_cams, g_col_acc, g_typ_acc = extract_features_and_eval_attr(model, val_g_loader, device)
        
        # 计算传统车辆 ReID 匹配度
        mAP, CMC = evaluate_reid(q_feats, q_vids, q_cams, g_feats, g_vids, g_cams)

        current_lr = optimizer.param_groups[0]['lr']
        
        # 极具美感的终端日志，一眼看尽所有关键指标
        print(f"Epoch [{epoch+1}/{args.epochs}] | LR: {current_lr:.6f} | TotalLoss: {avg_loss:.4f} \n"
              f"      🏆 [身份识别] mAP: {mAP:.4f} | Rank-1: {CMC[0]:.4f} | Rank-5: {CMC[4]:.4f} \n"
              f"      🎨 [属性感知(Query)] 颜色识别率: {q_col_acc:.1f}% | 车型识别率: {q_typ_acc:.1f}% \n"
              f"      🎨 [属性感知(Test) ] 颜色识别率: {g_col_acc:.1f}% | 车型识别率: {g_typ_acc:.1f}%")

        base_scheduler.step()
        
        # 保存最优模型
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(), 
                'num_classes': num_classes,
                'num_colors': num_colors,
                'num_types': num_types,
                'color_map': color_map, # 保存映射表，日后部署推理时需要用它反推属性名
                'type_map': type_map,
                'best_mAP': best_mAP
            }, best_ckpt_path)
            print("      [✔] 检测到 mAP 提升，最优多任务特征提取模型已保存！")

    # --- 7. 画图与收尾 ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Curve (Multi-Task ResNet50-IBN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plot_path = os.path.join(args.save_dir, 'loss_curve_mt.png')
    plt.savefig(plot_path)
    print(f"\n=> 🎉 多维度图谱感知网络训练全流程结束！最优模型保存至：{args.save_dir}")

if __name__ == '__main__':
    main()