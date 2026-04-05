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

import timm # 引入强大的图像模型库，用于加载 IBN

# ==========================================
# 1. 硬件环境检测与随机种子设定
# ==========================================
def check_device():
    print("\n" + "=" * 60)
    print("=> 🖥️ 硬件环境检测报告:")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"   [✔] 成功检测到 GPU 加速环境！({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"   [!] 警告: 未检测到 GPU，当前正在使用纯 CPU 运行！")
    print("=" * 60 + "\n")
    return device

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"=> 已固定全局随机种子 Seed: {seed}")

# ==========================================
# 2. 数据解析工具 (专门针对真实 Query 和 Test)
# ==========================================
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

# --- 新增/恢复：真实评测需要的解析函数 ---
def get_test_meta_map(test_xml):
    """解析 test_label.xml，建立图片名到 真实ID 和 摄像头ID 的映射字典"""
    items = parse_xml_compat(test_xml)
    vid_map, cam_map = {}, {}
    for item in items:
        img_name = item.get('imageName')
        vid_map[img_name] = item.get('vehicleID')
        cam_map[img_name] = item.get('cameraID')
    return vid_map, cam_map

def parse_query_meta(name):
    """解析 query 图片名提取真实信息，例如 0002_c002_00030600_0.jpg -> (0002, c002)"""
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "-1", "-1"

def read_txt_lines(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# ==========================================
# 3. Dataset 定义
# ==========================================
class ReIDDataset(Dataset):
    def __init__(self, img_dir, data_list, transform=None):
        self.img_dir = img_dir
        self.data_list = data_list 
        self.transform = transform

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        img_name, label_or_vid, cam_id = self.data_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label_or_vid, cam_id, img_name

# ==========================================
# 为三元损失设计的 PK 采样器
# ==========================================
class RandomIdentitySampler(torch.utils.data.sampler.Sampler):
    """
    专门为 ReID 设计的 PK 采样器
    保证每个 Batch 内包含 P 个 ID，每个 ID 包含 K 张图片
    """
    def __init__(self, data_list, batch_size, num_instances=4):
        self.data_list = data_list
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        
        # 解析传入的 train_list (图片名, 车辆ID, 摄像头ID)
        for index, item in enumerate(self.data_list):
            pid = item[1]  # 车辆 ID
            self.index_dic[pid].append(index)
            
        self.pids = list(self.index_dic.keys())
        
        # 估算一个 epoch 的总长度
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            # 如果某个 ID 的总图片不够 K 张，就随机重采样补齐，保证能凑够一对
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

# ==========================================
# 三元损失的定义
# ==========================================
# ==========================================
# 三元损失的定义 (终极稳定版)
# ==========================================
class TripletLoss(nn.Module):
    """
    带有在线难例挖掘 (Online Hard Example Mining) 的三元组损失函数
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        inputs: 骨干网络输出的特征 (Batch_Size, Feature_Dim)
        targets: 真实的身份标签 (Batch_Size)
        """
        # 必须转为 FP32，防止混合精度下的溢出
        inputs = inputs.float()
        n = inputs.size(0)
        
        # 🚨 核心修复点：放弃手写公式，使用 PyTorch 官方高度优化的 cdist 算欧氏距离
        # 自带梯度保护，彻底杜绝 NaN 导致模型卡死的问题！
        dist = torch.cdist(inputs, inputs, p=2.0)

        # 生成 Mask 区分正样本(同ID)和负样本(不同ID)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        # 难例挖掘：寻找每个 Anchor 的最难正样本和最难负样本
        for i in range(n):
            # Hardest Positive (距离最远的同类)
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  
            # Hardest Negative (距离最近的异类)
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0)) 

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # 计算 Margin Ranking Loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
# ==========================================
# 4. 核心网络：真正的 ResNet50-IBN-a
# ==========================================
class ResNet50IBN_ReID(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50IBN_ReID, self).__init__()
        
        # 1. 改为通过 torch.hub 加载官方的 IBN-a 预训练模型
        print("=> 正在通过 torch.hub 加载官方 resnet50_ibn_a ...")
        # 首次运行会自动从 GitHub 下载权重到本地缓存
        backbone = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        
        # 2. 逐层提取 backbone (舍弃掉原本用于 ImageNet 的 avgpool 和 fc)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # 3. 核心魔改：将最后一次下采样的 stride 改为 1 (生成 16x16 而非 8x8 的高分辨率特征图)
        self.layer4[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)
        
        # 4. ReID 自定义分类器与 BNNeck
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

        # 👇 在 __init__ 的最后加上这三行权重初始化代码
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        # 手动通过骨干网络的前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x) 
        
        # 全局平均池化并展平
        vbase = self.gap(features).view(features.size(0), -1)
        
        # 通过 BNNeck
        feat_bn = self.bottleneck(vbase)
        
        # 🚨 核心修改：评测/推理时，必须返回 feat_bn 而不是 vbase！
        return (self.classifier(feat_bn), vbase) if self.training else feat_bn

# ==========================================
# 5. 评测函数 (提取特征 & CMC/mAP)
# ==========================================
def extract_features(model, dataloader, device):
    model.eval()
    feats_list, vids_list, cams_list = [], [], []
    with torch.no_grad():
        for imgs, vids, cams, _ in dataloader:
            imgs = imgs.to(device)
            feats = model(imgs)
            feats = F.normalize(feats, p=2, dim=1) # 必须做 L2 归一化
            feats_list.append(feats.cpu())
            vids_list.extend(vids)
            cams_list.extend(cams)
    return torch.cat(feats_list, dim=0), vids_list, cams_list

def evaluate_reid(q_feats, q_vids, q_cams, g_feats, g_vids, g_cams):
    sim_matrix = torch.mm(q_feats, g_feats.t()).numpy()
    q_vids, q_cams, g_vids, g_cams = map(np.array, (q_vids, q_cams, g_vids, g_cams))
    num_q = sim_matrix.shape[0]
    indices = np.argsort(sim_matrix, axis=1)[:, ::-1]
    matches = (g_vids[indices] == q_vids[:, np.newaxis]).astype(np.int32)
    
    all_ap, all_cmc = [], []
    for q_idx in range(num_q):
        q_vid, q_cam = q_vids[q_idx], q_cams[q_idx]
        order = indices[q_idx]
        # 排除同 ID 且同摄像头的 junk 数据
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
# 6. 主流程：真实 Query 评测与微调
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
    best_ckpt_path = os.path.join(args.save_dir, 'best_resnet50_ibn.pth')
    
    # ---------------- 核心修改点：构建真实评测名单 ----------------
    print("\n=> 正在解析真实 Train, Query, Test 数据字典...")
    train_xml = os.path.join(args.root, 'train_label.xml')
    test_xml = os.path.join(args.root, 'test_label.xml')
    query_txt = os.path.join(args.root, 'name_query.txt')
    test_txt = os.path.join(args.root, 'name_test.txt')

    train_list, num_classes = get_train_data(train_xml)
    
    # 解析出底层映射关系
    test_img_to_vid, test_img_to_cam = get_test_meta_map(test_xml)
    query_names = read_txt_lines(query_txt)
    test_names = read_txt_lines(test_txt)
    
    # 生成真实的 List：(图片名, 车辆ID, 摄像头ID)
    real_q_list = [(name, parse_query_meta(name)[0], parse_query_meta(name)[1]) for name in query_names] 
    real_test_list = [(name, test_img_to_vid.get(name, "-1"), test_img_to_cam.get(name, "-1")) for name in test_names]
    
    print(f"   [Train] 训练样本数: {len(train_list)} (ID分类数: {num_classes})")
    print(f"   [Query] 真实查询数: {len(real_q_list)}")
    print(f"   [Test]  底库样本数: {len(real_test_list)}\n")
    # --------------------------------------------------------------

    # 图像预处理
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 🚨 Dataloader 彻底指向真实的文件夹
    # --- 核心修改：使用 PK 采样器替换原本的随机打乱 ---
    # 注意：使用了 sampler 之后，DataLoader 里面决不能再写 shuffle=True
    train_sampler = RandomIdentitySampler(train_list, args.batch_size, num_instances=4)
    train_loader = DataLoader(
        ReIDDataset(os.path.join(args.root, 'image_train'), train_list, transform_train), 
        batch_size=args.batch_size, 
        sampler=train_sampler,  # 👈 挂载刚刚写的采样器
        num_workers=4,
        drop_last=True          # 👈 保证每个 batch 严格等于 64，防止最后一个 batch 尺寸不对导致 Triplet 计算报错
    )
    val_q_loader = DataLoader(ReIDDataset(os.path.join(args.root, 'image_query'), real_q_list, transform_test), batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_g_loader = DataLoader(ReIDDataset(os.path.join(args.root, 'image_test'), real_test_list, transform_test), batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = ResNet50IBN_ReID(num_classes=num_classes).to(device)
# --- 1. 定义双重损失函数 ---
    criterion_id = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=0.3)
# 设定两个 Loss 的权重比例（通常 1:1 即可）
    weight_id = 1.0
    weight_triplet = 1.0
# -------------------------
    optimizer = optim.Adam(model.parameters(), lr=0.00035, weight_decay=5e-4)
    milestones = [70, 120]
    base_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    best_mAP = 0.0
    epoch_losses = [] 

    print("=> 🚀 开始训练 ResNet50-IBN，并将在真实 Query 集上评测...")
    from tqdm import tqdm
    for epoch in range(args.epochs):
        # --- 学习率 Warmup 逻辑 ---
        if epoch < 10:
            lr = 0.00035 * (epoch + 1) / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        # ------------------------
        
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=False)
        
        for imgs, labels, _, _ in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                # --- 2. 获取双输出 ---
                # logits 用于 ID Loss 分类；features (即 vbase) 用于 Triplet Loss 计算距离
                logits, features = model(imgs) 
                
                # --- 3. 计算组合 Loss ---
                loss_id = criterion_id(logits, labels)
                loss_triplet = criterion_triplet(features, labels)
                
                loss = weight_id * loss_id + weight_triplet * loss_triplet
                # ------------------------
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
    # 顺便在进度条里分开显示两个 loss，方便观察收敛情况
            pbar.set_postfix({
                'L_id': f"{loss_id.item():.3f}", 
                'L_tri': f"{loss_triplet.item():.3f}"
            })

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss) 
        
        # 使用真实 Query vs Test 进行评测
        q_feats, q_vids, q_cams = extract_features(model, val_q_loader, device)
        g_feats, g_vids, g_cams = extract_features(model, val_g_loader, device)
        mAP, CMC = evaluate_reid(q_feats, q_vids, q_cams, g_feats, g_vids, g_cams)

        # 先获取本轮实际使用的学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 将所有重要指标整合到一行打印，保持终端极其清爽
        print(f"Epoch [{epoch+1}/{args.epochs}] | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | "
              f"Real_mAP: {mAP:.4f} | Rank-1: {CMC[0]:.4f} | Rank-5: {CMC[4]:.4f} | Rank-10: {CMC[9]:.4f}")

# 🚨 核心修复：无条件步进 Scheduler，保持 PyTorch 内部计数器对齐！
        # (因为我们在下一轮开头还会重新计算 Warmup LR，所以它不会干扰预热逻辑)
        base_scheduler.step()
        
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(), 
                'num_classes': num_classes,
                'best_mAP': best_mAP
            }, best_ckpt_path)
            print("   [✔] 最优 IBN 模型已更新并保存！")

    # 绘制并保存 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Curve (ResNet50-IBN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plot_path = os.path.join(args.save_dir, 'loss_curve.png')
    plt.savefig(plot_path)
    print(f"\n=> 🎉 训练全流程结束！最优模型和 Loss 曲线已保存至：{args.save_dir}")

if __name__ == '__main__':
    main()