# File Name: train_resnet_ibn.py
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

import timm # Powerful image model library for loading IBN variants

# ==========================================
# 1. Hardware Detection & Random Seed Setup
# ==========================================
def check_device():
    print("\n" + "=" * 60)
    print("=>  Hardware Environment Detection Report:")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"   [✔] GPU acceleration detected! ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"   [!] Warning: GPU not detected, running on CPU mode!")
    print("=" * 60 + "\n")
    return device

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"=> Global random seed fixed at: {seed}")

# ==========================================
# 2. Data Parsing Tools (For Real Query and Test)
# ==========================================
def parse_xml_compat(xml_path):
    """
    Compatible XML parser to handle GBK encoding issues.
    Replaces GB2312 declarations with UTF-8 to ensure successful parsing.
    """
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

# --- Added/Restored: Parsing functions required for real evaluation ---
def get_test_meta_map(test_xml):
    """Parses test_label.xml to create mapping of imageName to realID and cameraID"""
    items = parse_xml_compat(test_xml)
    vid_map, cam_map = {}, {}
    for item in items:
        img_name = item.get('imageName')
        vid_map[img_name] = item.get('vehicleID')
        cam_map[img_name] = item.get('cameraID')
    return vid_map, cam_map

def parse_query_meta(name):
    """Extracts ground truth from query filenames, e.g., 0002_c002_00030600_0.jpg -> (0002, c002)"""
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "-1", "-1"

def read_txt_lines(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# ==========================================
# 3. Dataset Definition
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
# PK Sampler designed for Triplet Loss
# ==========================================
class RandomIdentitySampler(torch.utils.data.sampler.Sampler):
    """
    Random Identity Sampler for ReID.
    Ensures each batch contains P identities, each with K images.
    """
    def __init__(self, data_list, batch_size, num_instances=4):
        self.data_list = data_list
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        
        # Parse train_list (imageName, vehicleID, cameraID)
        for index, item in enumerate(self.data_list):
            pid = item[1]  # Vehicle ID
            self.index_dic[pid].append(index)
            
        self.pids = list(self.index_dic.keys())
        
        # Estimate total epoch length
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
            # If an ID has fewer than K images, resample to fill the gap
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
# Triplet Loss Definition
# ==========================================
# ==========================================
# Triplet Loss Definition (Stable Version)
# ==========================================
class TripletLoss(nn.Module):
    """
    Triplet Loss with Online Hard Example Mining (OHEM)
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        inputs: Features from backbone (Batch_Size, Feature_Dim)
        targets: Identity labels (Batch_Size)
        """
        # Convert to FP32 to prevent overflow in Mixed Precision training
        inputs = inputs.float()
        n = inputs.size(0)
        
        # Core Fix: Use PyTorch optimized cdist for Euclidean distance.
        # Built-in gradient protection avoids NaN issues that lead to model crashes.
        dist = torch.cdist(inputs, inputs, p=2.0)

        # Generate mask to distinguish positive (same ID) and negative (different ID) samples
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        # Hard Mining: Find hardest positive and hardest negative for each anchor
        for i in range(n):
            # Hardest Positive (furthest distance within same class)
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  
            # Hardest Negative (closest distance across different classes)
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0)) 

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Calculate Margin Ranking Loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

# ==========================================
# 4. Core Network: ResNet50-IBN-a
# ==========================================
class ResNet50IBN_ReID(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50IBN_ReID, self).__init__()
        
        # 1. Loading official IBN-a pretrained model via torch.hub
        print("=> Loading official resnet50_ibn_a via torch.hub ...")
        # Weights automatically download from GitHub to local cache on first run
        backbone = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=True)
        
        # 2. Extract backbone layers (discarding ImageNet-specific avgpool and fc)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # 3. Core modification: Change stride of the last downsampling block to 1 
        # (Generates 16x16 high-resolution feature maps instead of 8x8)
        self.layer4[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)
        
        # 4. Custom ReID Classifier and BNNeck
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

        # Initializing weights for the BNNeck and Classifier
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        # Manual forward pass through backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x) 
        
        # Global Average Pooling and Flattening
        vbase = self.gap(features).view(features.size(0), -1)
        
        # Pass through BNNeck
        feat_bn = self.bottleneck(vbase)
        
        #  Core Change: Evaluation/Inference must return feat_bn, not vbase!
        return (self.classifier(feat_bn), vbase) if self.training else feat_bn

# ==========================================
# 5. Evaluation Functions (Feature Extraction & CMC/mAP)
# ==========================================
def extract_features(model, dataloader, device):
    model.eval()
    feats_list, vids_list, cams_list = [], [], []
    with torch.no_grad():
        for imgs, vids, cams, _ in dataloader:
            imgs = imgs.to(device)
            feats = model(imgs)
            feats = F.normalize(feats, p=2, dim=1) # L2 Normalization is mandatory
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
        # Remove junk data (same ID and same Camera)
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
# 6. Main Process: Real Query Evaluation & Fine-tuning
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help="Dataset root directory")
    parser.add_argument('--epochs', type=int, default=30) 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='./IBNmodel_output')
    args = parser.parse_args()

    set_seed(42)
    device = check_device()
    os.makedirs(args.save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.save_dir, 'best_resnet50_ibn.pth')
    
    # ---------------- Core Modification: Build real evaluation manifest ----------------
    print("\n=> Parsing real Train, Query, and Test data dictionaries...")
    train_xml = os.path.join(args.root, 'train_label.xml')
    test_xml = os.path.join(args.root, 'test_label.xml')
    query_txt = os.path.join(args.root, 'name_query.txt')
    test_txt = os.path.join(args.root, 'name_test.txt')

    train_list, num_classes = get_train_data(train_xml)
    
    # Parse underlying mapping relationships
    test_img_to_vid, test_img_to_cam = get_test_meta_map(test_xml)
    query_names = read_txt_lines(query_txt)
    test_names = read_txt_lines(test_txt)
    
    # Generate real lists: (imageName, vehicleID, cameraID)
    real_q_list = [(name, parse_query_meta(name)[0], parse_query_meta(name)[1]) for name in query_names] 
    real_test_list = [(name, test_img_to_vid.get(name, "-1"), test_img_to_cam.get(name, "-1")) for name in test_names]
    
    print(f"   [Train] Sample count: {len(train_list)} (Class count: {num_classes})")
    print(f"   [Query] Query count: {len(real_q_list)}")
    print(f"   [Test]  Gallery count: {len(real_test_list)}\n")
    # --------------------------------------------------------------

    # Image Preprocessing
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataloader points to real directories
    # --- Core Change: Use PK Sampler instead of standard random shuffle ---
    # Note: When using a sampler, shuffle=True must be avoided in DataLoader.
    train_sampler = RandomIdentitySampler(train_list, args.batch_size, num_instances=4)
    train_loader = DataLoader(
        ReIDDataset(os.path.join(args.root, 'image_train'), train_list, transform_train), 
        batch_size=args.batch_size, 
        sampler=train_sampler,  # Load the custom sampler
        num_workers=4,
        drop_last=True          # Ensures batch size is strictly 64 to prevent Triplet calculation errors
    )
    val_q_loader = DataLoader(ReIDDataset(os.path.join(args.root, 'image_query'), real_q_list, transform_test), batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_g_loader = DataLoader(ReIDDataset(os.path.join(args.root, 'image_test'), real_test_list, transform_test), batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = ResNet50IBN_ReID(num_classes=num_classes).to(device)
# --- 1. Define Dual Loss Functions ---
    criterion_id = nn.CrossEntropyLoss()
    criterion_triplet = TripletLoss(margin=0.3)
# Set loss weights (usually 1:1)
    weight_id = 1.0
    weight_triplet = 1.0
# -------------------------
    optimizer = optim.Adam(model.parameters(), lr=0.00035, weight_decay=5e-4)
    milestones = [70, 120]
    base_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    best_mAP = 0.0
    epoch_losses = [] 

    print("=> 🚀 Starting training for ResNet50-IBN with real Query set evaluation...")
    from tqdm import tqdm
    for epoch in range(args.epochs):
        # --- Learning Rate Warmup Logic ---
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
                # --- 2. Get Dual Outputs ---
                # Logits used for ID Loss classification; features (vbase) used for Triplet distance calculation
                logits, features = model(imgs) 
                
                # --- 3. Compute Composite Loss ---
                loss_id = criterion_id(logits, labels)
                loss_triplet = criterion_triplet(features, labels)
                
                loss = weight_id * loss_id + weight_triplet * loss_triplet
                # ------------------------
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            # Display individual losses in the progress bar
            pbar.set_postfix({
                'L_id': f"{loss_id.item():.3f}", 
                'L_tri': f"{loss_triplet.item():.3f}"
            })

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss) 
        
        # Evaluate using real Query vs Test set
        q_feats, q_vids, q_cams = extract_features(model, val_q_loader, device)
        g_feats, g_vids, g_cams = extract_features(model, val_g_loader, device)
        mAP, CMC = evaluate_reid(q_feats, q_vids, q_cams, g_feats, g_vids, g_cams)

        # Get current effective learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Consolidated log for each epoch
        print(f"Epoch [{epoch+1}/{args.epochs}] | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | "
              f"Real_mAP: {mAP:.4f} | Rank-1: {CMC[0]:.4f} | Rank-5: {CMC[4]:.4f} | Rank-10: {CMC[9]:.4f}")

        #  Step the scheduler unconditionally to keep PyTorch counters aligned
        base_scheduler.step()
        
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(), 
                'num_classes': num_classes,
                'best_mAP': best_mAP
            }, best_ckpt_path)
            print("   [✔] Optimized IBN model updated and saved!")

    # Plot and save Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Curve (ResNet50-IBN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plot_path = os.path.join(args.save_dir, 'loss_curve.png')
    plt.savefig(plot_path)
    print(f"\n=>  Training complete! Optimized model and loss curve saved to: {args.save_dir}")

if __name__ == '__main__':
    main()
