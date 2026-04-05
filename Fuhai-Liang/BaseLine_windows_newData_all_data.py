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
# [NEW] Fix all random seeds for absolute reproducibility
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"=> Global random seed fixed at: {seed}")

# ==========================================
# 1. Data Parsing Tools (GBK/GB2312 Compatible)
# ==========================================
def parse_xml_compat(xml_path):
    """
    Parses XML files while handling GBK/GB2312 encoding issues.
    Replaces encoding declarations to UTF-8 to ensure successful parsing.
    """
    with open(xml_path, 'r', encoding='gbk', errors='ignore') as f:
        xml_content = f.read()
    xml_content = xml_content.replace('encoding="gb2312"', 'encoding="utf-8"')
    xml_content = xml_content.replace('encoding="GB2312"', 'encoding="utf-8"')
    root = ET.fromstring(xml_content)
    return root.find('Items').findall('Item')

def get_train_data(train_xml):
    """Parses training XML and maps vehicleIDs to continuous indices (0 to N-1)."""
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
    """
    Prepares validation query and gallery lists from the test XML.
    Used for monitoring model performance during the training phase.
    """
    items = parse_xml_compat(test_xml)
    id_dict = defaultdict(list)
    for item in items:
        id_dict[item.get('vehicleID')].append({
            'img_name': item.get('imageName'),
            'cam_id': item.get('cameraID')
        })
        
    unique_vids = sorted(list(id_dict.keys())) # Sort to maintain consistency
    random.seed(seed) # Ensure identical sampling across runs
    
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

# ===== [Mod 1] NEW: Parse Test Metadata (Returns vid and cam mapping dictionaries) =====
def get_test_meta_map(test_xml):
    items = parse_xml_compat(test_xml)
    vid_map, cam_map = {}, {}
    for item in items:
        img_name = item.get('imageName')
        vid_map[img_name] = item.get('vehicleID')
        cam_map[img_name] = item.get('cameraID')
    return vid_map, cam_map

# ===== [Mod 2] NEW: Extract vehicleID and cameraID from Query filename =====
def parse_query_meta(name):
    """
    Parses query image names, e.g., 0002_c002_00030600_0.jpg
    Returns (vehicleID, cameraID)
    """
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "-1", "-1"

def read_txt_lines(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# ==========================================
# 2. Dataset Definition
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
# 3. ResNet50 Baseline Model
# ==========================================
class ResNet50ReID(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50ReID, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the last two layers (AvgPool and FC) to get a feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        # BNNeck architecture for ReID: significantly improves performance
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        features = self.backbone(x)
        features = self.gap(features).view(features.size(0), -1)
        feat_bn = self.bottleneck(features)
        # Return both logits (for CrossEntropy) and features (for Triplet/Metric Learning) during training
        return (self.classifier(feat_bn), features) if self.training else feat_bn

# ==========================================
# 4. Feature Extraction & Standard ReID Evaluation
# ==========================================
def extract_features(model, dataloader, device):
    from tqdm import tqdm
    model.eval()
    feats_list, vids_list, cams_list, names_list = [], [], [], []
    with torch.no_grad():
        for imgs, vids, cams, names in tqdm(dataloader, desc="Extracting Features", leave=False):
            imgs = imgs.to(device)
            feats = model(imgs)
            # L2 Normalization is crucial for cosine similarity-based retrieval
            feats = F.normalize(feats, p=2, dim=1)
            feats_list.append(feats.cpu())
            vids_list.extend(vids)
            cams_list.extend(cams)
            names_list.extend(names)
    return torch.cat(feats_list, dim=0), vids_list, cams_list, names_list

def evaluate_reid(q_feats, q_vids, q_cams, g_feats, g_vids, g_cams):
    """Calculates Mean Average Precision (mAP) and CMC curve (Rank-N)."""
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
        # Standard ReID filtering: samples with same ID and same Camera are treated as junk
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
# 5. Main Process
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help="Root directory of dataset")
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

    # Path configurations
    train_dir = os.path.join(args.root, 'image_train')
    test_dir = os.path.join(args.root, 'image_test')
    query_dir = os.path.join(args.root, 'image_query')
    train_xml = os.path.join(args.root, 'train_label.xml')
    test_xml = os.path.join(args.root, 'test_label.xml')
    query_txt = os.path.join(args.root, 'name_query.txt')
    test_txt = os.path.join(args.root, 'name_test.txt')

    print("\n=> Parsing data dictionaries...")
    train_list, num_classes = get_train_data(train_xml)
    test_eval_q, test_eval_g = get_test_eval_data(test_xml, args.eval_k_queries, args.seed)
    
    query_names = read_txt_lines(query_txt)
    test_names = read_txt_lines(test_txt)
    
    # ===== [Mod 3] Apply parsing functions to build lists with ground truth labels =====
    test_img_to_vid, test_img_to_cam = get_test_meta_map(test_xml)
    # For query: use parse_query_meta to extract real vid and cam from filenames
    real_q_list = [(name, parse_query_meta(name)[0], parse_query_meta(name)[1]) for name in query_names] 
    # For gallery: map real vid and cam from the test_xml
    real_test_list = [(name, test_img_to_vid.get(name, "-1"), test_img_to_cam.get(name, "-1")) for name in test_names]

    print(f"   [Train] Sample count: {len(train_list)} (ID count: {num_classes})")
    print(f"   [Test Eval] Query count: {len(test_eval_q)}, Gallery count: {len(test_eval_g)}")
    print(f"   [Real Query] Full Ground Truth query count: {len(real_q_list)}\n")

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
    # Training Loop
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

        # --- Evaluate mAP and CMC on Test set after each Epoch ---
        print(f"\n=> Epoch {epoch+1} finished, starting Test evaluation...")
        q_feats, q_vids, q_cams, _ = extract_features(model, val_q_loader, device)
        g_feats, g_vids, g_cams, _ = extract_features(model, val_g_loader, device)
        mAP, CMC = evaluate_reid(q_feats, q_vids, q_cams, g_feats, g_vids, g_cams)
        
        print(f"   [Epoch {epoch+1} Metrics] mAP: {mAP:.4f} | Rank-1: {CMC[0]:.4f} | Rank-5: {CMC[4]:.4f} | Rank-10: {CMC[9]:.4f}")

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
            print(f"   [✔] New best mAP found ({best_mAP:.4f}), saved to {best_ckpt_path}\n")
        else:
            print(f"   [ ] No improvement over best mAP ({best_mAP:.4f} at Epoch {best_epoch})\n")

    # ==========================================
    # Final Retrieval Prediction & Excel Export
    # ==========================================
    print("=" * 60)
    print(f"=> Training complete! best epoch = {best_epoch}, best mAP = {best_mAP:.4f}")
    
    print(f"=> Loading best weights from {best_ckpt_path} ...")
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> best_baseline.pth loaded. Preparing Excel and final metrics for real Query...")
    print("=" * 60)

    real_q_loader = DataLoader(ReIDDataset(query_dir, real_q_list, transform_test), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    real_g_loader = DataLoader(ReIDDataset(test_dir, real_test_list, transform_test), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Extract features
    rq_feats, rq_vids, rq_cams, rq_names = extract_features(model, real_q_loader, device)
    rg_feats, rg_vids, rg_cams, rg_names = extract_features(model, real_g_loader, device)

    print("\n=> Calculating similarity and generating Excel report...")
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
    print(f"=> [Success] Excel prediction results saved to: {output_excel}")

    # ===== [Mod 4] NEW: Standard mAP/CMC evaluation on all full real queries =====
    print("\n=> ==========================================================")
    print(f"=> [Final Evaluation] Starting metrics calculation for {len(rq_names)} Query vs {len(rg_names)} Test samples")
    print("=> Standards: Matching based on vehicleID + cameraID filtering (same camera = junk)")
    
    mAP_final, CMC_final = evaluate_reid(rq_feats, rq_vids, rq_cams, rg_feats, rg_vids, rg_cams)
    
    print(f"   [Real Query Final Metrics] mAP: {mAP_final:.4f} | Rank-1: {CMC_final[0]:.4f} | Rank-5: {CMC_final[4]:.4f} | Rank-10: {CMC_final[9]:.4f}")
    print("=> ==========================================================\n")

if __name__ == '__main__':
    main()
