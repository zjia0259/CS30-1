#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIC21 Track2 ReID baseline with local validation split (P*K sampler)
- Train: image_train + train_label.xml (labeled)
- Local Val: split IDs from train_label.xml, build val_query/val_gallery from held-out IDs
- Model: ResNet50 + BNNeck + CE + BatchHard Triplet
- Each epoch prints val mAP / Rank@1/5/10

Folder structure (as your screenshot):
root/
  image_train/
  image_query/
  image_test/
  name_train.txt
  name_query.txt
  name_test.txt
  train_label.xml
  (optional) query_label.xml / test_label.xml (often unlabeled)
"""

import os
import time
import argparse
import random
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def cam_to_int(cam_str: str) -> int:
    if cam_str is None:
        return -1
    cam_str = cam_str.strip()
    if cam_str.startswith(("c", "C")):
        return int(cam_str[1:])
    # sometimes like "001"
    try:
        return int(cam_str)
    except Exception:
        return -1


def robust_parse_xml_items(xml_path: str) -> List[dict]:
    """
    Parse XML with possible gb2312/gbk declaration (Python3.12 issue).
    Return list of item.attrib dict.
    Expected (train_label.xml):
      <Item imageName="xxxx.jpg" vehicleID="0001" cameraID="c001" ... />
    """
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        raw = open(xml_path, "rb").read()
        decoded = None
        for enc in ("utf-8", "gbk", "gb2312", "latin1"):
            try:
                decoded = raw.decode(enc)
                break
            except Exception:
                continue
        if decoded is None:
            decoded = raw.decode("utf-8", errors="ignore")

        decoded = decoded.replace('encoding="gb2312"', 'encoding="utf-8"')
        decoded = decoded.replace("encoding='gb2312'", "encoding='utf-8'")
        decoded = decoded.replace('encoding="GB2312"', 'encoding="utf-8"')
        decoded = decoded.replace("encoding='GB2312'", "encoding='utf-8'")
        root = ET.fromstring(decoded)

    items = []
    for it in root.findall(".//Item"):
        items.append(dict(it.attrib))
    return items


def load_name_list(txt_path: str) -> List[str]:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]


# =========================================================
# Local split (ID-level) and val query/gallery construction
# =========================================================
def build_local_split_from_train_xml(
    root: str,
    train_xml: str,
    name_train_txt: str,
    val_id_count: int = 40,
    seed: int = 42,
    query_per_id: int = 1
):
    """
    Split vehicle IDs into train_ids and val_ids.
    Then build:
      train_list: [(img_name, label_int, cam_int)]
      val_query_list: [(img_name, vid_str, cam_int)]
      val_gallery_list: [(img_name, vid_str, cam_int)]
    Notes:
      - query/gallery are built from held-out IDs only
      - for each val ID, randomly pick query_per_id images as query, rest as gallery
    """
    xml_path = os.path.join(root, train_xml)
    items = robust_parse_xml_items(xml_path)

    # optional filter by name_train.txt (safer)
    name_train_path = os.path.join(root, name_train_txt)
    names_in_txt = set(load_name_list(name_train_path))

    # group by vehicleID
    id_dict = defaultdict(list)  # vid -> list of (img_name, cam_int)
    for it in items:
        img = it.get("imageName")
        vid = it.get("vehicleID")
        cam = it.get("cameraID")
        if not img or not vid:
            continue
        if img not in names_in_txt:
            continue
        id_dict[str(vid)].append((img, cam_to_int(cam)))

    all_ids = list(id_dict.keys())
    rnd = random.Random(seed)
    rnd.shuffle(all_ids)

    if val_id_count <= 0:
        val_id_count = max(1, int(0.05 * len(all_ids)))

    val_ids = set(all_ids[-val_id_count:])
    train_ids = [vid for vid in all_ids if vid not in val_ids]

    # map train IDs to contiguous labels
    id_map = {vid: i for i, vid in enumerate(train_ids)}

    train_list = []
    for vid in train_ids:
        label = id_map[vid]
        for (img, cam) in id_dict[vid]:
            train_list.append((img, label, cam))

    val_query_list = []
    val_gallery_list = []
    for vid in sorted(list(val_ids)):
        imgs = id_dict[vid]
        if len(imgs) == 0:
            continue
        # pick query_per_id images as query
        qk = min(query_per_id, len(imgs))
        q_idx = set(rnd.sample(range(len(imgs)), qk))
        for i, (img, cam) in enumerate(imgs):
            if i in q_idx:
                val_query_list.append((img, vid, cam))
            else:
                val_gallery_list.append((img, vid, cam))

        # edge case: if all images became query (rare when len=1)
        if len(val_gallery_list) == 0:
            # move last query to gallery to enable evaluation
            img, vid2, cam2 = val_query_list.pop()
            val_gallery_list.append((img, vid2, cam2))

    return train_list, val_query_list, val_gallery_list, len(train_ids)


# =========================================================
# Dataset
# =========================================================
class AICTrainDataset(Dataset):
    def __init__(self, root: str, train_list: List[Tuple[str, int, int]], transform=None):
        self.root = root
        self.img_dir = os.path.join(root, "image_train")
        self.data = train_list  # (img_name, label_int, cam_int)
        self.transform = transform

        self.labels = [x[1] for x in self.data]  # for PK sampler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        img_name, label, cam = self.data[idx]
        path = os.path.join(self.img_dir, img_name)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class AICValDataset(Dataset):
    def __init__(self, root: str, split: str, data_list: List[Tuple[str, str, int]], transform=None):
        """
        split: 'query' or 'gallery' (both read from image_train for local val)
        data_list: (img_name, vid_str, cam_int)
        """
        self.root = root
        self.img_dir = os.path.join(root, "image_train")
        self.split = split
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        img_name, vid, cam = self.data[idx]
        path = os.path.join(self.img_dir, img_name)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, vid, cam, img_name


# =========================================================
# P*K Sampler
# =========================================================
class PKSampler(Sampler[int]):
    """
    Each batch: P identities, K instances per identity.
    Here labels are contiguous ints already.
    """
    def __init__(self, labels: List[int], P: int, K: int, iters_per_epoch: int, seed: int = 42):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.P = P
        self.K = K
        self.iters_per_epoch = iters_per_epoch
        self.seed = seed

        self.index_dict = defaultdict(list)
        for idx, lab in enumerate(self.labels):
            self.index_dict[int(lab)].append(idx)
        self.unique_ids = list(self.index_dict.keys())
        if len(self.unique_ids) < P:
            raise ValueError(f"Not enough IDs for P={P}. Only {len(self.unique_ids)} labels exist.")

    def __len__(self):
        return self.iters_per_epoch * self.P * self.K

    def __iter__(self):
        rnd = random.Random(self.seed + int(time.time()))
        out = []
        for _ in range(self.iters_per_epoch):
            chosen = rnd.sample(self.unique_ids, self.P)
            for pid in chosen:
                pool = self.index_dict[pid]
                if len(pool) >= self.K:
                    idxs = rnd.sample(pool, self.K)
                else:
                    idxs = [rnd.choice(pool) for _ in range(self.K)]
                out.extend(idxs)
        return iter(out)


# =========================================================
# Model: ResNet50 + BNNeck
# =========================================================
class PaperBaseline(nn.Module):
    """
    backbone -> GAP -> feat_base (Triplet)
                 -> BNNeck -> feat_bn -> classifier (CE)
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        net = resnet50(weights=weights)

        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bnneck = nn.BatchNorm1d(2048)
        self.bnneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(2048, num_classes, bias=False)

    def forward(self, x):
        feat_map = self.backbone(x)
        feat_base = self.gap(feat_map).view(x.size(0), -1)
        feat_bn = self.bnneck(feat_base)

        if self.training:
            logits = self.classifier(feat_bn)
            return logits, feat_base, feat_bn
        else:
            return feat_bn


# =========================================================
# Triplet Loss: batch-hard
# =========================================================
class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.ranking = nn.MarginRankingLoss(margin=margin)

    @staticmethod
    def pairwise_distance(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        xx = (x * x).sum(dim=1, keepdim=True)
        dist = xx + xx.t() - 2.0 * (x @ x.t())
        dist = torch.clamp(dist, min=0.0)
        dist = torch.sqrt(dist + eps)
        return dist

    def forward(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = self.pairwise_distance(feat)
        N = dist.size(0)
        labels = labels.view(N, 1)

        is_pos = labels.eq(labels.t())
        is_neg = ~is_pos
        is_pos.fill_diagonal_(False)

        dist_pos = dist.clone()
        dist_pos[~is_pos] = -1e9
        hardest_pos, _ = dist_pos.max(dim=1)

        dist_neg = dist.clone()
        dist_neg[~is_neg] = 1e9
        hardest_neg, _ = dist_neg.min(dim=1)

        y = torch.ones_like(hardest_neg)
        return self.ranking(hardest_neg, hardest_pos, y)


# =========================================================
# ReID evaluation (mAP/CMC) with same-camera filtering
# =========================================================
def compute_mAP(index, good_index, junk_index):
    ap = 0.0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True).flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        old_precision = i * 1.0 / rows_good[i] if rows_good[i] != 0 else 1.0
        ap += d_recall * (old_precision + precision) / 2.0
    return ap, cmc


def evaluate_one_query(score_row, q_vid, q_cam, g_vids, g_cams):
    # sort high->low
    index = np.argsort(score_row)[::-1]

    query_index = np.argwhere(g_vids == q_vid)
    camera_index = np.argwhere(g_cams == q_cam)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=False)
    junk_index = np.intersect1d(query_index, camera_index)  # same id same cam

    return compute_mAP(index, good_index, junk_index)


def calculate_result(g_feat, g_vid, g_cam, q_feat, q_vid, q_cam):
    qf = torch.from_numpy(q_feat).float()
    gf = torch.from_numpy(g_feat).float()
    score = torch.mm(qf, gf.t()).numpy()

    CMC = torch.IntTensor(len(g_vid)).zero_()
    ap_sum = 0.0
    valid_q = 0

    g_vid = np.asarray(g_vid)
    g_cam = np.asarray(g_cam)
    q_vid = np.asarray(q_vid)
    q_cam = np.asarray(q_cam)

    for i in range(len(q_vid)):
        ap_tmp, cmc_tmp = evaluate_one_query(score[i], q_vid[i], q_cam[i], g_vid, g_cam)
        if cmc_tmp[0] == -1:
            continue
        CMC += cmc_tmp
        ap_sum += ap_tmp
        valid_q += 1

    CMC = CMC.float() / max(1, valid_q)
    mAP = ap_sum / max(1, valid_q)
    return mAP, CMC


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device, use_flip: bool):
    model.eval()
    feats_list = []
    vids_list = []
    cams_list = []

    def fliplr(x):
        inv_idx = torch.arange(x.size(3) - 1, -1, -1).long().to(x.device)
        return x.index_select(3, inv_idx)

    for imgs, vids, cams, _ in loader:
        imgs = imgs.to(device)
        f = model(imgs)
        if use_flip:
            f = f + model(fliplr(imgs))
        f = l2_normalize(f, dim=1)
        feats_list.append(f.cpu())
        vids_list.extend(list(vids))
        cams_list.extend(list(cams))

    feats = torch.cat(feats_list, dim=0).numpy().astype(np.float32)
    return feats, np.array(vids_list), np.array(cams_list)


# =========================================================
# Main train with local val
# =========================================================
def main():
    parser = argparse.ArgumentParser("AIC21 Track2 baseline with local val split (P*K)")
    parser.add_argument("--root", type=str, required=True, help="AIC21_Track2_ReID root")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)

    # P*K
    parser.add_argument("--P", type=int, default=8)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--iters_per_epoch", type=int, default=300)

    # loss/optim
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.1)

    # local val split
    parser.add_argument("--val_id_count", type=int, default=40)
    parser.add_argument("--query_per_id", type=int, default=1)

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--flip_eval", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--save_path", type=str, default="checkpoints/aic21_pk_baseline.pth")
    parser.add_argument("--save_every", type=int, default=1)

    args = parser.parse_args()

    if args.batch_size != args.P * args.K:
        raise ValueError(f"batch_size must equal P*K. Got batch_size={args.batch_size}, P={args.P}, K={args.K}")

    set_seed(args.seed)
    device = get_device()
    print(f"[Device] {device}", flush=True)

    ensure_dir(os.path.dirname(args.save_path) or "checkpoints")

    # 1) build split
    train_list, val_q_list, val_g_list, num_classes = build_local_split_from_train_xml(
        root=args.root,
        train_xml="train_label.xml",
        name_train_txt="name_train.txt",
        val_id_count=args.val_id_count,
        seed=args.seed,
        query_per_id=args.query_per_id
    )
    print(f"[Split] train_samples={len(train_list)} classes={num_classes} | val_query={len(val_q_list)} val_gallery={len(val_g_list)}",
          flush=True)

    # 2) transforms
    train_tf = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value="random"),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # 3) datasets/loaders
    train_set = AICTrainDataset(args.root, train_list, transform=train_tf)
    sampler = PKSampler(train_set.labels, P=args.P, K=args.K, iters_per_epoch=args.iters_per_epoch, seed=args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, drop_last=True)

    val_q_set = AICValDataset(args.root, "query", val_q_list, transform=test_tf)
    val_g_set = AICValDataset(args.root, "gallery", val_g_list, transform=test_tf)
    val_q_loader = DataLoader(val_q_set, batch_size=64, shuffle=False, num_workers=args.num_workers)
    val_g_loader = DataLoader(val_g_set, batch_size=64, shuffle=False, num_workers=args.num_workers)

    # 4) model/loss/optim
    model = PaperBaseline(num_classes=num_classes, pretrained=(not args.no_pretrained)).to(device)
    ce_loss = nn.CrossEntropyLoss()
    tri_loss = BatchHardTripletLoss(margin=args.margin)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # 5) train epochs + local eval
    best_map = -1.0
    best_path = args.save_path.replace(".pth", "_best.pth")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        ce_sum, tri_sum = 0.0, 0.0
        correct, total = 0, 0

        print(f"=== Epoch {ep}/{args.epochs} start ===", flush=True)

        for it, (imgs, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, feat_base, _ = model(imgs)

            loss_ce = ce_loss(logits, labels)
            loss_tri = tri_loss(l2_normalize(feat_base, dim=1), labels)
            loss = loss_ce + loss_tri

            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            ce_sum += loss_ce.item() * bs
            tri_sum += loss_tri.item() * bs
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += bs

            if args.log_every > 0 and (it % args.log_every == 0 or it == 1):
                done = it / max(1, len(train_loader))
                elapsed = time.time() - t0
                eta = elapsed / max(1e-6, done) - elapsed
                print(f"[E{ep:02d}] iter {it}/{len(train_loader)} | CE {ce_sum/max(1,total):.4f} TRI {tri_sum/max(1,total):.4f} ACC {correct/max(1,total):.4f} | eta {eta/60:.1f}m",
                      flush=True)

        scheduler.step()
        print(f"=== Epoch {ep} DONE | CE {ce_sum/max(1,total):.4f} TRI {tri_sum/max(1,total):.4f} ACC {correct/max(1,total):.4f} | time {(time.time()-t0)/60:.1f}m ===",
              flush=True)

        # local eval
        print(f"[Val] extracting features...", flush=True)
        q_feat, q_vid, q_cam = extract_features(model, val_q_loader, device, use_flip=args.flip_eval)
        g_feat, g_vid, g_cam = extract_features(model, val_g_loader, device, use_flip=args.flip_eval)

        mAP, CMC = calculate_result(g_feat, g_vid, g_cam, q_feat, q_vid, q_cam)
        print(f"[Val@E{ep}] mAP={mAP:.4f} R1={CMC[0].item():.4f} R5={CMC[4].item():.4f} R10={CMC[9].item():.4f}",
              flush=True)

        # save
        if ep % args.save_every == 0 or ep == args.epochs:
            torch.save({
                "epoch": ep,
                "state_dict": model.state_dict(),
                "class_num": num_classes,
                "args": vars(args),
            }, args.save_path)
            print(f"[Save] -> {args.save_path}", flush=True)

        if mAP > best_map:
            best_map = mAP
            torch.save({
                "epoch": ep,
                "state_dict": model.state_dict(),
                "class_num": num_classes,
                "best_mAP": best_map,
                "args": vars(args),
            }, best_path)
            print(f"[Best] mAP={best_map:.4f} -> {best_path}", flush=True)


if __name__ == "__main__":
    main()