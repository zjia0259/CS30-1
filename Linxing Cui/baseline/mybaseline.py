#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Paper baseline (NO data split, NO sampling):
# ResNet50 + BNNeck + CE + Batch-hard Triplet, P*K sampler
# Train on full image_train; Evaluate on full image_query/image_test.

import os
import time
import argparse
import random
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import scipy.io


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


def cam_to_int(cam_str: str) -> int:
    if cam_str is None:
        return -1
    cam_str = cam_str.strip()
    if cam_str.startswith(("c", "C")):
        return int(cam_str[1:])
    return int(cam_str)


def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def format_eta(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    s = int(seconds % 60)
    h = int(m // 60)
    m = m % 60
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:d}m{s:02d}s"


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def robust_parse_veri_xml(xml_path: str) -> Dict[str, Tuple[int, int]]:
    """
    Parse VeRi XML and handle gb2312/gbk declaration issues on Python 3.12.
    Return dict: imageName -> (vehicleID int, cameraID int)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
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

    meta: Dict[str, Tuple[int, int]] = {}
    for item in root.findall(".//Item"):
        name = item.attrib.get("imageName")
        vid = item.attrib.get("vehicleID")
        cam = item.attrib.get("cameraID")
        if name is None or vid is None or cam is None:
            continue
        meta[name] = (int(vid), cam_to_int(cam))
    return meta


# =========================================================
# Dataset (NO sampling, NO split)
# =========================================================
class VeRiDataset(Dataset):
    """
    VeRi directory structure:
      root/
        image_train/
        image_query/
        image_test/
        train_label_utf8.xml
        test_label_utf8.xml
        name_train.txt
        name_query.txt
        name_test.txt
    """
    def __init__(self, root: str, split: str, transform=None,
                 train_xml: str = "train_label_utf8.xml",
                 test_xml: str = "test_label_utf8.xml"):
        assert split in ("train", "query", "gallery")
        self.root = root
        self.split = split
        self.transform = transform

        if split == "train":
            self.img_dir = os.path.join(root, "image_train")
            self.name_txt = os.path.join(root, "name_train.txt")
            self.xml_path = os.path.join(root, train_xml)
        elif split == "query":
            self.img_dir = os.path.join(root, "image_query")
            self.name_txt = os.path.join(root, "name_query.txt")
            self.xml_path = os.path.join(root, test_xml)
        else:
            self.img_dir = os.path.join(root, "image_test")
            self.name_txt = os.path.join(root, "name_test.txt")
            self.xml_path = os.path.join(root, test_xml)

        with open(self.name_txt, "r", encoding="utf-8", errors="ignore") as f:
            names = [line.strip() for line in f if line.strip()]

        self.meta = robust_parse_veri_xml(self.xml_path)

        # filter names not in xml
        before = len(names)
        names = [n for n in names if n in self.meta]
        after = len(names)
        if after != before:
            print(f"[{split}] filtered {before-after} images not found in xml", flush=True)

        self.img_names = names

        if split == "train":
            vids = sorted({self.meta[n][0] for n in self.img_names})
            self.vid2idx = {vid: i for i, vid in enumerate(vids)}
            self.class_num = len(vids)
            self.labels = [self.vid2idx[self.meta[n][0]] for n in self.img_names]
        else:
            self.vid2idx = None
            self.class_num = None
            self.labels = None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx: int):
        name = self.img_names[idx]
        path = os.path.join(self.img_dir, name)
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        vid, cam = self.meta.get(name, (-1, -1))
        if self.split == "train":
            label = self.vid2idx[vid]
            return img, label
        else:
            return img, vid, cam, name


# =========================================================
# P*K Sampler (epoch length controlled by iters_per_epoch)
# =========================================================
class PKSampler(Sampler[int]):
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
            raise ValueError(f"Not enough identities for P={P}. Only {len(self.unique_ids)} IDs available.")

    def __len__(self):
        return self.iters_per_epoch * self.P * self.K

    def __iter__(self):
        rnd = random.Random(self.seed + int(time.time()))
        all_indices: List[int] = []
        for _ in range(self.iters_per_epoch):
            chosen_ids = rnd.sample(self.unique_ids, self.P)
            for pid in chosen_ids:
                pool = self.index_dict[pid]
                if len(pool) >= self.K:
                    idxs = rnd.sample(pool, self.K)
                else:
                    idxs = [rnd.choice(pool) for _ in range(self.K)]
                all_indices.extend(idxs)
        return iter(all_indices)


# =========================================================
# Model: ResNet50 + BNNeck (paper baseline)
# =========================================================
class PaperBaseline(nn.Module):
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
# Evaluation (CMC/mAP)
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


def evaluate_one_query(score_row, ql, qc, gl, gc):
    index = np.argsort(score_row)[::-1]
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=False)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)
    return compute_mAP(index, good_index, junk_index)


def calculate_result(gallery_feat, gallery_label, gallery_cam,
                     query_feat, query_label, query_cam,
                     result_file: Optional[str] = None):
    qf = torch.from_numpy(query_feat).float()
    gf = torch.from_numpy(gallery_feat).float()
    score = torch.mm(qf, gf.t()).numpy()

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap_sum = 0.0
    valid_q = 0

    for i in range(len(query_label)):
        ap_tmp, cmc_tmp = evaluate_one_query(score[i], query_label[i], query_cam[i],
                                             gallery_label, gallery_cam)
        if cmc_tmp[0] == -1:
            continue
        CMC += cmc_tmp
        ap_sum += ap_tmp
        valid_q += 1

    CMC = CMC.float() / max(1, valid_q)
    mAP = ap_sum / max(1, valid_q)

    out = f"Rank@1:{CMC[0].item():.6f} Rank@5:{CMC[4].item():.6f} Rank@10:{CMC[9].item():.6f} mAP:{mAP:.6f}"
    print(out, flush=True)
    if result_file:
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(out + "\n")
    return mAP, CMC


@torch.no_grad()
def extract_features(model: nn.Module, loader: DataLoader, device, use_flip: bool):
    model.eval()
    feats_list, vid_list, cam_list = [], [], []
    name_list = []

    def fliplr(x):
        inv_idx = torch.arange(x.size(3) - 1, -1, -1).long().to(x.device)
        return x.index_select(3, inv_idx)

    for imgs, vids, cams, names in loader:
        imgs = imgs.to(device)
        f = model(imgs)
        if use_flip:
            f = f + model(fliplr(imgs))
        f = l2_normalize(f, dim=1)

        feats_list.append(f.detach().cpu().float())
        vid_list.append(torch.tensor(vids, dtype=torch.long))
        cam_list.append(torch.tensor(cams, dtype=torch.long))
        name_list.extend(list(names))

    feats = torch.cat(feats_list, dim=0).numpy().astype(np.float32)
    vids = torch.cat(vid_list, dim=0).numpy().astype(np.int64)
    cams = torch.cat(cam_list, dim=0).numpy().astype(np.int64)
    return feats, vids, cams, name_list


# =========================================================
# Transforms
# =========================================================
def build_train_transform(input_size: int, use_random_erasing: bool):
    t = [
        transforms.Resize((input_size, input_size), interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]
    if use_random_erasing:
        t.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value="random"))
    return transforms.Compose(t)


def build_test_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# =========================================================
# Train/Test
# =========================================================
def train(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[Device] {device}", flush=True)

    if args.batch_size != args.P * args.K:
        raise ValueError(f"batch_size must equal P*K. Got batch_size={args.batch_size}, P={args.P}, K={args.K}")

    ensure_dir(os.path.dirname(args.save_path) or "checkpoints")

    train_tf = build_train_transform(args.input_size, use_random_erasing=(not args.no_re))
    train_set = VeRiDataset(args.root, "train", transform=train_tf,
                            train_xml=args.train_xml, test_xml=args.test_xml)

    sampler = PKSampler(train_set.labels, P=args.P, K=args.K, iters_per_epoch=args.iters_per_epoch, seed=args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    print(f"[Train] images={len(train_set)} class_num={train_set.class_num} iters/epoch={len(train_loader)}",
          flush=True)

    model = PaperBaseline(num_classes=train_set.class_num, pretrained=not args.no_pretrained).to(device)
    ce_loss = nn.CrossEntropyLoss()
    tri_loss = BatchHardTripletLoss(margin=args.margin)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Optional: full official eval during training (NO sampling, can be slow)
    if args.eval_every > 0:
        eval_tf = build_test_transform(args.input_size)
        eval_query_set = VeRiDataset(args.root, "query", transform=eval_tf,
                                     train_xml=args.train_xml, test_xml=args.test_xml)
        eval_gallery_set = VeRiDataset(args.root, "gallery", transform=eval_tf,
                                       train_xml=args.train_xml, test_xml=args.test_xml)
        eval_query_loader = DataLoader(eval_query_set, batch_size=args.eval_batch_size, shuffle=False,
                                       num_workers=args.num_workers)
        eval_gallery_loader = DataLoader(eval_gallery_set, batch_size=args.eval_batch_size, shuffle=False,
                                         num_workers=args.num_workers)

    best_map = -1.0
    best_path = args.save_path.replace(".pth", "_best.pth")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_ce_sum, loss_tri_sum = 0.0, 0.0
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
            loss_ce_sum += loss_ce.item() * bs
            loss_tri_sum += loss_tri.item() * bs
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += bs

            if args.log_every > 0 and (it % args.log_every == 0 or it == 1):
                avg_ce = loss_ce_sum / max(1, total)
                avg_tri = loss_tri_sum / max(1, total)
                avg_acc = correct / max(1, total)

                done = it / max(1, len(train_loader))
                elapsed = time.time() - t0
                eta = elapsed / max(1e-6, done) - elapsed

                print(f"[E{ep:02d}] iter {it}/{len(train_loader)} | CE {avg_ce:.4f} TRI {avg_tri:.4f} ACC {avg_acc:.4f} | eta {format_eta(eta)}",
                      flush=True)

        scheduler.step()

        avg_ce = loss_ce_sum / max(1, total)
        avg_tri = loss_tri_sum / max(1, total)
        avg_acc = correct / max(1, total)
        print(f"=== Epoch {ep} DONE | CE {avg_ce:.4f} TRI {avg_tri:.4f} ACC {avg_acc:.4f} | time {format_eta(time.time()-t0)} ===",
              flush=True)

        # Save periodic
        if ep % args.save_every == 0 or ep == args.epochs:
            torch.save({
                "epoch": ep,
                "state_dict": model.state_dict(),
                "class_num": train_set.class_num,
                "args": vars(args),
            }, args.save_path)
            print(f"[Save] -> {args.save_path}", flush=True)

        # Full eval (optional)
        if args.eval_every > 0 and (ep % args.eval_every == 0 or ep == args.epochs):
            print(f"[Eval] Epoch {ep} (FULL official split) ...", flush=True)
            q_feat, q_vid, q_cam, _ = extract_features(model, eval_query_loader, device, use_flip=args.flip_eval)
            g_feat, g_vid, g_cam, _ = extract_features(model, eval_gallery_loader, device, use_flip=args.flip_eval)
            mAP, CMC = calculate_result(g_feat, g_vid, g_cam, q_feat, q_vid, q_cam, result_file=None)
            print(f"[Eval@E{ep}] mAP={mAP:.4f} R1={CMC[0].item():.4f} R5={CMC[4].item():.4f}", flush=True)

            if mAP > best_map:
                best_map = mAP
                torch.save({
                    "epoch": ep,
                    "state_dict": model.state_dict(),
                    "class_num": train_set.class_num,
                    "best_mAP": best_map,
                    "args": vars(args),
                }, best_path)
                print(f"[Best] mAP={best_map:.4f} -> {best_path}", flush=True)


def test(args):
    set_seed(args.seed)
    device = get_device()
    print(f"[Device] {device}", flush=True)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    test_tf = build_test_transform(args.input_size)

    query_set = VeRiDataset(args.root, "query", transform=test_tf,
                            train_xml=args.train_xml, test_xml=args.test_xml)
    gallery_set = VeRiDataset(args.root, "gallery", transform=test_tf,
                              train_xml=args.train_xml, test_xml=args.test_xml)

    query_loader = DataLoader(query_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    class_num = ckpt["class_num"]

    model = PaperBaseline(num_classes=class_num, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    print(f"[Test] query={len(query_set)} gallery={len(gallery_set)}", flush=True)
    print("[Test] Extracting query features...", flush=True)
    q_feat, q_vid, q_cam, _ = extract_features(model, query_loader, device, use_flip=args.flip)

    print("[Test] Extracting gallery features...", flush=True)
    g_feat, g_vid, g_cam, _ = extract_features(model, gallery_loader, device, use_flip=args.flip)

    if args.mat_out:
        mat = {
            "query_f": q_feat,
            "gallery_f": g_feat,
            "query_label": q_vid.reshape(1, -1),
            "gallery_label": g_vid.reshape(1, -1),
            "query_cam": q_cam.reshape(1, -1),
            "gallery_cam": g_cam.reshape(1, -1),
        }
        scipy.io.savemat(args.mat_out, mat)
        print(f"[Saved] {args.mat_out}", flush=True)

    if args.result_file:
        if os.path.isfile(args.result_file):
            os.remove(args.result_file)

    print("[Eval] Computing CMC/mAP ...", flush=True)
    mAP, CMC = calculate_result(g_feat, g_vid, g_cam, q_feat, q_vid, q_cam, args.result_file)
    print(f"[Final] mAP={mAP:.4f} R1={CMC[0].item():.4f} R5={CMC[4].item():.4f} R10={CMC[9].item():.4f}",
          flush=True)


# =========================================================
# CLI
# =========================================================
def build_parser():
    p = argparse.ArgumentParser("Paper baseline (NO split, NO sampling) with P*K sampler")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--root", type=str, required=True)
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--input_size", type=int, default=256)
    tr.add_argument("--num_workers", type=int, default=0)

    tr.add_argument("--P", type=int, default=8)
    tr.add_argument("--K", type=int, default=4)
    tr.add_argument("--batch_size", type=int, default=32)

    tr.add_argument("--iters_per_epoch", type=int, default=350,
                    help="controls epoch time; 1100~full pass for batch=32, 350 is faster")

    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--weight_decay", type=float, default=5e-4)
    tr.add_argument("--step_size", type=int, default=20)
    tr.add_argument("--gamma", type=float, default=0.1)

    tr.add_argument("--margin", type=float, default=0.3)
    tr.add_argument("--log_every", type=int, default=20)
    tr.add_argument("--save_path", type=str, default="checkpoints/paper_pk_nosplit.pth")
    tr.add_argument("--save_every", type=int, default=5)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--train_xml", type=str, default="train_label_utf8.xml")
    tr.add_argument("--test_xml", type=str, default="test_label_utf8.xml")
    tr.add_argument("--no_pretrained", action="store_true")
    tr.add_argument("--no_re", action="store_true")

    # full eval only (optional)
    tr.add_argument("--eval_every", type=int, default=0, help="FULL eval every N epochs (0=off)")
    tr.add_argument("--eval_batch_size", type=int, default=64)
    tr.add_argument("--flip_eval", action="store_true")

    te = sub.add_parser("test")
    te.add_argument("--root", type=str, required=True)
    te.add_argument("--ckpt", type=str, required=True)
    te.add_argument("--batch_size", type=int, default=64)
    te.add_argument("--input_size", type=int, default=256)
    te.add_argument("--num_workers", type=int, default=0)
    te.add_argument("--flip", action="store_true")
    te.add_argument("--mat_out", type=str, default="pytorch_result.mat")
    te.add_argument("--result_file", type=str, default="baseline_result.txt")
    te.add_argument("--seed", type=int, default=42)
    te.add_argument("--train_xml", type=str, default="train_label_utf8.xml")
    te.add_argument("--test_xml", type=str, default="test_label_utf8.xml")

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()