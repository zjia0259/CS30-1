import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
# ==========================================
# 1. 核心感知网络：多任务 ResNet50-IBN
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
# 2. 图库 Dataset
# ==========================================
class GalleryDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_name

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 配置路径 (请根据你的实际情况修改)
    test_dir = './data/AIC21_Track2_ReID/image_test' 
    ckpt_path = './ResNet_IBN_output/best_resnet50_ibn_mt.pth'
    output_pt = './Data2_gallery_features.pt'

    # 1. 加载模型权重
    checkpoint = torch.load(ckpt_path, map_location='cpu',weights_only=False)
    model = ResNet50IBN_ReID_MultiTask(
        num_classes=checkpoint['num_classes'],
        num_colors=checkpoint['num_colors'],
        num_types=checkpoint['num_types']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # 2. 数据预处理与 DataLoader
    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = GalleryDataset(test_dir, transform_test)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # 3. 提取特征
    all_feats = []
    all_names = []
    
    print("=> 正在提取 Test 图库特征...")
    with torch.no_grad():
        for imgs, names in tqdm(dataloader):
            imgs = imgs.to(device)
            # 推理阶段，模型返回 (feat, color_logits, type_logits)
            feats, _, _ = model(imgs)
            # 必须做 L2 归一化，方便后续直接用矩阵乘法算余弦相似度
            feats = F.normalize(feats, p=2, dim=1) 
            all_feats.append(feats.cpu())
            all_names.extend(names)
            
    all_feats = torch.cat(all_feats, dim=0)
    
    # 4. 保存为 .pt 文件
    torch.save({
        'features': all_feats,
        'names': all_names
    }, output_pt)
    
    print(f"=> 🎉 成功保存图库特征! 共 {len(all_names)} 张图片，保存在 {output_pt}")

if __name__ == '__main__':
    main()