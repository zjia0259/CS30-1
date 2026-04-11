import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO

# ==========================================
# 1. 复制你的模型类到这里 (和上面一样)
# ==========================================
import torch.nn as nn
class ResNet50IBN_ReID_MultiTask(nn.Module):
    def __init__(self, num_classes, num_colors, num_types):
        super(ResNet50IBN_ReID_MultiTask, self).__init__()

        # ================== 修改这里 ==================
        print("=> 正在从本地完全离线加载 resnet50_ibn_a 代码...")
        local_repo_path = '/Users/liangshilin/Desktop/Capstone/Baseline/IBN/XingangPan_IBN-Net_master'
        
        # 核心改动：传入本地路径，并强制加上 source='local'
        backbone = torch.hub.load(local_repo_path, 'resnet50_ibn_a', pretrained=False, source='local')
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
# 2. 类别映射字典
# ==========================================
# 假设训练时的映射是按 1-10 和 1-9 的顺序映射到 0-9 和 0-8 索引的
COLOR_MAP = ["黄色 (Yellow)", "橙色 (Orange)", "绿色 (Green)", "灰色 (Gray)", 
             "红色 (Red)", "蓝色 (Blue)", "白色 (White)", "金色 (Golden)", 
             "棕色 (Brown)", "黑色 (Black)"]

TYPE_MAP = ["轿车 (Sedan)", "SUV", "面包车 (Van)", "掀背车 (Hatchback)", 
            "MPV", "皮卡 (Pickup)", "公交车 (Bus)", "卡车 (Truck)", "旅行车 (Estate)"]

# ==========================================
# 3. 初始化 FastAPI 与 加载模型
# ==========================================
app = FastAPI()

# 允许跨域请求 (为了让前端 HTML 能访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# 挂载静态文件目录 (关键：让前端能够直接通过 URL 加载 test 图库的图片)
# 注意：把 './My_ReID_Data/image_test' 换成你真实的图库路径
app.mount("/gallery_images", StaticFiles(directory="/Users/liangshilin/Desktop/Capstone/COMP5703/旧的思想/Capstone/数据/VeRi/image_test"), name="gallery")
# app.mount("/gallery_images", StaticFiles(directory="/Users/liangshilin/Desktop/Capstone/COMP5703/旧的思想/Capstone/数据/AIC21_Track2_ReID重识别/image_test"), name="gallery")

# 全局变量
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
gallery_feats = None
gallery_names = None
transform_test = transforms.Compose([
    transforms.Resize((256, 256)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.on_event("startup")
def load_assets():
    global model, gallery_feats, gallery_names
    print("=> 正在加载模型与图库特征...")
    
    # 加载模型
    ckpt = torch.load('/Users/liangshilin/Desktop/Capstone/Baseline/model/best_resnet50_ibn_mt.pth', map_location='cpu', weights_only=False )
    model = ResNet50IBN_ReID_MultiTask(ckpt['num_classes'], ckpt['num_colors'], ckpt['num_types'])
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    
    # 加载图库特征库
    gallery_data = torch.load('/Users/liangshilin/Desktop/Capstone/Baseline/model/Data1_gallery_features.pt', map_location=device)
    # gallery_data = torch.load('/Users/liangshilin/Desktop/Capstone/Baseline/model/Data2_gallery_features.pt', map_location=device)   
    gallery_feats = gallery_data['features']
    gallery_names = gallery_data['names']
    print("=> 后端引擎启动完毕！")

# ==========================================
# 4. 预测接口
# ==========================================
@app.post("/predict")
async def predict_vehicle(file: UploadFile = File(...)):
    # 读取前端传来的图片
    image_data = await file.read()
    img = Image.open(BytesIO(image_data)).convert('RGB')
    
    # 预处理
    img_tensor = transform_test(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 提取特征与属性
        q_feat, pred_color_logits, pred_type_logits = model(img_tensor)
        q_feat = F.normalize(q_feat, p=2, dim=1)
        
        # 解析颜色和车型 (取出概率最大的索引)
        color_idx = torch.argmax(pred_color_logits, dim=1).item()
        type_idx = torch.argmax(pred_type_logits, dim=1).item()
        
        pred_color_name = COLOR_MAP[color_idx]
        pred_type_name = TYPE_MAP[type_idx]
        
        # 计算特征相似度 (由于已做 L2 归一化，直接矩阵相乘就是余弦相似度)
        # sim_matrix 的 shape 是 [1, N]
        sim_matrix = torch.mm(q_feat, gallery_feats.t()).squeeze(0)
        
        # 取 Top 10 相似度最高的图片的索引
        top10_scores, top10_indices = torch.topk(sim_matrix, k=10)
        
        # 获取 Top 10 图片的名字
        top10_images = [gallery_names[idx.item()] for idx in top10_indices]
        
    return {
        "color": pred_color_name,
        "type": pred_type_name,
        "top10": top10_images
    }