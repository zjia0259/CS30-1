import torch

checkpoint = torch.load('/Users/hansen/Desktop/USYD/26s1/5703/project/A/best_module/best_resnet50_ibn.pth', map_location='cpu')

print(checkpoint.keys())

print(checkpoint['state_dict'].keys())