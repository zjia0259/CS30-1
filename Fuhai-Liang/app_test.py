import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import io

# from train_kg_gnn import ResNet50IBN_ReID
class MockResNet50IBN_ReID(nn.Module):
    """
    A lightweight mock of your ResNet50IBN_ReID to make this script runnable.
    It simulates returning the (logits, features) tuple.
    """
    def __init__(self, feature_dim=2048):
        super().__init__()
        # Dummy layer just to output the correct shape
        self.dummy = nn.Linear(3 * 256 * 256, feature_dim)

    def forward(self, x):
        # Flatten image and pass through dummy layer
        x = x.view(x.size(0), -1)
        features = self.dummy(x)
        logits = features # Dummy logits
        # Simulating your model's output structure: (logits, vbase)
        return (logits, features) 


def test_baseline_search_logic():
    print("=> 🚀 Starting Baseline Search Logic Test\n")
    device = torch.device("cpu") # Forced to CPU for easy testing


    print("=> [1/4] Initializing Baseline model...")
    car_encoder = MockResNet50IBN_ReID().to(device)
    car_encoder.eval()

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    print("=> [2/4] Generating Mock Gallery Features...")
    gallery_size = 500
    feature_dim = 2048
    
    # Randomly generate gallery features and normalize them
    mock_gallery_feats = np.random.randn(gallery_size, feature_dim).astype(np.float32)
    mock_gallery_feats = mock_gallery_feats / np.linalg.norm(mock_gallery_feats, axis=1, keepdims=True)
    
    # Generate mock filenames for the gallery (e.g., "0001_c001_123.jpg")
    mock_gallery_names = [f"{i:04d}_c001_123456.jpg" for i in range(gallery_size)]


    print("=> [3/4] Simulating user image upload...")
    # Create a dummy solid-color RGB image
    dummy_image = Image.new('RGB', (800, 600), color='red')
    
    # Simulate the FastAPI await file.read() process
    img_byte_arr = io.BytesIO()
    dummy_image.save(img_byte_arr, format='JPEG')
    image_data = img_byte_arr.getvalue()

    # App.py logic starts here:
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img_tensor = transform_test(img).unsqueeze(0).to(device)
    print(f"   [+] Image tensor shape ready: {img_tensor.shape}")

    print("=> [4/4] Executing Baseline Search (Feature extraction & matching)...")
    top_k = 10 # Let's fetch top 10 for testing
    
    with torch.no_grad():
        # 1. Forward pass
        res_car = car_encoder(img_tensor)
        
        # Extract vbase (handling the tuple output logic from your app.py)
        vbase = res_car[1] if isinstance(res_car, tuple) else res_car
        
        # 2. L2 Normalization (Crucial for cosine similarity via dot product)
        q_feat = torch.nn.functional.normalize(vbase, p=2, dim=1).cpu().numpy()
        
    # 3. Calculate similarity matrix (Query feat dot Gallery feats)
    sim_matrix = np.dot(q_feat, mock_gallery_feats.T)
    
    # 4. Sort indices to get highest scores first
    indices = np.argsort(sim_matrix[0])[::-1][:top_k]

    # 5. Format results
    results = []
    for rank, idx in enumerate(indices):
        results.append({
            "rank": rank + 1,
            "name": mock_gallery_names[idx],
            "score": float(sim_matrix[0][idx])
        })

    print("\n" + "="*50)
    for res in results[:5]:
        print(f"   Rank {res['rank']:02d} | Name: {res['name']} | Score: {res['score']:.4f}")
    print("="*50)

if __name__ == "__main__":
    test_baseline_search_logic()