# Vehicle Re-Identification System

Scene- and Metadata-Aware Knowledge Graph Learning for Robust Vehicle Re-Identification.

## Quick Start

### Requirements
- Docker + Docker Compose
- NVIDIA GPU + nvidia-container-toolkit

### 1. Prepare model weights

Place the trained model weights in the `best_module/` directory:

```
best_module/
├── best_resnet50_ibn.pth    # Visual encoder (ResNet-IBN)
└── best_kg_module.pth       # Spatial-temporal GNN module
```

### 2. Start the service

```bash
docker-compose up -d
```

The API will be available at `http://localhost:8000`.

### 3. Check status

```bash
# API health check
curl http://localhost:8000/health

# View logs
docker-compose logs -f api
```

### 4. Stop the service

```bash
docker-compose down
```

---

## Optional: VeRi Dataset Gallery

If using the VeRi benchmark dataset as the default gallery, place it at `data/VeRi/`:

```
data/
└── VeRi/
    ├── image_test/
    ├── name_test.txt
    └── test_label.xml
```

If no dataset is present, upload a custom gallery via the `/upload_gallery` API.

---

## API Reference

See [API_DOCS.md](API_DOCS.md) for full API documentation.

---

## Project Structure

```
.
├── app.py                  # FastAPI backend (maintained by Member D)
├── train_kg_gnn.py         # GNN model definitions (Member A)
├── frontend/               # Web UI (Member B)
│   └── mobile_query.html
├── best_module/            # Trained model weights (not in git)
├── Dockerfile              # Container build (Member D)
├── docker-compose.yml      # Service orchestration (Member D)
└── requirements.txt        # Python dependencies
```

---

## Team

| Member | Role |
|--------|------|
| Fuhai-Liang | Algorithm & Backend Lead |
| Member B | Frontend UI/UX |
| Claudia Duan | Data Engineering & API Edge |
| Sen Han | DevOps & Testing |
