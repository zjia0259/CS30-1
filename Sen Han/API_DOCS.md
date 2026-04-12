# API Documentation

Base URL: `http://localhost:8000`

---

## GET /health

Check if the service is running and the gallery is loaded.

**Response**
```json
{
  "status": "ok",
  "device": "cuda",
  "gallery_loaded": true,
  "gallery_source": "dynamic",
  "gallery_size": 11579
}
```

---

## GET /gallery/status

Get detailed information about the current gallery.

**Response**
```json
{
  "loaded": true,
  "source": "dynamic",
  "size": 100,
  "supports_graph": true,
  "supports_st": true,
  "metadata_summary": {
    "images": 100,
    "with_camera": 100,
    "with_timestamp": 100,
    "complete_for_st": true
  },
  "sample_urls": ["http://localhost:8000/gallery/veh0001_c001_...jpg"]
}
```

---

## POST /search

Search for matching vehicles given a query image.

**Request** (`multipart/form-data`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | image file | Yes | Query vehicle image |
| `mode` | string | No | `auto` (default), `baseline`, `gnn`, `st` |
| `camera_id` | string | No | Camera ID where query was captured (e.g. `c001`) |
| `frame_id` | string | No | Frame number or timestamp |
| `vehicle_id` | string | No | Known vehicle ID (if any) |

**Mode Selection**

| Mode | Description |
|------|-------------|
| `auto` | System picks best mode based on available metadata |
| `baseline` | Visual features only (73.6% accuracy, always available) |
| `gnn` | Visual + spatial graph reasoning (requires camera_id) |
| `st` | GNN + spatio-temporal penalty (requires camera_id + frame_id) |

> If metadata is missing, the system automatically falls back to `baseline` mode. The system never crashes.

**Response**
```json
{
  "status": "success",
  "requested_mode": "auto",
  "resolved_mode": "st",
  "degraded_reason": null,
  "gallery_source": "dynamic",
  "voted_uid": "0123",
  "results": [
    {
      "rank": 1,
      "name": "0123_c001_00001234_cam3_carA_001.jpg",
      "score": 0.9821,
      "image_url": "http://localhost:8000/gallery/0123_c001_...",
      "vehicle_id": "0123",
      "camera_id": 1,
      "frame_id": 1234,
      "source_name": "cam3_carA_001.jpg",
      "penalized_by_st": false
    }
  ]
}
```

**`degraded_reason` values**

| Value | Meaning |
|-------|---------|
| `null` | No degradation, requested mode used |
| `metadata_incomplete_auto_fallback` | Auto mode fell back to baseline (missing metadata) |
| `query_missing_camera` | GNN requested but no camera_id provided |
| `query_missing_time_or_camera` | ST requested but camera or frame missing |
| `gallery_missing_camera` | Gallery has no camera metadata |
| `gallery_missing_time_or_camera` | Gallery not suitable for ST mode |

---

## POST /upload_gallery

Upload a ZIP package to replace the current gallery and rebuild the feature cache.

**Request** (`multipart/form-data`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | .zip file | Yes | ZIP containing images + optional `metadata.json` |

**ZIP Structure**
```
gallery.zip
├── images/
│   ├── cam1_car001_001.jpg
│   └── cam2_car001_002.jpg
└── metadata.json
```

**metadata.json format**
```json
[
  {
    "filename": "cam1_car001_001.jpg",
    "vehicle_id": "car001",
    "camera_id": "c001",
    "timestamp": 12345
  }
]
```

> `metadata.json` is optional. Without it, the system uses baseline-only mode.

**Response**
```json
{
  "status": "success",
  "source": "dynamic",
  "images": 3,
  "supports_graph": true,
  "supports_st": true,
  "metadata_summary": {
    "images": 3,
    "with_camera": 3,
    "with_timestamp": 3,
    "complete_for_st": true
  }
}
```

---

## GET /trajectory_data/{vehicle_id}

Get the movement trajectory of a specific vehicle across cameras.

**Path Parameter**

| Parameter | Description |
|-----------|-------------|
| `vehicle_id` | Vehicle ID string (e.g. `0123`) |

**Response**
```json
{
  "vehicle_id": "0123",
  "data": [
    {
      "filename": "0123_c001_00001234_cam1_001.jpg",
      "frame": 1234,
      "cam": "c001",
      "image_url": "http://localhost:8000/gallery/0123_c001_..."
    }
  ]
}
```

---

## GET /mobile

Returns the mobile query web page (`frontend/mobile_query.html`).

---

## Static Files

Gallery images are served at `/gallery/{filename}`.

Example: `http://localhost:8000/gallery/0123_c001_00001234_cam1_001.jpg`
