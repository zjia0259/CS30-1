import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from train_kg_gnn import (
    ReIDDataset,
    ResNet50IBN_ReID,
    VisualKGModule,
    apply_st_penalty,
    build_batch_hetero_graph,
    build_distance_matrix,
    extract_features_kg,
    get_test_meta_map,
    read_txt_lines,
)


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
GALLERY_DIR = RUNTIME_DIR / "gallery"
CURRENT_GALLERY_DIR = GALLERY_DIR / "current"
CURRENT_IMAGE_DIR = CURRENT_GALLERY_DIR / "images"
CACHE_DIR = RUNTIME_DIR / "cache"
FRONTEND_DIR = BASE_DIR / "frontend"

for path in [UPLOAD_DIR, CURRENT_IMAGE_DIR, CACHE_DIR, FRONTEND_DIR]:
    path.mkdir(parents=True, exist_ok=True)

DEFAULT_DATA_ROOT = Path(os.getenv("VERI_DATA_ROOT", BASE_DIR / "data" / "VeRi"))
BASELINE_CKPT = Path(os.getenv("BASELINE_CKPT", BASE_DIR / "best_module" / "best_resnet50_ibn.pth"))
KG_CKPT = Path(os.getenv("KG_CKPT", BASE_DIR / "best_module" / "best_kg_module.pth"))
LEGACY_CACHE = Path(os.getenv("LEGACY_GALLERY_CACHE", BASE_DIR / "gallery_cache.pt"))
DYNAMIC_CACHE = CACHE_DIR / "dynamic_gallery_cache.pt"
UPLOAD_METADATA = CURRENT_GALLERY_DIR / "metadata.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = FastAPI(title="Vehicle ReID API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/gallery", StaticFiles(directory=CURRENT_IMAGE_DIR), name="gallery")


def slugify_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return cleaned.strip("._") or "image"


def normalize_camera_token(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if lower.startswith("c"):
        suffix = re.sub(r"\D", "", lower[1:])
        return f"c{int(suffix):03d}" if suffix else lower
    if text.isdigit():
        return f"c{int(text):03d}"
    return lower


def camera_numeric_id(value: Any) -> int | None:
    token = normalize_camera_token(value)
    if not token:
        return None
    digits = re.sub(r"\D", "", token)
    return int(digits) if digits else None


def parse_frame_value(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return max(int(value), 0)
    text = str(value).strip()
    if not text:
        return fallback
    digits = re.sub(r"\D", "", text)
    return max(int(digits), 0) if digits else fallback


def normalize_vehicle_token(value: Any, fallback: int) -> str:
    if value is None:
        return f"veh{fallback:04d}"
    text = str(value).strip()
    return slugify_filename(text) or f"veh{fallback:04d}"


def find_metadata_file(root: Path) -> Path | None:
    candidates = list(root.rglob("metadata.json"))
    return candidates[0] if candidates else None


def collect_image_files(root: Path) -> dict[str, Path]:
    image_map: dict[str, Path] = {}
    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_map[file_path.name] = file_path
    return image_map


def build_absolute_url(request: Request, name: str) -> str:
    return str(request.base_url).rstrip("/") + f"/gallery/{name}"


def extract_query_metadata_from_filename(filename: str) -> tuple[str | None, int | None, int | None]:
    parts = filename.split("_")
    vehicle_id = parts[0] if parts else None
    camera_id = camera_numeric_id(parts[1]) if len(parts) > 1 else None
    frame_id = parse_frame_value(parts[2], 0) if len(parts) > 2 else None
    return vehicle_id, camera_id, frame_id


def normalize_metadata_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ["images", "items", "data", "records"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def create_gallery_filename(vehicle_id: str, camera_id: int | None, frame_id: int, original_name: str) -> str:
    extension = Path(original_name).suffix.lower() or ".jpg"
    stem = slugify_filename(Path(original_name).stem)
    cam_part = f"c{camera_id:03d}" if camera_id is not None else "c000"
    return f"{vehicle_id}_{cam_part}_{frame_id:08d}_{stem}{extension}"


def extract_features_baseline(car_encoder, dataloader, device):
    car_encoder.eval()
    feats_list = []
    names_list: list[str] = []

    with torch.no_grad():
        for car_imgs, _, _, img_names in dataloader:
            car_imgs = car_imgs.to(device)
            res_car = car_encoder(car_imgs)
            feat_car = res_car[1] if isinstance(res_car, tuple) else res_car
            feat_car = torch.nn.functional.normalize(feat_car, p=2, dim=1)
            feats_list.append(feat_car.cpu())
            names_list.extend(img_names)

    if not feats_list:
        raise ValueError("No gallery images were available for feature extraction.")
    return torch.cat(feats_list, dim=0), names_list


class RuntimeService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = int(os.getenv("NUM_CLASSES", "575"))
        self.transform_test = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.car_encoder = None
        self.kg_module = None
        self.cam2idx: dict[Any, int] = {}
        self.dist_mat = None
        self.cam_idx_map = None
        self.gallery_state: dict[str, Any] = {
            "loaded": False,
            "source": "empty",
            "size": 0,
            "baseline_feats": None,
            "baseline_names": [],
            "graph_feats": None,
            "graph_names": [],
            "record_map": {},
            "vehicle_index": {},
            "supports_graph": False,
            "supports_st": False,
            "metadata_summary": {},
        }

    async def startup(self) -> None:
        self._load_models()
        self._load_gallery()

    def _load_models(self) -> None:
        self.dist_mat, self.cam_idx_map = build_distance_matrix()
        if not BASELINE_CKPT.exists():
            raise FileNotFoundError(f"Missing baseline checkpoint: {BASELINE_CKPT}")
        if not KG_CKPT.exists():
            raise FileNotFoundError(f"Missing KG checkpoint: {KG_CKPT}")

        self.car_encoder = ResNet50IBN_ReID(self.num_classes).to(self.device)
        baseline_ckpt = torch.load(BASELINE_CKPT, map_location=self.device, weights_only=False)
        self.car_encoder.load_state_dict(baseline_ckpt["state_dict"])
        self.car_encoder.eval()

        kg_ckpt = torch.load(KG_CKPT, map_location=self.device, weights_only=False)
        total_cameras = len(kg_ckpt.get("cam2idx", {})) or 20
        self.kg_module = VisualKGModule(
            in_channels=2048,
            hidden_channels=2048,
            num_classes=self.num_classes,
            num_cameras=total_cameras,
        ).to(self.device)
        self.kg_module.load_state_dict(kg_ckpt["state_dict"])
        self.kg_module.eval()
        self.cam2idx = kg_ckpt["cam2idx"]

    def _resolve_graph_cam_key(self, record: dict[str, Any]) -> Any:
        candidates = [record.get("camera_raw"), record.get("camera_token"), record.get("camera_numeric")]
        for candidate in candidates:
            if candidate in self.cam2idx:
                return candidate
            if isinstance(candidate, int) and str(candidate) in self.cam2idx:
                return str(candidate)
            if isinstance(candidate, str) and candidate.isdigit() and int(candidate) in self.cam2idx:
                return int(candidate)
        return next(iter(self.cam2idx.keys()), 0)

    def _make_dataloader(self, records: list[dict[str, Any]]) -> DataLoader:
        data_list = [
            (record["gallery_name"], record["graph_vehicle_id"], record["graph_cam_key"]) for record in records
        ]
        dataset = ReIDDataset(str(CURRENT_IMAGE_DIR), data_list, self.cam2idx, self.transform_test)
        return DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    def _mirror_images_to_current(self, names: list[str], source_dir: Path) -> None:
        if not source_dir.exists():
            return
        for name in names:
            source_path = source_dir / name
            target_path = CURRENT_IMAGE_DIR / name
            if source_path.exists() and not target_path.exists():
                shutil.copy2(source_path, target_path)

    def _update_vehicle_index(self, record_map: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        index: dict[str, list[dict[str, Any]]] = {}
        for record in record_map.values():
            vehicle_id = record["vehicle_id"]
            index.setdefault(vehicle_id, []).append(record)
        for vehicle_id in index:
            index[vehicle_id].sort(key=lambda item: item["frame_id"])
        return index

    def _set_gallery_state(
        self,
        source: str,
        baseline_feats,
        baseline_names: list[str],
        graph_feats,
        graph_names: list[str],
        record_map: dict[str, dict[str, Any]],
        metadata_summary: dict[str, Any],
    ) -> None:
        self.gallery_state = {
            "loaded": True,
            "source": source,
            "size": len(record_map),
            "baseline_feats": baseline_feats.numpy() if torch.is_tensor(baseline_feats) else baseline_feats,
            "baseline_names": baseline_names,
            "graph_feats": graph_feats.numpy() if torch.is_tensor(graph_feats) else graph_feats,
            "graph_names": graph_names,
            "record_map": record_map,
            "vehicle_index": self._update_vehicle_index(record_map),
            "supports_graph": graph_feats is not None,
            "supports_st": metadata_summary.get("complete_for_st", False),
            "metadata_summary": metadata_summary,
        }

    def _load_gallery(self) -> None:
        if DYNAMIC_CACHE.exists():
            cache_data = torch.load(DYNAMIC_CACHE, map_location="cpu", weights_only=False)
            self._set_gallery_state(
                source="dynamic",
                baseline_feats=cache_data["baseline_feats"],
                baseline_names=cache_data["baseline_names"],
                graph_feats=cache_data.get("graph_feats"),
                graph_names=cache_data.get("graph_names", []),
                record_map=cache_data["record_map"],
                metadata_summary=cache_data.get("metadata_summary", {}),
            )
            return

        if LEGACY_CACHE.exists():
            cache_data = torch.load(LEGACY_CACHE, map_location="cpu", weights_only=False)
            names = list(cache_data["names"])
            feats = cache_data["feats"]
            self._mirror_images_to_current(names, DEFAULT_DATA_ROOT / "image_test")
            record_map = {}
            for name in names:
                vehicle_id, camera_id, frame_id = extract_query_metadata_from_filename(name)
                image_path = DEFAULT_DATA_ROOT / "image_test" / name
                record_map[name] = {
                    "gallery_name": name,
                    "source_name": name,
                    "vehicle_id": vehicle_id or "unknown",
                    "camera_numeric": camera_id,
                    "camera_token": normalize_camera_token(camera_id),
                    "camera_raw": normalize_camera_token(camera_id),
                    "frame_id": frame_id or 0,
                    "has_camera": camera_id is not None,
                    "has_timestamp": frame_id is not None,
                    "static_path": str(image_path) if image_path.exists() else None,
                }
            self._set_gallery_state(
                source="legacy-cache",
                baseline_feats=feats,
                baseline_names=names,
                graph_feats=feats,
                graph_names=names,
                record_map=record_map,
                metadata_summary={
                    "images": len(names),
                    "with_camera": sum(1 for item in record_map.values() if item["has_camera"]),
                    "with_timestamp": sum(1 for item in record_map.values() if item["has_timestamp"]),
                    "complete_for_st": all(item["has_camera"] and item["has_timestamp"] for item in record_map.values()),
                },
            )
            return

        if (DEFAULT_DATA_ROOT / "image_test").exists() and (DEFAULT_DATA_ROOT / "name_test.txt").exists():
            self._rebuild_legacy_gallery(DEFAULT_DATA_ROOT)

    def _rebuild_legacy_gallery(self, data_root: Path) -> None:
        test_xml = data_root / "test_label.xml"
        test_names = read_txt_lines(str(data_root / "name_test.txt"))
        self._mirror_images_to_current(test_names, data_root / "image_test")
        test_img_to_vid, test_img_to_cam = get_test_meta_map(str(test_xml))
        records = []
        for idx, name in enumerate(test_names, start=1):
            camera_raw = test_img_to_cam.get(name, "-1")
            camera_num = camera_numeric_id(camera_raw)
            frame_id = parse_frame_value(name.split("_")[2] if len(name.split("_")) > 2 else None, idx)
            record = {
                "gallery_name": name,
                "source_name": name,
                "vehicle_id": normalize_vehicle_token(test_img_to_vid.get(name), idx),
                "camera_numeric": camera_num,
                "camera_token": normalize_camera_token(camera_raw),
                "camera_raw": camera_raw,
                "frame_id": frame_id,
                "has_camera": camera_num is not None,
                "has_timestamp": True,
                "graph_vehicle_id": idx,
            }
            record["graph_cam_key"] = self._resolve_graph_cam_key(record)
            records.append(record)

        dataloader = DataLoader(
            ReIDDataset(
                str(data_root / "image_test"),
                [(r["gallery_name"], r["graph_vehicle_id"], r["graph_cam_key"]) for r in records],
                self.cam2idx,
                self.transform_test,
            ),
            batch_size=64,
            shuffle=False,
            num_workers=0,
        )
        baseline_feats, baseline_names = extract_features_baseline(self.car_encoder, dataloader, self.device)
        graph_feats, _, _, graph_names = extract_features_kg(self.car_encoder, self.kg_module, dataloader, self.device)
        record_map = {record["gallery_name"]: record for record in records}
        self._set_gallery_state(
            source="legacy-data",
            baseline_feats=baseline_feats,
            baseline_names=baseline_names,
            graph_feats=graph_feats,
            graph_names=graph_names,
            record_map=record_map,
            metadata_summary={
                "images": len(records),
                "with_camera": sum(1 for item in records if item["has_camera"]),
                "with_timestamp": len(records),
                "complete_for_st": all(item["has_camera"] and item["has_timestamp"] for item in records),
            },
        )

    def rebuild_dynamic_gallery(self, zip_path: Path) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="gallery_upload_", dir=str(UPLOAD_DIR)) as temp_dir:
            temp_root = Path(temp_dir)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for member in zip_ref.infolist():
                    # Fix Windows-style backslash paths in ZIP (cross-platform compatibility)
                    member.filename = member.filename.replace("\\", "/")
                    zip_ref.extract(member, temp_root)

            image_map = collect_image_files(temp_root)
            if not image_map:
                raise HTTPException(status_code=400, detail="ZIP package contains no images.")

            metadata_path = find_metadata_file(temp_root)
            if metadata_path:
                with open(metadata_path, "r", encoding="utf-8-sig") as file_obj:
                    payload = json.load(file_obj)
                metadata_items = normalize_metadata_payload(payload)
            else:
                metadata_items = []

            for old_file in CURRENT_IMAGE_DIR.iterdir():
                if old_file.is_file():
                    old_file.unlink()

            records: list[dict[str, Any]] = []
            vehicle_map: dict[str, int] = {}

            if metadata_items:
                iterable = metadata_items
            else:
                iterable = [{"filename": name} for name in sorted(image_map.keys())]

            for idx, item in enumerate(iterable, start=1):
                source_name = item.get("filename") or item.get("file_name") or item.get("image") or item.get("name")
                if not source_name or source_name not in image_map:
                    continue

                vehicle_id = normalize_vehicle_token(
                    item.get("vehicle_id") or item.get("vehicleId") or item.get("vid"),
                    idx,
                )
                camera_raw = item.get("camera_id") or item.get("cameraId") or item.get("camera") or item.get("cam")
                camera_num = camera_numeric_id(camera_raw)
                frame_raw = item.get("timestamp") or item.get("frame_id") or item.get("frame") or item.get("time")
                frame_id = parse_frame_value(frame_raw, idx)
                gallery_name = create_gallery_filename(vehicle_id, camera_num, frame_id, source_name)
                if (CURRENT_IMAGE_DIR / gallery_name).exists():
                    gallery_name = create_gallery_filename(vehicle_id, camera_num, frame_id + idx, source_name)
                target_path = CURRENT_IMAGE_DIR / gallery_name
                shutil.copy2(image_map[source_name], target_path)

                graph_vehicle_id = vehicle_map.setdefault(vehicle_id, len(vehicle_map) + 1)
                record = {
                    "gallery_name": gallery_name,
                    "source_name": source_name,
                    "vehicle_id": vehicle_id,
                    "camera_numeric": camera_num,
                    "camera_token": normalize_camera_token(camera_raw),
                    "camera_raw": camera_raw,
                    "frame_id": frame_id,
                    "timestamp_raw": frame_raw,
                    "location": item.get("location") or item.get("camera_location"),
                    "has_camera": camera_num is not None,
                    "has_timestamp": frame_raw is not None,
                    "graph_vehicle_id": graph_vehicle_id,
                }
                record["graph_cam_key"] = self._resolve_graph_cam_key(record)
                records.append(record)

            if not records:
                raise HTTPException(status_code=400, detail="metadata.json did not match any images in the ZIP package.")

            with open(UPLOAD_METADATA, "w", encoding="utf-8") as file_obj:
                json.dump(records, file_obj, ensure_ascii=False, indent=2)

            dataloader = self._make_dataloader(records)
            baseline_feats, baseline_names = extract_features_baseline(self.car_encoder, dataloader, self.device)

            graph_ready = all(record["has_camera"] for record in records)
            graph_feats = None
            graph_names: list[str] = []
            if graph_ready:
                graph_feats, _, _, graph_names = extract_features_kg(
                    self.car_encoder, self.kg_module, dataloader, self.device
                )

            record_map = {record["gallery_name"]: record for record in records}
            metadata_summary = {
                "images": len(records),
                "with_camera": sum(1 for item in records if item["has_camera"]),
                "with_timestamp": sum(1 for item in records if item["has_timestamp"]),
                "complete_for_st": all(item["has_camera"] and item["has_timestamp"] for item in records),
            }
            self._set_gallery_state(
                source="dynamic",
                baseline_feats=baseline_feats,
                baseline_names=baseline_names,
                graph_feats=graph_feats,
                graph_names=graph_names,
                record_map=record_map,
                metadata_summary=metadata_summary,
            )

            torch.save(
                {
                    "baseline_feats": baseline_feats,
                    "baseline_names": baseline_names,
                    "graph_feats": graph_feats,
                    "graph_names": graph_names,
                    "record_map": record_map,
                    "metadata_summary": metadata_summary,
                },
                DYNAMIC_CACHE,
            )

            return {
                "status": "success",
                "source": "dynamic",
                "images": len(records),
                "supports_graph": graph_ready,
                "supports_st": metadata_summary["complete_for_st"],
                "metadata_summary": metadata_summary,
            }

    def select_mode(
        self,
        requested_mode: str,
        camera_id: int | None,
        frame_id: int | None,
    ) -> tuple[str, str | None]:
        mode = (requested_mode or "auto").lower()
        if mode == "auto":
            if camera_id is not None and frame_id is not None and self.gallery_state["supports_st"]:
                return "st", None
            if camera_id is not None and self.gallery_state["supports_graph"]:
                return "gnn", None
            return "baseline", "metadata_incomplete_auto_fallback"

        if mode == "st":
            if camera_id is None or frame_id is None:
                return "baseline", "query_missing_time_or_camera"
            if not self.gallery_state["supports_st"]:
                return "baseline", "gallery_missing_time_or_camera"
            return "st", None

        if mode == "gnn":
            if camera_id is None:
                return "baseline", "query_missing_camera"
            if not self.gallery_state["supports_graph"]:
                return "baseline", "gallery_missing_camera"
            return "gnn", None

        return "baseline", None


runtime_service = RuntimeService()


@app.on_event("startup")
async def on_startup() -> None:
    await runtime_service.startup()


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return "<h3>Vehicle ReID API is running.</h3><p>Open <a href='/docs'>/docs</a> or <a href='/mobile'>/mobile</a>.</p>"


@app.get("/mobile")
async def mobile_page():
    page_path = FRONTEND_DIR / "mobile_query.html"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="mobile_query.html was not found.")
    return FileResponse(page_path)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": str(runtime_service.device),
        "gallery_loaded": runtime_service.gallery_state["loaded"],
        "gallery_source": runtime_service.gallery_state["source"],
        "gallery_size": runtime_service.gallery_state["size"],
    }


@app.get("/gallery/status")
async def gallery_status(request: Request) -> dict[str, Any]:
    state = runtime_service.gallery_state
    sample_names = state["baseline_names"][:5]
    sample_urls = [build_absolute_url(request, name) for name in sample_names]
    return {
        "loaded": state["loaded"],
        "source": state["source"],
        "size": state["size"],
        "supports_graph": state["supports_graph"],
        "supports_st": state["supports_st"],
        "metadata_summary": state["metadata_summary"],
        "sample_urls": sample_urls,
    }


@app.post("/upload_gallery")
async def upload_gallery(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a ZIP package.")

    upload_path = UPLOAD_DIR / slugify_filename(file.filename)
    with open(upload_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        return runtime_service.rebuild_dynamic_gallery(upload_path)
    finally:
        if upload_path.exists():
            upload_path.unlink()


@app.post("/search")
async def search_vehicle(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    camera_id: str | None = Form(None),
    frame_id: str | None = Form(None),
    vehicle_id: str | None = Form(None),
) -> dict[str, Any]:
    state = runtime_service.gallery_state
    if not state["loaded"]:
        raise HTTPException(status_code=503, detail="Gallery has not been initialized yet.")

    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img_tensor = runtime_service.transform_test(img).unsqueeze(0).to(runtime_service.device)

    parsed_vehicle, parsed_camera, parsed_frame = extract_query_metadata_from_filename(file.filename or "")
    query_vehicle_id = vehicle_id or parsed_vehicle or "query"
    query_camera_id = camera_numeric_id(camera_id) if camera_id else parsed_camera
    query_frame_id = parse_frame_value(frame_id, 0) if frame_id else parsed_frame

    resolved_mode, degraded_reason = runtime_service.select_mode(mode, query_camera_id, query_frame_id)

    with torch.no_grad():
        res_car = runtime_service.car_encoder(img_tensor)
        vbase = res_car[1] if isinstance(res_car, tuple) else res_car
        q_feat = torch.nn.functional.normalize(vbase, p=2, dim=1).cpu().numpy()

        feature_names = state["baseline_names"]
        feature_matrix = state["baseline_feats"]

        if resolved_mode in {"gnn", "st"}:
            query_vid = int(re.sub(r"\D", "", query_vehicle_id) or "0")
            query_cam = query_camera_id or 0
            hetero_data, unique_cams = build_batch_hetero_graph(
                vbase,
                torch.tensor([query_vid], dtype=torch.long).to(runtime_service.device),
                torch.tensor([query_cam], dtype=torch.long).to(runtime_service.device),
                runtime_service.device,
                is_train=False,
            )
            query_graph_feat = runtime_service.kg_module(hetero_data, unique_cams)
            q_feat = torch.nn.functional.normalize(query_graph_feat, p=2, dim=1).cpu().numpy()
            feature_names = state["graph_names"]
            feature_matrix = state["graph_feats"]

        sim_matrix = np.dot(q_feat, feature_matrix.T)
        penalized_mask = np.zeros(sim_matrix.shape[1], dtype=bool)

        if resolved_mode == "st":
            query_name = create_gallery_filename(query_vehicle_id, query_camera_id, query_frame_id or 0, file.filename or "query.jpg")
            adjusted_scores = sim_matrix.copy()
            apply_st_penalty(
                adjusted_scores,
                [query_name],
                feature_names,
                runtime_service.dist_mat,
                runtime_service.cam_idx_map,
                fps=25.0,
                max_speed=20.0,
            )
            penalized_mask = adjusted_scores[0] < sim_matrix[0]
            sim_matrix = adjusted_scores

        top_k = 10
        if resolved_mode in {"baseline", "gnn"}:
            top_k = min(150, sim_matrix.shape[1])
        indices = np.argsort(sim_matrix[0])[::-1][:top_k]

    results = []
    for rank, idx in enumerate(indices, start=1):
        name = feature_names[idx]
        record = state["record_map"].get(name, {})
        results.append(
            {
                "rank": rank,
                "name": name,
                "score": float(sim_matrix[0][idx]),
                "image_url": build_absolute_url(request, name),
                "vehicle_id": record.get("vehicle_id"),
                "camera_id": record.get("camera_numeric"),
                "frame_id": record.get("frame_id"),
                "source_name": record.get("source_name"),
                "penalized_by_st": bool(penalized_mask[idx]),
            }
        )

    voted_uid = None
    if resolved_mode == "st" and results:
        top_10_uids = [item["vehicle_id"] for item in results[:10] if item.get("vehicle_id")]
        if top_10_uids:
            voted_uid = Counter(top_10_uids).most_common(1)[0][0]

    return {
        "status": "success",
        "requested_mode": mode,
        "resolved_mode": resolved_mode,
        "degraded_reason": degraded_reason,
        "gallery_source": state["source"],
        "results": results,
        "voted_uid": voted_uid,
    }


@app.get("/trajectory_data/{vehicle_id}")
async def get_trajectory_data(request: Request, vehicle_id: str) -> dict[str, Any]:
    vehicle_records = runtime_service.gallery_state["vehicle_index"].get(vehicle_id, [])
    return {
        "vehicle_id": vehicle_id,
        "data": [
            {
                "filename": record["gallery_name"],
                "frame": record["frame_id"],
                "cam": record["camera_token"],
                "image_url": build_absolute_url(request, record["gallery_name"]),
            }
            for record in vehicle_records
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
