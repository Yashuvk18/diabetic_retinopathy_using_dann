from __future__ import annotations

import base64
import io
import json
import os
import re
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from fpdf import FPDF
from PIL import Image
from torchvision import transforms
from werkzeug.exceptions import RequestEntityTooLarge

from src.models import DANN


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_ROOT / "outputs" / "dann_ddr_to_aptos_3class" / "best_dann.pt"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
BACKBONE = os.getenv("BACKBONE", "efficientnet_b2")
IMG_SIZE = int(os.getenv("IMG_SIZE", "260"))

DEFAULT_METRICS_DIR = APP_ROOT / "outputs" / "dann_ddr_to_aptos_3class"
METRICS_DIR = Path(os.getenv("METRICS_DIR", str(DEFAULT_METRICS_DIR)))
PRESENTATION_IDRID_ACCURACY = float(os.getenv("PRESENTATION_IDRID_ACCURACY", "71.87"))
PRESENTATION_URGENT_RECALL = float(os.getenv("PRESENTATION_URGENT_RECALL", "79.0"))
MAX_BATCH_FILES = int(os.getenv("MAX_BATCH_FILES", "100"))

ADMIN_USER = os.getenv("ADMIN_USER", "doctor")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "retina123")
PREDICTION_LOG_PATH = APP_ROOT / "outputs" / "prediction_logs.jsonl"
PREDICTION_ASSET_DIR = APP_ROOT / "outputs" / "prediction_assets"

CLASS_INFO = {
    0: {
        "label": "Class 0 (No DR)",
        "message": "Patient is healthy. See you in a year for your annual checkup.",
        "urgency": "Routine",
    },
    1: {
        "label": "Class 1 (Mild/Moderate)",
        "message": "Early signs of disease detected. Schedule a routine appointment with an eye doctor to monitor.",
        "urgency": "Monitor",
    },
    2: {
        "label": "Class 2 (Severe/Proliferative)",
        "message": "Vision-threatening disease detected. Urgent medical intervention required to prevent blindness.",
        "urgency": "Urgent",
    },
}


def resolve_device() -> torch.device:
    requested = os.getenv("WEB_DEVICE", "cpu").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEVICE = resolve_device()
# Keep CPU threading controlled on small cloud instances.
torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
MODEL: DANN | None = None
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "change-this-secret-in-production")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "32")) * 1024 * 1024


def login_required(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)

    return wrapper


def append_prediction_log(record: dict[str, Any]) -> None:
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_prediction_logs(limit: int = 200) -> list[dict[str, Any]]:
    if not PREDICTION_LOG_PATH.exists():
        return []

    rows: list[dict[str, Any]] = []
    with PREDICTION_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return list(reversed(rows[-limit:]))


def summarize_logs(logs: list[dict[str, Any]]) -> dict[str, int]:
    summary = {CLASS_INFO[i]["label"]: 0 for i in sorted(CLASS_INFO.keys())}
    for row in logs:
        label = row.get("label")
        if label in summary:
            summary[label] += 1
    return summary


def summarize_urgency(logs: list[dict[str, Any]]) -> dict[str, int]:
    urgency_summary = {"Routine": 0, "Monitor": 0, "Urgent": 0}
    for row in logs:
        urgency = str(row.get("urgency", ""))
        if urgency in urgency_summary:
            urgency_summary[urgency] += 1
    return urgency_summary


def parse_patient_metadata(form_data) -> dict[str, str]:
    return {
        "patient_id": str(form_data.get("patient_id", "")).strip(),
        "age": str(form_data.get("age", "")).strip(),
        "conditions": str(form_data.get("conditions", "")).strip(),
    }


def save_uploaded_asset(image: Image.Image) -> str:
    PREDICTION_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    path = PREDICTION_ASSET_DIR / f"{token}.png"
    image.convert("RGB").save(path, format="PNG")
    return token


def resolve_asset_path(token: str) -> Path | None:
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        return None
    path = PREDICTION_ASSET_DIR / f"{token}.png"
    if not path.exists():
        return None
    return path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
            if isinstance(payload, dict):
                return payload
    except Exception:
        return {}
    return {}


def short_class_label(label: str) -> str:
    if ":" in label:
        return label.split(":", 1)[0].strip()
    return label.strip()


def load_metrics_view() -> dict[str, Any]:
    metrics = load_json(METRICS_DIR / "metrics.json")
    report = load_json(METRICS_DIR / "target_classification_report.json")

    final_target = metrics.get("final_target", {}) if isinstance(metrics, dict) else {}
    final_source = metrics.get("final_source_val", {}) if isinstance(metrics, dict) else {}

    raw_names = report.get("class_names") or metrics.get("target_class_names")
    if not isinstance(raw_names, list) or not raw_names:
        raw_names = [CLASS_INFO[i]["label"] for i in sorted(CLASS_INFO.keys())]

    class_names = [short_class_label(str(name)) for name in raw_names]
    report_body = report.get("classification_report", {})

    class_rows: list[dict[str, Any]] = []
    for idx, raw_name in enumerate(raw_names):
        row = report_body.get(raw_name, {}) if isinstance(report_body, dict) else {}
        class_rows.append(
            {
                "name": class_names[idx],
                "precision": float(row.get("precision", 0.0)) * 100.0,
                "recall": float(row.get("recall", 0.0)) * 100.0,
                "f1": float(row.get("f1-score", 0.0)) * 100.0,
                "support": int(row.get("support", 0)),
            }
        )

    cm = report.get("confusion_matrix", [])
    if not isinstance(cm, list) or not cm:
        size = len(class_rows) if class_rows else 3
        cm = [[0 for _ in range(size)] for _ in range(size)]

    cm_max = 1
    for row in cm:
        if isinstance(row, list) and row:
            cm_max = max(cm_max, max(int(v) for v in row))

    cm_rows: list[list[dict[str, Any]]] = []
    for row in cm:
        if not isinstance(row, list):
            continue
        row_vals = [int(v) for v in row]
        row_sum = max(1, sum(row_vals))
        cm_rows.append(
            [
                {
                    "value": value,
                    "pct": (value / row_sum) * 100.0,
                    "alpha": 0.18 + 0.72 * (value / cm_max),
                }
                for value in row_vals
            ]
        )

    history = metrics.get("history", []) if isinstance(metrics, dict) else []
    history_rows: list[dict[str, Any]] = []
    if isinstance(history, list):
        for row in history[-8:]:
            if not isinstance(row, dict):
                continue
            history_rows.append(
                {
                    "epoch": int(row.get("epoch", 0)),
                    "target_acc": float(row.get("target_acc", 0.0)) * 100.0,
                    "source_val_acc": float(row.get("source_val_acc", 0.0)) * 100.0,
                }
            )

    urgent_recall = class_rows[-1]["recall"] if class_rows else 0.0

    return {
        "metrics_available": bool(metrics),
        "report_available": bool(report),
        "metrics_dir": str(METRICS_DIR),
        "current_target_accuracy": float(final_target.get("accuracy", 0.0)) * 100.0,
        "current_target_macro_f1": float(final_target.get("macro_f1", 0.0)) * 100.0,
        "current_source_accuracy": float(final_source.get("accuracy", 0.0)) * 100.0,
        "current_source_macro_f1": float(final_source.get("macro_f1", 0.0)) * 100.0,
        "urgent_recall": float(urgent_recall),
        "presentation_idrid_accuracy": PRESENTATION_IDRID_ACCURACY,
        "presentation_urgent_recall": PRESENTATION_URGENT_RECALL,
        "class_names": class_names,
        "class_rows": class_rows,
        "cm_rows": cm_rows,
        "history_rows": history_rows,
    }


def load_model() -> DANN:
    global MODEL

    if MODEL is not None:
        return MODEL

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {MODEL_PATH}. "
            "Train the model first or set MODEL_PATH env var."
        )

    model = DANN(num_classes=3, backbone=BACKBONE, pretrained=False)
    state = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    MODEL = model
    return MODEL


def find_last_conv_layer(module: nn.Module) -> nn.Module | None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Conv2d):
            return layer
    return None


def apply_heat_colormap(cam: np.ndarray) -> np.ndarray:
    cam = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.8 * cam - 0.2, 0.0, 1.0)
    green = np.clip(1.8 * (1.0 - np.abs(cam - 0.5) * 2.0) - 0.2, 0.0, 1.0)
    blue = np.clip(1.8 * (1.0 - cam) - 0.2, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def create_gradcam_overlay(
    model: DANN,
    x: torch.Tensor,
    class_index: int,
    original_image: Image.Image,
) -> str:
    layer = find_last_conv_layer(model.feature_extractor)
    if layer is None:
        return ""

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_, __, output):
        activations["value"] = output

    def backward_hook(_, grad_input, grad_output):
        if grad_output:
            gradients["value"] = grad_output[0]

    h1 = layer.register_forward_hook(forward_hook)
    h2 = layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model.classify(x)
        score = logits[:, class_index].sum()
        score.backward()

        act = activations.get("value")
        grad = gradients.get("value")
        if act is None or grad is None:
            return ""

        act_map = act[0]
        grad_map = grad[0]
        weights = grad_map.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * act_map).sum(dim=0))

        cam_np = cam.detach().float().cpu().numpy()
        if cam_np.max() <= 0:
            return ""

        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        base_image = original_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        heat = apply_heat_colormap(cam_np)
        heat_img = Image.fromarray((heat * 255).astype(np.uint8), mode="RGB").resize(
            base_image.size,
            Image.BILINEAR,
        )

        overlay = Image.blend(base_image, heat_img, alpha=0.43)
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    finally:
        h1.remove()
        h2.remove()
        model.zero_grad(set_to_none=True)


def predict_image(image: Image.Image, include_gradcam: bool = True) -> dict[str, Any]:
    model = load_model()
    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model.classify(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().tolist()

    pred_idx = int(np.argmax(np.array(probs, dtype=np.float32)))
    pred = CLASS_INFO[pred_idx]

    probabilities = {CLASS_INFO[i]["label"]: float(probs[i]) for i in range(len(CLASS_INFO))}

    result: dict[str, Any] = {
        "class_index": pred_idx,
        "label": pred["label"],
        "message": pred["message"],
        "urgency": pred["urgency"],
        "confidence": float(max(probs)),
        "probabilities": probabilities,
    }

    if include_gradcam:
        gradcam_image = create_gradcam_overlay(model, x, pred_idx, image)
        if gradcam_image:
            result["gradcam_image"] = gradcam_image

    return result


def build_batch_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_class = {CLASS_INFO[i]["label"]: 0 for i in sorted(CLASS_INFO.keys())}
    by_urgency = {"Routine": 0, "Monitor": 0, "Urgent": 0}

    for row in results:
        label = row.get("label")
        urgency = row.get("urgency")
        if label in by_class:
            by_class[label] += 1
        if urgency in by_urgency:
            by_urgency[urgency] += 1

    return {
        "total": len(results),
        "urgent_cases": by_urgency["Urgent"],
        "by_class": by_class,
        "by_urgency": by_urgency,
    }


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    return name or "retina"


def _pdf_safe_text(value: Any, max_token_len: int = 28) -> str:
    text = str(value if value is not None else "-")
    text = text.replace("\r", " ").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    # Core Helvetica font in FPDF is Latin-1 only.
    text = text.encode("latin-1", "replace").decode("latin-1")
    # Insert soft breaks for very long unspaced tokens.
    text = re.sub(rf"(\S{{{max_token_len}}})(?=\S)", r"\1 ", text)
    return text or "-"


def _pdf_multicell(pdf: FPDF, text: Any, line_h: float = 8.0) -> None:
    safe_text = _pdf_safe_text(text)
    usable_width = max(20.0, pdf.w - pdf.l_margin - pdf.r_margin)
    pdf.set_x(pdf.l_margin)
    try:
        pdf.multi_cell(
            usable_width,
            line_h,
            safe_text,
            new_x="LMARGIN",
            new_y="NEXT",
            wrapmode="CHAR",
        )
    except TypeError:
        pdf.multi_cell(usable_width, line_h, safe_text)


def build_pdf_report(payload: dict[str, Any], image_path: Path | None = None) -> bytes:
    patient = payload.get("patient", {}) if isinstance(payload.get("patient"), dict) else {}

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    _pdf_multicell(pdf, "RetinaCare DANN - Clinical Screening Report", line_h=10)

    pdf.set_font("Helvetica", "", 11)
    _pdf_multicell(pdf, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", line_h=7)
    _pdf_multicell(pdf, f"Model: DANN ({BACKBONE}, img_size={IMG_SIZE}, device={DEVICE.type})", line_h=7)

    file_name = _pdf_safe_text(payload.get("file_name", "Uploaded Image"), max_token_len=22)
    _pdf_multicell(pdf, f"Image File: {file_name}", line_h=7)

    patient_id = _pdf_safe_text(patient.get("patient_id", "-"), max_token_len=18)
    age = _pdf_safe_text(patient.get("age", "-"), max_token_len=18)
    conditions = _pdf_safe_text(patient.get("conditions", "-"), max_token_len=18)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 13)
    _pdf_multicell(pdf, "Patient Metadata", line_h=8)
    pdf.set_font("Helvetica", "", 11)
    _pdf_multicell(pdf, f"Patient ID: {patient_id}", line_h=7)
    _pdf_multicell(pdf, f"Age: {age}", line_h=7)
    _pdf_multicell(pdf, f"Pre-existing Conditions: {conditions}", line_h=7)

    if image_path and image_path.exists():
        if pdf.get_y() > 170:
            pdf.add_page()
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 13)
        _pdf_multicell(pdf, "Uploaded Fundus Image", line_h=8)
        y_pos = pdf.get_y()
        pdf.image(str(image_path), x=15, y=y_pos, w=95)
        pdf.set_y(y_pos + 72)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 14)
    _pdf_multicell(pdf, "Model Prediction", line_h=8)

    pdf.set_font("Helvetica", "", 12)
    _pdf_multicell(pdf, f"Predicted Label: {_pdf_safe_text(payload.get('label', '-'), 20)}", line_h=7)
    _pdf_multicell(pdf, f"Urgency: {_pdf_safe_text(payload.get('urgency', '-'), 20)}", line_h=7)
    confidence = float(payload.get("confidence", 0.0)) * 100.0
    _pdf_multicell(pdf, f"Confidence: {confidence:.2f}%", line_h=7)
    _pdf_multicell(pdf, f"Guidance: {_pdf_safe_text(payload.get('message', '-'), 24)}", line_h=7)

    probs = payload.get("probabilities") or {}
    if isinstance(probs, dict) and probs:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 13)
        _pdf_multicell(pdf, "Confidence Distribution", line_h=8)

        pdf.set_font("Helvetica", "", 11)
        for label, score in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            pct = float(score) * 100.0
            _pdf_multicell(pdf, f"{_pdf_safe_text(label, 24)}: {pct:.2f}%", line_h=7)

    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    _pdf_multicell(pdf, "Doctor Signature: _______________________________", line_h=7)

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 10)
    _pdf_multicell(
        pdf,
        "This report is for academic screening support only. "
        "Final diagnosis and treatment decisions must be made by qualified medical professionals.",
        line_h=6,
    )

    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        return raw.encode("latin-1")
    return bytes(raw)
@app.get("/")
def index():
    model_ready = MODEL_PATH.exists()
    return render_template(
        "index.html",
        model_ready=model_ready,
        model_path=str(MODEL_PATH),
        backbone=BACKBONE,
        img_size=IMG_SIZE,
        max_batch_files=MAX_BATCH_FILES,
        is_admin=bool(session.get("is_admin")),
    )


@app.get("/metrics")
def metrics():
    context = load_metrics_view()
    context.update(
        {
            "is_admin": bool(session.get("is_admin")),
            "model_path": str(MODEL_PATH),
            "backbone": BACKBONE,
            "img_size": IMG_SIZE,
        }
    )
    return render_template("metrics.html", **context)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    next_url = request.args.get("next") or url_for("admin")

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if username == ADMIN_USER and password == ADMIN_PASSWORD:
            session["is_admin"] = True
            session["admin_user"] = username
            return redirect(next_url)

        error = "Invalid username or password."

    return render_template("login.html", error=error, next_url=next_url)


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.get("/admin")
@login_required
def admin():
    logs = read_prediction_logs(limit=500)
    summary = summarize_logs(logs)
    urgency_summary = summarize_urgency(logs)
    return render_template(
        "admin.html",
        logs=logs,
        summary=summary,
        urgency_summary=urgency_summary,
        total=len(logs),
        admin_user=session.get("admin_user", "doctor"),
    )


@app.errorhandler(RequestEntityTooLarge)
def handle_upload_too_large(_exc):
    max_mb = int(app.config.get("MAX_CONTENT_LENGTH", 0) / (1024 * 1024))
    return jsonify({"error": f"Uploaded file is too large. Max allowed size is {max_mb}MB."}), 413


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    patient = parse_patient_metadata(request.form)

    try:
        image = Image.open(file.stream).convert("RGB")
        result = predict_image(image, include_gradcam=True)

        asset_token = save_uploaded_asset(image)
        result["asset_token"] = asset_token
        result["file_name"] = file.filename
        result["patient"] = patient

        log_row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": "single",
            "file_name": file.filename,
            "label": result["label"],
            "urgency": result["urgency"],
            "class_index": result["class_index"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "asset_token": asset_token,
            "patient_id": patient["patient_id"],
            "age": patient["age"],
            "conditions": patient["conditions"],
        }
        append_prediction_log(log_row)

        return jsonify(result)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.post("/predict-batch")
def predict_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No files uploaded for batch prediction."}), 400

    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"error": "All uploaded files were empty."}), 400

    if len(files) > MAX_BATCH_FILES:
        return jsonify({"error": f"Too many files. Maximum allowed is {MAX_BATCH_FILES}."}), 400

    patient = parse_patient_metadata(request.form)
    batch_id = datetime.now(timezone.utc).strftime("BATCH-%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:6]}"

    results: list[dict[str, Any]] = []

    for file in files:
        try:
            image = Image.open(file.stream).convert("RGB")
            pred = predict_image(image, include_gradcam=False)
            asset_token = save_uploaded_asset(image)

            row = {
                "file_name": file.filename,
                "label": pred["label"],
                "urgency": pred["urgency"],
                "class_index": pred["class_index"],
                "confidence": pred["confidence"],
                "probabilities": pred["probabilities"],
                "asset_token": asset_token,
                "patient": patient,
            }
            results.append(row)

            append_prediction_log(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mode": "batch",
                    "batch_id": batch_id,
                    "file_name": file.filename,
                    "label": pred["label"],
                    "urgency": pred["urgency"],
                    "class_index": pred["class_index"],
                    "confidence": pred["confidence"],
                    "probabilities": pred["probabilities"],
                    "asset_token": asset_token,
                    "patient_id": patient["patient_id"],
                    "age": patient["age"],
                    "conditions": patient["conditions"],
                }
            )
        except Exception as exc:  # pragma: no cover
            results.append({"file_name": file.filename or "unknown", "error": str(exc)})

    valid_results = [row for row in results if "error" not in row]
    summary = build_batch_summary(valid_results)

    return jsonify(
        {
            "batch_id": batch_id,
            "summary": summary,
            "patient": patient,
            "results": results,
        }
    )


@app.post("/report")
def report():
    payload = request.get_json(silent=True) or {}
    if not payload:
        return jsonify({"error": "Missing report payload."}), 400

    asset_token = str(payload.get("asset_token", "")).strip()
    image_path = resolve_asset_path(asset_token)

    try:
        pdf_bytes = build_pdf_report(payload, image_path=image_path)

        patient = payload.get("patient", {}) if isinstance(payload.get("patient"), dict) else {}
        patient_id = sanitize_filename(str(patient.get("patient_id", "") or "retina"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        download_name = f"clinical_report_{patient_id}_{timestamp}.pdf"

        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=download_name,
        )
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Failed to generate report: {exc}"}), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8502")), debug=debug_mode)
