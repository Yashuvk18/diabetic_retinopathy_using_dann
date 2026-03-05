# Diabetic Retinopathy Domain Adaptation

This project includes:

1. Source-only baseline (train on source, test on target).
2. DANN model (Domain-Adversarial Neural Network) for domain adaptation.
3. 3-class clinical triage mapping.
4. Flask web app with explainability and deployment-ready setup.

## 1) Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Key DANN Defaults

- label scheme: `three`
- classes: `3`
- epochs: `15`
- backbone: `efficientnet_b2`

## 3) 3-Class Clinical Labels

- Class 0 (No DR): Patient is healthy. See you in a year for your annual checkup.
- Class 1 (Mild/Moderate): Early signs of disease detected. Schedule a routine appointment with an eye doctor to monitor.
- Class 2 (Severe/Proliferative): Vision-threatening disease detected. Urgent medical intervention required to prevent blindness.

Label mapping from 5-class DR labels:

- `0 -> 0`
- `1 -> 1`
- `2 -> 1`
- `3 -> 2`
- `4 -> 2`

## 4) Recommended DANN Run (DDR -> APTOS)

```bash
python -m src.train_dann \
  --source-csv data/ddr/DR_grading.csv \
  --source-img-dir data/ddr/DR_grading \
  --source-image-col id_code \
  --source-label-col diagnosis \
  --target-train-csv data/aptos/target_train_unlabeled.csv \
  --target-train-img-dir data/aptos/train_images \
  --target-train-image-col id_code \
  --target-test-csv data/aptos/target_test.csv \
  --target-test-img-dir data/aptos/train_images \
  --target-test-image-col id_code \
  --target-test-label-col diagnosis \
  --target-test-default-ext .png \
  --label-scheme three \
  --num-classes 3 \
  --backbone efficientnet_b2 \
  --img-size 260 \
  --epochs 15 \
  --batch-size 16 \
  --lr 5e-5 \
  --feature-lr-mult 0.2 \
  --class-head-lr-mult 1.0 \
  --domain-lr-mult 0.25 \
  --domain-loss-weight 0.5 \
  --warmup-epochs 2 \
  --entropy-weight 0.005 \
  --label-smoothing 0.05 \
  --model-selection target_acc \
  --output-dir outputs/dann_ddr_to_aptos_3class
```

## 5) Website Features

- Single-image triage with confidence bars.
- Grad-CAM heatmap overlay (explainability).
- Patient metadata fields (`Patient ID`, `Age`, `Pre-existing Conditions`).
- Downloadable clinical PDF report (includes metadata, prediction, confidence, doctor-signature line).
- Batch processing for multiple scans with summary and downloadable CSV.
- Admin dashboard with prediction history and metadata audit trail.
- Dedicated model metrics page with confusion matrix + DANN architecture diagram.

## 6) Run Website Locally

```bash
MODEL_PATH=outputs/dann_ddr_to_aptos_3class/best_dann.pt \
BACKBONE=efficientnet_b2 \
IMG_SIZE=260 \
FLASK_SECRET_KEY=change_me \
ADMIN_USER=doctor \
ADMIN_PASSWORD=retina123 \
python web_app.py
```

Open:

- `http://127.0.0.1:8502/` (screening)
- `http://127.0.0.1:8502/metrics` (academic metrics tab)
- `http://127.0.0.1:8502/login` (doctor login)
- `http://127.0.0.1:8502/admin` (admin dashboard)

Optional env vars:

- `WEB_DEVICE=cpu|mps|cuda|auto`
- `MAX_BATCH_FILES=100`
- `METRICS_DIR=outputs/dann_ddr_to_aptos_3class`
- `PRESENTATION_IDRID_ACCURACY=71.87`
- `PRESENTATION_URGENT_RECALL=79`

## 7) Main API Endpoints

- `POST /predict` (single image + metadata)
- `POST /predict-batch` (multiple images + metadata)
- `POST /report` (PDF generation from prediction payload)

## 8) Docker

Build image:

```bash
docker build -t retina-dann-web .
```

Run container:

```bash
docker run --rm -p 8502:8502 \
  -e PORT=8502 \
  -e MODEL_PATH=outputs/dann_ddr_to_aptos_3class/best_dann.pt \
  -e BACKBONE=efficientnet_b2 \
  -e IMG_SIZE=260 \
  -e ADMIN_USER=doctor \
  -e ADMIN_PASSWORD=retina123 \
  -e FLASK_SECRET_KEY=change_me \
  retina-dann-web
```

Note: deployment must include the trained checkpoint at `MODEL_PATH`.
