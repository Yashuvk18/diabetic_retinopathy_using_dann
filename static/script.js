const input = document.getElementById("image-input");
const form = document.getElementById("predict-form");
const previewWrap = document.getElementById("preview-wrap");
const preview = document.getElementById("preview");
const dropzone = document.getElementById("dropzone");
const modeHint = document.getElementById("mode-hint");

const patientIdInput = document.getElementById("patient-id");
const patientAgeInput = document.getElementById("patient-age");
const patientConditionsInput = document.getElementById("patient-conditions");

const predictBtn = document.getElementById("predict-btn");
const forceBatchBtn = document.getElementById("force-batch-btn");
const downloadReportBtn = document.getElementById("download-report-btn");
const downloadBatchCsvBtn = document.getElementById("download-batch-csv-btn");

const resultState = document.getElementById("result-state");
const singleResult = document.getElementById("single-result");
const batchResult = document.getElementById("batch-result");

const resultBadge = document.getElementById("result-badge");
const resultUrgency = document.getElementById("result-urgency");
const resultLabel = document.getElementById("result-label");
const resultMessage = document.getElementById("result-message");
const resultPatient = document.getElementById("result-patient");
const probList = document.getElementById("prob-list");

const camPanel = document.getElementById("cam-panel");
const camImage = document.getElementById("cam-image");

const batchId = document.getElementById("batch-id");
const batchNote = document.getElementById("batch-note");
const batchSummary = document.getElementById("batch-summary");
const batchRows = document.getElementById("batch-body-rows");

let selectedFiles = [];
let latestPrediction = null;
let latestBatch = null;
let previewObjectUrl = "";

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function collectPatientMetadata() {
  return {
    patient_id: (patientIdInput?.value || "").trim(),
    age: (patientAgeInput?.value || "").trim(),
    conditions: (patientConditionsInput?.value || "").trim(),
  };
}

function patientLine(patient) {
  const parts = [];
  if (patient?.patient_id) parts.push(`Patient ID: ${patient.patient_id}`);
  if (patient?.age) parts.push(`Age: ${patient.age}`);
  if (patient?.conditions) parts.push(`Conditions: ${patient.conditions}`);
  return parts.join(" | ") || "Patient metadata not provided.";
}

function setLoading(isLoading, mode = "single") {
  predictBtn.disabled = isLoading;
  forceBatchBtn.disabled = isLoading;

  if (mode === "batch") {
    predictBtn.textContent = isLoading ? "Running Batch..." : "Run Triage";
    forceBatchBtn.textContent = isLoading ? "Running Batch..." : "Run Batch Triage";
  } else {
    predictBtn.textContent = isLoading ? "Analyzing..." : "Run Triage";
    forceBatchBtn.textContent = isLoading ? "Analyzing..." : "Run Batch Triage";
  }

  if (downloadReportBtn) downloadReportBtn.disabled = isLoading;
  if (downloadBatchCsvBtn) downloadBatchCsvBtn.disabled = isLoading;
}

function hideAllResults() {
  singleResult.hidden = true;
  batchResult.hidden = true;
}

function setState(text, isError = false) {
  resultState.hidden = false;
  resultState.textContent = text;
  resultState.style.background = isError ? "#ffe8e2" : "#f4f1ea";
  resultState.style.color = isError ? "#7a2f1f" : "#5a6b62";
  hideAllResults();
}

function renderProbabilities(probabilities) {
  probList.innerHTML = "";
  const entries = Object.entries(probabilities || {}).sort((a, b) => b[1] - a[1]);

  for (const [label, prob] of entries) {
    const item = document.createElement("div");
    item.className = "prob-item";

    const pct = Math.max(0, Math.min(100, Number(prob) * 100));

    item.innerHTML = `
      <div class="prob-head">
        <span>${escapeHtml(label)}</span>
        <span>${pct.toFixed(1)}%</span>
      </div>
      <div class="prob-bar">
        <div class="prob-fill" style="width:${pct}%"></div>
      </div>
    `;

    probList.appendChild(item);
  }
}

function updateModeHint() {
  if (!selectedFiles.length) {
    modeHint.textContent = "No files selected.";
    return;
  }

  if (selectedFiles.length === 1) {
    modeHint.textContent = "Single-image mode: prediction + Grad-CAM + PDF report.";
    return;
  }

  modeHint.textContent = `${selectedFiles.length} files selected: batch triage mode.`;
}

function showPreview(file) {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
  }
  previewObjectUrl = URL.createObjectURL(file);
  preview.src = previewObjectUrl;
  previewWrap.hidden = false;
}

function handleFiles(fileList) {
  selectedFiles = Array.from(fileList || []);
  latestPrediction = null;
  latestBatch = null;

  if (downloadReportBtn) {
    downloadReportBtn.hidden = true;
  }
  if (downloadBatchCsvBtn) {
    downloadBatchCsvBtn.hidden = true;
  }

  if (!selectedFiles.length) {
    previewWrap.hidden = true;
    updateModeHint();
    setState("Waiting for image upload...");
    return;
  }

  if (selectedFiles.length === 1) {
    showPreview(selectedFiles[0]);
    setState("Image selected. Click 'Run Triage'.");
  } else {
    previewWrap.hidden = true;
    setState(`${selectedFiles.length} images selected. Click 'Run Triage' for batch processing.`);
  }

  updateModeHint();
}

function showSingleResult(payload) {
  latestPrediction = payload;
  latestBatch = null;

  resultState.hidden = true;
  singleResult.hidden = false;
  batchResult.hidden = true;

  resultBadge.textContent = "Predicted";
  resultUrgency.textContent = `${payload.urgency || "Unknown"} priority`;
  resultLabel.textContent = payload.label || "-";
  resultMessage.textContent = payload.message || "-";

  const patient = payload.patient || collectPatientMetadata();
  resultPatient.textContent = patientLine(patient);

  renderProbabilities(payload.probabilities || {});

  if (payload.gradcam_image) {
    camImage.src = payload.gradcam_image;
    camPanel.hidden = false;
  } else {
    camPanel.hidden = true;
  }

  if (downloadReportBtn) {
    downloadReportBtn.hidden = false;
  }
  if (downloadBatchCsvBtn) {
    downloadBatchCsvBtn.hidden = true;
  }
}

function renderBatchSummary(summary) {
  const byUrgency = summary?.by_urgency || {};
  const byClass = summary?.by_class || {};

  const cards = [
    { title: "Total Processed", value: summary?.total || 0 },
    { title: "Urgent Cases", value: summary?.urgent_cases || 0 },
    { title: "Routine", value: byUrgency["Routine"] || 0 },
    { title: "Monitor", value: byUrgency["Monitor"] || 0 },
    { title: "Urgent", value: byUrgency["Urgent"] || 0 },
  ];

  for (const [label, count] of Object.entries(byClass)) {
    cards.push({ title: label, value: count });
  }

  batchSummary.innerHTML = cards
    .map(
      (card) => `
      <article class="batch-summary-card">
        <h4>${escapeHtml(card.title)}</h4>
        <p>${escapeHtml(card.value)}</p>
      </article>
    `,
    )
    .join("");
}

function renderBatchRows(rows) {
  batchRows.innerHTML = "";

  for (const row of rows) {
    const tr = document.createElement("tr");

    if (row.error) {
      tr.innerHTML = `
        <td>${escapeHtml(row.file_name || "-")}</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td class="status-error">${escapeHtml(row.error)}</td>
      `;
      batchRows.appendChild(tr);
      continue;
    }

    const confidence = Math.max(0, Math.min(100, Number(row.confidence || 0) * 100));

    tr.innerHTML = `
      <td>${escapeHtml(row.file_name || "-")}</td>
      <td>${escapeHtml(row.label || "-")}</td>
      <td>${escapeHtml(row.urgency || "-")}</td>
      <td>
        <div class="mini-bar-wrap">
          <div class="mini-bar" style="width:${confidence}%;"></div>
        </div>
        <span class="mini-bar-label">${confidence.toFixed(1)}%</span>
      </td>
      <td class="status-ok">OK</td>
    `;

    batchRows.appendChild(tr);
  }
}

function showBatchResult(payload) {
  latestBatch = payload;
  latestPrediction = null;

  resultState.hidden = true;
  singleResult.hidden = true;
  batchResult.hidden = false;

  batchId.textContent = payload.batch_id || "Batch";
  batchNote.textContent = patientLine(payload.patient || collectPatientMetadata());

  renderBatchSummary(payload.summary || {});
  renderBatchRows(payload.results || []);

  if (downloadReportBtn) {
    downloadReportBtn.hidden = true;
  }
  if (downloadBatchCsvBtn) {
    downloadBatchCsvBtn.hidden = !(payload.results && payload.results.length);
  }
}

async function parseJsonResponse(response) {
  const text = await response.text();
  let payload = null;

  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (_err) {
      payload = null;
    }
  }

  if (!response.ok) {
    const message = (payload && payload.error) || text || `Request failed with status ${response.status}`;
    throw new Error(message);
  }

  if (!payload) {
    throw new Error("Server returned empty/non-JSON response. Check deployment logs for backend errors.");
  }

  return payload;
}

async function runSinglePrediction() {
  if (!window.MODEL_READY) {
    setState("Model checkpoint is missing. Train model or set MODEL_PATH first.", true);
    return;
  }

  if (selectedFiles.length !== 1) {
    setState("Select exactly one image for single prediction mode.", true);
    return;
  }

  setLoading(true, "single");
  setState("Running model inference...");

  try {
    const metadata = collectPatientMetadata();
    const data = new FormData();
    data.append("image", selectedFiles[0]);
    data.append("patient_id", metadata.patient_id);
    data.append("age", metadata.age);
    data.append("conditions", metadata.conditions);

    const response = await fetch("/predict", {
      method: "POST",
      body: data,
    });

    const payload = await parseJsonResponse(response);
    showSingleResult(payload);
  } catch (error) {
    setState(error.message || "Prediction failed.", true);
  } finally {
    setLoading(false, "single");
  }
}

async function runBatchPrediction() {
  if (!window.MODEL_READY) {
    setState("Model checkpoint is missing. Train model or set MODEL_PATH first.", true);
    return;
  }

  if (!selectedFiles.length) {
    setState("Please upload one or more images first.", true);
    return;
  }

  if (selectedFiles.length > Number(window.MAX_BATCH_FILES || 100)) {
    setState(`Batch limit exceeded. Maximum allowed is ${window.MAX_BATCH_FILES}.`, true);
    return;
  }

  setLoading(true, "batch");
  setState(`Running batch inference on ${selectedFiles.length} images...`);

  try {
    const metadata = collectPatientMetadata();
    const data = new FormData();

    for (const file of selectedFiles) {
      data.append("images", file);
    }

    data.append("patient_id", metadata.patient_id);
    data.append("age", metadata.age);
    data.append("conditions", metadata.conditions);

    const response = await fetch("/predict-batch", {
      method: "POST",
      body: data,
    });

    const payload = await parseJsonResponse(response);
    showBatchResult(payload);
  } catch (error) {
    setState(error.message || "Batch prediction failed.", true);
  } finally {
    setLoading(false, "batch");
  }
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

input.addEventListener("change", (event) => {
  handleFiles(event.target.files);
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.style.borderColor = "#1f7a65";
});

dropzone.addEventListener("dragleave", () => {
  dropzone.style.borderColor = "#9bb9ad";
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.style.borderColor = "#9bb9ad";
  const files = event.dataTransfer?.files;
  if (files && files.length) {
    handleFiles(files);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!selectedFiles.length) {
    setState("Please upload an image first.", true);
    return;
  }

  if (selectedFiles.length === 1) {
    await runSinglePrediction();
    return;
  }

  await runBatchPrediction();
});

forceBatchBtn.addEventListener("click", async () => {
  if (!selectedFiles.length) {
    setState("Please upload images first for batch mode.", true);
    return;
  }
  await runBatchPrediction();
});

if (downloadReportBtn) {
  downloadReportBtn.addEventListener("click", async () => {
    if (!latestPrediction) {
      setState("Run single prediction first to download the clinical report.", true);
      return;
    }

    try {
      downloadReportBtn.disabled = true;
      downloadReportBtn.textContent = "Generating PDF...";

      const reportPayload = { ...latestPrediction };
      delete reportPayload.gradcam_image;

      const response = await fetch("/report", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(reportPayload),
      });

      if (!response.ok) {
        const text = await response.text();
        let payload = {};
        try {
          payload = JSON.parse(text);
        } catch (_err) {
          payload = {};
        }
        throw new Error(payload.error || text || "Failed to generate PDF report.");
      }

      const blob = await response.blob();
      downloadBlob(blob, "retina_clinical_report.pdf");
    } catch (error) {
      setState(error.message || "Failed to download PDF report.", true);
    } finally {
      downloadReportBtn.disabled = false;
      downloadReportBtn.textContent = "Download Clinical Report";
    }
  });
}

if (downloadBatchCsvBtn) {
  downloadBatchCsvBtn.addEventListener("click", () => {
    if (!latestBatch || !Array.isArray(latestBatch.results)) {
      setState("No batch results available to export.", true);
      return;
    }

    const rows = [
      ["batch_id", "file_name", "label", "urgency", "confidence", "status", "error"],
    ];

    for (const row of latestBatch.results) {
      rows.push([
        latestBatch.batch_id || "",
        row.file_name || "",
        row.label || "",
        row.urgency || "",
        row.confidence !== undefined ? (Number(row.confidence) * 100).toFixed(2) : "",
        row.error ? "failed" : "ok",
        row.error || "",
      ]);
    }

    const csv = rows
      .map((line) =>
        line
          .map((value) => {
            const escaped = String(value).replace(/"/g, '""');
            return `"${escaped}"`;
          })
          .join(","),
      )
      .join("\n");

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    downloadBlob(blob, "retina_batch_results.csv");
  });
}

updateModeHint();
setState("Waiting for image upload...");
