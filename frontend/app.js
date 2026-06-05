"use strict";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const detectBtn = document.getElementById("detect-btn");
const thresholdInput = document.getElementById("threshold");
const thresholdValue = document.getElementById("threshold-value");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const resultImg = document.getElementById("result-image");
const resultMeta = document.getElementById("result-meta");
const detCount = document.getElementById("det-count");
const detTableBody = document.querySelector("#det-table tbody");
const hwBadge = document.getElementById("hardware-badge");
const hwText = document.getElementById("hardware-text");

let selectedFile = null;

// ----- Hardware badge (from /api/health) -------------------------------------
function paintHardware(hw, label) {
  const isGpu = hw && hw.device_type === "GPU";
  hwBadge.classList.remove("badge-unknown", "badge-gpu", "badge-cpu");
  hwBadge.classList.add(isGpu ? "badge-gpu" : "badge-cpu");
  hwText.textContent = label || (isGpu ? "GPU" : "CPU");
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    paintHardware(data.hardware, data.hardware_label);
    hwBadge.title = `Model: ${data.model} | Inference device: ${data.hardware_label}`;
  } catch (err) {
    hwText.textContent = "hardware unknown";
  }
}

// ----- File selection --------------------------------------------------------
function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("Please choose an image file.", true);
    return;
  }
  selectedFile = file;
  detectBtn.disabled = false;
  setStatus(`Selected: ${file.name} (${(file.size / 1024).toFixed(0)} KB)`);
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});

thresholdInput.addEventListener("input", () => {
  thresholdValue.textContent = parseFloat(thresholdInput.value).toFixed(2);
});

// ----- Detect ----------------------------------------------------------------
function setStatus(msg, isError = false) {
  statusEl.innerHTML = msg;
  statusEl.classList.toggle("error", isError);
}

detectBtn.addEventListener("click", runDetection);

async function runDetection() {
  if (!selectedFile) return;
  detectBtn.disabled = true;
  setStatus('<span class="spinner"></span>Running inference&hellip;');

  const mode = document.querySelector('input[name="mode"]:checked').value;
  const form = new FormData();
  form.append("file", selectedFile);
  form.append("threshold", thresholdInput.value);
  form.append("mode", mode);

  try {
    const res = await fetch("/api/detect", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Request failed");
    }
    const data = await res.json();
    renderResult(data);
    setStatus(`Done &middot; ${data.count} object(s) found in ${data.inference_ms} ms.`);
  } catch (err) {
    setStatus("Error: " + err.message, true);
  } finally {
    detectBtn.disabled = false;
  }
}

function renderResult(data) {
  resultsEl.hidden = false;
  resultImg.src = data.annotated_image;

  const hw = data.hardware || {};
  const isGpu = hw.device_type === "GPU";
  paintHardware(hw, data.hardware_label);

  const modeLabel = { box: "Boxes", mask: "Masks", both: "Boxes + masks" }[data.mode] || data.mode;
  resultMeta.innerHTML = "";
  const chips = [
    { cls: isGpu ? "hw-gpu" : "hw-cpu", label: "Computed on", value: data.hardware_label },
    { label: "Model", value: data.model },
    { label: "Overlay", value: modeLabel },
    { label: "Inference", value: data.inference_ms + " ms" },
    { label: "Objects", value: data.count },
    { label: "Image", value: `${data.image_size.width}×${data.image_size.height}` },
  ];
  for (const c of chips) {
    const el = document.createElement("span");
    el.className = "chip" + (c.cls ? " " + c.cls : "");
    el.innerHTML = `${c.label}: <strong>${c.value}</strong>`;
    resultMeta.appendChild(el);
  }

  detCount.textContent = data.count;
  detTableBody.innerHTML = "";
  data.detections
    .sort((a, b) => b.confidence - a.confidence)
    .forEach((d, i) => {
      const tr = document.createElement("tr");
      const b = d.box;
      const pct = Math.round(d.confidence * 100);
      const boxCell = b ? `${b.x1}, ${b.y1}, ${b.x2}, ${b.y2}` : "&mdash;";
      const areaCell = d.mask_area != null ? d.mask_area.toLocaleString() : "&mdash;";
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td>${d.class_name}</td>
        <td class="conf">${pct}%<span class="bar" style="width:${Math.max(4, pct * 0.6)}px"></span></td>
        <td>${boxCell}</td>
        <td>${areaCell}</td>`;
      detTableBody.appendChild(tr);
    });
  resultsEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

loadHealth();
