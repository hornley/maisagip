const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileList = document.getElementById("file-list");
const previewWrap = document.getElementById("preview-wrap");
const previewsEl = document.getElementById("previews");
const inspectBtn = document.getElementById("inspect-btn");
const statusEl = document.getElementById("dz-status");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const confidenceThreshold = document.getElementById("confidence-threshold");
const confidenceThresholdValue = document.getElementById("confidence-threshold-value");

let selectedFiles = [];
let previewUrls = [];
let selectionGeneration = 0;
let mismatchResults = [];
let mismatchIndex = 0;

const mismatchDialog = document.getElementById("mismatch-dialog");
const mismatchClose = document.getElementById("mismatch-close");
const mismatchPrevious = document.getElementById("mismatch-previous");
const mismatchNext = document.getElementById("mismatch-next");
const mismatchThumbnails = document.getElementById("mismatch-thumbnails");

confidenceThreshold.addEventListener("input", () => {
  confidenceThresholdValue.textContent = `${confidenceThreshold.value}%`;
});

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) selectFiles([...fileInput.files]);
});

["dragover", "dragenter"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.style.background = "var(--green-100)";
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.style.background = "";
  })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files && e.dataTransfer.files.length) {
    selectFiles([...e.dataTransfer.files]);
  }
});

function selectFiles(files) {
  selectionGeneration += 1;
  if (mismatchDialog.open) mismatchDialog.close();
  mismatchResults = [];
  previewUrls.forEach((url) => URL.revokeObjectURL(url));
  previewUrls = files.map((file) => URL.createObjectURL(file));
  selectedFiles = files;
  errorEl.classList.add("hidden");
  resultEl.classList.add("hidden");
  previewsEl.innerHTML = "";
  files.forEach((f, index) => {
    const holder = document.createElement("div");
    holder.className = "pthumb";
    const img = document.createElement("img");
    img.src = previewUrls[index];
    const cap = document.createElement("span");
    cap.className = "pthumb-cap";
    cap.textContent = f.name;
    holder.append(img, cap);
    previewsEl.appendChild(holder);
  });
  fileList.textContent = `${files.length} photo(s) selected`;
  fileList.classList.remove("hidden");
  previewWrap.classList.remove("hidden");
  statusEl.textContent = "";
}

inspectBtn.addEventListener("click", async () => {
  if (!selectedFiles.length) return;
  const requestGeneration = selectionGeneration;
  const requestFiles = selectedFiles;
  inspectBtn.disabled = true;
  statusEl.textContent = "Analyzing ear…";
  const form = new FormData();
  requestFiles.forEach((f) => form.append("files", f));
  form.append("confidence_threshold", String(Number(confidenceThreshold.value) / 100));
  try {
    const res = await fetch("/inspect", { method: "POST", body: form });
    const data = await res.json();
    if (requestGeneration !== selectionGeneration) return;
    if (!res.ok) {
      const detail = typeof data.detail === "string" ? data.detail : `HTTP ${res.status}`;
      if (isStructuredMismatch(data, requestFiles.length)) {
        showMismatchDialog(detail, data.view_results);
        return;
      }
      throw new Error(detail);
    }
    render(data);
  } catch (err) {
    if (requestGeneration !== selectionGeneration) return;
    showError(err.message);
  } finally {
    inspectBtn.disabled = false;
    statusEl.textContent = "";
  }
});

function isStructuredMismatch(data, fileCount) {
  if (!data || typeof data.detail !== "string" || !Array.isArray(data.view_results)) return false;
  if (data.view_results.length !== fileCount || !fileCount) return false;
  const views = data.view_results.map((entry) => entry?.view);
  if (!views.every((view) => Number.isInteger(view) && view >= 1 && view <= fileCount)) return false;
  if (new Set(views).size !== fileCount || !views.every((view, index) => view === index + 1)) return false;
  return data.view_results.every((entry) =>
    entry && typeof entry.variety?.class === "string" && Array.isArray(entry.defects) &&
    entry.defects.every((defect) => defect && typeof defect.class === "string")
  );
}

function showMismatchDialog(detail, results) {
  mismatchResults = results;
  mismatchIndex = 0;
  showError(detail);
  document.getElementById("mismatch-message").textContent = detail;
  buildMismatchThumbnails();
  renderMismatchView();
  if (!mismatchDialog.open) mismatchDialog.showModal();
  mismatchDialog.focus();
}

function renderMismatchView() {
  const result = mismatchResults[mismatchIndex];
  if (!result) return;
  updateMismatchThumbnailState();
  const image = document.getElementById("mismatch-image");
  image.src = previewUrls[result.view - 1];
  image.alt = `Uploaded image for view ${result.view}`;
  document.getElementById("mismatch-view").textContent = `View ${result.view}`;
  document.getElementById("mismatch-variety-name").textContent = varietyDisplayName(result.variety.class);
  document.getElementById("mismatch-variety-class").textContent = `Canonical class: ${formatLabel(result.variety.class)}`;

  const counts = new Map();
  result.defects.forEach((defect) => {
    if (defect.class === "corn_ear") return;
    counts.set(defect.class, (counts.get(defect.class) || 0) + 1);
  });
  const defectList = document.getElementById("mismatch-defect-list");
  defectList.replaceChildren();
  if (!counts.size) {
    const empty = document.createElement("li");
    empty.className = "mismatch-no-defects";
    empty.textContent = "No visible defects detected.";
    defectList.appendChild(empty);
  } else {
    counts.forEach((count, className) => {
      const item = document.createElement("li");
      item.textContent = `${formatLabel(className)} ×${count}`;
      defectList.appendChild(item);
    });
  }
  document.getElementById("mismatch-counter").textContent = `${mismatchIndex + 1} of ${mismatchResults.length}`;
  mismatchPrevious.disabled = mismatchIndex === 0;
  mismatchNext.disabled = mismatchIndex === mismatchResults.length - 1;
}

function buildMismatchThumbnails() {
  mismatchThumbnails.replaceChildren();
  previewUrls.forEach((url, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "mismatch-thumbnail";
    const selected = index === mismatchIndex;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-label", `Select view ${index + 1}`);
    button.setAttribute("aria-pressed", String(selected));
    button.title = `View ${index + 1}`;
    const image = document.createElement("img");
    image.src = url;
    image.alt = "";
    button.appendChild(image);
    button.addEventListener("click", () => {
      mismatchIndex = index;
      renderMismatchView();
    });
    mismatchThumbnails.appendChild(button);
  });
}

function updateMismatchThumbnailState() {
  [...mismatchThumbnails.children].forEach((button, index) => {
    const selected = index === mismatchIndex;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function closeMismatchDialog() {
  if (mismatchDialog.open) mismatchDialog.close();
}

mismatchClose.addEventListener("click", closeMismatchDialog);
mismatchPrevious.addEventListener("click", () => {
  if (mismatchIndex > 0) {
    mismatchIndex -= 1;
    renderMismatchView();
  }
});
mismatchNext.addEventListener("click", () => {
  if (mismatchIndex < mismatchResults.length - 1) {
    mismatchIndex += 1;
    renderMismatchView();
  }
});
mismatchDialog.addEventListener("close", () => inspectBtn.focus());

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function pct(v) {
  return `${Math.round((v ?? 0) * 100)}%`;
}

function varietyDisplayName(className) {
  const names = {
    yellow_sweet_corn: "Sweet Fortune",
    white_corn: "Sweet Pearl",
  };
  if (typeof className !== "string") return "Unknown variety";
  return Object.prototype.hasOwnProperty.call(names, className)
    ? names[className]
    : className.replace(/_/g, " ");
}

function render(report) {
  resultEl.classList.remove("hidden");
  errorEl.classList.add("hidden");

  document.getElementById("variety-label").textContent = varietyDisplayName(report.variety.class);
  const inspectionDate = new Date(report.inspection_date);
  let formattedInspectionDate = report.inspection_date;
  if (!Number.isNaN(inspectionDate.getTime())) {
    formattedInspectionDate = new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(inspectionDate);
  }
  document.getElementById("inspection-date").textContent =
    `Inspected ${formattedInspectionDate} · ${report.view_count} views`;
  const annotatedImage = document.getElementById("annotated");
  annotatedImage.src = report.image_url;
  renderAnnotationHotspots(report);

  const warning = document.getElementById("warning");
  warning.classList.add("hidden");
  if (report.grade.needs_reinspection) {
    warning.textContent = report.warnings?.length
      ? report.warnings.join(" ")
      : "Low prediction confidence detected — consider re-inspecting with clearer photos.";
    warning.classList.remove("hidden");
  }

  const gradeValue = document.getElementById("grade-value");
  gradeValue.textContent = report.grade.grade_label;
  gradeValue.classList.toggle("below", report.grade.grade === "Reject");

  document.getElementById("coverage-value").textContent = pct(report.defect_coverage);
  document.getElementById("coverage-bar").style.width = pct(report.defect_coverage);
  document.getElementById("coverage-bar").style.background =
    report.defect_coverage >= 0.1 ? "var(--danger)" : "var(--gold)";

  const reasons = document.getElementById("grade-reasons");
  reasons.innerHTML = "";
  report.grade.grade_reasons.forEach((r) => {
    const li = document.createElement("li");
    li.textContent = r;
    reasons.appendChild(li);
  });

  const defectsBox = document.getElementById("defects");
  defectsBox.innerHTML = "";
  if (!report.defects.length) {
    const chip = document.createElement("span");
    chip.className = "chip none";
    chip.textContent = "No visible defects detected";
    defectsBox.appendChild(chip);
  } else {
    report.defects.forEach((d) => {
      const chip = document.createElement("span");
      chip.className = "chip defective";
      chip.innerHTML = `${d.class.replace(/_/g, " ")} <span class="conf-val">×${d.merged_count}</span>`;
      defectsBox.appendChild(chip);
    });
  }

  document.getElementById("ear-size").textContent = report.traits.ear_size;
  document.getElementById("completeness").textContent = pct(report.traits.kernel_completeness);
  document.getElementById("completeness-bar").style.width = pct(report.traits.kernel_completeness);

  renderRecommendations(report.recommendations);

  const viewsBox = document.getElementById("views");
  viewsBox.innerHTML = "";
  report.views.forEach((v) => {
    const card = document.createElement("figure");
    card.className = "view-card";
    const img = document.createElement("img");
    img.src = v.image_url;
    img.alt = `view ${v.view}`;
    const cap = document.createElement("figcaption");
    const lines = [
      `View ${v.view}`,
      varietyDisplayName(v.variety.class),
      `Kernel completeness: ${pct(v.traits.kernel_completeness)}`,
    ];
    lines.forEach((line, index) => {
      if (index) cap.appendChild(document.createElement("br"));
      cap.appendChild(document.createTextNode(line));
    });
    card.append(img, cap);
    viewsBox.appendChild(card);
  });

  const chip = document.getElementById("mode-chip");
  const modes = report.mode || {};
  chip.textContent = `classifier: ${modes.classifier} · detector: ${modes.detector}`;
  chip.classList.remove("hidden");

  resultEl.scrollIntoView({ behavior: "smooth" });
}

function formatLabel(value) {
  if (typeof value !== "string" || !value) return "Not provided";
  return value.replace(/_/g, " ");
}

function renderRecommendations(recommendations) {
  const card = document.getElementById("recommendations-card");
  const summary = document.getElementById("recommendations-summary");
  const list = document.getElementById("recommendations-list");
  card.classList.add("hidden");
  summary.textContent = "";
  list.replaceChildren();
  if (!recommendations || typeof recommendations !== "object" || !Array.isArray(recommendations.options)) return;

  recommendations.options.forEach((option) => {
    if (!option || typeof option !== "object" || option.status === "unavailable" ||
        typeof option.label !== "string" || !option.label) return;
    const item = document.createElement("li");
    item.className = "recommendation-item";

    const label = document.createElement("strong");
    label.className = "recommendation-label";
    label.textContent = option.label;
    item.appendChild(label);

    const price = option.price_estimate;
    if (price && typeof price === "object" &&
        typeof price.currency === "string" && typeof price.unit === "string" &&
        typeof price.low === "number" && Number.isFinite(price.low) &&
        typeof price.high === "number" && Number.isFinite(price.high) &&
        price.high >= price.low && typeof price.source === "string" && price.source.trim()) {
      const priceEl = document.createElement("p");
      priceEl.className = "recommendation-price";
      priceEl.textContent = `${price.currency} ${price.low}–${price.high} / ${price.unit} · source: ${price.source}`;
      item.appendChild(priceEl);
    }
    list.appendChild(item);
  });

  if (list.children.length) card.classList.remove("hidden");
}

function renderAnnotationHotspots(report) {
  const overlay = document.getElementById("annotation-hotspots");
  overlay.replaceChildren();
  const views = report.views || [];
  if (!views.length) return;

  const columns = views.length >= 2 ? 2 : 1;
  const cellWidth = Math.max(...views.map((view) => view.image_width || 0));
  const cellHeight = Math.max(...views.map((view) => view.image_height || 0));
  if (!cellWidth || !cellHeight) return;
  const gridWidth = cellWidth * columns;
  const gridHeight = cellHeight * Math.ceil(views.length / columns);

  views.forEach((view, index) => {
    const offsetX = (index % columns) * cellWidth;
    const offsetY = Math.floor(index / columns) * cellHeight;
    (view.defects || []).forEach((detection) => {
      const [x0, y0, x1, y1] = detection.box;
      if (x1 <= x0 || y1 <= y0) return;
      const hotspot = document.createElement("button");
      hotspot.type = "button";
      hotspot.className = "annotation-hotspot";
      hotspot.classList.add(detection.class === "corn_ear" ? "ear-hotspot" : "defect-hotspot");
      const name = detection.class.replace(/_/g, " ");
      const label = `${name} · ${pct(detection.confidence)} · view ${view.view}`;
      hotspot.setAttribute("aria-label", label);
      hotspot.style.left = `${((offsetX + x0) / gridWidth) * 100}%`;
      hotspot.style.top = `${((offsetY + y0) / gridHeight) * 100}%`;
      hotspot.style.width = `${((x1 - x0) / gridWidth) * 100}%`;
      hotspot.style.height = `${((y1 - y0) / gridHeight) * 100}%`;
      const tooltip = document.createElement("span");
      tooltip.className = "annotation-tooltip";
      tooltip.textContent = label;
      if (offsetX + x0 > gridWidth * 0.65) hotspot.classList.add("tooltip-right");
      if (offsetY + y0 > gridHeight - 60) hotspot.classList.add("tooltip-above");
      hotspot.appendChild(tooltip);
      overlay.appendChild(hotspot);
    });
  });
}
