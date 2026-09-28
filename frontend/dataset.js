const earIdInput = document.getElementById("ear-id");
const earOptions = document.getElementById("ear-options");
const nextEarBtn = document.getElementById("next-ear");
const varietySel = document.getElementById("variety");
const earForm = document.getElementById("ear-form");
const importBtn = document.getElementById("import-btn");
const importStatus = document.getElementById("import-status");
const errorEl = document.getElementById("error");
const statsEl = document.getElementById("stats");
const problemsWrap = document.getElementById("problems");
const problemsTitle = document.getElementById("problems-title");
const problemsIcon = document.getElementById("problems-icon");
const problemList = document.getElementById("problem-list");
const validateBtn = document.getElementById("validate-btn");
const earsBody = document.querySelector("#ears-table tbody");
const gtBody = document.querySelector("#gt-table tbody");
const earsCountEl = document.getElementById("ears-count");
const VIEW_ANGLES = ["0°", "90°", "180°", "270°"];
const bulkForm = document.getElementById("bulk-form");
const bulkVariety = document.getElementById("bulk-variety");
const bulkImages = document.getElementById("bulk-images");
const bulkSummary = document.getElementById("bulk-summary");
const bulkImportBtn = document.getElementById("bulk-import-btn");
const bulkStatus = document.getElementById("bulk-status");

const PV_CLASSES = ["corn_ear", "mold", "insect_damage", "discoloration", "deformity", "missing_kernels", "ear_decay"];
const LABEL_EDGE_EPSILON = 1e-4;
const PV_COLORS = {
  corn_ear: "#4caf50",
  mold: "#9c27b0",
  insect_damage: "#f44336",
  discoloration: "#ff9800",
  deformity: "#2196f3",
  missing_kernels: "#00bcd4",
  ear_decay: "#795548",
};

const pvModal = document.getElementById("preview-modal");
const pvCanvas = document.getElementById("pv-canvas");
const pvTitle = document.getElementById("pv-title");
const pvNote = document.getElementById("pv-note");
const pvLegend = document.getElementById("pv-legend");
const pvPrev = document.getElementById("pv-prev");
const pvNext = document.getElementById("pv-next");
const pvCounter = document.getElementById("pv-counter");
const pvState = { ear: null, view: 1, views: [] };

const annotatorModal = document.getElementById("annotator-modal");
const annotatorCanvas = document.getElementById("annotator-canvas");
const annotatorTitle = document.getElementById("annotator-title");
const annotatorClasses = document.getElementById("annotator-classes");
const annotatorViews = document.getElementById("annotator-views");
const annotatorStage = document.querySelector(".annotator-stage");
const annotatorZoomOut = document.getElementById("annotator-zoom-out");
const annotatorZoomFit = document.getElementById("annotator-zoom-fit");
const annotatorZoomIn = document.getElementById("annotator-zoom-in");
const annotatorFullscreen = document.getElementById("annotator-fullscreen");
const annotatorLoading = document.getElementById("annotator-loading");
const annotatorNote = document.getElementById("annotator-note");
const annotatorSave = document.getElementById("annotator-save");
const annotatorClose = document.getElementById("annotator-close");
const annotatorDelete = document.getElementById("annotator-delete");
const annotatorMode = document.getElementById("annotator-mode");
const annotatorBoxCount = document.getElementById("annotator-box-count");
const annotatorBoxList = document.getElementById("annotator-box-list");
const annotatorState = {
  ear: null,
  view: 1,
  views: [],
  image: null,
  boxes: [],
  selected: -1,
  classId: 0,
  dirty: false,
  drag: null,
  invalidLines: [],
  saving: false,
  zoom: 1,
  fitWidth: 0,
  fitHeight: 0,
};

let earsData = [];
let gtRowsData = [];

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function clearError() {
  errorEl.classList.add("hidden");
}

async function readBody(res) {
  const text = await res.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch {
    data = null;
  }
  return { data, text };
}

async function expectOk(res) {
  const { data, text } = await readBody(res);
  if (!res.ok) {
    const detail = (data && data.detail) || text || `HTTP ${res.status}`;
    throw new Error(detail);
  }
  return data;
}

async function loadAll() {
  clearError();
  const [stats, ears, gt] = await Promise.all([
    expectOk(await fetch("/dataset/stats")),
    expectOk(await fetch("/dataset/ears")),
    expectOk(await fetch("/dataset/ground-truth")),
  ]);
  earsData = ears;
  renderStats(stats);
  renderEars(ears, gt.rows);
  renderGroundTruth(gt.rows, ears);
  populateEarOptions(ears);
}

function renderStats(stats) {
  const incomplete = stats.incomplete_ears.length;
  const complete = stats.ears - incomplete;
  const completePct = stats.ears ? Math.round((complete / stats.ears) * 100) : 0;
  const labelPct = stats.detector_images ? Math.round((stats.detector_labels / stats.detector_images) * 100) : 0;

  const varieties = Object.entries(stats.classifier_images)
    .map(
      ([k, v]) => `
        <span class="mini-stat">
          <span class="mini-dot"></span>
          <span class="mini-label">${k.replace(/_/g, " ")}</span>
          <span class="mini-value">${v}</span>
        </span>`
    )
    .join("");

  const classes = Object.entries(stats.boxes_per_class)
    .map(([k, v]) => `<span class="class-chip">${k.replace(/_/g, " ")}<b>${v}</b></span>`)
    .join("");

  statsEl.innerHTML = `
    <div class="stats-grid">
      <article class="stat-card lead">
        <span class="stat-card-label">Ears in dataset</span>
        <span class="stat-card-value">${stats.ears}</span>
        <span class="stat-card-note ${incomplete ? "bad" : "good"}">${incomplete ? `${incomplete} incomplete<em> · ${stats.missing_labels.length} label missing</em>` : "all complete"}</span>
      </article>
      <article class="stat-card">
        <span class="stat-card-label">Detector images</span>
        <span class="stat-card-value">${stats.detector_images}</span>
        <span class="stat-card-note">${stats.missing_labels.length} missing labels</span>
      </article>
      <article class="stat-card">
        <span class="stat-card-label">Labels</span>
        <span class="stat-card-value">${stats.detector_labels}</span>
        <span class="stat-card-note">${labelPct}% labeled</span>
        <div class="bar"><div class="bar-fill" style="width: ${labelPct}%"></div></div>
      </article>
      <article class="stat-card">
        <span class="stat-card-label">Complete ears</span>
        <span class="stat-card-value">${complete}<em> / ${stats.ears}</em></span>
        <div class="bar"><div class="bar-fill" style="width: ${completePct}%"></div></div>
      </article>
    </div>
    <div class="stats-sub">
      <div class="sub-block">
        <h4>Classifier images by variety</h4>
        <div class="mini-stats">${varieties || '<span class="empty">nothing imported yet</span>'}</div>
      </div>
      <div class="sub-block">
        <h4>Detector boxes by class</h4>
        <div class="class-chips">${classes || '<span class="empty">no labels yet</span>'}</div>
      </div>
    </div>
  `;
}

function renderEars(ears, gtRows) {
  const gtIds = new Set(gtRows.map((r) => r.ear_id));
  earsCountEl.textContent = ears.length;
  if (!ears.length) {
    earsBody.innerHTML = `<tr><td colspan="6" class="empty-cell">No ears imported yet. Use the form above to add the first one.</td></tr>`;
    return;
  }
  earsBody.innerHTML = ears
    .map((e) => {
      const dots = [1, 2, 3, 4]
        .map((n) => `<span class="view-dot ${e.views.includes(n) ? "on" : ""}" title="view ${n}"></span>`)
        .join("");
      const labels =
        e.labels.length === e.views.length
          ? '<span class="badge ok">labeled</span>'
          : e.labels.length
            ? `<span class="badge warn">${e.labels.length}/${e.views.length} views</span>`
            : '<span class="badge danger">no labels</span>';
      const gt = gtIds.has(e.ear_id) ? '<span class="badge gt">recorded</span>' : '<span class="badge none">—</span>';
      const annotate = [1, 2, 3, 4]
        .map((n) => annotateCell(e, n))
        .join("");
      return `
        <tr>
          <td class="id-cell">${e.ear_id}</td>
          <td><span class="variety-tag">${e.variety.replace(/_/g, " ")}</span></td>
          <td><span class="view-dots">${dots}</span></td>
          <td>${labels}</td>
          <td>${gt}</td>
          <td class="col-action">
            <div class="row-actions">
              <button type="button" class="primary small canvas-open" data-ear="${e.ear_id}" data-view="${e.views[0] || 1}">Draw boxes</button>
              <button type="button" class="ghost small toggle-annotate" data-ear="${e.ear_id}">Annotate</button>
              <button type="button" class="ghost small danger ear-delete" data-ear="${e.ear_id}">Delete</button>
            </div>
          </td>
        </tr>
        <tr class="annotate-row hidden" data-ear="${e.ear_id}">
          <td colspan="6">
            <div class="annotate-grid">${annotate}</div>
            <div class="annotate-foot">
              <button type="button" class="ghost small annotate-save-all" data-ear="${e.ear_id}" disabled>Save labels</button>
              <span class="annotate-status"></span>
            </div>
          </td>
        </tr>`;
    })
    .join("");
  wireAnnotate();
}

function annotateCell(e, n) {
  const hasImage = e.views.includes(n);
  const hasLabel = e.labels.includes(n);
  const angle = `v${n} · ${VIEW_ANGLES[n - 1]}`;
  let body;
  if (!hasImage) {
    body = '<span class="badge none">no image</span>';
  } else if (hasLabel) {
    body = `
      <span class="badge ok">labeled</span>
      <span class="annotate-attach">
        <label class="annotation-pick"><input type="file" accept=".txt" class="annotate-file"><span>replace .txt</span></label>
        <span class="annotate-chosen"></span>
      </span>`;
  } else {
    body = `
      <span class="annotate-attach">
        <label class="annotation-pick"><input type="file" accept=".txt" class="annotate-file"><span>choose .txt</span></label>
        <span class="annotate-chosen"></span>
      </span>`;
  }
  const preview = hasImage
    ? `<button type="button" class="ghost small pv-open" data-ear="${e.ear_id}" data-view="${n}">Preview</button>`
    : "";
  const draw = hasImage
    ? `<button type="button" class="ghost small canvas-open" data-ear="${e.ear_id}" data-view="${n}">Draw</button>`
    : "";
  return `
      <div class="annotate-view" data-ear="${e.ear_id}" data-view="${n}">
        <span class="annotate-angle">${angle}</span>
        ${body}
        <div class="annotate-view-actions">${draw}${preview}</div>
      </div>`;
}

function wireAnnotate() {
  earsBody.querySelectorAll(".toggle-annotate").forEach((btn) => {
    btn.addEventListener("click", () => {
      const row = earsBody.querySelector(`tr.annotate-row[data-ear="${btn.dataset.ear}"]`);
      row.classList.toggle("hidden");
      btn.textContent = row.classList.contains("hidden") ? "Annotate" : "Close";
    });
  });
  earsBody.querySelectorAll(".annotate-file").forEach((input) => {
    input.addEventListener("change", () => {
      const row = input.closest(".annotate-row");
      const btn = row.querySelector(".annotate-save-all");
      const cell = input.closest(".annotate-view");
      const chosen = cell.querySelector(".annotate-chosen");
      chosen.textContent = input.files[0] ? input.files[0].name : "";
      updateAttachAll(row, btn);
    });
  });
  earsBody.querySelectorAll(".annotate-save-all").forEach((btn) => {
    btn.addEventListener("click", () => attachAllLabels(btn));
  });
  earsBody.querySelectorAll(".ear-delete").forEach((btn) => {
    btn.addEventListener("click", () => deleteEar(btn));
  });
  earsBody.querySelectorAll(".pv-open").forEach((btn) => {
    btn.addEventListener("click", () => openPreview(btn.dataset.ear, Number(btn.dataset.view)));
  });
  earsBody.querySelectorAll(".canvas-open").forEach((btn) => {
    btn.addEventListener("click", () => openAnnotator(btn.dataset.ear, Number(btn.dataset.view)));
  });
}

function updateAttachAll(row, btn) {
  const chosen = [...row.querySelectorAll(".annotate-view .annotate-file")].filter((i) => i.files.length);
  btn.disabled = chosen.length === 0;
  const status = row.querySelector(".annotate-status");
  status.textContent = chosen.length ? `${chosen.length} label${chosen.length === 1 ? "" : "s"} selected` : "";
  status.classList.remove("bad");
}

async function attachAllLabels(btn) {
  const row = btn.closest(".annotate-row");
  const earId = btn.dataset.ear;
  const cells = [...row.querySelectorAll(".annotate-view")].filter((c) => {
    const input = c.querySelector(".annotate-file");
    return input && input.files.length;
  });
  if (!cells.length) return;
  clearError();
  btn.disabled = true;
  const status = row.querySelector(".annotate-status");
  const form = new FormData();
  for (const cell of cells) {
    form.append("views", cell.dataset.view);
    form.append("labels", cell.querySelector(".annotate-file").files[0]);
  }
  status.textContent = "Saving…";
  try {
    const res = await fetch(`/dataset/ears/${encodeURIComponent(earId)}/labels/bulk`, {
      method: "PUT",
      body: form,
    });
    await expectOk(res);
    loadAll();
  } catch (err) {
    showError(err.message);
    status.textContent = `Attach failed: ${err.message}`;
    status.classList.add("bad");
    btn.disabled = false;
  }
}

async function deleteEar(btn) {
  const earId = btn.dataset.ear;
  if (!window.confirm(`Delete ${earId} with all its images, annotations, and ground truth? This cannot be undone.`)) return;
  clearError();
  btn.disabled = true;
  try {
    const res = await fetch(`/dataset/ears/${encodeURIComponent(earId)}`, { method: "DELETE" });
    await expectOk(res);
    loadAll();
  } catch (err) {
    showError(err.message);
    btn.disabled = false;
  }
}

function earById(id) {
  return earsData.find((e) => e.ear_id === id);
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function annotationLabel(classId) {
  return PV_CLASSES[classId].replace(/_/g, " ");
}

function parseYoloLabels(text) {
  const boxes = [];
  const invalidLines = [];
  text.split(/\r?\n/).forEach((line, index) => {
    if (!line.trim()) return;
    const parts = line.trim().split(/\s+/).map(Number);
    const [classId, x, y, width, height] = parts;
    const valid = parts.length === 5 && parts.every(Number.isFinite) && Number.isInteger(classId) &&
      classId >= 0 && classId < PV_CLASSES.length && width > 0 && height > 0 &&
      x >= 0 && x <= 1 && y >= 0 && y <= 1 && width <= 1 && height <= 1 &&
      x - width / 2 >= -LABEL_EDGE_EPSILON && x + width / 2 <= 1 + LABEL_EDGE_EPSILON &&
      y - height / 2 >= -LABEL_EDGE_EPSILON && y + height / 2 <= 1 + LABEL_EDGE_EPSILON;
    if (!valid) {
      invalidLines.push(index + 1);
      return;
    }
    boxes.push({ classId, x, y, width, height });
  });
  return { boxes, invalidLines };
}

function annotationPoint(event) {
  const rect = annotatorCanvas.getBoundingClientRect();
  return {
    x: clamp((event.clientX - rect.left) * (annotatorCanvas.width / rect.width), 0, annotatorCanvas.width),
    y: clamp((event.clientY - rect.top) * (annotatorCanvas.height / rect.height), 0, annotatorCanvas.height),
  };
}

function annotationBoxRect(box) {
  return {
    x: (box.x - box.width / 2) * annotatorCanvas.width,
    y: (box.y - box.height / 2) * annotatorCanvas.height,
    width: box.width * annotatorCanvas.width,
    height: box.height * annotatorCanvas.height,
  };
}

function annotationHitTest(point) {
  for (let i = annotatorState.boxes.length - 1; i >= 0; i -= 1) {
    const rect = annotationBoxRect(annotatorState.boxes[i]);
    if (point.x >= rect.x && point.x <= rect.x + rect.width && point.y >= rect.y && point.y <= rect.y + rect.height) {
      return i;
    }
  }
  return -1;
}

function updateAnnotationCursor(event) {
  if (!annotatorState.image || annotatorState.saving) {
    annotatorCanvas.style.cursor = "default";
    return;
  }
  const point = annotationPoint(event);
  const hit = annotationHitTest(point);
  if (hit < 0) {
    annotatorCanvas.style.cursor = "crosshair";
    return;
  }
  const handle = annotationHandleAt(point, annotationBoxRect(annotatorState.boxes[hit]));
  const cursors = { nw: "nwse-resize", n: "ns-resize", ne: "nesw-resize", e: "ew-resize", se: "nwse-resize", s: "ns-resize", sw: "nesw-resize", w: "ew-resize" };
  annotatorCanvas.style.cursor = handle ? cursors[handle] : "move";
}

function annotationHandleAt(point, rect) {
  const handles = {
    nw: [rect.x, rect.y],
    n: [rect.x + rect.width / 2, rect.y],
    ne: [rect.x + rect.width, rect.y],
    e: [rect.x + rect.width, rect.y + rect.height / 2],
    se: [rect.x + rect.width, rect.y + rect.height],
    s: [rect.x + rect.width / 2, rect.y + rect.height],
    sw: [rect.x, rect.y + rect.height],
    w: [rect.x, rect.y + rect.height / 2],
  };
  return Object.entries(handles).find(([, [x, y]]) => Math.hypot(point.x - x, point.y - y) < 14)?.[0] || null;
}

function setAnnotatorNote(message, bad = false) {
  annotatorNote.textContent = message;
  annotatorNote.classList.toggle("bad", bad);
}

function chooseAnnotatorClass(classId, applyToSelected = false) {
  annotatorState.classId = classId;
  if (applyToSelected && annotatorState.selected >= 0) {
    annotatorState.boxes[annotatorState.selected].classId = classId;
    annotatorState.dirty = true;
  }
  renderAnnotatorClasses();
  renderAnnotatorCanvas();
  renderAnnotatorBoxList();
  setAnnotatorNote(`Class ${classId} · ${annotationLabel(classId)} selected for the next box.`);
}

function renderAnnotatorClasses() {
  annotatorClasses.innerHTML = PV_CLASSES
    .map((name, classId) => `
      <button type="button" class="annotator-class ${classId === annotatorState.classId ? "selected" : ""}" data-class-id="${classId}" ${annotatorState.saving ? "disabled" : ""}>
        <span class="annotator-swatch" style="--box-color: ${PV_COLORS[name]}"></span>
        <span>${name.replace(/_/g, " ")}</span>
        <span class="annotator-class-id">${classId}</span>
      </button>`)
    .join("");
  annotatorClasses.querySelectorAll(".annotator-class").forEach((button) => {
    button.addEventListener("click", () => {
      chooseAnnotatorClass(Number(button.dataset.classId), true);
    });
  });
}

function renderAnnotatorViews() {
  annotatorViews.innerHTML = [1, 2, 3, 4]
    .map((view) => {
      const missing = !annotatorState.views.includes(view);
      return `
      <button type="button" class="annotator-view-tab ${view === annotatorState.view ? "selected" : ""} ${missing ? "missing" : ""}" data-view="${view}" ${missing || annotatorState.saving ? "disabled" : ""}>
        <span>v${view}</span><small>${VIEW_ANGLES[view - 1]}</small>
      </button>`;
    })
    .join("");
  annotatorViews.querySelectorAll(".annotator-view-tab").forEach((button) => {
    button.addEventListener("click", () => switchAnnotatorView(Number(button.dataset.view)));
  });
}

function renderAnnotatorBoxList() {
  annotatorBoxCount.textContent = annotatorState.boxes.length;
  annotatorDelete.disabled = annotatorState.selected < 0 || annotatorState.saving;
  annotatorMode.textContent = annotatorState.selected >= 0 ? "Edit selected" : "Draw";
  if (!annotatorState.boxes.length) {
    annotatorBoxList.innerHTML = '<span class="empty">No boxes yet.</span>';
    return;
  }
  annotatorBoxList.innerHTML = annotatorState.boxes
    .map((box, index) => {
      const name = annotationLabel(box.classId);
      const selected = index === annotatorState.selected ? " selected" : "";
      return `<button type="button" class="annotator-box-row${selected}" data-box-index="${index}" ${annotatorState.saving ? "disabled" : ""}>
        <span class="annotator-swatch" style="--box-color: ${PV_COLORS[PV_CLASSES[box.classId]]}"></span>
        <span>${name}</span><small>#${index + 1}</small>
      </button>`;
    })
    .join("");
  annotatorBoxList.querySelectorAll(".annotator-box-row").forEach((button) => {
    button.addEventListener("click", () => {
      annotatorState.selected = Number(button.dataset.boxIndex);
      renderAnnotatorCanvas();
      renderAnnotatorBoxList();
    });
  });
}

function setAnnotatorZoom(value) {
  if (!annotatorState.fitWidth || !annotatorState.fitHeight) return;
  annotatorState.zoom = clamp(value, 0.5, 3);
  annotatorCanvas.width = Math.round(annotatorState.fitWidth * annotatorState.zoom);
  annotatorCanvas.height = Math.round(annotatorState.fitHeight * annotatorState.zoom);
  annotatorZoomFit.textContent = `${Math.round(annotatorState.zoom * 100)}%`;
  annotatorZoomOut.disabled = annotatorState.zoom <= 0.5;
  annotatorZoomIn.disabled = annotatorState.zoom >= 3;
  renderAnnotatorCanvas();
}

function fitAnnotatorImage() {
  if (!annotatorState.image) return;
  const maxW = Math.min(1200, Math.max(360, annotatorStage.clientWidth - 20));
  const maxH = Math.min(700, Math.max(260, window.innerHeight * 0.58));
  const scale = Math.min(1, maxW / annotatorState.image.naturalWidth, maxH / annotatorState.image.naturalHeight);
  annotatorState.fitWidth = Math.max(1, Math.round(annotatorState.image.naturalWidth * scale));
  annotatorState.fitHeight = Math.max(1, Math.round(annotatorState.image.naturalHeight * scale));
  setAnnotatorZoom(1);
}

async function toggleAnnotatorFullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else if (annotatorStage.requestFullscreen) await annotatorStage.requestFullscreen();
    else setAnnotatorNote("Fullscreen is not supported by this browser.", true);
  } catch (err) {
    setAnnotatorNote(`Could not enter fullscreen: ${err.message}`, true);
  }
}

function syncAnnotatorFullscreen() {
  annotatorFullscreen.textContent = document.fullscreenElement ? "Exit full screen" : "Full screen";
  if (annotatorState.image) fitAnnotatorImage();
}

function renderAnnotatorCanvas() {
  if (!annotatorState.image) return;
  const ctx = annotatorCanvas.getContext("2d");
  ctx.clearRect(0, 0, annotatorCanvas.width, annotatorCanvas.height);
  ctx.drawImage(annotatorState.image, 0, 0, annotatorCanvas.width, annotatorCanvas.height);
  annotatorState.boxes.forEach((box, index) => {
    const rect = annotationBoxRect(box);
    const color = PV_COLORS[PV_CLASSES[box.classId]] || "#607d8b";
    const selected = index === annotatorState.selected;
    ctx.strokeStyle = color;
    ctx.lineWidth = selected ? 3 : 2;
    ctx.setLineDash(selected ? [] : [6, 4]);
    ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
    ctx.setLineDash([]);
    if (selected) {
      const label = `${annotationLabel(box.classId)} · ${index + 1}`;
      const fontSize = Math.max(12, Math.round(annotatorCanvas.width / 62));
      ctx.font = `700 ${fontSize}px ui-sans-serif, system-ui, sans-serif`;
      const labelHeight = fontSize + 8;
      const labelWidth = ctx.measureText(label).width + 12;
      ctx.fillStyle = color;
      ctx.fillRect(rect.x, Math.max(0, rect.y - labelHeight), labelWidth, labelHeight);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, rect.x + 6, Math.max(fontSize, rect.y - 6));
      const handles = [
        [rect.x, rect.y], [rect.x + rect.width / 2, rect.y], [rect.x + rect.width, rect.y],
        [rect.x + rect.width, rect.y + rect.height / 2], [rect.x + rect.width, rect.y + rect.height],
        [rect.x + rect.width / 2, rect.y + rect.height], [rect.x, rect.y + rect.height],
        [rect.x, rect.y + rect.height / 2],
      ];
      handles.forEach(([x, y]) => {
        ctx.fillStyle = "#fff";
        ctx.fillRect(x - 5, y - 5, 10, 10);
        ctx.strokeStyle = color;
        ctx.strokeRect(x - 5, y - 5, 10, 10);
      });
    }
  });
}

function renderAnnotator() {
  renderAnnotatorViews();
  renderAnnotatorClasses();
  renderAnnotatorBoxList();
  renderAnnotatorCanvas();
  annotatorSave.disabled = !annotatorState.image;
}

async function loadAnnotatorView(view) {
  const requestId = (annotatorState.requestId || 0) + 1;
  annotatorState.requestId = requestId;
  annotatorState.view = view;
  annotatorState.image = null;
  annotatorState.boxes = [];
  annotatorState.selected = -1;
  annotatorState.dirty = false;
  annotatorState.invalidLines = [];
  annotatorLoading.classList.remove("hidden");
  annotatorLoading.textContent = "Loading image…";
  annotatorCanvas.classList.add("hidden");
  annotatorSave.disabled = true;
  setAnnotatorNote("Loading view…");
  renderAnnotatorViews();
  renderAnnotatorBoxList();

  try {
    const imageRes = await fetch(`/dataset/ears/${encodeURIComponent(annotatorState.ear)}/views/${view}/image`, { cache: "no-store" });
    if (!imageRes.ok) throw new Error(`Could not load image (HTTP ${imageRes.status})`);
    const imageUrl = URL.createObjectURL(await imageRes.blob());
    const image = new Image();
    image.src = imageUrl;
    await image.decode();
    URL.revokeObjectURL(imageUrl);

    const labelRes = await fetch(`/dataset/ears/${encodeURIComponent(annotatorState.ear)}/views/${view}/label`, { cache: "no-store" });
    let labelText = "";
    if (labelRes.ok) labelText = await labelRes.text();
    else if (labelRes.status !== 404) throw new Error(`Could not load annotation (HTTP ${labelRes.status})`);

    if (requestId !== annotatorState.requestId) return;
    annotatorState.image = image;
    const parsed = parseYoloLabels(labelText);
    annotatorState.boxes = parsed.boxes;
    annotatorState.invalidLines = parsed.invalidLines;
    annotatorState.fitWidth = 0;
    annotatorState.fitHeight = 0;
    annotatorCanvas.classList.remove("hidden");
    annotatorLoading.classList.add("hidden");
    annotatorSave.disabled = false;
    if (annotatorState.invalidLines.length) {
      setAnnotatorNote(`Lines ${annotatorState.invalidLines.join(", ")} are invalid and will be replaced if you save.`, true);
    } else {
      setAnnotatorNote(labelText ? `${annotatorState.boxes.length} box${annotatorState.boxes.length === 1 ? "" : "es"} loaded.` : "Draw the corn ear box first.");
    }
    fitAnnotatorImage();
    renderAnnotator();
  } catch (err) {
    annotatorLoading.textContent = "Could not load this view.";
    setAnnotatorNote(err.message, true);
  }
}

async function switchAnnotatorView(view) {
  if (annotatorState.saving) return;
  if (view === annotatorState.view && annotatorState.image) return;
  if (annotatorState.dirty && !window.confirm("Discard unsaved boxes for this view?")) return;
  await loadAnnotatorView(view);
}

function cycleAnnotatorView(direction) {
  if (annotatorState.saving || annotatorState.views.length < 2) return;
  const currentIndex = annotatorState.views.indexOf(annotatorState.view);
  if (currentIndex < 0) return;
  const nextIndex = (currentIndex + direction + annotatorState.views.length) % annotatorState.views.length;
  switchAnnotatorView(annotatorState.views[nextIndex]);
}

async function openAnnotator(earId, view) {
  const ear = earById(earId);
  if (!ear || !ear.views.length) return;
  annotatorState.ear = earId;
  annotatorState.views = ear.views;
  annotatorState.classId = 0;
  annotatorTitle.textContent = `${earId} · draw detector boxes`;
  annotatorModal.showModal();
  renderAnnotatorViews();
  await loadAnnotatorView(ear.views.includes(view) ? view : ear.views[0]);
}

function deleteSelectedAnnotation() {
  if (annotatorState.selected < 0) return;
  annotatorState.boxes.splice(annotatorState.selected, 1);
  annotatorState.selected = -1;
  annotatorState.dirty = true;
  renderAnnotatorCanvas();
  renderAnnotatorBoxList();
}

function startAnnotationPointer(event) {
  if (!annotatorState.image || annotatorState.saving) return;
  event.preventDefault();
  const point = annotationPoint(event);
  const hit = annotationHitTest(point);
  annotatorState.selected = hit;
  if (hit >= 0) {
    const box = annotatorState.boxes[hit];
    const rect = annotationBoxRect(box);
    const handle = annotationHandleAt(point, rect);
    annotatorState.drag = {
      type: handle ? "resize" : "move",
      handle,
      start: point,
      startBox: { ...box },
    };
  } else {
    annotatorState.boxes.push({ classId: annotatorState.classId, x: point.x / annotatorCanvas.width, y: point.y / annotatorCanvas.height, width: 0, height: 0 });
    annotatorState.selected = annotatorState.boxes.length - 1;
    annotatorState.drag = { type: "draw", start: point };
    annotatorState.dirty = true;
  }
  annotatorCanvas.setPointerCapture(event.pointerId);
  renderAnnotatorCanvas();
  renderAnnotatorBoxList();
}

function moveAnnotationPointer(event) {
  const drag = annotatorState.drag;
  if (!drag || annotatorState.selected < 0) {
    updateAnnotationCursor(event);
    return;
  }
  const point = annotationPoint(event);
  const box = annotatorState.boxes[annotatorState.selected];
  annotatorState.dirty = true;
  if (drag.type === "draw") {
    const left = Math.min(drag.start.x, point.x) / annotatorCanvas.width;
    const top = Math.min(drag.start.y, point.y) / annotatorCanvas.height;
    const right = Math.max(drag.start.x, point.x) / annotatorCanvas.width;
    const bottom = Math.max(drag.start.y, point.y) / annotatorCanvas.height;
    box.x = (left + right) / 2;
    box.y = (top + bottom) / 2;
    box.width = right - left;
    box.height = bottom - top;
  } else if (drag.type === "move") {
    const dx = (point.x - drag.start.x) / annotatorCanvas.width;
    const dy = (point.y - drag.start.y) / annotatorCanvas.height;
    box.x = clamp(drag.startBox.x + dx, box.width / 2, 1 - box.width / 2);
    box.y = clamp(drag.startBox.y + dy, box.height / 2, 1 - box.height / 2);
  } else if (drag.type === "resize") {
    const left = drag.startBox.x - drag.startBox.width / 2;
    const top = drag.startBox.y - drag.startBox.height / 2;
    const right = left + drag.startBox.width;
    const bottom = top + drag.startBox.height;
    const nextLeft = drag.handle.includes("w") ? clamp(point.x / annotatorCanvas.width, 0, right - 0.005) : left;
    const nextRight = drag.handle.includes("e") ? clamp(point.x / annotatorCanvas.width, left + 0.005, 1) : right;
    const nextTop = drag.handle.includes("n") ? clamp(point.y / annotatorCanvas.height, 0, bottom - 0.005) : top;
    const nextBottom = drag.handle.includes("s") ? clamp(point.y / annotatorCanvas.height, top + 0.005, 1) : bottom;
    box.x = (nextLeft + nextRight) / 2;
    box.y = (nextTop + nextBottom) / 2;
    box.width = nextRight - nextLeft;
    box.height = nextBottom - nextTop;
  }
  renderAnnotatorCanvas();
}

function finishAnnotationPointer() {
  const drag = annotatorState.drag;
  if (!drag) return;
  const box = annotatorState.boxes[annotatorState.selected];
  if (drag.type === "draw" && (!box || box.width < 0.005 || box.height < 0.005)) {
    annotatorState.boxes.splice(annotatorState.selected, 1);
    annotatorState.selected = -1;
  }
  annotatorState.drag = null;
  annotatorCanvas.style.cursor = "crosshair";
  renderAnnotatorCanvas();
  renderAnnotatorBoxList();
}

async function saveAnnotatorView() {
  if (!annotatorState.image) return;
  if (!annotatorState.boxes.some((box) => box.classId === 0)) {
    setAnnotatorNote("Add one corn ear box before saving.", true);
    annotatorState.classId = 0;
    renderAnnotatorClasses();
    return;
  }
  if (annotatorState.invalidLines.length && !window.confirm("This label has invalid lines. Saving will replace them with the boxes currently shown. Continue?")) return;
  const text = annotatorState.boxes
    .map((box) => `${box.classId} ${box.x.toFixed(6)} ${box.y.toFixed(6)} ${box.width.toFixed(6)} ${box.height.toFixed(6)}`)
    .join("\n") + "\n";
  const form = new FormData();
  form.append("label", new Blob([text], { type: "text/plain" }), `${annotatorState.ear}_v${annotatorState.view}.txt`);
  annotatorState.saving = true;
  annotatorSave.disabled = true;
  annotatorClose.disabled = true;
  renderAnnotatorViews();
  setAnnotatorNote("Saving annotation…");
  try {
    const res = await fetch(`/dataset/ears/${encodeURIComponent(annotatorState.ear)}/labels/${annotatorState.view}`, {
      method: "PUT",
      body: form,
    });
    await expectOk(res);
    annotatorState.dirty = false;
    annotatorState.invalidLines = [];
    setAnnotatorNote("Saved. Continue with the next view when ready.");
    await loadAll();
  } catch (err) {
    setAnnotatorNote(err.message, true);
  } finally {
    annotatorState.saving = false;
    annotatorSave.disabled = false;
    annotatorClose.disabled = false;
    renderAnnotatorViews();
  }
}

annotatorCanvas.addEventListener("pointerdown", startAnnotationPointer);
annotatorCanvas.addEventListener("pointermove", moveAnnotationPointer);
annotatorCanvas.addEventListener("pointerup", finishAnnotationPointer);
annotatorCanvas.addEventListener("pointercancel", finishAnnotationPointer);
annotatorZoomOut.addEventListener("click", () => setAnnotatorZoom(annotatorState.zoom - 0.25));
annotatorZoomIn.addEventListener("click", () => setAnnotatorZoom(annotatorState.zoom + 0.25));
annotatorZoomFit.addEventListener("click", fitAnnotatorImage);
annotatorFullscreen.addEventListener("click", toggleAnnotatorFullscreen);
document.addEventListener("fullscreenchange", syncAnnotatorFullscreen);
annotatorDelete.addEventListener("click", deleteSelectedAnnotation);
annotatorSave.addEventListener("click", saveAnnotatorView);
annotatorClose.addEventListener("click", () => {
  if (annotatorState.saving) return;
  if (!annotatorState.dirty || window.confirm("Discard unsaved boxes for this view?")) annotatorModal.close();
});
annotatorModal.addEventListener("cancel", (event) => {
  if (annotatorState.saving || (annotatorState.dirty && !window.confirm("Discard unsaved boxes for this view?"))) event.preventDefault();
});
window.addEventListener("beforeunload", (event) => {
  if (!annotatorModal.open || !annotatorState.dirty) return;
  // Browsers show their own generic confirmation message for tab/window close.
  event.preventDefault();
  event.returnValue = "";
});
window.addEventListener("keydown", (event) => {
  const target = event.target;
  const editingText = target instanceof HTMLElement && target.closest("input, textarea, select, [contenteditable='true']");
  if (annotatorModal.open && !editingText && event.altKey && !event.ctrlKey && !event.metaKey && /^Digit[1-4]$/.test(event.code)) {
    if (!annotatorState.saving) {
      const view = Number(event.code.slice(-1));
      if (annotatorState.views.includes(view)) switchAnnotatorView(view);
    }
    event.preventDefault();
    return;
  }
  if (annotatorModal.open && !editingText && !event.altKey && !event.ctrlKey && !event.metaKey && (event.key === "[" || event.key === "]")) {
    cycleAnnotatorView(event.key === "[" ? -1 : 1);
    event.preventDefault();
    return;
  }
  if (annotatorModal.open && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    if (!annotatorState.saving) saveAnnotatorView();
    return;
  }
  if (annotatorModal.open && /^[1-7]$/.test(event.key)) {
    if (annotatorState.saving) return;
    chooseAnnotatorClass(Number(event.key) - 1);
    event.preventDefault();
    return;
  }
  if (annotatorModal.open && annotatorState.selected >= 0 && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
    if (annotatorState.saving) return;
    const box = annotatorState.boxes[annotatorState.selected];
    const step = event.shiftKey ? 0.01 : 0.005;
    if (event.key === "ArrowLeft") event.shiftKey ? box.width = clamp(box.width - step, 0.005, box.width) : box.x = clamp(box.x - step, box.width / 2, 1 - box.width / 2);
    if (event.key === "ArrowRight") event.shiftKey ? box.width = clamp(box.width + step, 0.005, 2 * Math.min(box.x, 1 - box.x)) : box.x = clamp(box.x + step, box.width / 2, 1 - box.width / 2);
    if (event.key === "ArrowUp") event.shiftKey ? box.height = clamp(box.height - step, 0.005, box.height) : box.y = clamp(box.y - step, box.height / 2, 1 - box.height / 2);
    if (event.key === "ArrowDown") event.shiftKey ? box.height = clamp(box.height + step, 0.005, 2 * Math.min(box.y, 1 - box.y)) : box.y = clamp(box.y + step, box.height / 2, 1 - box.height / 2);
    annotatorState.dirty = true;
    event.preventDefault();
    renderAnnotatorCanvas();
    return;
  }
  if (annotatorModal.open && (event.key === "Delete" || event.key === "Backspace")) {
    const target = event.target;
    const editingText = target instanceof HTMLElement && target.closest("input, textarea, select, [contenteditable='true']");
    if (!editingText) {
      event.preventDefault();
      deleteSelectedAnnotation();
    }
  }
});

async function openPreview(earId, view) {
  const ear = earById(earId);
  if (!ear) return;
  pvState.ear = earId;
  pvState.view = view;
  pvState.views = ear.views;
  pvModal.showModal();
  await renderPreview();
}

async function renderPreview() {
  const { ear, view, views } = pvState;
  pvTitle.textContent = `${ear} · v${view} · ${VIEW_ANGLES[view - 1]}`;
  pvCounter.textContent = `view ${view} of ${views.length}`;
  pvLegend.innerHTML = "";
  const hasView = views.includes(view);
  pvNote.textContent = hasView ? "Loading image…" : "No image for this view.";
  pvCanvas.classList.toggle("hidden", !hasView);
  const idx = views.indexOf(view);
  pvNext.disabled = idx === -1 || idx >= views.length - 1;
  pvPrev.disabled = idx <= 0;
  if (!hasView) return;

  let img;
  try {
    img = new Image();
    const res = await fetch(`/dataset/ears/${encodeURIComponent(ear)}/views/${view}/image`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    img.src = URL.createObjectURL(await res.blob());
    await img.decode();
  } catch (err) {
    pvNote.textContent = `Could not load image: ${err.message}`;
    return;
  }

  const maxW = 760;
  const maxH = Math.min(540, window.innerHeight * 0.68);
  const scale = Math.min(1, maxW / img.naturalWidth, maxH / img.naturalHeight);
  const w = Math.round(img.naturalWidth * scale);
  const h = Math.round(img.naturalHeight * scale);
  pvCanvas.width = w;
  pvCanvas.height = h;
  const ctx = pvCanvas.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);

  try {
    const res = await fetch(`/dataset/ears/${encodeURIComponent(ear)}/views/${view}/label`, { cache: "no-store" });
    if (res.status === 404) {
      pvNote.textContent = "No annotation attached to this view.";
      return;
    }
    if (!res.ok) {
      const body = await readBody(res);
      throw new Error((body.data && body.data.detail) || body.text || `HTTP ${res.status}`);
    }
    const text = await res.text();
    const counts = {};
    let rendered = 0;
    for (const line of text.split("\n")) {
      const parts = line.trim().split(/\s+/).map(Number);
      if (parts.length < 5 || parts.some((v) => !Number.isFinite(v))) continue;
      const [cls, x, y, bw, bh] = parts;
      if (cls < 0 || cls >= PV_CLASSES.length) continue;
      const name = PV_CLASSES[cls].replace(/_/g, " ");
      const x0 = (x - bw / 2) * w;
      const y0 = (y - bh / 2) * h;
      const bwpx = bw * w;
      const bhpx = bh * h;
      const color = PV_COLORS[PV_CLASSES[cls]] || "#607d8b";
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.16;
      ctx.fillRect(x0, y0, bwpx, bhpx);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(3, w / 220);
      ctx.strokeRect(x0, y0, bwpx, bhpx);
      ctx.font = `${Math.max(11, Math.round(w / 60))}px ui-sans-serif, system-ui, sans-serif`;
      const tw = ctx.measureText(name).width + 8;
      const th = 16;
      ctx.fillStyle = color;
      ctx.fillRect(x0, Math.max(0, y0 - th), tw, th);
      ctx.fillStyle = "#fff";
      ctx.fillText(name, x0 + 4, Math.max(th - 4, y0 - 4));
      counts[name] = (counts[name] || 0) + 1;
      rendered += 1;
    }
    pvNote.textContent = rendered ? `${rendered} annotation box${rendered === 1 ? "" : "es"} shown` : "No drawable annotation boxes found.";
    for (const [name, n] of Object.entries(counts)) {
      const li = document.createElement("li");
      li.className = "legend-item";
      const dot = document.createElement("span");
      dot.className = "legend-dot";
      const key = Object.keys(PV_COLORS).find((k) => k.replace(/_/g, " ") === name);
      dot.style.background = PV_COLORS[key] || "#607d8b";
      li.append(dot, document.createTextNode(`${name} · ${n}`));
      pvLegend.appendChild(li);
    }
  } catch (err) {
    pvNote.textContent = `Could not load annotation: ${err.message}`;
  }
}

function changePreview(delta) {
  const { views } = pvState;
  const idx = views.indexOf(pvState.view);
  const target = views[Math.min(Math.max(idx + delta, 0), views.length - 1)];
  if (target) {
    pvState.view = target;
    renderPreview();
  }
}

pvPrev.addEventListener("click", () => changePreview(-1));
pvNext.addEventListener("click", () => changePreview(1));
pvModal.querySelector(".pv-close").addEventListener("click", () => pvModal.close());
pvModal.addEventListener("click", (e) => {
  if (e.target === pvModal) pvModal.close();
});

function gradeClass(grade) {
  if (grade === "Extra Class") return "grade-extra";
  if (grade === "Class I") return "grade-i";
  if (grade === "Class II") return "grade-ii";
  if (grade) return "grade-reject";
  return "grade-unset";
}

function renderGroundTruth(gtRows, ears) {
  const rows = [...gtRows];
  const known = new Set(gtRows.map((r) => r.ear_id));
  ears.forEach((e) => {
    if (!known.has(e.ear_id)) {
      rows.push({
        ear_id: e.ear_id,
        variety: e.variety,
        coverage: "0",
        mold_count: "0",
        insect_count: "0",
        missing_count: "0",
        other_count: "0",
        grade: "",
      });
    }
  });
  rows.sort((a, b) => a.ear_id.localeCompare(b.ear_id));
  gtRowsData = rows;
  if (!rows.length) {
    gtBody.innerHTML = `<tr><td colspan="9" class="empty-cell">Nothing to grade yet. Import an ear to enable ground-truth entry.</td></tr>`;
    return;
  }
  gtBody.innerHTML = rows
    .map(
      (r) => `
      <tr data-ear="${r.ear_id}">
        <td class="id-cell">${r.ear_id}</td>
        <td><span class="gt-variety">${r.variety}</span></td>
        <td><input class="gt-input gt-coverage" type="number" min="0" max="1" step="0.01" value="${r.coverage}" title="Defect coverage (0–1)"></td>
        <td><input class="gt-input gt-count" type="number" min="0" step="1" value="${r.mold_count}"></td>
        <td><input class="gt-input gt-count" type="number" min="0" step="1" value="${r.insect_count}"></td>
        <td><input class="gt-input gt-count" type="number" min="0" step="1" value="${r.missing_count}"></td>
        <td><input class="gt-input gt-count" type="number" min="0" step="1" value="${r.other_count}"></td>
        <td><span class="grade-chip gt-grade ${gradeClass(r.grade)}">${r.grade || "unset"}</span></td>
        <td><button type="button" class="ghost gt-save">Save</button></td>
      </tr>`
    )
    .join("");
  gtBody.querySelectorAll(".gt-save").forEach((btn) => {
    btn.addEventListener("click", () => saveGroundTruthRow(btn));
  });
  gtBody.querySelectorAll(".gt-count, .gt-coverage").forEach((input) => {
    input.addEventListener("change", () => {
      const row = input.closest("tr");
      const grade = computeGrade(row);
      const gradeEl = row.querySelector(".gt-grade");
      gradeEl.textContent = grade;
      gradeEl.className = `grade-chip gt-grade ${gradeClass(grade)}`;
    });
  });
}

function computeGrade(row) {
  const coverage = parseFloat(row.querySelector(".gt-coverage").value) || 0;
  const mold = parseInt(row.querySelectorAll(".gt-count")[0].value, 10) || 0;
  const insect = parseInt(row.querySelectorAll(".gt-count")[1].value, 10) || 0;
  if (mold > 0 || insect > 0 || coverage >= 0.1) return "Reject";
  if (coverage >= 0.05) return "Class II";
  if (coverage > 0) return "Class I";
  return "Extra Class";
}

async function saveGroundTruthRow(btn) {
  const row = btn.closest("tr");
  const counts = row.querySelectorAll(".gt-count");
  const payload = {
    ear_id: row.dataset.ear,
    variety: row.querySelector(".gt-variety").textContent,
    coverage: row.querySelector(".gt-coverage").value,
    mold_count: counts[0].value,
    insect_count: counts[1].value,
    missing_count: counts[2].value,
    other_count: counts[3].value,
  };
  btn.disabled = true;
  try {
    const res = await fetch("/dataset/ground-truth", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await expectOk(res);
    const gradeEl = row.querySelector(".gt-grade");
    gradeEl.textContent = data.row.grade;
    gradeEl.className = `grade-chip gt-grade ${gradeClass(data.row.grade)}`;
    clearError();
    loadAll();
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
  }
}

function populateEarOptions(ears) {
  earOptions.innerHTML = ears.map((e) => `<option value="${e.ear_id}">`).join("");
}

nextEarBtn.addEventListener("click", () => {
  const nums = earsData
    .map((e) => parseInt(e.ear_id.replace("ear", ""), 10))
    .filter((n) => Number.isFinite(n));
  const next = (nums.length ? Math.max(...nums) : 0) + 1;
  earIdInput.value = `ear${String(next).padStart(3, "0")}`;
});

function inspectBulkFiles(files) {
  const groups = new Map();
  const invalid = [];
  const pattern = /^ear(\d{3})_v([1-4])\.(heic|heif|jpg|jpeg|png)$/i;
  files.forEach((file) => {
    const match = file.name.match(pattern);
    if (!match) {
      invalid.push(file.name);
      return;
    }
    const sourceEar = `ear${match[1]}`.toLowerCase();
    const view = Number(match[2]);
    const views = groups.get(sourceEar) || new Map();
    if (views.has(view)) {
      invalid.push(`${file.name} (duplicate view)`);
    } else {
      views.set(view, file);
    }
    groups.set(sourceEar, views);
  });
  const incomplete = [...groups.entries()]
    .map(([sourceEar, views]) => ({ sourceEar, missing: [1, 2, 3, 4].filter((v) => !views.has(v)) }))
    .filter((entry) => entry.missing.length);
  return { groups, invalid, incomplete };
}

function updateBulkSummary() {
  const files = [...(bulkImages.files || [])];
  const result = inspectBulkFiles(files);
  if (!files.length) {
    bulkSummary.textContent = "No bulk files selected.";
  } else if (result.invalid.length) {
    bulkSummary.textContent = `Fix ${result.invalid.length} invalid or duplicate filename${result.invalid.length === 1 ? "" : "s"}: ${result.invalid.join(", ")}`;
  } else if (result.incomplete.length) {
    const details = result.incomplete
      .map((entry) => `${entry.sourceEar} missing v${entry.missing.join(", v")}`)
      .join("; ");
    bulkSummary.textContent = `Incomplete groups: ${details}`;
  } else {
    bulkSummary.textContent = `${result.groups.size} ear group${result.groups.size === 1 ? "" : "s"} · ${files.length} view${files.length === 1 ? "" : "s"} ready to import`;
  }
  bulkImportBtn.disabled = !files.length || result.invalid.length > 0 || result.incomplete.length > 0;
}

bulkImages.addEventListener("change", updateBulkSummary);

bulkForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  updateBulkSummary();
  if (bulkImportBtn.disabled) return;
  const form = new FormData();
  form.append("variety", bulkVariety.value);
  [...bulkImages.files].forEach((file) => form.append("images", file));
  bulkImportBtn.disabled = true;
  bulkStatus.textContent = "Importing ears…";
  try {
    const res = await fetch("/dataset/bulk-ears", { method: "POST", body: form });
    const data = await expectOk(res);
    bulkStatus.textContent = `Imported ${data.ears_imported} ears · ${data.views_imported} view(s).`;
    bulkForm.reset();
    updateBulkSummary();
    await loadAll();
  } catch (err) {
    showError(err.message);
    bulkImportBtn.disabled = false;
  }
});

document.querySelectorAll(".view-slot").forEach((slot) => {
  const inputs = slot.querySelectorAll('input[type="file"]');
  inputs.forEach((input) =>
    input.addEventListener("change", () => {
      const image = slot.querySelector('input[accept*="image"]');
      const label = slot.querySelector('input[accept=".txt"]');
      const status = slot.querySelector(".slot-status");
      const parts = [];
      if (image.files.length) parts.push("Image");
      if (label.files.length) parts.push("Annotation");
      status.textContent = parts.length ? parts.join(" · ") : "";
      status.className = parts.length ? "slot-status ok" : "slot-status";
    })
  );
});

earForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  const slots = [...document.querySelectorAll(".view-slot")];
  const images = slots
    .map((s) => s.querySelector('input[accept*="image"]').files[0])
    .filter(Boolean);
  if (!images.length) {
    showError("Select at least one photo.");
    return;
  }
  const labels = slots
    .map((s) => s.querySelector('input[accept=".txt"]').files[0])
    .filter(Boolean);
  if (labels.length > images.length) {
    showError("More labels than photos.");
    return;
  }
  const form = new FormData();
  form.append("ear_id", earIdInput.value.trim() || "auto");
  form.append("variety", varietySel.value);
  images.forEach((f) => form.append("images", f));
  labels.forEach((f) => form.append("labels", f));

  importBtn.disabled = true;
  importStatus.textContent = "Importing…";
  try {
    const res = await fetch("/dataset/ears", { method: "POST", body: form });
    const data = await expectOk(res);
    importStatus.textContent = `Imported ${data.ear_id} · ${data.views_added} view(s), ${data.labels_added} label(s).`;
    earForm.reset();
    earIdInput.value = "auto";
    document.querySelectorAll(".slot-status").forEach((s) => (s.textContent = ""));
    loadAll();
  } catch (err) {
    showError(err.message);
  } finally {
    importBtn.disabled = false;
  }
});

validateBtn.addEventListener("click", async () => {
  validateBtn.disabled = true;
  try {
    const res = await fetch("/dataset/validate");
    const data = await expectOk(res);
    if (data.ok) {
      problemsTitle.textContent = "Dataset looks good";
      problemsIcon.className = "problems-icon ok";
      problemList.innerHTML = "<li>No validation problems found.</li>";
    } else {
      const n = data.problems.length;
      problemsTitle.textContent = `${n} validation problem${n === 1 ? "" : "s"}`;
      problemsIcon.className = "problems-icon warn";
      problemList.innerHTML = data.problems.map((p) => `<li>${p}</li>`).join("");
    }
    problemsWrap.classList.remove("hidden");
  } catch (err) {
    showError(err.message);
  } finally {
    validateBtn.disabled = false;
  }
});

const exportGtBtn = document.getElementById("export-gt");
exportGtBtn.addEventListener("click", () => {
  if (!gtRowsData.length) {
    showError("Nothing to export yet.");
    return;
  }
  clearError();
  const header = "ear_id,variety,coverage,mold_count,insect_count,missing_count,other_count,grade";
  const lines = gtRowsData
    .map((r) => [r.ear_id, r.variety, r.coverage, r.mold_count, r.insect_count, r.missing_count, r.other_count, r.grade].join(","));
  const blob = new Blob([[header, ...lines].join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "ground_truth.csv";
  a.click();
  URL.revokeObjectURL(url);
});

loadAll().catch((err) => showError(err.message));
