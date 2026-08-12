const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileList = document.getElementById("file-list");
const previewWrap = document.getElementById("preview-wrap");
const previewsEl = document.getElementById("previews");
const inspectBtn = document.getElementById("inspect-btn");
const statusEl = document.getElementById("dz-status");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");

let selectedFiles = [];

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
  selectedFiles = files;
  errorEl.classList.add("hidden");
  resultEl.classList.add("hidden");
  previewsEl.innerHTML = "";
  files.forEach((f) => {
    const holder = document.createElement("div");
    holder.className = "pthumb";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(f);
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
  inspectBtn.disabled = true;
  statusEl.textContent = "Analyzing ear…";
  const form = new FormData();
  selectedFiles.forEach((f) => form.append("files", f));
  try {
    const res = await fetch("/inspect", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    render(data);
  } catch (err) {
    showError(err.message);
  } finally {
    inspectBtn.disabled = false;
    statusEl.textContent = "";
  }
});

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function pct(v) {
  return `${Math.round((v ?? 0) * 100)}%`;
}

function render(report) {
  resultEl.classList.remove("hidden");
  errorEl.classList.add("hidden");

  document.getElementById("variety-label").textContent = report.variety.class.replace("_", " ");
  document.getElementById("overall-conf").textContent = `Overall confidence ${pct(report.grade.confidence)}`;
  document.getElementById("inspection-date").textContent =
    `Inspected ${report.inspection_date} · ${report.view_count} view(s)`;
  document.getElementById("annotated").src = report.image_url;

  const warning = document.getElementById("warning");
  warning.classList.add("hidden");
  if (report.grade.needs_reinspection) {
    warning.textContent = "Low prediction confidence detected — consider re-inspecting with clearer photos.";
    warning.classList.remove("hidden");
  }

  const gradeValue = document.getElementById("grade-value");
  gradeValue.textContent = report.grade.grade_label;
  gradeValue.classList.toggle("below", !report.grade.grade);

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
      const views = d.views_seen.map((v) => v + 1).join(",");
      chip.innerHTML = `${d.class.replace(/_/g, " ")} <span class="conf-val">×${d.merged_count} · cov ${pct(d.coverage)} · views ${views}</span>`;
      defectsBox.appendChild(chip);
    });
  }

  document.getElementById("ear-size").textContent =
    `${report.traits.ear_size} (${pct(report.traits.ear_size_fraction)})`;
  document.getElementById("completeness").textContent = pct(report.traits.kernel_completeness);
  document.getElementById("completeness-bar").style.width = pct(report.traits.kernel_completeness);

  document.getElementById("util-value").textContent = report.grade.utilization.recommendation.replace(/_/g, " ");
  document.getElementById("util-reason").textContent = report.grade.utilization.reason;

  const viewsBox = document.getElementById("views");
  viewsBox.innerHTML = "";
  report.views.forEach((v) => {
    const card = document.createElement("figure");
    card.className = "view-card";
    const img = document.createElement("img");
    img.src = v.image_url;
    img.alt = `view ${v.view}`;
    const defects = v.defects.filter((d) => d.class !== "corn_ear");
    const cap = document.createElement("figcaption");
    cap.textContent = `View ${v.view} · ${v.variety.class.replace("_", " ")}` +
      (defects.length ? ` · ${defects.map((d) => d.class.replace(/_/g, " ")).join(", ")}` : " · clean");
    card.append(img, cap);
    viewsBox.appendChild(card);
  });

  const chip = document.getElementById("mode-chip");
  const modes = report.mode || {};
  chip.textContent = `classifier: ${modes.classifier} · detector: ${modes.detector}`;
  chip.classList.remove("hidden");

  resultEl.scrollIntoView({ behavior: "smooth" });
}