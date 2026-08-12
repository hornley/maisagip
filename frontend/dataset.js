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
const problemList = document.getElementById("problem-list");
const validateBtn = document.getElementById("validate-btn");
const earsBody = document.querySelector("#ears-table tbody");
const gtBody = document.querySelector("#gt-table tbody");

let earsData = [];

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function clearError() {
  errorEl.classList.add("hidden");
}

async function loadAll() {
  clearError();
  const [stats, ears, gt] = await Promise.all([
    fetch("/dataset/stats").then((r) => r.json()),
    fetch("/dataset/ears").then((r) => r.json()),
    fetch("/dataset/ground-truth").then((r) => r.json()),
  ]);
  earsData = ears;
  renderStats(stats);
  renderEars(ears, gt.rows);
  renderGroundTruth(gt.rows, ears);
  populateEarOptions(ears);
}

function renderStats(stats) {
  const byVariety = Object.entries(stats.classifier_images)
    .map(([k, v]) => `<span class="stat"><span class="stat-label">${k}</span><span class="stat-value">${v}</span></span>`)
    .join("");
  const boxes = Object.entries(stats.boxes_per_class)
    .map(([k, v]) => `<span class="stat"><span class="stat-label">${k}</span><span class="stat-value">${v}</span></span>`)
    .join("");
  statsEl.innerHTML = `
    <span class="stat"><span class="stat-label">Ears</span><span class="stat-value">${stats.ears}</span></span>
    ${byVariety}
    <span class="stat"><span class="stat-label">Detector images</span><span class="stat-value">${stats.detector_images}</span></span>
    <span class="stat"><span class="stat-label">Labels</span><span class="stat-value">${stats.detector_labels}</span></span>
    <span class="stat"><span class="stat-label">Missing labels</span><span class="stat-value">${stats.missing_labels.length}</span></span>
    <span class="stat"><span class="stat-label">Incomplete ears</span><span class="stat-value">${stats.incomplete_ears.length}</span></span>
    ${boxes}
  `;
}

function renderEars(ears, gtRows) {
  const gtIds = new Set(gtRows.map((r) => r.ear_id));
  earsBody.innerHTML = ears
    .map(
      (e) => `
      <tr>
        <td>${e.ear_id}</td>
        <td>${e.variety}</td>
        <td>${e.views.join(", ")}</td>
        <td>${e.labels.length ? `<span class="ok">${e.labels.join(", ")}</span>` : `<span class="warn">missing</span>`}</td>
        <td>${gtIds.has(e.ear_id) ? '<span class="ok">yes</span>' : ""}</td>
      </tr>`
    )
    .join("");
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
  gtBody.innerHTML = rows
    .map(
      (r) => `
      <tr data-ear="${r.ear_id}">
        <td class="id-cell">${r.ear_id}</td>
        <td><span class="gt-variety">${r.variety}</span></td>
        <td><input class="gt-input gt-coverage" type="number" min="0" max="1" step="0.01" value="${r.coverage}"></td>
        <td><input class="gt-input gt-count" type="number" min="0" step="1" value="${r.mold_count}"></td>
        <td><input class="gt-input gt-count" type="number" min="0" step="1" value="${r.insect_count}"></td>
        <td><input class="gt-input gt-count" type="number" min="0" step="1" value="${r.missing_count}"></td>
        <td><input class="gt-input gt-count" type="number" min="0" step="1" value="${r.other_count}"></td>
        <td><span class="gt-grade">${r.grade}</span></td>
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
      row.querySelector(".gt-grade").textContent = grade;
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
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    row.querySelector(".gt-grade").textContent = data.row.grade;
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

document.querySelectorAll(".view-slot").forEach((slot) => {
  const inputs = slot.querySelectorAll('input[type="file"]');
  inputs.forEach((input) =>
    input.addEventListener("change", () => {
      const image = slot.querySelector('input[accept*="image"]');
      const label = slot.querySelector('input[accept=".txt"]');
      const status = slot.querySelector(".slot-status");
      const parts = [];
      if (image.files.length) parts.push("photo");
      if (label.files.length) parts.push("label");
      status.textContent = parts.length ? parts.join(" + ") : "";
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
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    importStatus.textContent = `Imported ${data.ear_id} (${data.views_added} view(s), ${data.labels_added} label(s)).`;
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
    const data = await res.json();
    if (data.ok) {
      problemList.innerHTML = "<li>Dataset OK.</li>";
    } else {
      problemList.innerHTML = data.problems.map((p) => `<li>${p}</li>`).join("");
    }
    problemsWrap.classList.remove("hidden");
  } catch (err) {
    showError(err.message);
  } finally {
    validateBtn.disabled = false;
  }
});

loadAll().catch((err) => showError(err.message));
