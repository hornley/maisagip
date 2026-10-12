const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

class FakeClassList {
  constructor() {
    this.values = new Set();
  }

  add(...names) { names.forEach((name) => this.values.add(name)); }
  remove(...names) { names.forEach((name) => this.values.delete(name)); }
  contains(name) { return this.values.has(name); }
  toggle(name, force) {
    const shouldAdd = force === undefined ? !this.values.has(name) : force;
    if (shouldAdd) this.values.add(name); else this.values.delete(name);
    return shouldAdd;
  }
}

class FakeElement {
  constructor(document, tagName = "div") {
    this.ownerDocument = document;
    this.tagName = tagName;
    this.children = [];
    this.listeners = {};
    this.classList = new FakeClassList();
    this.style = {};
    this.attributes = {};
    this.disabled = false;
    this.open = false;
    this.value = "";
    this._textContent = "";
    this._innerHTML = "";
  }

  set textContent(value) { this._textContent = String(value); this.children = []; }
  get textContent() { return this._textContent; }
  set innerHTML(value) { this._innerHTML = String(value); this.children = []; }
  get innerHTML() { return this._innerHTML; }
  append(...items) { this.children.push(...items); }
  appendChild(item) { this.children.push(item); return item; }
  replaceChildren(...items) { this.children = items; }
  addEventListener(type, listener) {
    (this.listeners[type] ||= []).push(listener);
  }
  dispatchEvent(type, event = {}) {
    (this.listeners[type] || []).forEach((listener) => listener(event));
  }
  click() {
    this.ownerDocument.activeElement = this;
    this.dispatchEvent("click", { currentTarget: this, target: this });
  }
  focus() { this.ownerDocument.activeElement = this; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name]; }
  showModal() { this.open = true; }
  close() {
    this.open = false;
    this.dispatchEvent("close");
  }
  scrollIntoView() {}
}

function createHarness() {
  const elements = new Map();
  const document = {
    activeElement: null,
    createElement: (tagName) => new FakeElement(document, tagName),
    getElementById: (id) => {
      if (!elements.has(id)) elements.set(id, new FakeElement(document));
      return elements.get(id);
    },
  };
  const ids = [
    "dropzone", "file-input", "file-list", "preview-wrap", "previews", "inspect-btn", "dz-status",
    "error", "result", "confidence-threshold", "confidence-threshold-value", "mismatch-dialog",
    "mismatch-close", "mismatch-previous", "mismatch-next", "mismatch-thumbnails", "mismatch-message",
    "mismatch-image", "mismatch-view", "mismatch-variety-name", "mismatch-variety-class",
    "mismatch-defect-list", "mismatch-counter", "variety-label", "inspection-date", "annotated",
    "annotation-hotspots", "warning", "grade-value", "coverage-value", "coverage-bar", "grade-reasons",
    "defects", "ear-size", "completeness", "completeness-bar", "recommendations-card",
    "recommendations-summary", "recommendations-list", "views", "mode-chip",
  ];
  ids.forEach((id) => elements.set(id, new FakeElement(document)));
  elements.get("confidence-threshold").value = "50";

  const revoked = [];
  let urlNumber = 0;
  const context = {
    console,
    document,
    URL: {
      createObjectURL: () => `blob:test-${++urlNumber}`,
      revokeObjectURL: (url) => revoked.push(url),
    },
    FormData: class { append() {} },
    fetch: () => Promise.reject(new Error("fetch not configured")),
    setTimeout,
    clearTimeout,
  };
  context.window = context;
  vm.createContext(context);
  const source = fs.readFileSync(path.join(__dirname, "..", "app.js"), "utf8");
  vm.runInContext(source, context, { filename: "frontend/app.js" });
  vm.runInContext(`globalThis.appTest = {
    isStructuredMismatch,
    selectFiles,
    showMismatchDialog,
  };`, context);
  return { context, elements, revoked };
}

function result(view, varietyClass, defects) {
  return { view, variety: { class: varietyClass }, defects };
}

test("validates an aligned structured mismatch response", () => {
  const { context } = createHarness();
  const valid = {
    detail: "Views disagree.",
    view_results: [result(1, "white_corn", []), result(2, "yellow_sweet_corn", [])],
  };
  assert.equal(context.appTest.isStructuredMismatch(valid, 2), true);
  assert.equal(context.appTest.isStructuredMismatch({ ...valid, view_results: [valid.view_results[1], valid.view_results[0]] }, 2), false);
  assert.equal(context.appTest.isStructuredMismatch({ ...valid, view_results: [result(1, "white_corn", [{ class: 4 }]), valid.view_results[1]] }, 2), false);
});

test("selecting a thumbnail preserves its focused button and updates details", () => {
  const { context, elements } = createHarness();
  const indexHtml = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
  assert.match(indexHtml, /id="mismatch-thumbnails"[^>]*role="group"/);
  const files = [{ name: "one.jpg" }, { name: "two.jpg" }];
  context.appTest.selectFiles(files);
  context.appTest.showMismatchDialog("Views disagree.", [
    result(1, "white_corn", [{ class: "mold", confidence: 0.8, box: [0, 0, 1, 1] }, { class: "mold" }, { class: "corn_ear" }]),
    result(2, "yellow_sweet_corn", []),
  ]);

  const strip = elements.get("mismatch-thumbnails");
  const firstButton = strip.children[0];
  const secondButton = strip.children[1];
  assert.equal(strip.children.length, 2);
  assert.equal(firstButton.getAttribute("aria-pressed"), "true");
  assert.match(elements.get("mismatch-defect-list").children[0].textContent, /mold ×2/);

  secondButton.click();
  assert.equal(context.document.activeElement, secondButton);
  assert.equal(strip.children[1], secondButton);
  assert.equal(firstButton.getAttribute("aria-pressed"), "false");
  assert.equal(secondButton.getAttribute("aria-pressed"), "true");
  assert.equal(elements.get("mismatch-view").textContent, "View 2");
  assert.equal(elements.get("mismatch-variety-name").textContent, "Sweet Fortune");
  assert.equal(elements.get("mismatch-defect-list").children[0].textContent, "No visible defects detected.");
});

test("ignores stale inspection responses and revokes replaced preview URLs", async () => {
  const harness = createHarness();
  const { context, elements, revoked } = harness;
  let resolveFetch;
  context.fetch = () => new Promise((resolve) => { resolveFetch = resolve; });
  context.appTest.selectFiles([{ name: "old.jpg" }]);
  const oldUrl = context.document.getElementById("previews").children[0].children[0].src;
  elements.get("inspect-btn").click();
  context.appTest.selectFiles([{ name: "new.jpg" }]);
  assert.deepEqual(revoked, [oldUrl]);

  resolveFetch({
    ok: false,
    status: 400,
    json: async () => ({ detail: "Old mismatch", view_results: [result(1, "white_corn", [])] }),
  });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(elements.get("mismatch-dialog").open, false);
  assert.equal(elements.get("error").classList.contains("hidden"), true);
});
