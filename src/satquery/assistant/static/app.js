const byId = (id) => document.getElementById(id);
let classNames = [];

function setText(target, value) {
  target.textContent = value == null ? "" : String(value);
}

function appendQuestionControls(form) {
  const fragment = byId("question-template").content.cloneNode(true);
  form.querySelector(".question-block").replaceWith(fragment);
  form.querySelectorAll(".examples button").forEach((button) => {
    button.addEventListener("click", () => {
      form.elements.question.value = button.textContent;
      form.elements.question.focus();
    });
  });
}

appendQuestionControls(byId("guided-form"));
appendQuestionControls(byId("zip-form"));

document.querySelectorAll(".mode-button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".mode-button").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".input-mode").forEach((item) => item.classList.add("hidden"));
    button.classList.add("active");
    byId(button.dataset.panel === "guided-panel" ? "guided-form" : "zip-form").classList.remove("hidden");
  });
});

function updateSar(number) {
  const modality = document.querySelector(`[data-modality="${number}"]`).value;
  const field = document.querySelector(`[data-sar="${number}"]`);
  field.classList.toggle("hidden", modality !== "SAR");
  field.querySelector("select").disabled = modality !== "SAR";
}

document.querySelectorAll("[data-modality]").forEach((select) => {
  select.addEventListener("change", () => updateSar(select.dataset.modality));
});
updateSar("1");

byId("second-observation").addEventListener("change", (event) => {
  const fieldset = byId("observation-2");
  fieldset.disabled = !event.target.checked;
  fieldset.classList.toggle("hidden", !event.target.checked);
  if (event.target.checked) updateSar("2");
});

function setRunState(label, kind) {
  const target = byId("run-state");
  setText(target, label);
  target.className = `run-state ${kind}`;
}

function formatValue(value) {
  if (typeof value === "number") {
    if (Number.isInteger(value)) return value.toLocaleString();
    return Math.abs(value) < 1 ? value.toFixed(4) : value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  if (Array.isArray(value)) return value.map((item) => formatValue(item)).join(", ");
  if (value && typeof value === "object") {
    if (typeof value.name === "string" && typeof value.estimated_fraction === "number") {
      return `${value.name} ${(value.estimated_fraction * 100).toFixed(2)}%`;
    }
    return JSON.stringify(value);
  }
  return String(value ?? "—");
}

function renderMeasurements(measurements) {
  const table = byId("measurements");
  table.replaceChildren();
  const rows = Array.isArray(measurements)
    ? measurements
    : measurements && Array.isArray(measurements.classes)
      ? measurements.classes
      : measurements && typeof measurements === "object"
        ? [measurements]
        : [];
  if (!rows.length) {
    const caption = document.createElement("caption");
    setText(caption, "No measurements were produced.");
    table.append(caption);
    return;
  }
  const keys = [...new Set(rows.flatMap((row) => Object.keys(row)))].filter((key) => {
    return rows.some((row) => typeof row[key] !== "object" || Array.isArray(row[key]));
  });
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  keys.forEach((key) => {
    const th = document.createElement("th");
    setText(th, key.replaceAll("_", " "));
    headerRow.append(th);
  });
  head.append(headerRow);
  const body = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    keys.forEach((key) => {
      const td = document.createElement("td");
      setText(td, formatValue(row[key]));
      tr.append(td);
    });
    body.append(tr);
  });
  table.append(head, body);
}

function findGrid(evidence) {
  const items = Array.isArray(evidence) ? evidence : evidence && typeof evidence === "object" ? [evidence] : [];
  for (const item of items) {
    for (const key of ["fraction_grid", "estimated_fraction_grid", "dominant_class_index_grid", "class_delta_grid"]) {
      if (Array.isArray(item[key]) && item[key].length === 15) return { values: item[key], key };
    }
  }
  return null;
}

function gridColor(value, kind) {
  if (kind === "dominant_class_index_grid") {
    const hue = (Number(value) * 47 + 35) % 360;
    return `hsl(${hue} 45% 48%)`;
  }
  const normalized = kind === "class_delta_grid" ? (Number(value) + 1) / 2 : Number(value);
  const lightness = 94 - Math.max(0, Math.min(1, normalized)) * 67;
  return `hsl(145 36% ${lightness}%)`;
}

function renderGrid(evidence) {
  const found = findGrid(evidence);
  const section = byId("grid-section");
  const grid = byId("evidence-grid");
  grid.replaceChildren();
  section.classList.toggle("hidden", !found);
  if (!found) return;
  found.values.forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "grid-cell";
      button.style.backgroundColor = gridColor(value, found.key);
      button.setAttribute("aria-label", `Row ${rowIndex + 1}, column ${columnIndex + 1}: ${formatValue(value)}`);
      button.addEventListener("click", () => {
        setText(byId("cell-readout"), `r${rowIndex + 1} · c${columnIndex + 1} · ${formatValue(value)}`);
      });
      grid.append(button);
    });
  });
  const legend = byId("legend");
  legend.replaceChildren();
  if (found.key === "dominant_class_index_grid" && classNames.length) {
    legend.classList.add("class-legend");
    classNames.forEach((item) => {
      const entry = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.style.backgroundColor = gridColor(item.index, found.key);
      setText(entry, `${item.index} · ${item.name}`);
      entry.prepend(swatch);
      legend.append(entry);
    });
    return;
  }
  legend.classList.remove("class-legend");
  const low = document.createElement("span");
  const ramp = document.createElement("span");
  const high = document.createElement("span");
  ramp.className = "legend-ramp";
  setText(low, found.key === "class_delta_grid" ? "decrease" : "low");
  setText(high, found.key === "dominant_class_index_grid" ? "class index" : found.key === "class_delta_grid" ? "increase" : "high");
  legend.append(low, ramp, high);
}

function renderResult(report) {
  byId("empty-result").classList.add("hidden");
  byId("error-box").classList.add("hidden");
  byId("result").classList.remove("hidden");
  setText(byId("deterministic-answer"), report.deterministic_answer || report.answer);
  renderMeasurements(report.measurements);
  renderGrid(report.evidence);

  const limitations = byId("limitations");
  limitations.replaceChildren();
  (report.limitations || []).forEach((value) => {
    const item = document.createElement("li");
    setText(item, value);
    limitations.append(item);
  });
  setText(byId("trace"), JSON.stringify(report.trace, null, 2));

  const preview = byId("source-preview");
  if (report.downloads && report.downloads.preview) {
    preview.src = report.downloads.preview;
    preview.classList.remove("hidden");
  } else {
    preview.removeAttribute("src");
    preview.classList.add("hidden");
  }
  const downloads = byId("downloads");
  downloads.replaceChildren();
  Object.entries(report.downloads || {}).forEach(([label, url]) => {
    if (typeof url !== "string" || !url.startsWith("/api/results/")) return;
    const link = document.createElement("a");
    link.href = url;
    setText(link, `Download ${label}`);
    downloads.append(link);
  });
  setRunState(report.abstained ? "Abstained" : "Complete", "done");
}

async function submit(endpoint, formData) {
  setRunState("Analyzing…", "working");
  byId("error-box").classList.add("hidden");
  try {
    const response = await fetch(endpoint, { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Analysis failed");
    renderResult(payload);
  } catch (error) {
    byId("result").classList.add("hidden");
    byId("empty-result").classList.add("hidden");
    const box = byId("error-box");
    setText(box, error.message || "Analysis failed");
    box.classList.remove("hidden");
    setRunState("Needs attention", "error");
  }
}

for (const form of [byId("guided-form"), byId("zip-form")]) {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    submit(form.dataset.endpoint, new FormData(form));
  });
}

byId("demo-form").addEventListener("submit", (event) => {
  event.preventDefault();
  submit("/api/demo", new FormData(event.currentTarget));
});

fetch("/api/status")
  .then((response) => response.json())
  .then((status) => {
    classNames = status.classes || [];
    const service = byId("service-status");
    service.className = "status-chip ready";
    setText(service, "Local service ready");
    setText(byId("capability-status"), `Models: ${status.capabilities.join(" · ") || "none"}`);
    setText(byId("provider-status"), status.providers.openai ? "OpenAI configured" : "OpenAI unavailable · local works");
    byId("demo-card").classList.toggle("hidden", !status.demo.available);
    setText(
      byId("demo-label"),
      `${status.demo.capability} cached features · demo_fit_all: ${status.demo.demo_fit_all ? "yes" : "no"}`
    );
  })
  .catch(() => {
    const service = byId("service-status");
    service.className = "status-chip";
    setText(service, "Service unavailable");
  });
