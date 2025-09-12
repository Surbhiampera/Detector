// 🌐 Define API base URL once here
// Change port 8000 if your FastAPI backend runs on a different one
const API_BASE = `http://127.0.0.1:8000`;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

async function fetchJSON(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  } catch (err) {
    console.error(`Error fetching ${url}:`, err);
    return null;
  }
}

function qs(params) {
  const u = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") u.append(k, v);
  });
  return u.toString();
}

function currentFilters() {
  return {
    drug: $("#filter-drug")?.value.trim(),
    event: $("#filter-event")?.value.trim(),
    sex: $("#filter-sex")?.value,
    min_age: $("#filter-min-age")?.value,
    max_age: $("#filter-max-age")?.value,
  };
}

async function loadKPIs() {
  const data = await fetchJSON(`${API_BASE}/api/kpis?${qs(currentFilters())}`);
  if (!data) return;
  $("#total-reports").textContent = data.total_reports;
  $("#flagged-events").textContent = data.flagged_events;
  $("#new-signals").textContent = data.new_signals_detected;
  const list = $("#top-risk-drugs");
  list.innerHTML = "";
  data.top_risk_drugs.forEach((d) => {
    const li = document.createElement("li");
    li.textContent = `${d.drug_name}: ${d.score.toFixed(2)}`;
    list.appendChild(li);
  });
}

let chartEventsTime, chartPie, chartSignalTrend;

async function loadCharts() {
  const events = await fetchJSON(
    `${API_BASE}/api/events?${qs({ ...currentFilters(), limit: 2000 })}`
  );
  if (!events) return;

  const byMonth = {};
  events.items.forEach((r) => {
    const m = r.date_reported?.slice(0, 7) || "";
    if (m) byMonth[m] = (byMonth[m] || 0) + 1;
  });
  const labels = Object.keys(byMonth).sort();
  const values = labels.map((k) => byMonth[k]);

  const ctx1 = document.getElementById("chart-events-time");
  chartEventsTime = new Chart(ctx1, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Reports", data: values, borderColor: "#2563eb", fill: false },
      ],
    },
    options: { responsive: true, plugins: { legend: { display: false } } },
  });

  const byDrug = {};
  events.items.forEach((r) => {
    byDrug[r.drug_name] = (byDrug[r.drug_name] || 0) + 1;
  });
  const drugLabels = Object.keys(byDrug).slice(0, 8);
  const drugValues = drugLabels.map((k) => byDrug[k]);
  const ctx2 = document.getElementById("chart-pie");
  chartPie = new Chart(ctx2, {
    type: "pie",
    data: { labels: drugLabels, datasets: [{ data: drugValues }] },
    options: { responsive: true },
  });

  const sig = await fetchJSON(
    `${API_BASE}/api/signals?${qs(currentFilters())}`
  );
  if (!sig) return;
  const ctx3 = document.getElementById("chart-signal-trend");
  const seriesDatasets = Object.values(sig.series).map((s) => ({
    label: s.label,
    data: s.points.map((p) => ({ x: p.month, y: p.count })),
  }));
  chartSignalTrend = new Chart(ctx3, {
    type: "line",
    data: { datasets: seriesDatasets },
    options: { parsing: false, scales: { x: { type: "timeseries" } } },
  });
}

async function loadExplorer() {
  const params = { ...currentFilters(), limit: 100 };
  const data = await fetchJSON(`${API_BASE}/api/events?${qs(params)}`);
  if (!data) return;

  const tbody = $("#events-table tbody");
  tbody.innerHTML = "";
  data.items.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.drug_name}</td>
      <td>${r.indication}</td>
      <td>${r.adverse_event}</td>
      <td>${r.age}</td>
      <td>${r.sex}</td>
      <td>${r.outcome}</td>
      <td>${r.date_reported}</td>
      <td>${r.signal_score.toFixed(1)}</td>
      <td>
        ${
          r.patient_id !== undefined && r.patient_id !== null
            ? `<button class="btn-download" data-id="${r.patient_id}">Download</button>`
            : ""
        }
      </td>
    `;
    tbody.appendChild(tr);
  });

  tbody.addEventListener("click", async (e) => {
    const target = e.target;
    if (target && target.classList.contains("btn-download")) {
      const pid = target.getAttribute("data-id");
      if (!pid) return;
      try {
        const url = `${API_BASE}/api/patient/${encodeURIComponent(
          pid
        )}/report?format=csv`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `patient_${pid}.csv`;
        document.body.appendChild(a);
        a.click();
        URL.revokeObjectURL(a.href);
        a.remove();
      } catch (err) {
        console.error(err);
        alert(err.message || String(err));
      }
    }
  });
}

async function loadHeatmapAndSignals() {
  const heat = await fetchJSON(
    `${API_BASE}/api/heatmap?${qs(currentFilters())}`
  );
  if (!heat) return;

  const container = $("#heatmap");
  container.innerHTML = "";
  const drugs = [...new Set(heat.cells.map((c) => c.drug_name))];
  const events = [...new Set(heat.cells.map((c) => c.adverse_event))];
  const grid = document.createElement("div");
  grid.className = "grid";
  grid.appendChild(document.createElement("div"));
  events.forEach((ev) => {
    const d = document.createElement("div");
    d.className = "hdr";
    d.textContent = ev;
    grid.appendChild(d);
  });
  drugs.forEach((dr) => {
    const rowHdr = document.createElement("div");
    rowHdr.className = "hdr";
    rowHdr.textContent = dr;
    grid.appendChild(rowHdr);
    events.forEach((ev) => {
      const cell = document.createElement("div");
      cell.className = "cell";
      const match = heat.cells.find(
        (c) => c.drug_name === dr && c.adverse_event === ev
      );
      const intensity = match ? match.intensity : 0;
      cell.style.background = `rgba(220, 38, 38, ${intensity})`;
      cell.title = `${dr} x ${ev} (${match ? match.count : 0})`;
      grid.appendChild(cell);
    });
  });
  container.appendChild(grid);

  const sig = await fetchJSON(`${API_BASE}/api/signals`);
  if (!sig) return;
  const ul = $("#top-signals");
  ul.innerHTML = "";
  sig.top_new_signals.forEach((s) => {
    const li = document.createElement("li");
    li.textContent = `${s.drug_name} - ${
      s.adverse_event
    } (slope ${s.slope.toFixed(3)})`;
    ul.appendChild(li);
  });
}

function buildHighlightedHTML(text, spans) {
  if (!spans || spans.length === 0) return text;
  let html = "";
  let cursor = 0;
  spans.forEach((s, idx) => {
    const pre = text.slice(cursor, s.start);
    const term = text.slice(s.start, s.end);
    html += pre
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    html += `<span class="highlight" data-term="${term}">${term}</span>`;
    cursor = s.end;
  });
  html += text
    .slice(cursor)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return html;
}

async function showEventDetails(term) {
  try {
    const data = await fetchJSON(
      `${API_BASE}/api/event_stats?${qs({ event: term })}`
    );
    const panel = $("#nlp-details");
    const byDrug = data.by_drug
      .map((d) => `<li>${d.drug_name}: ${d.count}</li>`)
      .join("\n");
    const cases = data.cases
      .map(
        (c) => `<li>#${c.patient_id} — ${c.drug_name} — ${c.date_reported}</li>`
      )
      .join("\n");
    panel.innerHTML = `
      <div><strong>Event:</strong> ${data.event}</div>
      <div><strong>Total cases:</strong> ${data.total} (last 30d: ${data.recent_30d})</div>
      <div class="sub"><strong>Top drugs</strong><ul>${byDrug}</ul></div>
      <div class="sub"><strong>Recent cases</strong><ul>${cases}</ul></div>
    `;
  } catch (e) {
    console.error(e);
  }
}

async function loadNLP() {
  const data = await fetchJSON(`${API_BASE}/api/nlp?limit=20`);
  if (!data) return;
  const ul = $("#nlp-list");
  ul.innerHTML = "";
  data.insights.forEach((i) => {
    const li = document.createElement("li");
    const header = `<div class="note-header"><strong>${
      i.patient_id ?? ""
    }</strong> ${i.date_reported} — severity ${i.severity_score}</div>`;
    const body = `<div class="note-body">${buildHighlightedHTML(
      i.note || "",
      i.spans || []
    )}</div>`;
    li.innerHTML = `<div class="note">${header}${body}</div>`;
    ul.appendChild(li);
  });
}

async function loadNLP() {
  const data = await fetchJSON(`${API_BASE}/api/nlp?limit=20`);
  if (!data) return;
  const ul = $("#nlp-list");
  ul.innerHTML = "";
  data.insights.forEach((i) => {
    const li = document.createElement("li");
    const header = `<div class="note-header"><strong>${
      i.patient_id ?? ""
    }</strong> ${i.date_reported} — severity ${i.severity_score}</div>`;
    const body = `<div class="note-body">${buildHighlightedHTML(
      i.note || "",
      i.spans || []
    )}</div>`;
    li.innerHTML = `<div class="note">${header}${body}</div>`;
    ul.appendChild(li);
  });
}

async function loadAlerts() {
  const data = await fetchJSON(`${API_BASE}/api/alerts`);
  if (!data) return;
  const ul = $("#alerts-list");
  ul.innerHTML = "";
  data.alerts.forEach((a) => {
    const li = document.createElement("li");
    li.className = `alert ${a.severity}`;
    li.textContent = `⚠️ ${a.message}`;
    ul.appendChild(li);
  });
}

function wireFilters() {
  $("#btn-apply").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    const orig = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Applying...";
    try {
      await Promise.all([
        loadExplorer(),
        loadKPIs(),
        loadCharts(),
        loadHeatmapAndSignals(),
        loadNLP(),
        loadAlerts(),
      ]);
    } finally {
      btn.disabled = false;
      btn.textContent = orig;
    }
  });

  $("#btn-upload").addEventListener("click", async () => {
    const fileInput = $("#file-csv");
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      alert("Please choose a CSV file first.");
      return;
    }
    const fd = new FormData();
    fd.append("file", file);
    const btn = $("#btn-upload");
    const orig = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Uploading...";
    try {
      const res = await fetch(`${API_BASE}/api/upload_csv`, {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      alert(`Upload complete: ${data.rows} rows`);
      await Promise.all([
        loadKPIs(),
        loadCharts(),
        loadExplorer(),
        loadHeatmapAndSignals(),
        loadNLP(),
        loadAlerts(),
      ]);
    } catch (e) {
      console.error(e);
      alert(e.message || String(e));
    } finally {
      btn.disabled = false;
      btn.textContent = orig;
      fileInput.value = "";
    }
  });
}

async function init() {
  await loadKPIs();
  await loadCharts();
  await loadExplorer();
  await loadHeatmapAndSignals();
  await loadNLP();
  await loadAlerts();
  wireFilters();

  // Set up event delegation for NLP highlights
  $("#nlp-list").addEventListener("click", (e) => {
    const target = e.target;
    if (target && target.classList.contains("highlight")) {
      const term = target.getAttribute("data-term");
      showEventDetails(term);
    }
  });

  setInterval(() => {
    loadKPIs();
    loadCharts();
    loadHeatmapAndSignals();
    loadNLP();
    loadAlerts();
  }, 60000);
}

window.addEventListener("DOMContentLoaded", init);
