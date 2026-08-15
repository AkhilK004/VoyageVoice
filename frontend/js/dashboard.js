/**
 * dashboard.js — Evaluation Dashboard logic.
 *
 * Fetches data from REST API, renders Chart.js charts,
 * populates tables and analysis cards.
 *
 * Views: overview | latency | quality | sessions | analysis
 */

// ── State ─────────────────────────────────────────────────────────────────────
let globalMetrics = null;
let allSessions = [];
let analysis = null;
let activeView = "overview";
const charts = {};

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadAll();
});

async function loadAll() {
  showLoading(true);
  try {
    const [metricsRes, sessionsRes, analysisRes] = await Promise.all([
      fetch("/api/metrics").then(r => r.json()),
      fetch("/api/conversations").then(r => r.json()),
      fetch("/api/analysis").then(r => r.json()),
    ]);
    globalMetrics = metricsRes;
    allSessions = sessionsRes.sessions || [];
    analysis = analysisRes;

    document.getElementById("total-sessions-badge").textContent =
      `${allSessions.length} session${allSessions.length !== 1 ? "s" : ""}`;

    renderOverview();
    renderLatency();
    renderQuality();
    renderSessions();
    renderAnalysis();
  } catch (e) {
    console.error("Failed to load dashboard data:", e);
  } finally {
    showLoading(false);
  }
}

// ── View Switcher ─────────────────────────────────────────────────────────────
const VIEW_META = {
  overview:  { title: "Overview",  sub: "Aggregate metrics across all sessions" },
  latency:   { title: "Latency",   sub: "STT → LLM → TTS → E2E pipeline timing" },
  quality:   { title: "Quality",   sub: "Relevance, naturalness & hallucination scores" },
  sessions:  { title: "Sessions",  sub: "Individual call sessions with drill-down" },
  analysis:  { title: "Analysis",  sub: "Failure patterns and improvement recommendations" },
};

function showView(name) {
  activeView = name;
  document.querySelectorAll(".view").forEach(el => el.classList.add("hidden"));
  document.getElementById(`view-${name}`)?.classList.remove("hidden");

  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle("active", el.dataset.view === name);
  });

  const meta = VIEW_META[name] || {};
  document.getElementById("page-title").textContent = meta.title || name;
  document.getElementById("page-sub").textContent = meta.sub || "";
}

function showLoading(show) {
  document.getElementById("loading").style.display = show ? "flex" : "none";
}

// ── Overview ──────────────────────────────────────────────────────────────────
function renderOverview() {
  const m = globalMetrics;
  if (!m || !m.latency) return;

  setVal("val-e2e",          fmt(m.latency.e2e_avg_ms) + " ms");
  setVal("sub-e2e",          `P95: ${fmt(m.latency.e2e_p95_ms)} ms`);
  setVal("val-relevance",    fmt(m.quality?.avg_relevance, 1) || "—");
  setVal("val-hallucination",fmtPct(m.quality?.hallucination_rate));
  setVal("sub-hall-count",   `${m.quality?.scored_turns || 0} scored turns`);
  setVal("val-interruptions",m.total_interruptions ?? "0");
  setVal("sub-turns",        `across ${m.total_turns || 0} turns`);

  // Color the E2E card based on performance
  colorCard("stat-e2e", m.latency.e2e_avg_ms, 1500, 3000);
  colorCard("stat-relevance", (5 - (m.quality?.avg_relevance || 3)) * 1000, 1000, 2500); // inverted
  colorCard("stat-hallucination", (m.quality?.hallucination_rate || 0) * 10000, 500, 1500);

  // E2E latency over time chart
  renderLine("chart-e2e", m.chart_data?.e2e_series || [], {
    label: "E2E Latency (ms)",
    color: "#6c63ff",
    threshold: 2000,
  });

  // Quality scores donut
  const rel = m.quality?.avg_relevance || 0;
  const nat = m.quality?.avg_naturalness || 0;
  renderRadar("chart-quality-overview", ["Relevance", "Naturalness", "Low Hallucin.", "Speed"],
    [rel, nat, 5 - (m.quality?.hallucination_rate || 0) * 5, Math.max(0, 5 - (m.latency.e2e_avg_ms || 2000) / 1000)]
  );
}

// ── Latency ───────────────────────────────────────────────────────────────────
function renderLatency() {
  const m = globalMetrics;
  if (!m || !m.latency) return;

  setVal("val-stt",   fmt(m.latency.stt_avg_ms) + " ms");
  setVal("val-llm",   fmt(m.latency.llm_ttft_avg_ms) + " ms");
  setVal("val-tts",   fmt(m.latency.tts_avg_ms) + " ms");
  setVal("val-e2e-2", fmt(m.latency.e2e_avg_ms) + " ms");

  // Stacked bar — breakdown per turn
  const series = m.chart_data?.stt_series || [];
  const llmSeries = m.chart_data?.llm_series || [];
  const labels = series.map((_, i) => `T${i + 1}`);
  renderStackedBar("chart-latency-breakdown", labels, [
    { label: "STT (ms)", data: series, color: "#06b6d4" },
    { label: "LLM TTFT (ms)", data: llmSeries, color: "#6c63ff" },
  ]);

  // Histograms
  renderHistogram("chart-stt-hist", m.chart_data?.stt_series || [], "STT (ms)", "#06b6d4");
  renderHistogram("chart-e2e-hist", m.chart_data?.e2e_series || [], "E2E (ms)", "#6c63ff");
}

// ── Quality ───────────────────────────────────────────────────────────────────
function renderQuality() {
  const m = globalMetrics;
  if (!m) return;

  // Relevance distribution (1-5 buckets)
  renderScoreDist("chart-relevance-dist", m.chart_data?.relevance_series || [], "Relevance", "#a78bfa");
  renderScoreDist("chart-natural-dist", m.chart_data?.relevance_series || [], "Naturalness", "#06b6d4");

  // Quality trend line
  renderLine("chart-quality-trend", m.chart_data?.relevance_series || [], {
    label: "Relevance Score",
    color: "#a78bfa",
    yMin: 0, yMax: 5,
  });
}

// ── Sessions Table ────────────────────────────────────────────────────────────
function renderSessions() {
  const tbody = document.getElementById("sessions-tbody");
  if (!allSessions.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty-row">No sessions yet. Start a call!</td></tr>`;
    return;
  }

  tbody.innerHTML = allSessions.map(s => {
    const dt = s.start_time ? new Date(s.start_time * 1000).toLocaleTimeString() : "—";
    const lat = s.e2e_avg_ms ? `${Math.round(s.e2e_avg_ms)}ms` : "—";
    const rel = s.avg_relevance ? s.avg_relevance.toFixed(1) : "—";
    const status = s.escalated
      ? `<span class="badge badge-red">Escalated</span>`
      : `<span class="badge badge-green">Complete</span>`;
    const latClass = s.e2e_avg_ms > 3000 ? "badge-red" : s.e2e_avg_ms > 1500 ? "badge-amber" : "badge-green";

    return `<tr onclick="loadDrilldown('${s.session_id}')">
      <td><code style="font-size:10px;color:var(--muted)">${(s.session_id || "").slice(0,18)}</code></td>
      <td>${dt}</td>
      <td>${s.turn_count || 0}</td>
      <td><span class="badge ${latClass}">${lat}</span></td>
      <td>${rel}</td>
      <td>${s.total_interruptions || 0}</td>
      <td>${status}</td>
      <td><button class="btn-detail" onclick="event.stopPropagation();loadDrilldown('${s.session_id}')">Detail →</button></td>
    </tr>`;
  }).join("");
}

async function loadDrilldown(sessionId) {
  const res = await fetch(`/api/conversations/${sessionId}`);
  if (!res.ok) return;
  const data = await res.json();

  const drilldown = document.getElementById("drilldown");
  const body = document.getElementById("drilldown-body");
  document.getElementById("drilldown-title").textContent = `Session: ${sessionId.slice(0, 20)}`;

  body.innerHTML = (data.turns || []).map(t => {
    const latencies = t.latencies || {};
    const quality = t.quality || {};
    const hallTag = quality.hallucination
      ? `<span class="tag tag-red">Hallucinated</span>`
      : quality.hallucination === false
      ? `<span class="tag tag-green">No hallucination</span>`
      : "";
    const intTag = t.was_interrupted ? `<span class="tag tag-red">Interrupted</span>` : "";

    return `<div class="turn-card">
      <div class="turn-header">
        <span>Turn ${t.turn_id}</span>
        <span>${t.state || ""}</span>
      </div>
      <div class="turn-user">👤 ${t.content?.user_text || "(no transcript)"}</div>
      <div class="turn-agent">🤖 ${t.content?.agent_text || "(no response)"}</div>
      <div class="turn-meta">
        <span class="tag">STT ${Math.round(latencies.stt_ms || 0)}ms</span>
        <span class="tag">LLM ${Math.round(latencies.llm_ttft_ms || 0)}ms</span>
        <span class="tag">TTS ${Math.round(latencies.tts_first_byte_ms || 0)}ms</span>
        <span class="tag">E2E ${Math.round(latencies.e2e_ms || 0)}ms</span>
        ${quality.relevance_score ? `<span class="tag">Rel ${quality.relevance_score}</span>` : ""}
        ${hallTag} ${intTag}
      </div>
    </div>`;
  }).join("") || "<div class='empty-state'>No turns in this session.</div>";

  drilldown.classList.remove("hidden");
  drilldown.scrollIntoView({ behavior: "smooth" });
}

function closeDrilldown() {
  document.getElementById("drilldown").classList.add("hidden");
}

// ── Analysis ──────────────────────────────────────────────────────────────────
function renderAnalysis() {
  if (!analysis) return;

  const grid = document.getElementById("analysis-grid");
  const patterns = analysis.patterns || [];

  if (!patterns.length) {
    grid.innerHTML = `<div class="empty-state">No sessions to analyse yet. Start some calls first.</div>`;
    return;
  }

  grid.innerHTML = patterns.map(p => {
    const examples = p.examples?.length
      ? `<div class="pattern-examples">Examples: "${p.examples.slice(0,2).join('", "')}"</div>`
      : "";
    return `<div class="pattern-card">
      <div class="pattern-header">
        <span class="severity-badge severity-${p.severity}">${p.severity.toUpperCase()}</span>
        <strong style="font-size:13px">${p.type.replace(/_/g, " ").toUpperCase()}</strong>
        ${p.percentage != null ? `<span class="pattern-pct" style="margin-left:auto">${p.percentage}%</span>` : ""}
      </div>
      <div class="pattern-desc">${p.description}</div>
      ${examples}
    </div>`;
  }).join("");

  const recs = analysis.recommendations || [];
  if (recs.length) {
    const card = document.getElementById("recommendations-card");
    card.style.display = "block";
    document.getElementById("rec-list").innerHTML = recs.map(r => `<li>${r}</li>`).join("");
  }
}

// ── Chart Helpers ─────────────────────────────────────────────────────────────
const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: false }, tooltip: { titleColor: "#f1f5f9", bodyColor: "#94a3b8" } },
  scales: {
    x: { ticks: { color: "#64748b", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.04)" } },
    y: { ticks: { color: "#64748b", font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.06)" } },
  },
};

function renderLine(id, data, opts = {}) {
  destroyChart(id);
  const ctx = document.getElementById(id)?.getContext("2d");
  if (!ctx || !data.length) return;

  const labels = data.map((_, i) => `T${i + 1}`);
  charts[id] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: opts.label || "Value",
        data,
        borderColor: opts.color || "#6c63ff",
        backgroundColor: (opts.color || "#6c63ff") + "22",
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointBackgroundColor: opts.color || "#6c63ff",
      }],
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        ...CHART_DEFAULTS.scales,
        y: { ...CHART_DEFAULTS.scales.y, min: opts.yMin, max: opts.yMax },
      },
    },
  });
}

function renderStackedBar(id, labels, datasets) {
  destroyChart(id);
  const ctx = document.getElementById(id)?.getContext("2d");
  if (!ctx) return;

  charts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels.length ? labels : ["No data"],
      datasets: datasets.map(d => ({
        label: d.label,
        data: d.data,
        backgroundColor: d.color + "bb",
        borderRadius: 3,
      })),
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { legend: { display: true, labels: { color: "#94a3b8", font: { size: 11 } } }, tooltip: CHART_DEFAULTS.plugins.tooltip },
      scales: { ...CHART_DEFAULTS.scales, x: { ...CHART_DEFAULTS.scales.x, stacked: true }, y: { ...CHART_DEFAULTS.scales.y, stacked: true } },
    },
  });
}

function renderHistogram(id, data, label, color) {
  destroyChart(id);
  const ctx = document.getElementById(id)?.getContext("2d");
  if (!ctx || !data.length) return;

  // Bin into 10 buckets
  const min = Math.min(...data), max = Math.max(...data);
  const binSize = Math.max(1, Math.ceil((max - min) / 10));
  const bins = {};
  data.forEach(v => {
    const bin = Math.floor((v - min) / binSize) * binSize + min;
    bins[bin] = (bins[bin] || 0) + 1;
  });
  const sorted = Object.entries(bins).sort((a, b) => +a[0] - +b[0]);

  charts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: sorted.map(([k]) => `${Math.round(+k)}ms`),
      datasets: [{ label, data: sorted.map(([, v]) => v), backgroundColor: color + "99", borderRadius: 3 }],
    },
    options: { ...CHART_DEFAULTS },
  });
}

function renderScoreDist(id, data, label, color) {
  destroyChart(id);
  const ctx = document.getElementById(id)?.getContext("2d");
  if (!ctx) return;

  const buckets = [1, 2, 3, 4, 5];
  const counts = buckets.map(b =>
    data.filter(v => v !== null && Math.round(v) === b).length
  );

  charts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: buckets.map(b => `${b} ★`),
      datasets: [{ label, data: counts, backgroundColor: color + "99", borderRadius: 4 }],
    },
    options: { ...CHART_DEFAULTS, scales: { ...CHART_DEFAULTS.scales, y: { ...CHART_DEFAULTS.scales.y, min: 0 } } },
  });
}

function renderRadar(id, labels, values) {
  destroyChart(id);
  const ctx = document.getElementById(id)?.getContext("2d");
  if (!ctx) return;

  charts[id] = new Chart(ctx, {
    type: "radar",
    data: {
      labels,
      datasets: [{
        label: "Agent Performance",
        data: values.map(v => Math.max(0, Math.min(5, v || 0))),
        borderColor: "#6c63ff",
        backgroundColor: "#6c63ff22",
        pointBackgroundColor: "#a78bfa",
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          min: 0, max: 5,
          ticks: { color: "#64748b", font: { size: 9 }, stepSize: 1 },
          grid: { color: "rgba(255,255,255,0.06)" },
          pointLabels: { color: "#94a3b8", font: { size: 11 } },
        },
      },
    },
  });
}

// ── Utility ───────────────────────────────────────────────────────────────────
function destroyChart(id) {
  if (charts[id]) { charts[id].destroy(); delete charts[id]; }
}

function setVal(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val ?? "—";
}

function fmt(val, dec = 0) {
  if (val == null || isNaN(val)) return "—";
  return Number(val).toFixed(dec);
}

function fmtPct(val) {
  if (val == null) return "—";
  return (val * 100).toFixed(1) + "%";
}

function colorCard(id, val, warnThreshold, badThreshold) {
  const card = document.getElementById(id);
  if (!card || val == null) return;
  if (val > badThreshold) card.style.borderColor = "rgba(239,68,68,0.3)";
  else if (val > warnThreshold) card.style.borderColor = "rgba(245,158,11,0.3)";
  else card.style.borderColor = "rgba(34,197,94,0.2)";
}

// Expose for onclick
window.showView = showView;
window.loadAll = loadAll;
window.loadDrilldown = loadDrilldown;
window.closeDrilldown = closeDrilldown;
