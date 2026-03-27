/* ============================================================
   Farmers EA Analytics — Dashboard JavaScript
   ============================================================ */

// ── App State ───────────────────────────────────────────────────────────────
const state = {
  activeTab: 'tab-overview',
  pipelineStatus: 'idle',
  pollTimer: null,
  ws: null,
  dataLoaded: false,
};

const PLOTLY_LAYOUT = {
  paper_bgcolor: '#21262D',
  plot_bgcolor:  '#21262D',
  font: { color: '#E6EDF3', family: 'Inter, sans-serif', size: 12 },
  margin: { t: 30, r: 20, b: 40, l: 60 },
  xaxis: { gridcolor: '#30363D', zerolinecolor: '#30363D' },
  yaxis: { gridcolor: '#30363D', zerolinecolor: '#30363D' },
  legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } },
  colorway: ['#E8290B','#58A6FF','#3FB950','#F5A623','#BC8CFF','#F85149'],
};

const PLOTLY_CFG = { responsive: true, displayModeBar: false };

const COLORS = {
  red: '#E8290B', blue: '#58A6FF', green: '#3FB950',
  gold: '#F5A623', purple: '#BC8CFF', warn: '#F85149',
};

// ── Utilities ───────────────────────────────────────────────────────────────
const fmtCurrency = (v) => {
  if (v == null) return '—';
  const abs = Math.abs(v);
  if (abs >= 1e9) return (v < 0 ? '-' : '') + '$' + (abs / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6) return (v < 0 ? '-' : '') + '$' + (abs / 1e6).toFixed(1) + 'M';
  if (abs >= 1e3) return (v < 0 ? '-' : '') + '$' + (abs / 1e3).toFixed(1) + 'K';
  return '$' + v.toFixed(0);
};

const fmtPct = (v) => v == null ? '—' : (v * 100).toFixed(1) + '%';
const fmtNumber = (v) => v == null ? '—' : Number(v).toLocaleString();
const fmtPctSigned = (v) => v == null ? '—' : (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%';

const priorityClass = (cls) => {
  if (!cls) return 'priority-maint';
  if (cls.includes('High'))   return 'priority-high';
  if (cls.includes('Medium')) return 'priority-medium';
  if (cls.includes('Low'))    return 'priority-low';
  return 'priority-maint';
};

const rankBadge = (r) => {
  const cls = r === 1 ? 'rank-1' : r === 2 ? 'rank-2' : r === 3 ? 'rank-3' : 'rank-n';
  return `<span class="rank-badge ${cls}">${r}</span>`;
};

const el = (id) => document.getElementById(id);

// ── Tab Navigation ──────────────────────────────────────────────────────────
const switchTab = (tabId) => {
  state.activeTab = tabId;
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  document.querySelectorAll('.tab-panel').forEach(panel => {
    panel.classList.toggle('active', panel.id === tabId);
  });
};

// ── Pipeline Control ────────────────────────────────────────────────────────
const runPipeline = async () => {
  const btn = el('btn-run');
  btn.disabled = true;
  btn.textContent = 'Running…';

  const progBar = el('pipeline-progress');
  progBar.classList.remove('hidden');
  el('progress-fill').style.width = '0%';

  try {
    await fetch('/api/pipeline/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ shock_params: null, forecast_horizon: 12 }),
    });
    startPolling();
  } catch (e) {
    console.error('Pipeline start failed:', e);
    btn.disabled = false;
    btn.textContent = '▶ Run Pipeline';
  }
};

const startPolling = () => {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(pollStatus, 1500);
};

const pollStatus = async () => {
  try {
    const res = await fetch('/api/pipeline/status');
    const data = await res.json();
    updatePipelineStatus(data);
    if (data.state === 'complete' || data.state === 'error') {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      if (data.state === 'complete') loadAllData();
    }
  } catch (e) { console.error('Poll error:', e); }
};

const updatePipelineStatus = (data) => {
  state.pipelineStatus = data.state;
  const dot = el('status-dot');
  const txt = el('status-text');
  const btn = el('btn-run');
  const fill = el('progress-fill');
  const bar = el('pipeline-progress');

  dot.className = 'status-dot ' + (data.state || 'idle');
  if (data.state === 'running') {
    txt.textContent = data.current_agent || 'Running…';
    bar.classList.remove('hidden');
    fill.style.width = (data.progress || 0) + '%';
    btn.disabled = true;
    btn.textContent = 'Running…';
  } else if (data.state === 'complete') {
    txt.textContent = 'Complete';
    fill.style.width = '100%';
    setTimeout(() => bar.classList.add('hidden'), 2000);
    btn.disabled = false;
    btn.textContent = '▶ Run Pipeline';
  } else if (data.state === 'error') {
    txt.textContent = 'Error';
    btn.disabled = false;
    btn.textContent = '▶ Run Pipeline';
  } else {
    txt.textContent = 'Idle';
    btn.disabled = false;
    btn.textContent = '▶ Run Pipeline';
  }
};

// ── WebSocket ───────────────────────────────────────────────────────────────
const connectWS = () => {
  try {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    state.ws = new WebSocket(`${proto}://${location.host}/ws/pipeline`);
    state.ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      if (msg.type === 'status')   updatePipelineStatus(msg.data);
      if (msg.type === 'progress') {
        el('progress-fill').style.width = (msg.progress || 0) + '%';
        el('status-text').textContent = msg.message || '';
      }
      if (msg.type === 'complete') {
        updatePipelineStatus({ state: 'complete', progress: 100 });
        loadAllData();
      }
      if (msg.type === 'error') {
        updatePipelineStatus({ state: 'error' });
      }
    };
    state.ws.onclose = () => setTimeout(connectWS, 5000);
  } catch (e) { console.warn('WS failed:', e); }
};

// ── Load All Data ───────────────────────────────────────────────────────────
const loadAllData = async () => {
  state.dataLoaded = true;
  await Promise.allSettled([
    loadKPIs(),
    loadPortfolioTrend(),
    loadStateMap(),
    loadForecasts(),
    loadOutliers(),
    loadOpportunities(),
    loadLeaderboard(),
    loadCausal(),
    loadTwin(),
    loadShap(),
    loadInsights(),
    loadReport(),
  ]);
};

// ── KPIs ────────────────────────────────────────────────────────────────────
const loadKPIs = async () => {
  try {
    const res = await fetch('/api/data/kpis');
    const d = await res.json();
    setKPI('kpi-gwp',      'Total GWP',           fmtCurrency(d.gwp),         `${d.gwp_attainment}% of target`);
    setKPI('kpi-pif',      'Policies in Force',    fmtNumber(d.pif),           `${d.pif_attainment}% of target`);
    setKPI('kpi-upside',   'GWP Opportunity',      fmtCurrency(d.gwp_upside),  `${d.high_priority_agents} high-priority agents`);
    setKPI('kpi-agents',   'Active Agents',        fmtNumber(d.n_agents),       `${fmtNumber(d.n_policies)} policies`);
    setKPI('kpi-renewal',  'Renewal Rate',         fmtPct(d.renewal_rate),      `Cancel: ${fmtPct(d.cancellation_rate)}`);
    setKPI('kpi-outliers', 'Outliers Flagged',     fmtNumber(d.outliers_flagged), `${fmtPct(d.anomaly_rate)} anomaly rate`);
  } catch (e) { console.error('KPI load error:', e); }
};

const setKPI = (id, label, value, sub) => {
  const card = el(id);
  if (!card) return;
  card.querySelector('.kpi-label').textContent = label;
  card.querySelector('.kpi-value').textContent = value;
  card.querySelector('.kpi-sub').textContent = sub;
};

// ── Portfolio Trend ─────────────────────────────────────────────────────────
const loadPortfolioTrend = async () => {
  try {
    const res = await fetch('/api/data/portfolio-trend');
    const data = await res.json();
    if (!data.length) return;
    const x = data.map(d => d.year_month);
    Plotly.newPlot('chart-portfolio-trend', [
      { x, y: data.map(d => d.gwp),  name: 'GWP',  type: 'scatter', mode: 'lines', line: { color: COLORS.red, width: 2.5 } },
      { x, y: data.map(d => d.pif),  name: 'PIF',  type: 'scatter', mode: 'lines', line: { color: COLORS.blue, width: 2 }, yaxis: 'y2' },
    ], {
      ...PLOTLY_LAYOUT,
      yaxis:  { ...PLOTLY_LAYOUT.yaxis, title: 'GWP ($)' },
      yaxis2: { ...PLOTLY_LAYOUT.yaxis, title: 'PIF', overlaying: 'y', side: 'right', gridcolor: 'transparent' },
    }, PLOTLY_CFG);
  } catch (e) { console.error(e); }
};

// ── State Map ───────────────────────────────────────────────────────────────
const loadStateMap = async () => {
  try {
    const res = await fetch('/api/data/state-map');
    const data = await res.json();
    if (!data.length) return;
    const sorted = data.sort((a, b) => a.total_gwp - b.total_gwp);
    Plotly.newPlot('chart-state-map', [{
      y: sorted.map(d => d.state),
      x: sorted.map(d => d.total_gwp),
      type: 'bar',
      orientation: 'h',
      marker: { color: sorted.map((_, i) => {
        const t = i / (sorted.length - 1 || 1);
        return `rgb(${Math.round(88 + 144 * t)}, ${Math.round(166 - 80 * t)}, ${Math.round(255 - 244 * t)})`;
      })},
      hovertemplate: '%{y}: %{x:$,.0f}<extra></extra>',
    }], { ...PLOTLY_LAYOUT, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Total GWP ($)' } }, PLOTLY_CFG);
  } catch (e) { console.error(e); }
};

// ── Forecasts ───────────────────────────────────────────────────────────────
const loadForecasts = async () => {
  try {
    const res = await fetch('/api/data/forecasts');
    const data = await res.json();
    renderForecastChart(data, 'portfolio_gwp', 'gwp', 'chart-forecast-gwp');
    renderForecastChart(data, 'portfolio_pif', 'pif', 'chart-forecast-pif');
  } catch (e) { console.error(e); }
};

const renderForecastChart = (data, key, metric, chartId) => {
  const series = data[key];
  if (!series || !series.length) return;
  const x = series.map(d => d.ds);
  const fcCol = metric + '_forecast';
  const loCol = metric + '_lower';
  const hiCol = metric + '_upper';
  const traces = [
    { x, y: series.map(d => d[fcCol]), name: 'Forecast', mode: 'lines', line: { color: COLORS.red, width: 2.5 } },
  ];
  if (series[0][loCol] != null) {
    traces.push(
      { x, y: series.map(d => d[hiCol]), name: 'Upper CI', mode: 'lines', line: { width: 0 }, showlegend: false },
      { x, y: series.map(d => d[loCol]), name: 'Lower CI', fill: 'tonexty', mode: 'lines',
        line: { width: 0 }, fillcolor: 'rgba(232,41,11,0.12)', showlegend: false },
    );
  }
  Plotly.newPlot(chartId, traces, { ...PLOTLY_LAYOUT, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: metric.toUpperCase() + ' ($)' } }, PLOTLY_CFG);
};

// ── Outliers ────────────────────────────────────────────────────────────────
const loadOutliers = async () => {
  try {
    const res = await fetch('/api/data/outliers');
    const { records, summary } = await res.json();
    const tbody = el('outlier-tbody');
    if (!tbody) return;
    tbody.innerHTML = records.slice(0, 100).map(r => `
      <tr>
        <td>${r.agent_name || r.agent_id || '—'}</td>
        <td>${r.state || '—'}</td>
        <td>${r.year_month || '—'}</td>
        <td>${fmtCurrency(r.gwp)}</td>
        <td>${fmtPct(r.cancellation_rate)}</td>
        <td><span class="priority-pill ${r.outlier_type === 'GWP Crash' ? 'priority-high' : r.outlier_type === 'GWP Spike' ? 'priority-medium' : 'priority-low'}">${r.outlier_type || '—'}</span></td>
        <td>${r.outlier_score || '—'}</td>
      </tr>`).join('');

    // Pie chart
    if (summary.by_type) {
      const labels = Object.keys(summary.by_type);
      const values = Object.values(summary.by_type);
      Plotly.newPlot('chart-outlier-types', [{
        labels, values, type: 'pie', hole: 0.45,
        marker: { colors: [COLORS.red, COLORS.blue, COLORS.gold, COLORS.green, COLORS.purple] },
        textinfo: 'label+percent', textfont: { size: 11 },
      }], { ...PLOTLY_LAYOUT, showlegend: false, margin: { t: 10, r: 10, b: 10, l: 10 } }, PLOTLY_CFG);
    }
  } catch (e) { console.error(e); }
};

// ── Opportunities ───────────────────────────────────────────────────────────
const loadOpportunities = async () => {
  try {
    const res = await fetch('/api/data/opportunities');
    const { records } = await res.json();
    const tbody = el('opportunity-tbody');
    if (!tbody || !records.length) return;
    tbody.innerHTML = records.slice(0, 50).map(r => `
      <tr>
        <td>${rankBadge(r.rank)}</td>
        <td>${r.agent_name || r.agent_id}</td>
        <td>${r.state}</td>
        <td>${r.district}</td>
        <td>${fmtCurrency(r.gwp)}</td>
        <td>${(r.composite_score || 0).toFixed(3)}</td>
        <td>${fmtCurrency(r.gwp_upside_est)}</td>
        <td><span class="priority-pill ${priorityClass(r.opportunity_class)}">${r.opportunity_class || '—'}</span></td>
      </tr>`).join('');

    // Scatter
    Plotly.newPlot('chart-opportunity-scatter', [{
      x: records.map(r => r.composite_score),
      y: records.map(r => r.gwp_upside_est),
      text: records.map(r => r.agent_name || r.agent_id),
      mode: 'markers',
      marker: { size: 7, color: records.map(r => r.composite_score), colorscale: [[0,'#58A6FF'],[1,'#E8290B']], showscale: true, colorbar: { title: 'Score', thickness: 12 } },
      hovertemplate: '<b>%{text}</b><br>Score: %{x:.3f}<br>Upside: %{y:$,.0f}<extra></extra>',
    }], {
      ...PLOTLY_LAYOUT,
      xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Composite Score' },
      yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'GWP Upside ($)' },
    }, PLOTLY_CFG);
  } catch (e) { console.error(e); }
};

// ── Leaderboard ─────────────────────────────────────────────────────────────
const loadLeaderboard = async () => {
  try {
    const res = await fetch('/api/data/leaderboard');
    const data = await res.json();
    const tbody = el('leaderboard-tbody');
    if (!tbody || !data.length) return;
    tbody.innerHTML = data.map(r => `
      <tr>
        <td>${rankBadge(r.rank)}</td>
        <td>${r.agent_name || r.agent_id}</td>
        <td>${r.state}</td>
        <td>${fmtCurrency(r.gwp)}</td>
        <td>${(r.composite_score || 0).toFixed(3)}</td>
        <td>${fmtCurrency(r.gwp_upside_est)}</td>
        <td><span class="priority-pill ${priorityClass(r.opportunity_class)}">${r.opportunity_class || '—'}</span></td>
      </tr>`).join('');
  } catch (e) { console.error(e); }
};

// ── Causal ──────────────────────────────────────────────────────────────────
const loadCausal = async () => {
  try {
    const res = await fetch('/api/data/causal');
    const { ate_results, causal_summary } = await res.json();
    const listEl = el('ate-list');
    const summaryEl = el('causal-summary-text');
    if (summaryEl) summaryEl.textContent = causal_summary || 'No causal summary available.';

    if (!ate_results || !listEl) return;
    const entries = Object.entries(ate_results);
    listEl.innerHTML = entries.map(([label, r]) => {
      const ate = r.ate;
      const p = r.p_value;
      const sig = p != null && p < 0.05;
      const pctW = Math.min(100, Math.abs(ate || 0) / (Math.max(...entries.map(e => Math.abs(e[1].ate || 1)))) * 100);
      return `
        <div class="ate-row">
          <div class="ate-label">${label}</div>
          <div class="ate-bar-wrap"><div class="ate-bar ${ate >= 0 ? 'pos' : 'neg'}" style="width:${pctW}%"></div></div>
          <div class="ate-value ${ate >= 0 ? 'text-green' : 'text-red'}">${ate != null ? fmtCurrency(ate) : '—'}</div>
          <span class="ate-sig ${sig ? 'sig' : 'nonsig'}">${sig ? 'Significant' : 'Not sig.'}</span>
        </div>`;
    }).join('');

    // Bar chart
    Plotly.newPlot('chart-ate', [{
      y: entries.map(e => e[0]),
      x: entries.map(e => e[1].ate || 0),
      type: 'bar', orientation: 'h',
      marker: { color: entries.map(e => (e[1].ate || 0) >= 0 ? COLORS.green : COLORS.warn) },
      hovertemplate: '%{y}: %{x:$,.0f}<extra></extra>',
    }], { ...PLOTLY_LAYOUT, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'ATE ($)' } }, PLOTLY_CFG);
  } catch (e) { console.error(e); }
};

// ── Digital Twin ────────────────────────────────────────────────────────────
const loadTwin = async () => {
  try {
    const res = await fetch('/api/data/twin');
    const { scenarios, summary } = await res.json();
    const summaryEl = el('twin-summary-text');
    if (summaryEl) summaryEl.textContent = summary || '';

    if (!scenarios || !scenarios.length) return;
    const scenarioNames = [...new Set(scenarios.map(s => s.scenario))];
    const scColors = { baseline: COLORS.blue, optimistic: COLORS.green, stress: COLORS.warn };

    // GWP
    Plotly.newPlot('chart-twin-gwp',
      scenarioNames.map(sc => {
        const pts = scenarios.filter(s => s.scenario === sc);
        return { x: pts.map(p => p.month), y: pts.map(p => p.total_gwp), name: sc, mode: 'lines', line: { color: scColors[sc] || COLORS.purple, width: 2 } };
      }),
      { ...PLOTLY_LAYOUT, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'GWP ($)' } }, PLOTLY_CFG);

    // PIF
    Plotly.newPlot('chart-twin-pif',
      scenarioNames.map(sc => {
        const pts = scenarios.filter(s => s.scenario === sc);
        return { x: pts.map(p => p.month), y: pts.map(p => p.total_pif), name: sc, mode: 'lines', line: { color: scColors[sc] || COLORS.purple, width: 2 } };
      }),
      { ...PLOTLY_LAYOUT, yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'PIF' } }, PLOTLY_CFG);
  } catch (e) { console.error(e); }
};

// ── SHAP ────────────────────────────────────────────────────────────────────
const loadShap = async () => {
  try {
    const res = await fetch('/api/data/shap');
    const { global_importance } = await res.json();
    if (!global_importance || !global_importance.length) return;
    const sorted = [...global_importance].sort((a, b) => a.importance - b.importance);
    Plotly.newPlot('chart-shap', [{
      y: sorted.map(d => d.feature),
      x: sorted.map(d => d.importance),
      type: 'bar', orientation: 'h',
      marker: { color: sorted.map((_, i) => {
        const t = i / (sorted.length - 1 || 1);
        return `rgb(${Math.round(88 + 144 * t)}, ${Math.round(166 - 117 * t)}, ${Math.round(255 - 204 * t)})`;
      })},
    }], { ...PLOTLY_LAYOUT, xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'SHAP Importance' }, margin: { ...PLOTLY_LAYOUT.margin, l: 140 } }, PLOTLY_CFG);
  } catch (e) { console.error(e); }
};

// ── Insights ────────────────────────────────────────────────────────────────
const loadInsights = async () => {
  try {
    const res = await fetch('/api/data/insights');
    const { insights, learning_update } = await res.json();
    const listEl = el('insights-list');
    if (listEl && insights) {
      listEl.innerHTML = insights.map(t =>
        `<div class="insight-item">${t.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')}</div>`
      ).join('');
    }
    const learnEl = el('learning-status');
    if (learnEl && learning_update) {
      const bias = learning_update.bias_corrections || {};
      learnEl.innerHTML = `
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;">
          <div class="impact-tile"><div class="impact-label">GWP Bias</div><div class="impact-value neutral">${(bias.portfolio_gwp || 0).toFixed(4)}</div></div>
          <div class="impact-tile"><div class="impact-label">PIF Bias</div><div class="impact-value neutral">${(bias.portfolio_pif || 0).toFixed(4)}</div></div>
          <div class="impact-tile"><div class="impact-label">Memories</div><div class="impact-value neutral">${learning_update.n_memories || 0}</div></div>
          <div class="impact-tile"><div class="impact-label">Feedback</div><div class="impact-value neutral">${Object.keys(learning_update.feedback || {}).length} metrics</div></div>
        </div>`;
    }
  } catch (e) { console.error(e); }
};

// ── Report ──────────────────────────────────────────────────────────────────
const loadReport = async () => {
  try {
    const res = await fetch('/api/report');
    const { report } = await res.json();
    const body = el('report-body');
    if (body) body.textContent = report || 'No report generated.';
  } catch (e) { console.error(e); }
};

// ── Shock Simulation ────────────────────────────────────────────────────────
const sliderIds = ['slider-conversion','slider-retention','slider-pricing','slider-producers','slider-discount'];
const valIds    = ['val-conversion','val-retention','val-pricing','val-producers','val-discount'];
const paramKeys = ['conversion_delta','retention_delta','pricing_delta','producer_count_delta','discount_delta'];

const initSliders = () => {
  sliderIds.forEach((sid, i) => {
    const slider = el(sid);
    if (!slider) return;
    slider.addEventListener('input', () => {
      el(valIds[i]).textContent = fmtPctSigned(parseFloat(slider.value));
    });
  });
};

const resetSliders = () => {
  sliderIds.forEach((sid, i) => {
    const slider = el(sid);
    if (slider) { slider.value = 0; el(valIds[i]).textContent = '+0.0%'; }
  });
  const grid = el('shock-impact');
  if (grid) grid.innerHTML = '';
};

const runShockSimulation = async () => {
  const body = {};
  sliderIds.forEach((sid, i) => { body[paramKeys[i]] = parseFloat(el(sid)?.value || 0); });

  try {
    const res = await fetch('/api/simulate/shock', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const { shock_results, shock_summary } = await res.json();
    if (!shock_results || !shock_results.scenarios) return;

    // Impact tiles for user scenario
    const user = shock_results.scenarios.user_scenario?.impact;
    const grid = el('shock-impact');
    if (grid && user) {
      grid.innerHTML = [
        { label: 'GWP Impact', value: user.gwp_delta, pct: user.gwp_delta_pct },
        { label: 'PIF Impact', value: user.pif_delta, pct: user.pif_delta_pct },
        { label: 'NB Change',  value: user.nb_delta },
        { label: 'Renewal Change', value: user.renewal_delta },
      ].map(t => {
        const cls = t.value > 0 ? 'positive' : t.value < 0 ? 'negative' : 'neutral';
        return `<div class="impact-tile">
          <div class="impact-label">${t.label}</div>
          <div class="impact-value ${cls}">${fmtCurrency(t.value)}${t.pct != null ? ' (' + fmtPctSigned(t.pct) + ')' : ''}</div>
        </div>`;
      }).join('');
    }

    // Comparison bar chart
    const scNames = Object.keys(shock_results.scenarios);
    Plotly.newPlot('chart-shock-comparison', [{
      x: scNames.map(s => s.replace(/_/g, ' ')),
      y: scNames.map(s => shock_results.scenarios[s]?.impact?.gwp_delta_pct || 0),
      type: 'bar',
      marker: { color: scNames.map(s => {
        const v = shock_results.scenarios[s]?.impact?.gwp_delta_pct || 0;
        return v >= 0 ? COLORS.green : COLORS.warn;
      })},
      text: scNames.map(s => fmtPctSigned(shock_results.scenarios[s]?.impact?.gwp_delta_pct || 0)),
      textposition: 'outside', textfont: { size: 11 },
      hovertemplate: '%{x}: %{y:.1%}<extra></extra>',
    }], {
      ...PLOTLY_LAYOUT,
      yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'GWP Delta (%)', tickformat: '.0%' },
    }, PLOTLY_CFG);
  } catch (e) { console.error('Shock sim error:', e); }
};

// make global for onclick handlers in HTML
window.runShockSimulation = runShockSimulation;
window.resetSliders = resetSliders;

// ── Initialisation ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Tab click handlers
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Run button
  el('btn-run')?.addEventListener('click', runPipeline);

  // Sliders
  initSliders();

  // WebSocket
  connectWS();

  // Check if pipeline already ran
  try {
    const res = await fetch('/api/pipeline/status');
    const data = await res.json();
    updatePipelineStatus(data);
    if (data.state === 'complete') loadAllData();
  } catch (e) { /* server not ready */ }
});
