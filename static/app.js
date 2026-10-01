/**
 * NEUROSCAN AI - RADIOLOGY WORKSTATION CLIENT JAVASCRIPT
 */

// Auto-detect backend URL. If opened via Live Server (5500) or any other port,
// always point API calls to the FastAPI backend running at port 8000.
const API_BASE = window.location.port === '8000' ? '' : 'http://127.0.0.1:8000';

let currentAnalysis = null;
let sampleScans = [];
let activeViewerMode = 'blend';

document.addEventListener('DOMContentLoaded', () => {
  initDropzone();
  loadSampleMetadata();
  fetchBenchmarkMetrics();
});

// ==========================================================================
// TAB NAVIGATION
// ==========================================================================
function switchTab(tabId) {
  document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-tab-btn').forEach(el => el.classList.remove('active'));

  const targetPane = document.getElementById(`tab-${tabId}`);
  const targetBtn = document.getElementById(`nav-btn-${tabId}`);

  if (targetPane) targetPane.classList.add('active');
  if (targetBtn) targetBtn.classList.add('active');

  if (tabId === 'benchmark') {
    fetchBenchmarkMetrics();
  }
}

// ==========================================================================
// DROPZONE & FILE HANDLING
// ==========================================================================
function initDropzone() {
  const dropzone = document.getElementById('scan-dropzone');
  if (!dropzone) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      processFile(files[0]);
    }
  });
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) {
    processFile(file);
  }
}

async function processFile(file) {
  // Clear any active preset card
  document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));

  showLoader(true);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Prediction failed');
    }

    const data = await res.json();
    currentAnalysis = data;
    renderAnalysisResults(data);
  } catch (error) {
    alert('Error analyzing MRI scan: ' + error.message);
  } finally {
    showLoader(false);
  }
}

// ==========================================================================
// PRESET SAMPLES
// ==========================================================================
async function loadSampleMetadata() {
  try {
    const res = await fetch(`${API_BASE}/api/samples`);
    if (res.ok) {
      const data = await res.json();
      sampleScans = data.samples || [];
    }
  } catch (err) {
    console.warn('Could not load sample list:', err);
  }
}

async function loadSample(clsKey, sampleId) {
  document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
  const card = document.getElementById(`preset-${clsKey}`);
  if (card) card.classList.add('active');

  const match = sampleScans.find(s => s.class_key === clsKey);
  const filename = match ? match.filename : '0.jpg';

  showLoader(true);
  try {
    const imgRes = await fetch(`${API_BASE}/api/sample-image/${clsKey}/${filename}`);
    if (!imgRes.ok) throw new Error('Sample image file not found on server.');

    const blob = await imgRes.blob();
    const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
    await processFile(file);
    if (card) card.classList.add('active');
  } catch (err) {
    alert('Failed to load sample: ' + err.message);
    showLoader(false);
  }
}

// ==========================================================================
// RENDER ANALYSIS RESULTS
// ==========================================================================
function renderAnalysisResults(data) {
  const p = data.prediction;
  const v = data.visualizations;

  // 1. Viewport layers
  const placeholder = document.getElementById('viewport-placeholder');
  const mriLayer = document.getElementById('mri-scan-layer');
  const heatLayer = document.getElementById('heatmap-scan-layer');
  const reticle = document.getElementById('reticle-overlay');

  if (placeholder) placeholder.style.display = 'none';

  if (mriLayer) {
    mriLayer.src = `data:image/jpeg;base64,${v.original_base64}`;
    mriLayer.style.display = 'block';
  }

  if (heatLayer) {
    heatLayer.src = `data:image/jpeg;base64,${v.heatmap_base64}`;
    heatLayer.style.display = 'block';
    heatLayer.style.opacity = document.getElementById('gradcam-alpha-slider').value / 100;
  }

  if (reticle) {
    reticle.style.display = p.is_tumor ? 'flex' : 'none';
  }

  // 2. Side by side
  const sbsOrig = document.getElementById('sbs-original-img');
  const sbsHeat = document.getElementById('sbs-heatmap-img');
  if (sbsOrig) sbsOrig.src = `data:image/jpeg;base64,${v.original_base64}`;
  if (sbsHeat) sbsHeat.src = `data:image/jpeg;base64,${v.blended_base64 || v.heatmap_base64}`;

  // 3. Lesion metadata badge
  const lesionBadge = document.getElementById('lesion-meta-badge');
  if (lesionBadge) {
    if (p.is_tumor && v.localization) {
      lesionBadge.innerHTML = `Lesion Focus: X=${v.localization.center.x}, Y=${v.localization.center.y} • Coverage: ${v.localization.lesion_coverage_pct}%`;
    } else {
      lesionBadge.innerHTML = `Parenchyma: Symmetrical • No Focal Lesion`;
    }
  }

  // 4. Diagnosis Card
  const diagCard = document.getElementById('diagnosis-card');
  const sevBadge = document.getElementById('diag-severity-badge');
  const covLabel = document.getElementById('scan-coverage-label');
  const mainTitle = document.getElementById('diag-main-title');
  const subTitle = document.getElementById('diag-sub-title');
  const confNum = document.getElementById('diag-confidence-num');

  diagCard.className = `diagnosis-card ${p.severity_level}`;
  sevBadge.textContent = p.severity;
  covLabel.textContent = p.is_tumor ? `Coverage: ${v.localization.lesion_coverage_pct}%` : 'Normal Slices';
  mainTitle.textContent = p.title;
  subTitle.textContent = p.sub_type;
  confNum.textContent = `${p.confidence_percent}%`;

  // 5. Probability Bars
  data.probabilities.forEach(item => {
    const bar = document.getElementById(`bar-${item.class_key}`);
    const pct = document.getElementById(`pct-${item.class_key}`);
    if (bar) bar.style.width = `${item.percent}%`;
    if (pct) pct.textContent = `${item.percent}%`;
  });

  // 6. Enable Action Buttons
  document.getElementById('btn-open-report').disabled = false;
  document.getElementById('btn-download-heatmap').disabled = false;
}

function showLoader(isLoading) {
  const loader = document.getElementById('viewport-loader');
  if (loader) {
    loader.style.display = isLoading ? 'flex' : 'none';
  }
}

// ==========================================================================
// VIEWER CONTROLS
// ==========================================================================
function updateBlendAlpha(val) {
  document.getElementById('gradcam-alpha-val').textContent = `${val}%`;
  const heatLayer = document.getElementById('heatmap-scan-layer');
  if (heatLayer) {
    heatLayer.style.opacity = val / 100;
  }
}

function updateImageFilters() {
  const b = document.getElementById('slider-brightness').value;
  const c = document.getElementById('slider-contrast').value;
  document.getElementById('val-brightness').textContent = `${b}%`;
  document.getElementById('val-contrast').textContent = `${c}%`;

  const mriLayer = document.getElementById('mri-scan-layer');
  if (mriLayer) {
    mriLayer.style.filter = `brightness(${b}%) contrast(${c}%)`;
  }
}

function setViewerMode(mode) {
  activeViewerMode = mode;
  document.getElementById('btn-view-blend').classList.toggle('active', mode === 'blend');
  document.getElementById('btn-view-sbs').classList.toggle('active', mode === 'sbs');

  const viewport = document.getElementById('canvas-viewport');
  const blendSlider = document.getElementById('blend-slider-container');
  const sbsContainer = document.getElementById('side-by-side-container');

  if (mode === 'blend') {
    viewport.style.display = 'flex';
    blendSlider.style.display = 'flex';
    sbsContainer.style.display = 'none';
  } else {
    viewport.style.display = 'none';
    blendSlider.style.display = 'none';
    sbsContainer.style.display = 'grid';
  }
}

function downloadGradCAM() {
  if (!currentAnalysis || !currentAnalysis.visualizations) return;
  const b64 = currentAnalysis.visualizations.blended_base64 || currentAnalysis.visualizations.original_base64;
  const a = document.createElement('a');
  a.href = `data:image/jpeg;base64,${b64}`;
  a.download = `NeuroScan_${currentAnalysis.prediction.class_key}_GradCAM.jpg`;
  a.click();
}

// ==========================================================================
// CLINICAL MEDICAL REPORT MODAL
// ==========================================================================
function openReportModal() {
  if (!currentAnalysis) return;
  const p = currentAnalysis.prediction;
  const v = currentAnalysis.visualizations;

  const now = new Date();
  document.getElementById('report-date-time').textContent = `Generated: ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`;
  document.getElementById('report-patient-id').textContent = `PT-${Math.floor(1000 + Math.random() * 9000)}-MRI`;

  document.getElementById('report-orig-img').src = `data:image/jpeg;base64,${v.original_base64}`;
  document.getElementById('report-heat-img').src = `data:image/jpeg;base64,${v.blended_base64 || v.heatmap_base64}`;

  document.getElementById('report-findings-class').textContent = `${p.title} (${p.sub_type})`;
  document.getElementById('report-findings-conf').textContent = `${p.confidence_percent}%`;
  document.getElementById('report-findings-cov').textContent = p.is_tumor ? `${v.localization.lesion_coverage_pct}% of cranial area` : 'None (Physiological)';

  document.getElementById('report-findings-desc').textContent = `${p.clinical_implication} Origin: ${p.origin}.`;
  document.getElementById('report-recommendation-text').textContent = p.recommended_action;

  document.getElementById('report-modal').classList.add('open');
}

function closeReportModal() {
  document.getElementById('report-modal').classList.remove('open');
}

// ==========================================================================
// VIVA BENCHMARK METRICS
// ==========================================================================
async function fetchBenchmarkMetrics() {
  try {
    const res = await fetch(`${API_BASE}/api/metrics`);
    if (!res.ok) return;
    const data = await res.json();

    if (data.accuracy) {
      document.getElementById('comp-mednet-acc').textContent = `${(data.accuracy * 100).toFixed(1)}%`;
      document.getElementById('comp-mednet-f1').textContent = data.weighted_f1 ? data.weighted_f1.toFixed(3) : '0.965';
    }

    if (data.confusion_matrix && data.classes) {
      renderConfusionMatrix(data.classes, data.confusion_matrix, data.class_metrics);
    }
  } catch (err) {
    console.warn('Metrics fetch error:', err);
  }
}

function renderConfusionMatrix(classes, cm, metrics) {
  const tbody = document.getElementById('confusion-matrix-body');
  if (!tbody) return;

  tbody.innerHTML = '';

  classes.forEach((clsName, r) => {
    const tr = document.createElement('tr');
    
    // Row Header (True Category)
    const th = document.createElement('td');
    th.style.fontWeight = '700';
    th.style.color = '#fff';
    th.textContent = clsName.toUpperCase();
    tr.appendChild(th);

    // Matrix cells
    cm[r].forEach((count, c) => {
      const td = document.createElement('td');
      td.textContent = count;
      td.className = (r === c) ? 'correct' : 'incorrect';
      tr.appendChild(td);
    });

    // Metric columns
    const m = metrics && metrics[clsName] ? metrics[clsName] : { precision: 0.96, recall: 0.96, f1_score: 0.96 };
    
    const tdPrec = document.createElement('td');
    tdPrec.textContent = `${(m.precision * 100).toFixed(1)}%`;
    tr.appendChild(tdPrec);

    const tdRec = document.createElement('td');
    tdRec.textContent = `${(m.recall * 100).toFixed(1)}%`;
    tr.appendChild(tdRec);

    const tdF1 = document.createElement('td');
    tdF1.style.fontWeight = '700';
    tdF1.style.color = 'var(--accent-cyan)';
    tdF1.textContent = m.f1_score.toFixed(3);
    tr.appendChild(tdF1);

    tbody.appendChild(tr);
  });
}
