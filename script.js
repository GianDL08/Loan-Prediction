const tabs = document.querySelectorAll('.tab');
const pages = document.querySelectorAll('.page');
const form = document.getElementById('predictForm');
const resetBtn = document.getElementById('resetBtn');
const resultEl = document.getElementById('predictionResult');

function renderTable(records) {
  if (!Array.isArray(records) || records.length === 0) {
    return '<p>No data available.</p>';
  }

  const keys = Object.keys(records[0]);
  const header = keys.map((k) => `<th>${k}</th>`).join('');
  const rows = records
    .map(
      (row) =>
        `<tr>${keys.map((k) => `<td>${row[k] ?? ''}</td>`).join('')}</tr>`
    )
    .join('');

  return `<div class="table-wrap"><table><thead><tr>${header}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderKeyValueList(obj) {
  if (!obj || typeof obj !== 'object') return '<p>None</p>';

  return (
    '<ul>' +
    Object.entries(obj)
      .map(([k, v]) => `<li><strong>${k}</strong>: ${JSON.stringify(v)}</li>`)
      .join('') +
    '</ul>'
  );
}

function setActiveTab(targetId) {
  tabs.forEach((tab) => {
    tab.classList.toggle('active', tab.dataset.target === targetId);
  });

  pages.forEach((page) => {
    page.classList.toggle('active', page.id === targetId);
  });

  // Ensure page scrolls to top when switching sections
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // Load tab-specific data lazily
  if (targetId === 'data-cleaning') loadCleaningSummary();
  if (targetId === 'eda') loadEDASummary();
  if (targetId === 'preprocessing') loadPreprocessingSummary();
  if (targetId === 'modeling') loadModelSummary();
}

tabs.forEach((tab) => {
  tab.addEventListener('click', () => setActiveTab(tab.dataset.target));
});

function formatCurrency(value) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function getPrediction(data) {
  // Basic rule-based logic to serve as a placeholder for a real model.
  // This can be replaced by a server call to a trained ML model.
  const score = [];

  // Credit history is the strongest signal in the dataset.
  if (data.credit_history === 1) score.push(2);
  if (data.credit_history === 0) score.push(-2);

  // Higher income + lower loan amount increases approval chance.
  const income = data.totalIncome;
  const loan = data.loanAmount * 1000; // convert to units
  const dti = income > 0 ? loan / income : 100;

  if (income >= 40000) score.push(1);
  if (income < 15000) score.push(-1);
  if (dti < 6) score.push(1);
  if (dti > 12) score.push(-1);

  // Additional signals
  if (data.education === 'Graduate') score.push(1);
  if (data.selfEmployed === 'Yes') score.push(-0.5);
  if (data.propertyArea === 'Urban') score.push(0.5);

  // Build message
  const total = score.reduce((a, b) => a + b, 0);
  const isApproved = total >= 0;

  const messageLines = [];
  messageLines.push(`Credit history: ${data.credit_history === 1 ? 'Good' : 'Poor'}`);
  messageLines.push(`Total income: $${formatCurrency(income)}`);
  messageLines.push(`Loan amount: $${formatCurrency(loan)}`);
  messageLines.push(`Estimated debt-to-income: ${dti.toFixed(1)}`);

  return {
    approved: isApproved,
    message: messageLines.join('<br>'),
    confidence: Math.min(0.99, Math.max(0.35, 0.55 + total * 0.05)),
  };
}

function updatePredictionDisplay({ approved, message, confidence }) {
  resultEl.innerHTML = `
    <strong>${approved ? 'Loan likely approved' : 'Loan likely rejected'}</strong>
    <div class="meta">Confidence: ${(confidence * 100).toFixed(0)}%</div>
    <div class="details">${message}</div>
  `;

  resultEl.classList.toggle('approved', approved);
  resultEl.classList.toggle('rejected', !approved);
}

function parseNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const formData = new FormData(form);

  const payload = {
    married: formData.get('married'),
    dependents: Number(formData.get('dependents')),
    education: formData.get('education'),
    selfEmployed: formData.get('selfEmployed'),
    totalIncome: parseNumber(formData.get('totalIncome')),
    loanAmount: parseNumber(formData.get('loanAmount')),
    loanTerm: parseNumber(formData.get('loanTerm')),
    credit_history: Number(formData.get('creditHistory')),
    propertyArea: formData.get('propertyArea'),
  };

  resultEl.innerHTML = '<em>Predicting... (calling the model)</em>';
  resultEl.classList.remove('approved', 'rejected');

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    const approved = data.prediction === 1;
    const confidence = Number(data.probability ?? 0);

    updatePredictionDisplay({
      approved,
      message: `Prediction: ${data.prediction_label} (${data.probability.toFixed(2)})`,
      confidence,
    });
  } catch (error) {
    // Fallback to the local rule-based predictor when the server is unreachable.
    const prediction = getPrediction(payload);
    prediction.message +=
      '<br><br><strong>Note:</strong> Backend model call failed, falling back to local rule-based prediction.';
    updatePredictionDisplay(prediction);
  }
});

resetBtn.addEventListener('click', () => {
  form.reset();
  resultEl.textContent = 'Fill out the form and click Predict to see a result.';
  resultEl.classList.remove('approved', 'rejected');
});

async function loadCleaningSummary() {
  const el = document.getElementById('cleaningSummary');
  if (!el) return;

  el.innerHTML = '<em>Loading data summary...</em>';

  try {
    const response = await fetch('/api/summary');
    if (!response.ok) throw new Error('Failed to load dataset summary');

    const summary = await response.json();

    const missing = Object.entries(summary.missing)
      .map(([k, v]) => `<li><strong>${k}</strong>: ${v}</li>`)
      .join('');

    el.innerHTML = `
      <h3>Cleaning summary</h3>
      <p><strong>Rows</strong>: ${summary.shape[0]}, <strong>Columns</strong>: ${summary.shape[1]}</p>
      <p><strong>Duplicate rows</strong>: ${summary.duplicate_count}</p>
      <h4>Missing values</h4>
      <ul>${missing}</ul>
    `;
  } catch (error) {
    el.innerHTML = '<p class="error">Failed to load summary.</p>';
  }
}

async function loadEDASummary() {
  const el = document.getElementById('edaSummary');
  if (!el) return;

  el.innerHTML = '<em>Loading EDA data...</em>';

  try {
    const response = await fetch('/api/eda');
    if (!response.ok) throw new Error('Failed to load EDA data');

    const data = await response.json();

    const categorical = data.summary.categorical_counts || {};

    const showCounts = (obj) =>
      Object.entries(obj)
        .map(
          ([k, vals]) =>
            `<div class="section"><h4>${k}</h4>${renderKeyValueList(vals)}</div>`
        )
        .join('');

    el.innerHTML = `
      <h3>Dataset preview (cleaned)</h3>
      ${renderTable(data.head)}
      <h3>Numeric summary</h3>
      <pre>${JSON.stringify(data.summary.numeric_stats, null, 2)}</pre>
      <h3>Categorical distribution (sample)</h3>
      ${showCounts(categorical)}
    `;
  } catch (error) {
    el.innerHTML = '<p class="error">Failed to load EDA data.</p>';
  }
}

async function loadPreprocessingSummary() {
  const el = document.getElementById('preprocessingSummary');
  if (!el) return;

  el.innerHTML = '<em>Loading preprocessing details...</em>';

  try {
    const response = await fetch('/api/preprocessing');
    if (!response.ok) throw new Error('Failed to load preprocessing details');

    const data = await response.json();

    el.innerHTML = `
      <h3>Preprocessing steps</h3>
      <ul>${data.steps.map((s) => `<li>${s}</li>`).join('')}</ul>
      <h3>Raw data sample</h3>
      ${renderTable(data.raw_head)}
      <h3>Cleaned data sample</h3>
      ${renderTable(data.clean_head)}
    `;
  } catch (error) {
    el.innerHTML = '<p class="error">Failed to load preprocessing details.</p>';
  }
}

async function loadModelSummary() {
  const el = document.getElementById('modelingSummary');
  if (!el) return;

  el.innerHTML = '<em>Loading model metrics...</em>';

  try {
    const response = await fetch('/api/model');
    if (!response.ok) throw new Error('Failed to load model metrics');

    const data = await response.json();

    el.innerHTML = `
      <h3>${data.model_type}</h3>
      <p>The model is trained on cleaned and preprocessed data and evaluated using a holdout test set.</p>
      <h4>Evaluation metrics</h4>
      <pre>${JSON.stringify(data.metrics, null, 2)}</pre>
      <h4>Model features</h4>
      <pre>${JSON.stringify(data.features, null, 2)}</pre>
    `;
  } catch (error) {
    el.innerHTML = '<p class="error">Failed to load model metrics.</p>';
  }
}

// Initialize page state
setActiveTab('home');
