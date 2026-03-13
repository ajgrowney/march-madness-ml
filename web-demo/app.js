// Simple TF.js demo loader for a saved Layers model.
const MODEL_PATH = 'models/saved_linear_tfjs/model.json';

let model = null;
let inBrowserModel = null;
let inBrowserSchema = null;

async function init() {
  const statusEl = document.getElementById('model-status');
  try {
    statusEl.textContent = 'loading...';
    model = await tf.loadLayersModel(MODEL_PATH);
    statusEl.textContent = 'loaded';
    document.getElementById('predict').disabled = false;
    document.getElementById('random').disabled = false;
    // enable in-browser training once saved model is available
    document.getElementById('train-in').disabled = false;
    document.getElementById('compare').disabled = false;
    document.getElementById('instantiate').disabled = false;
    document.getElementById('export').disabled = false;
  } catch (err) {
    console.error(err);
    statusEl.textContent = `error: ${err.message}`;
  }
}

function readInputs() {
  const f1 = parseFloat(document.getElementById('f1').value) || 0;
  const f2 = parseFloat(document.getElementById('f2').value) || 0;
  const f3 = parseFloat(document.getElementById('f3').value) || 0;
  return [f1, f2, f3];
}

function showPrediction(val) {
  document.getElementById('prediction').textContent = Number(val).toFixed(4);
}

document.addEventListener('DOMContentLoaded', () => {
  init();

  document.getElementById('predict').addEventListener('click', async () => {
    if (!model) return;
    const x = readInputs();
    const t = tf.tensor2d([x]);
    const pred = model.predict(t);
    const v = (await pred.data())[0];
    showPrediction(v);
    t.dispose();
    pred.dispose();
  });

  document.getElementById('random').addEventListener('click', () => {
    document.getElementById('f1').value = (Math.random() - 0.5) * 2;
    document.getElementById('f2').value = (Math.random() - 0.5) * 2;
    document.getElementById('f3').value = (Math.random() - 0.5) * 2;
  });

  // Instantiate in-browser model from selector
  document.getElementById('instantiate').addEventListener('click', () => {
    const type = document.getElementById('model-type').value;
    const inputDim = Math.max(1, parseInt(document.getElementById('input-dim').value, 10) || 1);
    inBrowserModel = buildInBrowserModel(inputDim, type);
    inBrowserSchema = { type, inputDim };
    document.getElementById('inModelStatus').textContent = 'instantiated';
    document.getElementById('train-in').disabled = false;
    document.getElementById('compare').disabled = false;
    document.getElementById('export').disabled = false;
  });

  // Export in-browser model
  document.getElementById('export').addEventListener('click', async () => {
    if (!inBrowserModel) return alert('Instantiate a model first');
    const name = `inbrowser_${inBrowserSchema?.type || 'model'}`;
    try {
      await inBrowserModel.save(`downloads://${name}`);
    } catch (err) {
      console.error('Export failed', err);
      alert('Export failed: ' + err.message);
    }
  });

  // In-browser model controls
  document.getElementById('train-in').addEventListener('click', async () => {
    const epochs = parseInt(document.getElementById('epochs').value, 10) || 30;
    const batch = parseInt(document.getElementById('batch').value, 10) || 64;
    document.getElementById('inModelStatus').textContent = 'training...';
    await trainInBrowserModel(epochs, batch);
    document.getElementById('inModelStatus').textContent = 'trained';
  });

  document.getElementById('compare').addEventListener('click', async () => {
    if (!model || !inBrowserModel) {
      alert('Both saved model and in-browser model must be available. Train the in-browser model first.');
      return;
    }
    document.getElementById('inModelStatus').textContent = 'comparing...';
    await compareModels();
    document.getElementById('inModelStatus').textContent = 'ready';
  });
});

function buildInBrowserModel(inputDim, type = 'linear') {
  const m = tf.sequential();
  if (type === 'logistic') {
    m.add(tf.layers.dense({ inputShape: [inputDim], units: 1, activation: 'sigmoid' }));
    m.compile({ optimizer: tf.train.adam(0.01), loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  } else {
    // linear regression
    m.add(tf.layers.dense({ inputShape: [inputDim], units: 1, activation: 'linear' }));
    m.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError', metrics: ['mae'] });
  }
  return m;
}

function generateSyntheticData(n = 1000, inputDim = 3) {
  const X = [];
  const y = [];
  const trueW = Array.from({ length: inputDim }, () => (Math.random() - 0.5) * 2);
  const bias = (Math.random() - 0.5) * 0.5;
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < inputDim; j++) row.push((Math.random() - 0.5) * 2);
    X.push(row);
    const noise = (Math.random() * 2 - 1) * 0.5;
    const val = row.reduce((acc, v, idx) => acc + v * trueW[idx], 0) + bias + noise;
    y.push(val);
  }
  return { X, y };
}

async function trainInBrowserModel(epochs = 30, batchSize = 64) {
  const schema = inBrowserSchema || { type: 'linear', inputDim: 3 };
  const type = schema.type || 'linear';
  const inputDim = schema.inputDim || 3;

  // Generate appropriate synthetic data depending on model type
  let X, y;
  if (type === 'logistic') {
    ({ X, y } = generateSyntheticClassificationData(1200, inputDim));
  } else {
    ({ X, y } = generateSyntheticData(1200, inputDim));
  }

  const Xtr = tf.tensor2d(X.slice(0, 1000));
  const ytr = tf.tensor2d(y.slice(0, 1000), [1000, 1]);
  if (!inBrowserModel) inBrowserModel = buildInBrowserModel(inputDim, type);

  // clear previous log
  const tbody = document.getElementById('training-log-body');
  tbody.innerHTML = '';

  // Disable buttons while training
  document.getElementById('train-in').disabled = true;
  document.getElementById('compare').disabled = true;

  await inBrowserModel.fit(Xtr, ytr, {
    epochs,
    batchSize,
    verbose: 0,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const row = document.createElement('tr');
        const eCell = document.createElement('td');
        eCell.textContent = String(epoch + 1);
        const lossCell = document.createElement('td');
        lossCell.textContent = (logs.loss ?? NaN).toFixed(4);
        const metricCell = document.createElement('td');
        // show mae or accuracy depending on model
        const primary = (logs.mae != null) ? logs.mae : logs.accuracy;
        metricCell.textContent = (primary ?? NaN).toFixed(4);
        row.appendChild(eCell);
        row.appendChild(lossCell);
        row.appendChild(metricCell);
        tbody.appendChild(row);
        await tf.nextFrame();
      }
    }
  });

  // Re-enable buttons
  document.getElementById('train-in').disabled = false;
  document.getElementById('compare').disabled = false;
  Xtr.dispose();
  ytr.dispose();
}

// Synthetic classification data generator
function generateSyntheticClassificationData(n = 1000, inputDim = 3) {
  const X = [];
  const y = [];
  const weights = [];
  for (let i = 0; i < inputDim; i++) weights.push((Math.random() - 0.5) * 2);
  const bias = (Math.random() - 0.5) * 0.5;
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < inputDim; j++) row.push((Math.random() - 0.5) * 2);
    const lin = row.reduce((acc, v, idx) => acc + v * weights[idx], 0) + bias + (Math.random() - 0.5) * 0.5;
    const label = lin > 0 ? 1 : 0;
    X.push(row);
    y.push(label);
  }
  return { X, y };
}

async function compareModels() {
  const schema = inBrowserSchema || { type: 'linear', inputDim: 3 };
  const inputDim = schema.inputDim || 3;
  const { X, y } = generateSyntheticData(300, inputDim);
  const Xt = tf.tensor2d(X);
  const yt = tf.tensor2d(y, [y.length, 1]);

  // saved model preds
  const savedPred = model.predict(Xt);
  const inPred = inBrowserModel.predict(Xt);

  const mseSaved = tf.losses.meanSquaredError(yt, savedPred).dataSync()[0];
  const mseIn = tf.losses.meanSquaredError(yt, inPred).dataSync()[0];

  // sample single-row prediction (from inputs)
  const sample = tf.tensor2d([readInputs()]);
  const ps = (await model.predict(sample).data())[0];
  const pi = (await inBrowserModel.predict(sample).data())[0];

  document.getElementById('mse-saved').textContent = mseSaved.toFixed(4);
  document.getElementById('mse-in').textContent = mseIn.toFixed(4);
  document.getElementById('pred-saved').textContent = ps.toFixed(4);
  document.getElementById('pred-in').textContent = pi.toFixed(4);

  // cleanup
  Xt.dispose();
  yt.dispose();
  savedPred.dispose();
  inPred.dispose();
  sample.dispose();
}

export {};
