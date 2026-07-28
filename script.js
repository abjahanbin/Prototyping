import * as faceapi from 'face-api.js';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const status = document.getElementById('status');
const expressionsEl = document.getElementById('expressions');
const ledgerEl = document.getElementById('ledger');
const ledgerHeaderEl = document.getElementById('ledgerHeader');

// Served from public/models — Vite copies public/ as-is, so this path
// works unchanged in dev and in the built output.
const MODEL_URL = `${import.meta.env.BASE_URL}models`;

const ACCENT = '#7fd1ff';
const ALERT_COLOR = '#ff4d4d';
const THREAT_ALERT_THRESHOLD = 55;
const HUD_FONT = 'ui-monospace, "Cascadia Code", "Courier New", monospace';

// Offscreen buffer used to sample/glitch small sub-regions of the face each frame.
const glitchBuffer = document.createElement('canvas');
const glitchBufferCtx = glitchBuffer.getContext('2d');

let latestDetections = [];
let activeGlitches = []; // { x, y, width, height, expiresAt }
let latestMetrics = null; // metrics for the primary (first) tracked face
let threatHistory = []; // scrolling trace samples, oldest first
const THREAT_HISTORY_LENGTH = 160;
let ledgerCount = 0;

let modelsLoaded = false;
let running = false;
let mediaStream = null;
let rafId = null;
let detectTimeoutId = null;
let glitchTimeoutId = null;
let ledgerTimeoutId = null;

startBtn.addEventListener('click', () => {
  if (running) {
    stopCamera();
  } else {
    startCamera();
  }
});

async function startCamera() {
  startBtn.disabled = true;

  try {
    if (!modelsLoaded) {
      status.textContent = 'Loading face models...';
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
      ]);
      modelsLoaded = true;
    }

    status.textContent = 'Requesting camera...';
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = mediaStream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    running = true;
    status.textContent = 'Running.';
    startBtn.textContent = 'Stop Camera';
    startBtn.disabled = false;

    rafId = requestAnimationFrame(renderFrame);
    detectLoop();
    scheduleNextGlitch();
    scheduleNextLedgerEntry();
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    console.error(err);
    startBtn.disabled = false;
  }
}

function stopCamera() {
  running = false;

  if (rafId !== null) cancelAnimationFrame(rafId);
  if (detectTimeoutId !== null) clearTimeout(detectTimeoutId);
  if (glitchTimeoutId !== null) clearTimeout(glitchTimeoutId);
  if (ledgerTimeoutId !== null) clearTimeout(ledgerTimeoutId);
  rafId = detectTimeoutId = glitchTimeoutId = ledgerTimeoutId = null;

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  video.srcObject = null;

  latestDetections = [];
  activeGlitches = [];
  latestMetrics = null;
  threatHistory = [];
  ledgerCount = 0;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  expressionsEl.textContent = '';
  ledgerEl.textContent = '';
  ledgerHeaderEl.textContent = 'SUBJECTS ON FILE: 000';

  status.textContent = 'Camera stopped.';
  startBtn.textContent = 'Start Camera';
}

async function detectLoop() {
  if (!running) return;

  latestDetections = await faceapi
    .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks(true)
    .withFaceExpressions();

  if (!running) return;

  for (const det of latestDetections) {
    if (det.landmarks) {
      det.threatMetrics = computeMetrics(det.landmarks);
      det.threatScore = aggregateThreatScore(det.threatMetrics);
    }
  }

  const primary = latestDetections.find((d) => d.landmarks);
  if (primary) {
    latestMetrics = primary.threatMetrics;
    threatHistory.push(primary.threatScore);
    if (threatHistory.length > THREAT_HISTORY_LENGTH) threatHistory.shift();
  } else {
    latestMetrics = null;
  }

  updateExpressionText();
  detectTimeoutId = setTimeout(detectLoop, 100);
}

function renderFrame(t) {
  if (!running) return;

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  drawGlitchBursts(t);
  drawHud(t);
  drawThreatTrace(t);

  rafId = requestAnimationFrame(renderFrame);
}

// --- ambient glitch scanning artifacts -----------------------------------
// Spawn rate/placement is random and unrelated to emotion — but a flagged
// (high-threat-score) face gets glitched faster, harder, and more often,
// as if the system is reacting to its own alert.

function scheduleNextGlitch() {
  if (!running) return;

  const anyFlagged = latestDetections.some((d) => (d.threatScore ?? 0) >= THREAT_ALERT_THRESHOLD);
  const delay = anyFlagged ? 70 + Math.random() * 110 : 250 + Math.random() * 500;

  glitchTimeoutId = setTimeout(() => {
    if (!running) return;
    spawnGlitch();
    scheduleNextGlitch();
  }, delay);
}

function spawnGlitch() {
  const withLandmarks = latestDetections.filter((d) => d.landmarks);
  if (withLandmarks.length === 0) return;

  const flagged = withLandmarks.filter((d) => (d.threatScore ?? 0) >= THREAT_ALERT_THRESHOLD);
  const pool = flagged.length > 0 ? flagged : withLandmarks;
  const burstCount = flagged.length > 0 ? 2 : 1;

  for (let i = 0; i < burstCount; i++) {
    const det = pool[Math.floor(Math.random() * pool.length)];
    const boxes = getFeatureBoxes(det.landmarks);
    const box = boxes[Math.floor(Math.random() * boxes.length)];

    const width = Math.floor(box.width);
    const height = Math.floor(box.height);
    if (width <= 6 || height <= 6) continue;

    const x = Math.max(0, Math.floor(box.x));
    const y = Math.max(0, Math.floor(box.y));

    activeGlitches.push({
      x, y, width, height,
      intense: flagged.length > 0,
      expiresAt: performance.now() + (flagged.length > 0 ? 160 + Math.random() * 260 : 120 + Math.random() * 200),
    });
  }
}

function getFeatureBoxes(landmarks) {
  return [
    paddedBoundingBox(landmarks.getLeftEye(), 6),
    paddedBoundingBox(landmarks.getRightEye(), 6),
    paddedBoundingBox(landmarks.getLeftEyeBrow(), 4),
    paddedBoundingBox(landmarks.getRightEyeBrow(), 4),
    paddedBoundingBox(landmarks.getNose(), 5),
    paddedBoundingBox(landmarks.getMouth(), 5),
  ];
}

function drawGlitchBursts(t) {
  activeGlitches = activeGlitches.filter((g) => g.expiresAt > t);

  for (const g of activeGlitches) {
    const { x, y, width, height } = g;
    if (x + width > canvas.width || y + height > canvas.height) continue;

    glitchBuffer.width = width;
    glitchBuffer.height = height;
    glitchBufferCtx.drawImage(video, x, y, width, height, 0, 0, width, height);

    const imageData = glitchBufferCtx.getImageData(0, 0, width, height);
    applyGlitch(imageData, width, height, g.intense);
    glitchBufferCtx.putImageData(imageData, 0, 0);

    ctx.drawImage(glitchBuffer, x, y);
  }
}

// banded horizontal displacement + chromatic aberration
function applyGlitch(imageData, width, height, intense) {
  const src = imageData.data;
  const out = new Uint8ClampedArray(src.length);

  const bandHeight = 4;
  const jitterAmount = intense ? 1.3 : 0.8;
  for (let y = 0; y < height; y++) {
    const band = Math.floor(y / bandHeight);
    const jitter = Math.floor((hash01(band * 97) - 0.5) * width * jitterAmount);
    for (let x = 0; x < width; x++) {
      const sx = clamp(x - jitter, 0, width - 1);
      const si = (y * width + sx) * 4;
      const di = (y * width + x) * 4;
      out[di] = src[si];
      out[di + 1] = src[si + 1];
      out[di + 2] = src[si + 2];
      out[di + 3] = 255;
    }
  }

  const final = new Uint8ClampedArray(out.length);
  const shift = intense ? 15 : 9;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const di = (y * width + x) * 4;
      const rx = clamp(x + shift, 0, width - 1);
      const bx = clamp(x - shift, 0, width - 1);
      final[di] = out[(y * width + rx) * 4];
      final[di + 1] = out[di + 1];
      final[di + 2] = out[(y * width + bx) * 4 + 2];
      final[di + 3] = 255;
    }
  }

  imageData.data.set(final);
}

// --- HUD: scanning readout across face + individual features -------------

function drawHud(t) {
  for (const det of latestDetections) {
    const box = det.detection.box;
    const flagged = (det.threatScore ?? 0) >= THREAT_ALERT_THRESHOLD;

    drawCornerBrackets(box.x, box.y, box.width, box.height, Math.min(24, box.width * 0.15), 2);
    drawMainPanel(det, box, t, flagged);

    if (det.landmarks) {
      drawLandmarkMesh(det.landmarks);
      drawFeatureReticles(det.landmarks);
      drawFaceRadarChart(det.threatMetrics, box);
    }
  }
}

function drawFaceRadarChart(metrics, box) {
  const radius = 46;
  const spacing = radius * 2 + 24;

  const fitsRight = box.x + box.width + spacing < canvas.width;
  const fitsLeft = box.x - spacing > 0;
  const cx = fitsRight
    ? box.x + box.width + radius + 16
    : fitsLeft
      ? box.x - radius - 16
      : clamp(box.x + box.width / 2, radius + 4, canvas.width - radius - 4);
  const cy = clamp(box.y + box.height / 2, radius + 18, canvas.height - radius - 8);

  drawRadarChartAt(metrics, cx, cy, radius);
}

function drawLandmarkMesh(landmarks) {
  const jaw = landmarks.getJawOutline();
  const leftBrow = landmarks.getLeftEyeBrow();
  const rightBrow = landmarks.getRightEyeBrow();
  const nose = landmarks.getNose();
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  const mouth = landmarks.getMouth();
  const outerMouth = mouth.slice(0, 12);
  const innerMouth = mouth.slice(12);

  ctx.strokeStyle = 'rgba(127, 209, 255, 0.45)';
  ctx.lineWidth = 1;
  drawPolyline(jaw, false);
  drawPolyline(leftBrow, false);
  drawPolyline(rightBrow, false);
  drawPolyline(nose, false);
  drawPolyline(leftEye, true);
  drawPolyline(rightEye, true);
  drawPolyline(outerMouth, true);
  drawPolyline(innerMouth, true);

  ctx.fillStyle = ACCENT;
  const allPoints = [...jaw, ...leftBrow, ...rightBrow, ...nose, ...leftEye, ...rightEye, ...mouth];
  for (const p of allPoints) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 1.6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawPolyline(points, closed) {
  ctx.beginPath();
  points.forEach((p, i) => {
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  if (closed) ctx.closePath();
  ctx.stroke();
}

function drawMainPanel(det, box, t, flagged) {
  const top = topExpressionOf(det.expressions).slice(0, 2);
  const lines = [
    `SCAN ${pulseDots(t)}`,
    ...top.map((e) => `${e.name.toUpperCase().padEnd(9)} ${String(Math.round(e.probability * 100)).padStart(3)}%`),
  ];
  if (flagged) {
    lines.push('⚠ THREAT FLAGGED');
    lines.push(dominantJustification(det.threatMetrics).sentence);
  }

  const color = flagged ? ALERT_COLOR : ACCENT;

  const font = `12px ${HUD_FONT}`;
  ctx.font = font;
  const textWidth = Math.max(...lines.map((l) => ctx.measureText(l).width));
  const panelHeight = lines.length * 14 + 6;
  const panelY = box.y - panelHeight - 6;

  ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
  ctx.fillRect(box.x, panelY, textWidth + 12, panelHeight);
  ctx.strokeStyle = color;
  ctx.lineWidth = flagged ? 1.5 + Math.abs(Math.sin(t / 200)) : 1;
  ctx.strokeRect(box.x, panelY, textWidth + 12, panelHeight);

  ctx.fillStyle = color;
  lines.forEach((line, i) => {
    ctx.fillText(line, box.x + 6, panelY + 14 + i * 14);
  });
}

function pulseDots(t) {
  const n = Math.floor(t / 300) % 4;
  return '.'.repeat(n).padEnd(3);
}

function drawFeatureReticles(landmarks) {
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  const leftBrow = landmarks.getLeftEyeBrow();
  const rightBrow = landmarks.getRightEyeBrow();
  const nose = landmarks.getNose();
  const mouth = landmarks.getMouth();
  const jaw = landmarks.getJawOutline();

  drawEyeReticle(leftEye, 'EYE_L');
  drawEyeReticle(rightEye, 'EYE_R');
  drawBrowReticle(leftBrow, leftEye, 'BROW_L');
  drawBrowReticle(rightBrow, rightEye, 'BROW_R');
  drawNoseReticle(nose);
  drawMouthReticle(mouth);
  drawJawReadout(jaw);
}

function drawEyeReticle(points, label) {
  const box = paddedBoundingBox(points, 6);
  drawCornerBrackets(box.x, box.y, box.width, box.height, Math.min(8, box.width * 0.3), 1.5);

  const aperture = Math.round(eyeAperture(points));
  drawTag(`${label} ${String(aperture).padStart(3)}%`, box.x, box.y - 4, 'left');
}

function drawBrowReticle(browPoints, eyePoints, label) {
  const box = paddedBoundingBox(browPoints, 4);
  drawCornerBrackets(box.x, box.y, box.width, box.height, Math.min(7, box.width * 0.25), 1.5);

  const raise = Math.round(browRaisePx(browPoints, eyePoints));
  drawTag(`${label} ${raise >= 0 ? '+' : ''}${raise}px`, box.x, box.y - 4, 'left');
}

function drawNoseReticle(points) {
  const box = paddedBoundingBox(points, 5);
  drawCornerBrackets(box.x, box.y, box.width, box.height, Math.min(8, box.width * 0.3), 1.5);

  const cx = Math.round(average(points.map((p) => p.x)));
  const cy = Math.round(average(points.map((p) => p.y)));

  drawTag(`NOSE ${cx},${cy}`, box.x, box.y + box.height + 14, 'left');
}

function drawMouthReticle(points) {
  const box = paddedBoundingBox(points, 5);
  drawCornerBrackets(box.x, box.y, box.width, box.height, Math.min(10, box.width * 0.25), 1.5);

  const aperture = Math.round(mouthAperture(points));
  drawTag(`MOUTH ${String(aperture).padStart(3)}%`, box.x, box.y + box.height + 14, 'left');
}

function drawJawReadout(points) {
  const chin = points[8];
  const angleDeg = jawTiltDeg(points);

  drawTag(`TILT ${angleDeg >= 0 ? '+' : ''}${angleDeg.toFixed(1)}°`, chin.x - 30, chin.y + 18, 'left');
}

// --- shared landmark metrics (feed reticle labels + threat scoring) -------

function eyeAperture(points) {
  const widthDist = dist(points[0], points[3]);
  const openDist = (dist(points[1], points[5]) + dist(points[2], points[4])) / 2;
  return clamp((openDist / widthDist) * 220, 0, 100);
}

function mouthAperture(points) {
  const widthDist = dist(points[0], points[6]);
  const openDist = dist(points[3], points[9]);
  return clamp((openDist / widthDist) * 130, 0, 100);
}

function browRaisePx(browPoints, eyePoints) {
  const browCenterY = average(browPoints.map((p) => p.y));
  const eyeCenterY = average(eyePoints.map((p) => p.y));
  return eyeCenterY - browCenterY;
}

function centroid(points) {
  return { x: average(points.map((p) => p.x)), y: average(points.map((p) => p.y)) };
}

// distance between eye centers — used to scale pixel measurements so they
// stay meaningful regardless of how close the face is to the camera
function interocularDistance(leftEye, rightEye) {
  return dist(centroid(leftEye), centroid(rightEye)) || 1;
}

function jawTiltDeg(jawPoints) {
  const left = jawPoints[0];
  const right = jawPoints[16];
  return (Math.atan2(right.y - left.y, right.x - left.x) * 180) / Math.PI;
}

// mirror-symmetry of the jawline around the nose bridge's x position
function symmetryScore(landmarks) {
  const jaw = landmarks.getJawOutline();
  const centerX = landmarks.getNose()[0].x;
  const faceWidth = dist(jaw[0], jaw[16]) || 1;

  let totalDeviation = 0;
  const pairCount = 8;
  for (let i = 0; i < pairCount; i++) {
    const leftDist = Math.abs(jaw[i].x - centerX);
    const rightDist = Math.abs(jaw[16 - i].x - centerX);
    totalDeviation += Math.abs(leftDist - rightDist);
  }
  const normalizedDeviation = totalDeviation / pairCount / faceWidth;
  return clamp(100 - normalizedDeviation * 400, 0, 100);
}

// --- threat scoring (pseudo-scientific by design — see notes on aggregateThreatScore) --

function computeMetrics(landmarks) {
  const leftEye = landmarks.getLeftEye();
  const rightEye = landmarks.getRightEye();
  const leftBrow = landmarks.getLeftEyeBrow();
  const rightBrow = landmarks.getRightEyeBrow();
  const mouth = landmarks.getMouth();

  const aperture = (eyeAperture(leftEye) + eyeAperture(rightEye)) / 2;

  // brow-to-eye gap relative to interocular distance — a ratio, so it holds
  // steady whether the face fills the frame or sits far from the camera
  const iod = interocularDistance(leftEye, rightEye);
  const browRaiseAvg = (browRaisePx(leftBrow, leftEye) + browRaisePx(rightBrow, rightEye)) / 2;
  const browRaiseRatio = browRaiseAvg / iod;
  const BROW_FURROWED = 0.12; // ratio at a hard furrow -> tension 100
  const BROW_RELAXED = 0.4; // ratio at rest/raised -> tension 0
  const tension = clamp(
    100 - ((browRaiseRatio - BROW_FURROWED) / (BROW_RELAXED - BROW_FURROWED)) * 100,
    0,
    100,
  );

  const oral = mouthAperture(mouth);
  const tiltDeg = jawTiltDeg(landmarks.getJawOutline());
  const tilt = clamp(Math.abs(tiltDeg) * 12, 0, 100);
  const symmetry = symmetryScore(landmarks);

  return { symmetry, tension, aperture, oral, tilt };
}

// The weights below are not calibrated to anything — that's the point. Real
// "threat"/"risk" scoring systems bury an equally arbitrary formula under a
// confident-looking number; this one is left visibly invented, right down to
// the small random jitter, rather than pretending to be a stable measurement.
function aggregateThreatScore(metrics) {
  const base =
    0.35 * metrics.tension +
    0.25 * (100 - metrics.symmetry) +
    0.15 * metrics.oral +
    0.15 * metrics.aperture +
    0.1 * metrics.tilt;
  const jitter = (Math.random() - 0.5) * 8;
  return clamp(base + jitter, 0, 100);
}

// Invented "reasoning" attached to a flag — sounds like a determination,
// but it's just naming whichever arbitrary input happened to score highest.
const JUSTIFICATIONS = {
  tension: { sentence: 'ELEVATED BROW TENSION — PATTERN MATCH: CLASS 2', code: 'CLASS 2' },
  symmetry: { sentence: 'ASYMMETRY EXCEEDS NOMINAL RANGE — RISK PROFILE 4B', code: 'PROFILE 4B' },
  oral: { sentence: 'ORAL APERTURE ANOMALY — FLAGGED FOR REVIEW', code: 'REVIEW' },
  aperture: { sentence: 'OCULAR DILATION IRREGULAR — RISK PROFILE 2C', code: 'PROFILE 2C' },
  tilt: { sentence: 'CRANIAL TILT DEVIATION — PATTERN MATCH: CLASS 7', code: 'CLASS 7' },
};

function dominantJustification(metrics) {
  const contributions = {
    tension: 0.35 * metrics.tension,
    symmetry: 0.25 * (100 - metrics.symmetry),
    oral: 0.15 * metrics.oral,
    aperture: 0.15 * metrics.aperture,
    tilt: 0.1 * metrics.tilt,
  };
  const topKey = Object.entries(contributions).sort((a, b) => b[1] - a[1])[0][0];
  return JUSTIFICATIONS[topKey];
}

// --- radar chart: raw axis readings, drawn beside each tracked face ------

const RADAR_AXES = [
  { key: 'symmetry', label: 'SYMM' },
  { key: 'tension', label: 'TENS' },
  { key: 'oral', label: 'ORAL' },
  { key: 'aperture', label: 'OCUL' },
  { key: 'tilt', label: 'TILT' },
];

function drawRadarChartAt(metrics, cx, cy, radius) {
  ctx.font = `10px ${HUD_FONT}`;
  ctx.fillStyle = ACCENT;
  ctx.fillText('BIOMETRIC INDEX', cx - radius, cy - radius - 10);

  const axisCount = RADAR_AXES.length;
  const angleFor = (i) => -Math.PI / 2 + (i / axisCount) * Math.PI * 2;

  // rings + spokes
  ctx.strokeStyle = 'rgba(127, 209, 255, 0.25)';
  ctx.lineWidth = 1;
  for (const frac of [0.33, 0.66, 1]) {
    ctx.beginPath();
    for (let i = 0; i <= axisCount; i++) {
      const a = angleFor(i % axisCount);
      const px = cx + Math.cos(a) * radius * frac;
      const py = cy + Math.sin(a) * radius * frac;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  for (let i = 0; i < axisCount; i++) {
    const a = angleFor(i);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(a) * radius, cy + Math.sin(a) * radius);
    ctx.stroke();
  }

  const values = RADAR_AXES.map((axis) => metrics[axis.key]);

  ctx.beginPath();
  values.forEach((v, i) => {
    const a = angleFor(i);
    const r = radius * (clamp(v, 0, 100) / 100);
    const px = cx + Math.cos(a) * r;
    const py = cy + Math.sin(a) * r;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.fillStyle = 'rgba(127, 209, 255, 0.25)';
  ctx.fill();
  ctx.strokeStyle = ACCENT;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.font = `10px ${HUD_FONT}`;
  ctx.fillStyle = ACCENT;
  RADAR_AXES.forEach((axis, i) => {
    const a = angleFor(i);
    const labelR = radius + 12;
    const px = cx + Math.cos(a) * labelR;
    const py = cy + Math.sin(a) * labelR;
    ctx.fillText(axis.label, px - ctx.measureText(axis.label).width / 2, py + 3);
  });
}

// --- scrolling polygraph-style threat trace -------------------------------

function drawThreatTrace(t) {
  const bandHeight = 74;
  const bandY = canvas.height - bandHeight;

  ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
  ctx.fillRect(0, bandY, canvas.width, bandHeight);

  ctx.strokeStyle = 'rgba(127, 209, 255, 0.2)';
  ctx.lineWidth = 1;
  for (const frac of [0.25, 0.5, 0.75]) {
    const y = bandY + bandHeight * frac;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  if (threatHistory.length > 1) {
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const stepX = canvas.width / (THREAT_HISTORY_LENGTH - 1);
    const offset = THREAT_HISTORY_LENGTH - threatHistory.length;
    threatHistory.forEach((v, i) => {
      const x = (offset + i) * stepX;
      const y = bandY + bandHeight - (clamp(v, 0, 100) / 100) * bandHeight;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  const current = threatHistory.length > 0 ? Math.round(threatHistory[threatHistory.length - 1]) : null;
  const label = current === null ? 'THREAT INDEX  --  NO SIGNAL' : `THREAT INDEX  ${String(current).padStart(3)}%`;

  ctx.font = `12px ${HUD_FONT}`;
  ctx.fillStyle = ACCENT;
  ctx.fillText(label, 10, bandY + 16);

  if (current !== null && Math.floor(t / 500) % 2 === 0) {
    ctx.fillStyle = '#ff5f5f';
    ctx.beginPath();
    ctx.arc(canvas.width - 46, bandY + 12, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText('REC', canvas.width - 36, bandY + 16);
  }
}

// --- scrolling subject ledger (ephemeral — nothing is stored) ------------

function scheduleNextLedgerEntry() {
  if (!running) return;

  const delay = 1800 + Math.random() * 1400;
  ledgerTimeoutId = setTimeout(() => {
    if (!running) return;
    appendLedgerEntry();
    scheduleNextLedgerEntry();
  }, delay);
}

function appendLedgerEntry() {
  const time = new Date().toLocaleTimeString();
  let line;

  let id = null;
  let flaggedEntry = false;

  if (latestMetrics) {
    ledgerCount += 1;
    const score = Math.round(aggregateThreatScore(latestMetrics));
    // Deliberately low and jittery — the system never rules out a repeat
    // subject, it just logs a new record anyway. Never actually 0: it
    // always leaves itself a little doubt, then ignores it.
    const matchConf = Math.round(4 + Math.random() * 22);
    id = `SUBJECT_${String(ledgerCount).padStart(3, '0')}`;
    flaggedEntry = score >= THREAT_ALERT_THRESHOLD;
    const reason = flaggedEntry ? `  REASON: ${dominantJustification(latestMetrics).code}` : '';
    line = `${time}  ${id.padEnd(13)} CONF ${String(matchConf).padStart(2)}%  THREAT ${String(score).padStart(3)}%${reason}`;
    ledgerHeaderEl.textContent = `SUBJECTS ON FILE: ${String(ledgerCount).padStart(3, '0')}`;
  } else {
    line = `${time}  ---  NO SUBJECT IN FRAME`;
  }

  appendLedgerRow(line);

  // The system sometimes walks a flag back — but the original entry above
  // is never edited or removed. The correction doesn't undo the record.
  if (flaggedEntry && Math.random() < 0.5) {
    const retractDelay = 2500 + Math.random() * 3000;
    setTimeout(() => {
      if (!running) return;
      appendLedgerRow(`${new Date().toLocaleTimeString()}  ${id.padEnd(13)} >> FLAG WITHDRAWN — INSUFFICIENT CONFIDENCE`);
    }, retractDelay);
  }
}

function appendLedgerRow(text) {
  const row = document.createElement('div');
  row.textContent = text;
  ledgerEl.appendChild(row);

  while (ledgerEl.childNodes.length > 40) {
    ledgerEl.removeChild(ledgerEl.firstChild);
  }
  ledgerEl.scrollTop = ledgerEl.scrollHeight;
}

// --- small drawing helpers -------------------------------------------------

function drawCornerBrackets(x, y, width, height, armLen, lineWidth) {
  ctx.strokeStyle = ACCENT;
  ctx.lineWidth = lineWidth;

  const corner = (sx, sy, dx, dy) => {
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(sx + dx * armLen, sy);
    ctx.moveTo(sx, sy);
    ctx.lineTo(sx, sy + dy * armLen);
    ctx.stroke();
  };

  corner(x, y, 1, 1);
  corner(x + width, y, -1, 1);
  corner(x, y + height, 1, -1);
  corner(x + width, y + height, -1, -1);
}

function drawTag(text, x, y, align) {
  ctx.font = `10px ${HUD_FONT}`;
  const textWidth = ctx.measureText(text).width;
  const tagX = align === 'left' ? x : x - textWidth;

  ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
  ctx.fillRect(tagX - 2, y - 10, textWidth + 4, 13);
  ctx.fillStyle = ACCENT;
  ctx.fillText(text, tagX, y);
}

function paddedBoundingBox(points, padding) {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs) - padding;
  const maxX = Math.max(...xs) + padding;
  const minY = Math.min(...ys) - padding;
  const maxY = Math.max(...ys) + padding;
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function average(values) {
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

// --- expression summary (below canvas) -------------------------------------

function updateExpressionText() {
  if (latestDetections.length === 0) {
    expressionsEl.textContent = 'No face detected.';
    return;
  }
  expressionsEl.innerHTML = latestDetections
    .map((det, i) => {
      const [top] = topExpressionOf(det.expressions);
      return `Face ${i + 1}: <b>${top.name}</b> (${Math.round(top.probability * 100)}%)`;
    })
    .join(' &nbsp;|&nbsp; ');
}

function topExpressionOf(expressions) {
  return Object.entries(expressions)
    .map(([name, probability]) => ({ name, probability }))
    .sort((a, b) => b.probability - a.probability);
}

// --- small math helpers ---------------------------------------------------

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

// deterministic 0..1 pseudo-random from an integer
function hash01(n) {
  const x = Math.sin(n * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}
