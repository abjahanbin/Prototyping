import * as faceapi from 'face-api.js';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const status = document.getElementById('status');
const expressionsEl = document.getElementById('expressions');

// Served from public/models — Vite copies public/ as-is, so this path
// works unchanged in dev and in the built output.
const MODEL_URL = `${import.meta.env.BASE_URL}models`;

// Offscreen buffer used to sample/distort just the face region each frame.
const faceBuffer = document.createElement('canvas');
const faceBufferCtx = faceBuffer.getContext('2d');

let latestDetections = [];

startBtn.addEventListener('click', startCamera);

async function startCamera() {
  startBtn.disabled = true;

  try {
    status.textContent = 'Loading face models...';
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
    ]);

    status.textContent = 'Requesting camera...';
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    status.textContent = 'Running.';
    requestAnimationFrame(renderFrame);
    detectLoop();
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    console.error(err);
    startBtn.disabled = false;
  }
}

async function detectLoop() {
  latestDetections = await faceapi
    .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceExpressions();

  updateExpressionText();
  setTimeout(detectLoop, 100);
}

function renderFrame(t) {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  for (const det of latestDetections) {
    applyFaceDistortion(det, t);
  }
  drawLabels();

  requestAnimationFrame(renderFrame);
}

// --- per-face distortion dispatch -------------------------------------

const PIXEL_DISTORTIONS = {
  angry: glitchDistort,
  sad: meltDistort,
  surprised: bulgeDistort,
  fearful: pixelateDistort,
  disgusted: noiseHueDistort,
};

function applyFaceDistortion(det, t) {
  const box = det.detection.box;
  const x = Math.max(0, Math.floor(box.x));
  const y = Math.max(0, Math.floor(box.y));
  const width = Math.min(Math.floor(box.width), canvas.width - x);
  const height = Math.min(Math.floor(box.height), canvas.height - y);
  if (width <= 4 || height <= 4) return;

  const [top] = topExpressionOf(det.expressions);

  if (top.name === 'happy') {
    zoomPulseDistort(x, y, width, height, t);
    return;
  }
  if (top.name === 'neutral') {
    return; // leave the plain video feed showing through
  }

  const distortFn = PIXEL_DISTORTIONS[top.name];
  if (!distortFn) return;

  faceBuffer.width = width;
  faceBuffer.height = height;
  faceBufferCtx.drawImage(video, x, y, width, height, 0, 0, width, height);

  const imageData = faceBufferCtx.getImageData(0, 0, width, height);
  distortFn(imageData, width, height, t);
  faceBufferCtx.putImageData(imageData, 0, 0);

  ctx.drawImage(faceBuffer, x, y);
}

// happy: confined breathing zoom, drawn directly (no per-pixel warp needed)
function zoomPulseDistort(x, y, width, height, t) {
  const scale = 1 + 0.18 * Math.sin(t / 250);
  const dw = width * scale;
  const dh = height * scale;
  const dx = x - (dw - width) / 2;
  const dy = y - (dh - height) / 2;

  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, width, height);
  ctx.clip();
  ctx.drawImage(video, x, y, width, height, dx, dy, dw, dh);
  ctx.restore();
}

// angry: banded horizontal glitch + chromatic aberration
function glitchDistort(imageData, width, height) {
  const src = imageData.data;
  const out = new Uint8ClampedArray(src.length);

  const bandHeight = 4;
  for (let y = 0; y < height; y++) {
    const band = Math.floor(y / bandHeight);
    const jitter = Math.floor((hash01(band * 97 + Math.floor(y / bandHeight)) - 0.5) * width * 0.8);
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

  // chromatic aberration: shift red/blue channels horizontally against the banded result
  const final = new Uint8ClampedArray(out.length);
  const shift = 9;
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

// sad: vertical drip/melt, columns smear downward with a slow drift
function meltDistort(imageData, width, height, t) {
  const src = new Uint8ClampedArray(imageData.data);
  const out = imageData.data;

  for (let x = 0; x < width; x++) {
    const drip = height * 0.65 * hash01(x) * (0.5 + 0.5 * Math.sin(t / 900 + x * 0.08));
    for (let y = 0; y < height; y++) {
      const sy = clamp(Math.floor(y - drip), 0, height - 1);
      const si = (sy * width + x) * 4;
      const di = (y * width + x) * 4;
      out[di] = src[si];
      out[di + 1] = src[si + 1];
      out[di + 2] = src[si + 2];
      out[di + 3] = 255;
    }
  }
}

// surprised: radial bulge, pulsing outward like a wide-eyed lens warp
function bulgeDistort(imageData, width, height, t) {
  const src = new Uint8ClampedArray(imageData.data);
  const out = imageData.data;

  const cx = width / 2;
  const cy = height / 2;
  const maxR = Math.max(cx, cy);
  const power = 0.32 + 0.15 * Math.sin(t / 250);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const r = Math.sqrt(dx * dx + dy * dy) / maxR;
      const di = (y * width + x) * 4;

      if (r > 1) {
        out[di] = src[di];
        out[di + 1] = src[di + 1];
        out[di + 2] = src[di + 2];
        out[di + 3] = 255;
        continue;
      }

      const rn = Math.pow(r, power) || 0;
      const scale = r === 0 ? 1 : rn / r;
      const sx = clamp(Math.floor(cx + dx * scale), 0, width - 1);
      const sy = clamp(Math.floor(cy + dy * scale), 0, height - 1);
      const si = (sy * width + sx) * 4;

      out[di] = src[si];
      out[di + 1] = src[si + 1];
      out[di + 2] = src[si + 2];
      out[di + 3] = 255;
    }
  }
}

// fearful: chunky block pixelation with a small per-frame shake
function pixelateDistort(imageData, width, height, t) {
  const src = new Uint8ClampedArray(imageData.data);
  const out = imageData.data;
  const block = 20;

  const shakeX = Math.round((hash01(Math.floor(t / 80)) - 0.5) * 14);
  const shakeY = Math.round((hash01(Math.floor(t / 80) + 500) - 0.5) * 14);

  for (let by = 0; by < height; by += block) {
    for (let bx = 0; bx < width; bx += block) {
      let r = 0, g = 0, b = 0, count = 0;
      for (let y = by; y < Math.min(by + block, height); y++) {
        for (let x = bx; x < Math.min(bx + block, width); x++) {
          const i = (y * width + x) * 4;
          r += src[i]; g += src[i + 1]; b += src[i + 2];
          count++;
        }
      }
      r = r / count; g = g / count; b = b / count;

      for (let y = by; y < Math.min(by + block, height); y++) {
        for (let x = bx; x < Math.min(bx + block, width); x++) {
          const sx = clamp(x + shakeX, 0, width - 1);
          const sy = clamp(y + shakeY, 0, height - 1);
          const di = (sy * width + sx) * 4;
          out[di] = r; out[di + 1] = g; out[di + 2] = b; out[di + 3] = 255;
        }
      }
    }
  }
}

// disgusted: hue rotation + luminance noise
function noiseHueDistort(imageData, width, height, t) {
  const data = imageData.data;
  const hueShift = 160 + 90 * Math.sin(t / 500);

  for (let i = 0; i < data.length; i += 4) {
    const [h, s, l] = rgbToHsl(data[i], data[i + 1], data[i + 2]);
    const noisyL = clamp(l + (Math.random() - 0.5) * 0.35, 0, 1);
    const [r, g, b] = hslToRgb((h + hueShift / 360) % 1, s, noisyL);
    data[i] = r; data[i + 1] = g; data[i + 2] = b; data[i + 3] = 255;
  }
}

// --- labels -------------------------------------------------------------

function drawLabels() {
  for (const det of latestDetections) {
    const { x, y, width, height } = det.detection.box;
    const [topExpression] = topExpressionOf(det.expressions);

    ctx.strokeStyle = '#7fd1ff';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);

    const label = `${topExpression.name} (${Math.round(topExpression.probability * 100)}%)`;
    ctx.font = '16px system-ui, sans-serif';
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(x, y - 22, textWidth + 8, 20);
    ctx.fillStyle = '#7fd1ff';
    ctx.fillText(label, x + 4, y - 7);
  }
}

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

// deterministic 0..1 pseudo-random from an integer, stable across frames
function hash01(n) {
  const x = Math.sin(n * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  let h, s;
  const l = (max + min) / 2;

  if (max === min) {
    h = s = 0;
  } else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      default: h = (r - g) / d + 4; break;
    }
    h /= 6;
  }
  return [h, s, l];
}

function hslToRgb(h, s, l) {
  if (s === 0) {
    const v = Math.round(l * 255);
    return [v, v, v];
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const r = hueToRgb(p, q, h + 1 / 3);
  const g = hueToRgb(p, q, h);
  const b = hueToRgb(p, q, h - 1 / 3);
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function hueToRgb(p, q, t) {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1 / 6) return p + (q - p) * 6 * t;
  if (t < 1 / 2) return q;
  if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
  return p;
}
