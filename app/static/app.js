/* Matchcall — demo, live run and training console. */
'use strict';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtDur = s => {
  s = Math.round(s || 0);
  return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
};
const debounce = (fn, ms = 180) => {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
};
async function getJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(url + ' → ' + r.status);
  return r.json();
}

const C = {
  s1: '#3987e5', s2: '#d95926', s3: '#199e70', s4: '#c98500',
  sky: '#4FE0D0', ink: '#EAF7EE', ink2: '#B9CDBF', muted: '#8FA99B',
  grid: '#1C3A2C', axis: '#2A4C3A',
};

if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
  $$('animateMotion').forEach(n => n.remove());
}

/* ================================================================ charts */

function niceMax(v) {
  if (!Number.isFinite(v) || v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  return Math.ceil(v / p) * p;
}

function lineChart(host, cfg) {
  const width = Math.max(280, host.clientWidth || 540);
  const height = cfg.height || 186;
  const pad = { l: 36, r: 48, t: 8, b: 22 };
  const series = (cfg.series || []).map(s => ({
    ...s,
    points: (s.points || []).filter(p => Number.isFinite(p.y)),
  })).filter(s => s.points.length);

  host.innerHTML = '';
  const cap = document.createElement('figcaption');
  cap.innerHTML = `<span class="ctitle">${esc(cfg.title)}</span>`;
  if (series.length > 1) {
    const lg = document.createElement('div');
    lg.className = 'legend';
    lg.innerHTML = series.map(s =>
      `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('');
    cap.appendChild(lg);
  }
  host.appendChild(cap);

  const body = document.createElement('div');
  body.className = 'chart-body';
  host.appendChild(body);

  if (!series.length) {
    const e = document.createElement('div');
    e.className = 'empty-chart';
    e.textContent = cfg.empty || 'Waiting for data.';
    body.appendChild(e);
    return;
  }

  const xs = series.flatMap(s => s.points.map(p => p.x));
  const ys = series.flatMap(s => s.points.map(p => p.y));
  let xmin = Math.min(...xs), xmax = Math.max(...xs);
  if (xmin === xmax) { xmin -= 0.5; xmax += 0.5; }
  const yTop = cfg.yMax != null ? cfg.yMax : niceMax(Math.max(...ys) * 1.1);
  const x = v => pad.l + (v - xmin) / (xmax - xmin) * (width - pad.l - pad.r);
  const y = v => height - pad.b - (v / yTop) * (height - pad.t - pad.b);
  const yFmt = cfg.yFmt || (v => v.toFixed(2));
  const xFmt = cfg.xFmt || (v => Math.round(v));

  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', cfg.title);
  body.appendChild(svg);

  const add = (tag, attrs, parent = svg) => {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    parent.appendChild(n);
    return n;
  };

  // y grid + ticks
  for (let i = 0; i <= 4; i++) {
    const v = yTop * i / 4;
    add('line', { class: 'grid', x1: pad.l, x2: width - pad.r, y1: y(v), y2: y(v) });
    add('text', { class: 'tick', x: pad.l - 6, y: y(v) + 3, 'text-anchor': 'end' })
      .textContent = yFmt(v);
  }
  // x ticks
  const nx = Math.min(6, Math.max(2, series[0].points.length));
  for (let i = 0; i < nx; i++) {
    const v = xmin + (xmax - xmin) * i / (nx - 1);
    add('text', { class: 'tick', x: x(v), y: height - 6, 'text-anchor': 'middle' })
      .textContent = xFmt(v);
  }
  add('line', { class: 'axis', x1: pad.l, x2: width - pad.r, y1: height - pad.b, y2: height - pad.b });

  // lines + end labels
  const ends = [];
  for (const s of series) {
    const d = s.points.map((p, i) =>
      `${i ? 'L' : 'M'}${x(p.x).toFixed(1)} ${y(p.y).toFixed(1)}`).join(' ');
    add('path', { class: 'line', d, stroke: s.color });
    if (s.points.length <= 8) {
      for (const p of s.points) {
        add('circle', {
          cx: x(p.x), cy: y(p.y), r: 3, fill: s.color,
          stroke: '#0A2118', 'stroke-width': 1.5,
        });
      }
    }
    const last = s.points[s.points.length - 1];
    ends.push({ s, p: last, yy: y(last.y) });
  }
  ends.sort((a, b) => a.yy - b.yy);
  let lastY = -Infinity;
  for (const e of ends) {
    let yy = e.yy;
    if (yy - lastY < 11) yy = lastY + 11;
    lastY = yy;
    add('text', { class: 'endlabel', x: width - pad.r + 6, y: yy + 3 })
      .textContent = yFmt(e.p.y);
  }

  // hover layer
  const tip = document.createElement('div');
  tip.className = 'tip';
  body.appendChild(tip);
  const vline = add('line', {
    x1: 0, x2: 0, y1: pad.t, y2: height - pad.b,
    stroke: C.axis, 'stroke-width': 1, opacity: 0,
  });
  const dots = series.map(s => add('circle', {
    r: 3.5, fill: s.color, stroke: '#0A2118', 'stroke-width': 2, opacity: 0,
  }));
  const overlay = add('rect', {
    x: pad.l, y: pad.t, width: width - pad.l - pad.r,
    height: height - pad.t - pad.b, fill: 'transparent',
  });

  const allX = [...new Set(series.flatMap(s => s.points.map(p => p.x)))].sort((a, b) => a - b);
  overlay.addEventListener('pointermove', ev => {
    const box = svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) * (width / box.width);
    let best = allX[0], bd = Infinity;
    for (const v of allX) {
      const d = Math.abs(x(v) - px);
      if (d < bd) { bd = d; best = v; }
    }
    vline.setAttribute('x1', x(best));
    vline.setAttribute('x2', x(best));
    vline.setAttribute('opacity', 0.7);
    const rows = [];
    series.forEach((s, i) => {
      const p = s.points.find(q => q.x === best);
      if (!p) { dots[i].setAttribute('opacity', 0); return; }
      dots[i].setAttribute('cx', x(p.x));
      dots[i].setAttribute('cy', y(p.y));
      dots[i].setAttribute('opacity', 1);
      rows.push(`<div class="row"><i style="background:${s.color}"></i>${esc(s.name)} <b>${yFmt(p.y)}</b></div>`);
    });
    tip.innerHTML = `<div style="color:${C.muted}">${esc(cfg.xLabel || 'x')} ${xFmt(best)}</div>${rows.join('')}`;
    tip.style.opacity = 1;
    const bx = (x(best) / width) * box.width;
    const left = Math.min(Math.max(bx + 10, 4), box.width - tip.offsetWidth - 4);
    tip.style.left = left + 'px';
    tip.style.top = '6px';
  });
  overlay.addEventListener('pointerleave', () => {
    tip.style.opacity = 0;
    vline.setAttribute('opacity', 0);
    dots.forEach(d => d.setAttribute('opacity', 0));
  });
}

/* ================================================================== demo */

let reel = null;
let trackingUrl = null;
let yolo26Url = null;

function momentHTML(m) {
  const kind = String(m.kind || 'moment');
  return `<button class="moment ${esc(kind)}" data-start="${m.reel_start}" data-n="${m.n}">
    <span class="min">${esc(m.minute)}′</span>
    <span class="txt"><b>${esc(m.team)}</b><span>${esc(m.commentary)}</span></span>
    <span class="kind">${esc(kind)}</span>
  </button>`;
}

async function loadDemo() {
  let d;
  try { d = await getJSON('/api/demo'); } catch { return; }
  if (!d.available) return;
  reel = d;
  const goals = (d.moments || []).filter(m => m.kind === 'goal').length;
  $('#heroMoments').textContent = d.clips ?? d.moments.length;
  $('#heroGoals').textContent = goals;
  $('#heroSeconds').textContent = Math.round(d.reel_seconds || 0) + ' s';
  $('#statClips').textContent = d.clips ?? d.moments.length;
  $('#statSeconds').textContent = Math.round(d.reel_seconds || 0) + ' s';
  $('#statGoals').textContent = goals;
  $('#reelMeta').textContent = `${d.clips} × ${Math.round(d.clip_len)} s`;
  $('#reelSub').textContent = `Argentina 2–2 France, second half. ${d.clips} moments, ${Math.round(d.clip_len)} s each, voiced by Fish Audio ${d.fish_model || ''}.`;
  $('#pipeGpt2').textContent = `${d.gpt2_lines ?? 0} GPT-2`;
  $('#pipeFallback').textContent = `· ${d.template_lines ?? 0} template fallback`;
  $('#pipeVoiced').textContent = `${d.clips} / ${d.clips} voiced`;

  const list = $('#momentList');
  list.innerHTML = (d.moments || []).map(momentHTML).join('');
  list.addEventListener('click', ev => {
    const b = ev.target.closest('.moment');
    if (!b) return;
    seekReel(parseFloat(b.dataset.start));
  });
  markActive(0);
}

function seekReel(t) {
  const v = $('#reelVideo');
  v.currentTime = Math.max(0, t + 0.01);
  v.play().catch(() => {});
  v.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function markActive(i) {
  $$('#momentList .moment').forEach((el, j) =>
    el.classList.toggle('active', j === i));
}

function setupReel() {
  const v = $('#reelVideo');
  // Resolve the sources at click time: the API appends a version query to the
  // current reel, and hardcoding the old filename made the tab reload the
  // pre-v4 video (and its old audio).
  const tabs = [
    ['#tabReel', () => (reel && reel.video_url)
      || '/media/commented_highlights/commented_highlights_v4.mp4'],
    ['#tabTracking', () => trackingUrl
      || '/media/commented_highlights/annotated_goal_possession.mp4'],
    ['#tabYolo26', () => yolo26Url
      || '/media/commented_highlights/yolo26n_comparison/annotated_goal_possession.mp4'],
  ];
  const notes = {
    '#tabReel': '18 s clips · Fish-voiced reel',
    '#tabTracking': 'Aryaman YOLOv5lu · players, ball, teams, possession',
    '#tabYolo26': 'YOLO26n · 2.39M params · 640 px · comparison overlay',
  };
  tabs.forEach(([sel, getSrc]) => {
    $(sel).addEventListener('click', () => {
      tabs.forEach(([s]) => $(s).setAttribute('aria-pressed', String(s === sel)));
      $('#trackingNote').textContent = notes[sel] || '';
      v.src = getSrc();
      v.load();
      v.play().catch(() => {});
    });
  });
  v.addEventListener('timeupdate', () => {
    if (!reel) return;
    const clip = reel.clip_len || 10;
    markActive(Math.min(reel.moments.length - 1, Math.floor(v.currentTime / clip)));
  });
}

/* ============================================================== training */

let trainingTimer = null;

function metric(v, k) {
  return `<div class="metric"><div class="v">${v}</div><div class="k">${esc(k)}</div></div>`;
}

function renderTraining(d) {
  const y = d.yolo || {};
  const l = d.llm || {};
  const t = d.tracking || {};
  trackingUrl = t.annotated_url || trackingUrl;
  yolo26Url = t.yolo26_url || yolo26Url;

  const state = y.status || 'queued';
  const running = state === 'running' || state === 'starting';
  const epoch = y.epochs_done || 0;
  const total = y.epochs_total || 100;
  const label = state === 'done' ? 'training complete'
    : state === 'stalled' ? 'training stalled'
    : state === 'paused' ? `paused at epoch ${epoch}/${total}`
    : state === 'running' ? `epoch ${epoch}/${total}`
    : 'starting';
  $('#statusText').textContent = `Fish s2.1-pro-free · YOLOv5lu ${label}`;
  $('#statusDot').className = 'dot ' + (state === 'done' ? '' : running ? 'live' : 'warn');
  $('#trainStatus').textContent = `YOLOv5lu ${label} · GPT-2 LoRA v4 done${l.final_loss != null ? ' · loss ' + l.final_loss : ''}`;

  const badge = $('#yoloBadge');
  badge.textContent = state === 'done' ? 'complete'
    : state === 'stalled' ? 'stalled' : state === 'paused' ? 'paused'
    : state === 'running' ? 'training' : 'starting';
  badge.className = 'badge ' + (state === 'done' ? 'done' : (state === 'stalled' || state === 'paused') ? 'stall' : 'live');

  const b = y.best_metrics || y.last_metrics || {};
  const ds = y.dataset || {};
  $('#yoloMetrics').innerHTML =
    metric(b.mAP50 != null ? b.mAP50.toFixed(3) : '—', 'mAP50') +
    metric(b.mAP50_95 != null ? b.mAP50_95.toFixed(3) : '—', 'mAP50–95') +
    metric(y.best_epoch ?? '—', 'best epoch') +
    metric(`${ds.train ?? '—'} / ${ds.val ?? '—'}`, 'train / val images');

  const rows = y.rows || [];
  lineChart($('#lossChart'), {
    title: 'Training losses',
    xLabel: 'epoch',
    series: [
      { name: 'box', color: C.s1, points: rows.map(r => ({ x: r.epoch, y: r.box_loss })) },
      { name: 'cls', color: C.s2, points: rows.map(r => ({ x: r.epoch, y: r.cls_loss })) },
      { name: 'dfl', color: C.s3, points: rows.map(r => ({ x: r.epoch, y: r.dfl_loss })) },
    ],
    empty: running ? 'First epoch running — curves appear after the first validation.' : 'No results yet.',
  });
  lineChart($('#mapChart'), {
    title: 'Validation metrics',
    xLabel: 'epoch',
    yMax: 1,
    yFmt: v => v.toFixed(2),
    series: [
      { name: 'precision', color: C.s1, points: rows.map(r => ({ x: r.epoch, y: r.precision })) },
      { name: 'recall', color: C.s2, points: rows.map(r => ({ x: r.epoch, y: r.recall })) },
      { name: 'mAP50', color: C.s3, points: rows.map(r => ({ x: r.epoch, y: r.mAP50 })) },
      { name: 'mAP50–95', color: C.s4, points: rows.map(r => ({ x: r.epoch, y: r.mAP50_95 })) },
    ],
    empty: running ? 'Validation metrics appear after epoch 1.' : 'No results yet.',
  });

  const assets = y.assets || {};
  const pics = ['results.png', 'labels.jpg', 'train_batch0.jpg', 'confusion_matrix.png']
    .filter(n => assets[n]);
  $('#yoloAssets').innerHTML = pics.map(n =>
    `<div class="asset"><img loading="lazy" src="${assets[n]}" alt="${esc(n)}"><span>${esc(n)}</span></div>`
  ).join('');

  $('#yoloLog').textContent = (y.log_tail || '').trim() || 'No log yet.';
  $('#yoloLinks').innerHTML = [
    y.task_url && `<a href="${y.task_url}" target="_blank" rel="noopener">ClearML task</a>`,
    y.project_url && `<a href="${y.project_url}" target="_blank" rel="noopener">ClearML project</a>`,
    assets['results.csv'] && `<a href="${assets['results.csv']}" download>results.csv</a>`,
    assets['args.yaml'] && `<a href="${assets['args.yaml']}" download>args.yaml</a>`,
    y.weights_ready && `<a href="${y.task_url}" target="_blank" rel="noopener">best.pt in the model registry</a>`,
  ].filter(Boolean).join('');

  $('#llmBadge').textContent = l.status === 'done' ? 'trained' : 'missing';
  $('#llmBadge').className = 'badge ' + (l.status === 'done' ? 'done' : 'stall');
  $('#llmMetrics').innerHTML =
    metric(l.final_loss != null ? l.final_loss.toFixed(3) : '—', 'final epoch loss') +
    metric(l.steps ?? '—', 'steps') +
    metric(l.epochs ?? '—', 'epochs') +
    metric((l.corpus && (l.corpus.lines ?? l.corpus.examples)) ?? '—', 'corpus lines');

  lineChart($('#llmChart'), {
    title: 'GPT-2 LoRA v4 — training loss',
    xLabel: 'step',
    height: 168,
    series: [{ name: 'loss', color: C.sky,
               points: (l.loss_history || []).map(h => ({ x: h.step, y: h.loss })) }],
    empty: 'No loss history.',
  });

  const ev = l.eval || {};
  if (ev.gpt2_accepted != null) {
    $('#llmNote').innerHTML =
      `<b>${esc(ev.gpt2_accepted)} / ${esc(ev.total ?? '—')}</b> lines passed the grounding validator on held-out SoccerReplay matches; ` +
      `${esc(ev.template_fallback ?? '—')} fell back to grounded templates. The adapter trains cleanly, but its free-form lines drifted, so the validator refuses them by design.`;
  } else {
    $('#llmNote').innerHTML =
      '<b>Validator-guarded.</b> The adapter is trained; lines that invent teams, scores or players are rejected and replaced with grounded templates.';
  }
  const ckpt = (l.trainer_state || '').split(/[\\/]/).slice(-2, -1)[0];
  const rank = ((l.method || '').match(/r=(\d+)/) || [])[1];
  $('#llmLinks').innerHTML = [
    l.task_url && `<a href="${l.task_url}" target="_blank" rel="noopener">ClearML task</a>`,
    ckpt && `<span class="muted" style="font-size:11px">${esc(ckpt)}${rank ? ' · rank ' + esc(rank) : ''}</span>`,
  ].filter(Boolean).join('');

  // pipeline card + tracking note reflect the real model
  if (t.weights_ready) {
    $('#pipeTracker').textContent = 'Aryaman YOLOv5lu';
    $('#trackingNote').textContent = 'Aryaman YOLOv5lu · players, ball, teams, possession';
  } else {
    $('#pipeTracker').textContent = 'YOLOv5lu · training';
    $('#trackingNote').textContent = 'FootballVision checkpoint · Aryaman YOLOv5lu training';
  }

  const cl = d.clearml || {};
  const tasks = cl.tasks || [];
  if (tasks.length) {
    const tbl = $('#clearmlTable');
    tbl.hidden = false;
    tbl.innerHTML = `<thead><tr><th>Task</th><th>Type</th><th>Status</th><th>Link</th></tr></thead><tbody>` +
      tasks.map(t => `<tr><td>${esc(t.name)}</td><td>${esc(t.type || '')}</td>` +
        `<td>${esc(t.status || '')}</td><td><a href="${esc(t.url)}" target="_blank" rel="noopener">open</a></td></tr>`).join('') +
      '</tbody>';
  }

  if (running) {
    if (!trainingTimer) trainingTimer = setInterval(loadTraining, 30000);
  } else if (trainingTimer) {
    clearInterval(trainingTimer); trainingTimer = null;
  }
}

async function loadTraining() {
  if (document.hidden) return;
  let d;
  try { d = await getJSON('/api/training'); } catch { return; }
  renderTraining(d);
}

/* ================================================================ upload */

let jobId = null, poll = null, localUrl = null;
let liveES = null, liveQueue = [], liveTimer = null, allSegs = [];

const STAGE_OF = {
  probe: [0], sampling: [0], detect: [1], labeling: [1], classifier: [1],
  writing: [2, 3], voice: [4], render: [4], done: [5],
};

function setRail(raw) {
  const active = STAGE_OF[raw] || [];
  $$('.rail-step').forEach((el, i) => {
    const st = el.querySelector('.state');
    el.classList.remove('active', 'done');
    if (raw === 'done') { el.classList.add('done'); st.textContent = 'done'; }
    else if (active.includes(i)) { el.classList.add('active'); st.textContent = 'working'; }
    else if (active.length && i < Math.max(...active)) { el.classList.add('done'); st.textContent = 'done'; }
    else { st.textContent = 'waiting'; }
  });
}

function showFile(file) {
  const drop = $('#drop');
  if (!file) return;
  drop.innerHTML = `<b>${esc(file.name)}</b><span class="file">${(file.size / 1e6).toFixed(1)} MB</span><span>click to change</span>`;
}

function setupUpload() {
  const drop = $('#drop'), fileIn = $('#file'), form = $('#callForm');
  const open = () => fileIn.click();
  drop.addEventListener('click', open);
  drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('over');
    if (e.dataTransfer.files.length) { fileIn.files = e.dataTransfer.files; showFile(fileIn.files[0]); }
  });
  fileIn.addEventListener('change', () => showFile(fileIn.files[0]));

  form.addEventListener('submit', async e => {
    e.preventDefault();
    if (!fileIn.files.length) { $('#runMsg').textContent = 'Choose a video first.'; return; }
    const file = fileIn.files[0];
    const fd = new FormData();
    fd.append('file', file);
    fd.append('home', $('#home').value || 'Home');
    fd.append('away', $('#away').value || 'Away');
    fd.append('t0', $('#t0').value || '0');
    fd.append('dur', $('#dur').value || '');
    fd.append('orig_audio', $('#orig').checked);
    fd.append('score_home', $('#scoreHome').value || '0');
    fd.append('score_away', $('#scoreAway').value || '0');
    fd.append('minute_offset', $('#minuteOffset').value || '0');

    stopLive();
    if (localUrl) URL.revokeObjectURL(localUrl);
    localUrl = URL.createObjectURL(file);
    const v = $('#outVideo');
    v.src = localUrl;
    $('#result').hidden = false;
    $('#runEmpty').hidden = true;
    $('#runStats').innerHTML = '';
    $('#download').style.display = 'none';
    $('#transcript').innerHTML = '';
    setRail('');
    $('#bar').style.width = '0%';
    $('#runMsg').textContent = 'Uploading…';
    $('#log').hidden = false;
    $('#log').textContent = '';
    $('#liveCap').hidden = false;
    $('#liveText').textContent = 'LIVE — commentary is generated as the clip plays';
    v.play().catch(() => {});
    $('#go').disabled = true;
    $('#go').textContent = 'Processing…';
    allSegs = [];

    try {
      const r = await fetch('/api/upload', { method: 'POST', body: fd });
      if (!r.ok) throw new Error(await r.text());
      ({ job_id: jobId } = await r.json());
      $('#runMsg').textContent = 'Queued — waiting for the worker.';
      poll = setInterval(tick, 1500);
      startLive();
    } catch (err) {
      $('#runMsg').textContent = 'Upload failed: ' + err.message;
      resetButton();
    }
  });

  $('#outVideo').addEventListener('ended', stopLive);
}

function resetButton() {
  $('#go').disabled = false;
  $('#go').textContent = 'Call it';
}

function renderTranscript(lines) {
  const el = $('#transcript');
  el.innerHTML = (lines || []).map(t => {
    const m = String(t).match(/^\[(\d+:\d+)\]\s*\(([^|)]+)\|?([^)]*)\)\s*(.*)$/);
    if (!m) return `<div class="tline">${esc(t)}</div>`;
    return `<div class="tline"><span class="ts">${esc(m[1])}</span>` +
      `<span class="tag">${esc(m[2].trim())}</span>${esc(m[4])}</div>`;
  }).join('');
}

async function tick() {
  if (!jobId) return;
  let j;
  try { j = await getJSON('/api/job/' + jobId); } catch { return; }
  $('#bar').style.width = (j.progress * 100).toFixed(1) + '%';
  setRail(j.stage);
  $('#runMsg').textContent = j.stage === 'done'
    ? 'Done — the clip is ready.'
    : `${j.stage}${j.message ? ' — ' + j.message : ''}`;
  const log = $('#log');
  log.textContent = (j.log || []).join('\n');
  log.scrollTop = log.scrollHeight;
  if (j.transcript && j.transcript.length) renderTranscript(j.transcript);

  if (j.status === 'done' && j.video_url) {
    clearInterval(poll); poll = null;
    stopLive();
    $('#liveCap').hidden = true;
    const v = $('#outVideo');
    v.src = j.video_url;
    v.load();
    const dl = $('#download');
    dl.href = j.video_url;
    dl.style.display = 'inline-block';
    const s = j.result || {};
    $('#runStats').innerHTML =
      `<div class="stat"><div class="v">${fmtDur(s.clip_duration_s)}</div><div class="k">clip</div></div>` +
      `<div class="stat"><div class="v">${s.segments ?? '—'}</div><div class="k">lines</div></div>` +
      `<div class="stat"><div class="v">${fmtDur(s.speech_seconds)}</div><div class="k">speech</div></div>` +
      `<div class="stat"><div class="v">${s.realtime_ratio ?? '—'}×</div><div class="k">realtime</div></div>`;
    resetButton();
  }
  if (j.status === 'error' || j.status === 'cancelled') {
    clearInterval(poll); poll = null;
    stopLive();
    $('#liveCap').hidden = true;
    $('#runMsg').textContent = j.status === 'error'
      ? 'Failed: ' + (j.error || 'unknown error')
      : 'Cancelled.';
    resetButton();
  }
}

function startLive() {
  liveES = new EventSource(`/api/job/${jobId}/live`);
  liveES.onmessage = e => {
    liveQueue.push(JSON.parse(e.data));
    if (!liveTimer) liveTimer = setInterval(pumpLive, 250);
  };
  liveES.addEventListener('end', () => { if (liveES) { liveES.close(); liveES = null; } });
}

function pumpLive() {
  const now = $('#outVideo').currentTime;
  liveQueue = liveQueue.filter(seg => {
    if (now >= seg.t0 - 0.2) {
      if (now - seg.t0 < 7) playSeg(seg);
      return false;
    }
    return true;
  });
  if (!liveQueue.length && !liveES && liveTimer) {
    clearInterval(liveTimer); liveTimer = null;
  }
}

function playSeg(seg) {
  allSegs.push(seg);
  const a = new Audio(seg.audio_url);
  a.play().catch(() => {});
  $('#liveText').textContent = `LIVE · ${seg.text}`;
  a.addEventListener('ended', () => {
    if ($('#liveText').textContent.startsWith('LIVE ·')) {
      $('#liveText').textContent = 'LIVE — commentary is generated as the clip plays';
    }
  });
}

function stopLive() {
  if (liveES) { liveES.close(); liveES = null; }
  if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
  liveQueue = [];
}

/* ================================================================== init */

loadDemo();
loadTraining();
setupReel();
setupUpload();
window.addEventListener('resize', debounce(() => {
  if (trainingTimer) loadTraining();
  else loadTraining();
}, 240));
