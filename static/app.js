"use strict";
const $ = (id) => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
const mask = document.createElement("canvas"), mctx = mask.getContext("2d", { willReadFrequently: true });
const tint = document.createElement("canvas"), tctx = tint.getContext("2d");

const prefs = Object.assign({ ornD: 3, height: 2.25, quality: "best", border: "none", space: 0, grid: false, mat: "11.5x11.5", text: "none", line2: "", font: "serif", tsize: 0.3 }, readPrefs());
let fonts = []; // lettering choices this computer has
let items = [], cur = null, photo = null, outlinePath = null;
let view = { s: 1, x: 0, y: 0 }, tool = "add", mode = "overlay";
let drag = null, space = false, pointer = null, raf = 0, loadToken = 0;
let tintStale = true, tintMode = "";
let queue = Promise.resolve(); // server calls run one after another, so the last answer is the newest

function readPrefs() { try { return JSON.parse(localStorage.getItem("prefs") || "{}"); } catch { return {}; } }
function savePrefs() { try { localStorage.setItem("prefs", JSON.stringify(prefs)); } catch {} }

async function api(method, url, body) {
  const opt = { method };
  if (body instanceof Blob || body instanceof FormData) opt.body = body;
  else if (body) { opt.body = JSON.stringify(body); opt.headers = { "Content-Type": "application/json" }; }
  let r;
  try { r = await fetch(url, opt); }
  catch { throw new Error("Silhouette Maker is not running. Start it again, then reload this page."); }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `Something went wrong (${r.status}).`);
  return data;
}
function enqueue(fn) { queue = queue.then(fn).catch((e) => busy(e.message, 6000)); return queue; }

let busyTimer = 0;
function busy(text, ms) {
  clearTimeout(busyTimer);
  $("busy").textContent = text || "";
  $("busy").hidden = !text;
  if (text && ms) busyTimer = setTimeout(() => busy(""), ms);
}

/* ---------- list ---------- */
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function flipT(it) { const b = it.outline.bbox; return it.settings.flip ? `translate(${b[0] + b[2]} 0) scale(-1 1)` : ""; }
function thumb(it) {
  const b = it.outline.bbox;
  if (!b) return `<svg viewBox="0 0 1 1"></svg>`;
  const w = b[2] - b[0], h = b[3] - b[1], p = Math.max(w, h) * 0.08, m = Math.max(w, h) + 2 * p;
  return `<svg viewBox="${b[0] - (m - w) / 2} ${b[1] - (m - h) / 2} ${m} ${m}"><path fill-rule="evenodd" transform="${flipT(it)}" d="${it.outline.d}"/></svg>`;
}
function kidHtml(it) { return `<div class="kid${it === cur ? " sel" : ""}" data-id="${it.id}">${thumb(it)}<span>${esc(it.name)}</span></div>`; }
function refreshItem(it) { // one child changed: redraw only its entry
  const el = $("list").querySelector(`[data-id="${it.id}"]`);
  if (el) el.outerHTML = kidHtml(it); else renderList();
}
function renderList() {
  $("list").innerHTML = items.map(kidHtml).join("");
  $("empty").hidden = items.length > 0;
  for (const id of ["bar", "panel", "hint"]) $(id).hidden = !cur;
  $("gallery").hidden = !!cur || !items.length;
  document.body.classList.toggle("home", !cur);
  if (!cur) renderGallery();
}

/* ---------- gallery ---------- */
const picked = new Set();
function shown() {
  const q = $("q").value.trim().toLowerCase(), f = $("filter").value;
  const list = items.filter((it) => (!q || it.name.toLowerCase().includes(q)) && (f === "all" || (f === "done") === it.done));
  if ($("sort").value === "name") list.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  else list.sort((a, b) => b.added - a.added);
  return list;
}
function targets() { return picked.size ? items.filter((it) => picked.has(it.id)) : shown(); }
function renderGallery() {
  for (const id of [...picked]) if (!items.some((it) => it.id === id)) picked.delete(id);
  const list = shown(), n = picked.size, photos = prefs.gView === "photo";
  $("grid").innerHTML = list.map((it) =>
    `<div class="card${picked.has(it.id) ? " picked" : ""}" data-id="${it.id}" title="Open ${esc(it.name)}">` +
    `<div class="art">${photos && it.photo ? `<img loading="lazy" alt="" src="/api/items/${it.id}/thumb?v=${it.rev}">` : thumb(it)}</div>` +
    `<input type="checkbox" class="pick" title="Select"${picked.has(it.id) ? " checked" : ""}>` +
    `<div class="cap"><b>${esc(it.name)}</b><small>${it.outline.bbox ? `${it.outline.height_in.toFixed(2)} in tall` : "nothing to cut yet"}` +
    `${it.done ? ' · <span class="ok">Finished ✓</span>' : ""}</small></div></div>`).join("") ||
    `<div class="none">No photos match.</div>`;
  $("gCount").textContent = list.length === items.length ? `${items.length} photo${items.length === 1 ? "" : "s"}` : `${list.length} of ${items.length}`;
  $("gView").textContent = photos ? "Show silhouettes" : "Show photos";
  $("gAll").textContent = n && list.every((it) => picked.has(it.id)) ? "Select none" : "Select all";
  $("gZip").textContent = n ? `Download ${n} SVG${n > 1 ? "s" : ""}` : "Download all SVGs";
  $("gSheet").textContent = n ? `Sheet of ${n}` : "Sheet of all";
  $("gDel").textContent = n ? `Delete ${n}` : "Delete";
  $("gDel").disabled = $("gDone").disabled = !n;
  const withPhoto = n ? targets().filter((it) => it.photo).length : 0;
  $("gForget").textContent = withPhoto ? `Delete ${withPhoto} photo${withPhoto > 1 ? "s" : ""} only` : "Delete photos only";
  $("gForget").disabled = !withPhoto;
  $("gProof").disabled = !targets().some((it) => it.outline.bbox);
  $("gDone").textContent = n && targets().every((it) => it.done) ? "Mark not finished" : "Mark finished";
  $("gZip").disabled = $("gSheet").disabled = !targets().some((it) => it.outline.bbox);
}
$("grid").addEventListener("click", (e) => {
  const card = e.target.closest(".card");
  if (!card) return;
  const id = card.dataset.id;
  if (e.target.classList.contains("pick")) { e.target.checked ? picked.add(id) : picked.delete(id); renderGallery(); }
  else select(items.find((it) => it.id === id));
});
for (const id of ["q", "filter", "sort"]) $(id).addEventListener("input", renderGallery);
$("gView").onclick = () => { prefs.gView = prefs.gView === "photo" ? "shape" : "photo"; savePrefs(); renderGallery(); };
$("gAll").onclick = () => {
  const list = shown(), all = picked.size && list.every((it) => picked.has(it.id));
  picked.clear();
  if (!all) list.forEach((it) => picked.add(it.id));
  renderGallery();
};
function idsQuery() { return "?ids=" + targets().filter((it) => it.outline.bbox).map((it) => it.id).join(","); }
$("gZip").onclick = () => download("/api/svgs.zip" + idsQuery() + "&" + exportQuery());
$("mat").value = prefs.mat;
if (!$("mat").value) $("mat").value = prefs.mat = "11.5x11.5"; // a saved size this version no longer offers
$("mat").onchange = (e) => { prefs.mat = e.target.value; savePrefs(); };
$("gSheet").onclick = () => {
  const [w, h] = prefs.mat.split("x");
  download(`/api/sheet${idsQuery()}&w=${w}&h=${h}&${exportQuery()}`);
};
$("gDone").onclick = () => {
  const list = targets(), done = !list.every((it) => it.done);
  enqueue(async () => {
    for (const it of list) { await api("PATCH", `/api/items/${it.id}`, { done }); it.done = done; }
    renderList();
  });
};
$("gDel").onclick = () => {
  const list = targets();
  if (!confirm(`Delete ${list.length} photo${list.length > 1 ? "s" : ""} and ${list.length > 1 ? "their silhouettes" : "its silhouette"} from this computer?`)) return;
  enqueue(async () => {
    for (const it of list) { await api("DELETE", `/api/items/${it.id}`); items.splice(items.indexOf(it), 1); }
    renderList();
  });
};
$("gProof").onclick = async () => {
  flushSettings();
  await queue;
  window.open(`/proof${idsQuery()}&${exportQuery()}`, "_blank");
};
$("gForget").onclick = () => {
  const list = targets().filter((it) => it.photo);
  if (!confirm(`Delete the original photo${list.length > 1 ? "s" : ""} of ${list.length} ${list.length > 1 ? "children" : "child"} and keep the silhouette${list.length > 1 ? "s" : ""}? They cannot be cut out again afterwards.`)) return;
  enqueue(async () => {
    for (const it of list) Object.assign(it, await api("POST", `/api/items/${it.id}/forget`), { name: it.name });
    renderList();
    busy(`${list.length} photo${list.length > 1 ? "s" : ""} deleted. The silhouettes are kept.`, 5000);
  });
};
$("backupBtn").onclick = () => download("/api/backup");
$("restoreBtn").onclick = () => $("restoreFile").click();
$("restoreFile").onchange = (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f) return;
  busy("Restoring…");
  enqueue(async () => {
    const fd = new FormData();
    fd.append("backup", f);
    const r = await api("POST", "/api/restore", fd);
    items = await api("GET", "/api/items");
    if (cur) cur = items.find((it) => it.id === cur.id) || null;
    renderList();
    busy(`${r.added} photo${r.added === 1 ? "" : "s"} restored` + (r.skipped ? `, ${r.skipped} already here` : "") + ".", 6000);
  });
};
$("list").addEventListener("click", (e) => {
  const el = e.target.closest(".kid");
  if (el) select(items.find((it) => it.id === el.dataset.id));
});

/* ---------- loading an item ---------- */
function loadImage(url) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = url; });
}
async function loadMask(it) { putMask(await loadImage(`/api/items/${it.id}/mask?t=${Date.now()}`), it); }
function putMask(im, it) { // a black-and-white picture -> the editable shape
  mask.width = tint.width = it.w; mask.height = tint.height = it.h;
  mctx.globalCompositeOperation = "source-over";
  mctx.drawImage(im, 0, 0);
  const d = mctx.getImageData(0, 0, it.w, it.h), a = d.data;
  for (let i = 0; i < a.length; i += 4) { a[i + 3] = a[i] > 127 ? 255 : 0; a[i] = a[i + 1] = a[i + 2] = 255; }
  mctx.putImageData(d, 0, 0);
  tintStale = true;
}
async function select(it) {
  const token = ++loadToken;
  if (!it) {
    cur = null; photo = null; renderList(); draw();
    if (location.hash) history.pushState(null, "", location.pathname); // so Back reopens it
    return;
  }
  flushSettings();
  await queue;
  await openItem(it, token);
}
async function openItem(it, token) {
  const im = it.photo ? await loadImage(`/api/items/${it.id}/photo?v=${it.rev}`) : null;
  if (token !== loadToken) return;
  cur = it; photo = im;
  if (location.hash !== "#" + it.id) history.pushState(null, "", "#" + it.id); // Back closes; a reload reopens
  await loadMask(it);
  await startHist("As opened");
  renderList(); fillPanel(); showOutline(); resize(); fit();
}

/* ---------- panel ---------- */
function fillPanel() {
  const s = cur.settings;
  $("name").value = cur.name;
  $("height").value = s.height_in; $("ornD").value = prefs.ornD;
  $("done").checked = cur.done;
  $("flip").checked = s.flip; $("cutOn").checked = s.cut_on;
  $("smooth").value = s.smooth; $("detail").value = s.detail_mm; $("tilt").value = s.tilt_deg || 0;
  $("quality").value = prefs.quality;
  $("border").value = prefs.border; $("space").value = prefs.space; $("space").disabled = prefs.border === "circle";
  $("text").value = prefs.text; $("line2").value = prefs.line2; $("font").value = prefs.font; $("tsize").value = prefs.tsize;
  $("line2").disabled = prefs.text !== "both"; $("textMore").hidden = prefs.text === "none";
  $("redo").disabled = $("quality").disabled = !cur.photo;
  $("forget").hidden = !cur.photo;
  labels();
}
function labels() {
  $("smoothV").textContent = cur.settings.smooth;
  $("tiltV").textContent = `${(+cur.settings.tilt_deg || 0).toFixed(1)}°`;
  $("detailV").textContent = cur.settings.detail_mm ? `${(+cur.settings.detail_mm).toFixed(1)} mm` : "everything";
}
function showOutline() {
  const o = cur.outline;
  outlinePath = o.d ? new Path2D(o.d) : null;
  const D = +prefs.ornD || 3, pad = 0.15;
  let shape = "";
  if (o.bbox) {
    // same layout as the files: shape, a small gap, then the lines of text, all centred on the ornament
    const b = o.bbox, k = cur.settings.height_in / (b[3] - b[1]), lines = textLines(cur);
    const block = lines.length ? 0.12 + lines.reduce((a, l) => a + 1.2 * l.size, 0) : 0;
    let y = o.height_in / 2 - block / 2 + 0.12;
    const css = (fonts.find((f) => f.key === prefs.font) || {}).css || "";
    const text = lines.map((l) => {
      const t = `<text x="0" y="${y + 0.78 * l.size}" font-size="${l.size}" text-anchor="middle" fill="#3a3a3a" style="${css}">${esc(l.text)}</text>`;
      y += 1.2 * l.size;
      return t;
    }).join("");
    shape = `<path fill-rule="evenodd" transform="translate(0 ${-block / 2}) scale(${cur.settings.flip ? -k : k} ${k}) translate(${-(b[0] + b[2]) / 2} ${-(b[1] + b[3]) / 2})" d="${o.d}"/>` + text;
  }
  $("orn").setAttribute("viewBox", `${-D / 2 - pad} ${-D / 2 - pad} ${D + 2 * pad} ${D + 2 * pad}`);
  $("orn").innerHTML = `<circle r="${D / 2}" fill="#d9b27c" stroke="#b08850" stroke-width="${D * 0.008}"/>` +
    `<circle cy="${-D / 2 + D * 0.06}" r="${D * 0.02}" fill="#17191c"/>` + shape;
  if (!o.bbox) $("size").textContent = "Nothing left to cut. Paint with Add, or move the neckline.";
  else {
    const big = Math.hypot(o.width_in, o.height_in) > D;
    $("size").textContent = `${o.height_in.toFixed(2)} in tall × ${o.width_in.toFixed(2)} in wide` + (big ? " · may not fit the ornament" : "");
  }
  $("svgBtn").disabled = $("pngBtn").disabled = !o.bbox;
  refreshItem(cur); draw();
}
let patchTimer = 0, patchWaiting = null; // patchWaiting: the photo whose change has not been sent yet
function pushSettings(delay = 180) {
  clearTimeout(patchTimer);
  patchWaiting = cur;
  patchTimer = setTimeout(flushSettings, delay);
}
function flushSettings() { // send the waiting change now
  clearTimeout(patchTimer);
  const it = patchWaiting;
  if (!it) return;
  patchWaiting = null;
  enqueue(async () => {
    const r = await api("PATCH", `/api/items/${it.id}`, { name: it.name, done: it.done, settings: it.settings });
    it.outline = r.outline;
    if (it === cur) showOutline(); else renderList();
  });
}
addEventListener("pagehide", () => { // closing the tab mid-typing: still save it
  const it = patchWaiting;
  if (it) fetch(`/api/items/${it.id}`, { method: "PATCH", keepalive: true, headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: it.name, done: it.done, settings: it.settings }) });
});
function bind(id, ev, fn) { $(id).addEventListener(ev, (e) => { if (cur) fn(e.target); }); }
bind("name", "input", (t) => { cur.name = t.value; pushSettings(400); if (prefs.text !== "none") showOutline(); });
bind("height", "change", (t) => {
  const v = Math.min(24, Math.max(0.25, +t.value || 2.25));
  t.value = v; cur.settings.height_in = prefs.height = v; savePrefs(); pushSettings(0); record("Size");
});
$("sizeAll").onclick = () => {
  const h = cur.settings.height_in;
  if (items.length < 2) return busy("This is the only photo so far.", 3000);
  if (!confirm(`Make all ${items.length} photos ${h} in tall?`)) return;
  flushSettings();
  busy("Resizing everyone…");
  enqueue(async () => {
    for (const r of await api("POST", "/api/height", { height_in: h })) {
      const it = items.find((i) => i.id === r.id);
      if (it) { it.settings.height_in = r.settings.height_in; it.outline = r.outline; }
    }
    prefs.height = h; savePrefs(); // new photos get it too
    renderList();
    if (cur) { fillPanel(); showOutline(); }
    busy(`All ${items.length} photos are now ${h} in tall.`, 4000);
  });
};
bind("ornD", "change", (t) => { prefs.ornD = Math.min(24, Math.max(0.5, +t.value || 3)); t.value = prefs.ornD; savePrefs(); showOutline(); });
bind("done", "change", (t) => { cur.done = t.checked; pushSettings(0); });
bind("flip", "change", (t) => { cur.settings.flip = t.checked; pushSettings(0); record("Face the other way"); });
bind("cutOn", "change", (t) => { cur.settings.cut_on = t.checked; pushSettings(0); draw(); record(t.checked ? "Neckline on" : "Neckline off"); });
bind("tilt", "change", () => record("Straighten")); // "change" fires once, when the slider is let go
bind("smooth", "change", () => record("Smoothing"));
bind("detail", "change", () => record("Thinnest detail"));
bind("tilt", "input", (t) => { cur.settings.tilt_deg = +t.value; labels(); draw(); pushSettings(); });
bind("tilt", "dblclick", (t) => { t.value = 0; cur.settings.tilt_deg = 0; labels(); draw(); pushSettings(0); record("Straighten"); });
bind("smooth", "input", (t) => { cur.settings.smooth = +t.value; labels(); pushSettings(); });
bind("detail", "input", (t) => { cur.settings.detail_mm = +t.value; labels(); pushSettings(); });
$("border").addEventListener("change", (e) => { prefs.border = e.target.value; $("space").disabled = prefs.border === "circle"; savePrefs(); });
$("space").addEventListener("change", (e) => { prefs.space = Math.min(6, Math.max(0, +e.target.value || 0)); e.target.value = prefs.space; savePrefs(); });
function exportQuery() {
  return `border=${prefs.border}&space=${prefs.space}&orn=${prefs.ornD}` +
    (fonts.length && prefs.text !== "none" ? `&text=${prefs.text}&line2=${encodeURIComponent(prefs.line2)}&font=${prefs.font}&tsize=${prefs.tsize}` : "");
}
function textLines(it) { // [{text, size in inches}] under the shape, as the files will have
  if (!fonts.length || prefs.text === "none") return [];
  const out = [], size = +prefs.tsize || 0.3;
  if (it.name.trim()) out.push({ text: it.name.trim(), size });
  if (prefs.text === "both" && prefs.line2.trim()) out.push({ text: prefs.line2.trim(), size: size * 0.8 });
  return out;
}
function textChanged() { savePrefs(); if (cur) { fillPanel(); showOutline(); } }
$("text").addEventListener("change", (e) => { prefs.text = e.target.value; textChanged(); });
$("font").addEventListener("change", (e) => { prefs.font = e.target.value; textChanged(); });
$("line2").addEventListener("input", (e) => { prefs.line2 = e.target.value; savePrefs(); if (cur) showOutline(); });
$("tsize").addEventListener("change", (e) => { prefs.tsize = Math.min(2, Math.max(0.1, +e.target.value || 0.3)); textChanged(); });
$("quality").addEventListener("change", (e) => { prefs.quality = e.target.value; savePrefs(); });

async function download(url) {
  flushSettings(); // the file must match what is on screen
  await queue;
  const a = document.createElement("a");
  a.href = url; a.download = "";
  document.body.append(a); a.click(); a.remove();
}
$("svgBtn").onclick = () => download(`/api/items/${cur.id}/export.svg?${exportQuery()}`);
$("pngBtn").onclick = () => download(`/api/items/${cur.id}/export.png?${exportQuery()}`);
$("redo").onclick = () => {
  if (!confirm("Run the automatic cut-out again? It replaces your brush edits and becomes the new starting point. (History can still take you back while this photo is open.)")) return;
  const it = cur;
  busy("Cutting out again…");
  enqueue(async () => {
    it.outline = await api("POST", `/api/items/${it.id}/redo`, { quality: prefs.quality });
    if (it === cur) { await loadMask(it); showOutline(); pushHist("New cut-out", await maskBlob(), JSON.stringify(it.settings)); }
    busy(it.outline.note || "", 8000);
  });
};
$("forget").onclick = () => {
  if (!confirm(`Delete the original photo of "${cur.name}" and keep the silhouette? It cannot be cut out again afterwards.`)) return;
  const it = cur;
  enqueue(async () => {
    Object.assign(it, await api("POST", `/api/items/${it.id}/forget`), { name: it.name });
    if (it === cur) { photo = null; fillPanel(); draw(); }
    renderList();
    busy("Photo deleted. The silhouette is kept.", 5000);
  });
};
$("del").onclick = () => {
  if (!confirm(`Delete "${cur.name}" and its photo from this computer?`)) return;
  const it = cur;
  enqueue(async () => {
    await api("DELETE", `/api/items/${it.id}`);
    const i = items.indexOf(it);
    items.splice(i, 1);
    if (it === cur) await select(null);
    else renderList();
  });
};

/* ---------- adding photos ---------- */
$("addBtn").onclick = () => $("file").click();
$("file").onchange = (e) => { addFiles([...e.target.files]); e.target.value = ""; };
async function addFiles(files) {
  files = files.filter((f) => f.type.startsWith("image/") || /\.(jpe?g|png|webp|bmp|heic)$/i.test(f.name));
  let first = null, failed = 0, note = "";
  for (let i = 0; i < files.length; i++) {
    const text = `Cutting out ${i + 1} of ${files.length}…`;
    busy(text);
    const slow = setTimeout(() => busy(text + " The first time, the cut-out model is downloaded (up to 1 GB), so this can take a few minutes."), 30000);
    const fd = new FormData();
    fd.append("photo", files[i]); fd.append("quality", prefs.quality); fd.append("height_in", prefs.height);
    try {
      const it = await api("POST", "/api/items", fd);
      items.push(it); first = first || it; note = it.note || note; renderList();
    } catch (e) { failed++; note = e.message; }
    clearTimeout(slow);
  }
  busy(failed > 1 ? `${failed} photos could not be added. ${note}` : note, 8000);
  if (first && !cur && files.length - failed === 1) select(first); // one photo: straight into the editor
}
addEventListener("dragover", (e) => { e.preventDefault(); document.body.classList.add("drag"); });
addEventListener("dragleave", (e) => { if (!e.relatedTarget) document.body.classList.remove("drag"); });
addEventListener("drop", (e) => { e.preventDefault(); document.body.classList.remove("drag"); addFiles([...e.dataTransfer.files]); });

/* ---------- canvas ---------- */
function resize() {
  const r = $("stage").getBoundingClientRect(), dpr = devicePixelRatio || 1;
  cv.width = Math.max(1, Math.round(r.width * dpr)); cv.height = Math.max(1, Math.round(r.height * dpr));
  draw();
}
new ResizeObserver(resize).observe($("stage"));
function fit() {
  if (!cur) return;
  const r = $("stage").getBoundingClientRect();
  view.s = Math.min(r.width / cur.w, r.height / cur.h) * 0.94;
  view.x = (r.width - cur.w * view.s) / 2; view.y = (r.height - cur.h * view.s) / 2;
  draw();
}
$("fit").onclick = fit;
function draw() { if (!raf) raf = requestAnimationFrame(render); }
function cutPts() {
  const c = cur.settings.cut;
  if (!c) return null;
  const p = c.map(([x, y]) => ({ x: x * cur.w, y: y * cur.h }));
  // the middle dot sits on the curve; p[1] is the pull point behind it
  const mid = { x: 0.25 * p[0].x + 0.5 * p[1].x + 0.25 * p[2].x, y: 0.25 * p[0].y + 0.5 * p[1].y + 0.25 * p[2].y };
  return { p, handles: [p[0], mid, p[2]] };
}
function render() {
  raf = 0;
  const dpr = devicePixelRatio || 1;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!cur) return;
  ctx.setTransform(dpr * view.s, 0, 0, dpr * view.s, dpr * view.x, dpr * view.y);
  const px = 1 / view.s; // one screen pixel, in photo pixels
  // the photo and painted shape are shown straightened; the cut line and neckline already are
  ctx.save();
  ctx.translate(cur.w / 2, cur.h / 2); ctx.rotate(tiltRad()); ctx.translate(-cur.w / 2, -cur.h / 2);
  if (mode === "shape") { ctx.fillStyle = "#f4efe6"; ctx.fillRect(0, 0, cur.w, cur.h); }
  else if (photo) ctx.drawImage(photo, 0, 0);
  else { ctx.fillStyle = "#e9e4da"; ctx.fillRect(0, 0, cur.w, cur.h); } // the photo was deleted
  if (mode !== "photo") {
    if (tintStale || tintMode !== mode) {
      tctx.globalCompositeOperation = "copy"; tctx.drawImage(mask, 0, 0);
      tctx.globalCompositeOperation = "source-in";
      tctx.fillStyle = mode === "shape" ? "#d3cbbd" : "#2f7df6"; tctx.fillRect(0, 0, tint.width, tint.height);
      tintStale = false; tintMode = mode;
    }
    ctx.globalAlpha = mode === "shape" ? 1 : 0.5; ctx.drawImage(tint, 0, 0); ctx.globalAlpha = 1;
  }
  ctx.restore();
  if (prefs.grid) {
    const step = Math.min(cur.w, cur.h) / 8, lines = (from, pick) => {
      ctx.beginPath();
      for (let x = from(cur.w); x <= cur.w; x += pick(cur.w)) { ctx.moveTo(x, 0); ctx.lineTo(x, cur.h); }
      for (let y = from(cur.h); y <= cur.h; y += pick(cur.h)) { ctx.moveTo(0, y); ctx.lineTo(cur.w, y); }
      ctx.lineWidth = 2.4 * px; ctx.strokeStyle = "rgba(0,0,0,.35)"; ctx.stroke();
    };
    lines((n) => (n / 2) % step, () => step);
    ctx.lineWidth = 1 * px; ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.stroke();
    lines((n) => n / 2, (n) => n); // the two centre lines, stronger
    ctx.lineWidth = 1.4 * px; ctx.strokeStyle = "#7fd1ff"; ctx.stroke();
  }
  if (outlinePath && mode === "shape") {
    // grey = painted area that will not be cut (below the neckline, wisps); black = the cut piece
    ctx.fillStyle = "#111"; ctx.fill(outlinePath, "evenodd");
  } else if (outlinePath) {
    ctx.lineJoin = "round";
    ctx.lineWidth = 3.5 * px; ctx.strokeStyle = "rgba(0,0,0,.55)"; ctx.stroke(outlinePath);
    ctx.lineWidth = 1.6 * px; ctx.strokeStyle = "#ffb347"; ctx.stroke(outlinePath);
  }
  const c = cur.settings.cut_on && cutPts();
  if (c) {
    const [a, q, b] = c.p, L = a.x <= b.x ? a : b, R = a.x <= b.x ? b : a;
    ctx.beginPath(); ctx.moveTo(-1e5, L.y); ctx.lineTo(L.x, L.y);
    ctx.moveTo(a.x, a.y); ctx.quadraticCurveTo(q.x, q.y, b.x, b.y);
    ctx.moveTo(R.x, R.y); ctx.lineTo(1e5, R.y);
    ctx.setLineDash([7 * px, 5 * px]);
    ctx.lineWidth = 3 * px; ctx.strokeStyle = "rgba(0,0,0,.5)"; ctx.stroke();
    ctx.lineWidth = 1.5 * px; ctx.strokeStyle = "#fff"; ctx.stroke();
    ctx.setLineDash([]);
    if (tool === "cut") for (const h of c.handles) {
      ctx.beginPath(); ctx.arc(h.x, h.y, 7 * px, 0, 7);
      ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = 2 * px; ctx.strokeStyle = "#f0a23b"; ctx.stroke();
    }
  }
  if (pointer && (tool === "add" || tool === "erase") && !space) {
    ctx.beginPath(); ctx.arc(pointer.x, pointer.y, (+$("brush").value / 2) * px, 0, 7);
    ctx.lineWidth = 2.5 * px; ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.stroke();
    ctx.lineWidth = 1 * px; ctx.strokeStyle = "#fff"; ctx.stroke();
  }
}

function pt(e) {
  const r = cv.getBoundingClientRect();
  return { x: (e.clientX - r.left - view.x) / view.s, y: (e.clientY - r.top - view.y) / view.s };
}
function tiltRad() { return ((cur && +cur.settings.tilt_deg) || 0) * Math.PI / 180; }
function toPhoto(p) { // a point on the straightened picture -> the same spot on the stored photo
  const t = -tiltRad(), x = p.x - cur.w / 2, y = p.y - cur.h / 2;
  return { x: x * Math.cos(t) - y * Math.sin(t) + cur.w / 2, y: x * Math.sin(t) + y * Math.cos(t) + cur.h / 2 };
}
function dab(a, b) {
  mctx.globalCompositeOperation = tool === "add" ? "source-over" : "destination-out";
  mctx.strokeStyle = "#fff"; mctx.lineCap = mctx.lineJoin = "round";
  mctx.lineWidth = +$("brush").value / view.s;
  mctx.beginPath(); mctx.moveTo(a.x, a.y); mctx.lineTo(b.x + 0.01, b.y + 0.01); mctx.stroke();
  tintStale = true;
}
function maskBlob() { // the painted shape as a small black-and-white PNG
  const c = document.createElement("canvas");
  c.width = mask.width; c.height = mask.height;
  const x = c.getContext("2d");
  x.fillStyle = "#000"; x.fillRect(0, 0, c.width, c.height);
  x.drawImage(mask, 0, 0);
  return new Promise((res) => c.toBlob(res, "image/png"));
}

/* ---------- history ----------
   One entry per step since the photo was opened: the shape (a PNG, a few tens
   of KB) and the settings. hist[hi] is where you are; later ones are undone. */
let hist = [], hi = -1;
const HIST_MAX = 100;
function pushHist(label, blob, settings) {
  hist.length = hi + 1; // a new step drops the undone ones
  hist.push({ label, blob, settings, t: new Date() });
  if (hist.length > HIST_MAX) hist.splice(1, 1); // the first entry always stays
  hi = hist.length - 1;
  renderHist();
}
async function startHist(label) {
  hist = []; hi = -1;
  pushHist(label, await maskBlob(), JSON.stringify(cur.settings));
}
function record(label) { // a step that changed settings only
  const it = cur, settings = JSON.stringify(it.settings);
  enqueue(async () => { if (it === cur && hi >= 0) pushHist(label, hist[hi].blob, settings); });
}
function commitMask(label) { // a brush stroke: remember it and save it
  const it = cur;
  enqueue(async () => {
    if (it !== cur) return; // the shared mask canvas now holds another photo
    const blob = await maskBlob();
    pushHist(label, blob, JSON.stringify(it.settings));
    it.outline = await api("PUT", `/api/items/${it.id}/mask`, blob);
    if (it === cur) showOutline(); else renderList();
  });
}
function goTo(i, relative = false) { // relative: i steps from wherever we are when its turn comes
  const it = cur;
  enqueue(async () => {
    if (relative) i += hi; // worked out here, so quick repeated Undo presses each count
    if (it !== cur || i < 0 || i >= hist.length || i === hi) return;
    const st = hist[i], maskChanged = st.blob !== hist[hi].blob;
    hi = i;
    if (maskChanged) {
      putMask(await createImageBitmap(st.blob), it);
      await api("PUT", `/api/items/${it.id}/mask`, st.blob);
    }
    it.settings = JSON.parse(st.settings);
    it.outline = (await api("PATCH", `/api/items/${it.id}`, { settings: it.settings })).outline;
    if (it === cur) { fillPanel(); showOutline(); renderHist(); } else renderList();
  });
}
function renderHist() {
  $("undo").disabled = hi <= 0;
  $("redoStep").disabled = hi >= hist.length - 1;
  $("histBtn").textContent = hist.length > 1 ? `History (${hist.length - 1})` : "History";
  $("histList").innerHTML = hist.map((h, i) =>
    `<div class="h${i === hi ? " cur" : i > hi ? " undone" : ""}" data-i="${i}"><span>${i ? i + ". " : ""}${esc(h.label)}</span>` +
    `<small>${h.t.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}</small></div>`).reverse().join("");
}
function showHist(on) { $("hist").hidden = !on; $("histBtn").classList.toggle("on", on); }
$("histBtn").onclick = () => showHist($("hist").hidden);
$("histList").addEventListener("click", (e) => { const el = e.target.closest(".h"); if (el) goTo(+el.dataset.i); });
$("redoStep").onclick = () => goTo(1, true);
$("resetBtn").onclick = () => {
  if (!confirm("Put this photo back to its first automatic cut-out and starting settings? (History can still take you back while this photo is open.)")) return;
  const it = cur;
  busy("Putting it back…");
  enqueue(async () => {
    const r = await api("POST", `/api/items/${it.id}/reset`, { quality: prefs.quality });
    Object.assign(it, r, { name: it.name, done: it.done });
    if (it === cur) {
      await loadMask(it); fillPanel(); showOutline();
      pushHist("Reset to how it started", await maskBlob(), JSON.stringify(it.settings));
    } else renderList();
    busy(r.note || "", 8000);
  });
};
cv.addEventListener("contextmenu", (e) => e.preventDefault());
cv.addEventListener("pointerdown", (e) => {
  if (!cur) return;
  cv.setPointerCapture(e.pointerId);
  const p = pt(e);
  if (e.button !== 0 || space) { drag = { kind: "pan", cx: e.clientX, cy: e.clientY, x: view.x, y: view.y }; return; }
  if (tool === "cut") {
    const c = cutPts();
    if (!c) return;
    const i = c.handles.findIndex((h) => Math.hypot(h.x - p.x, h.y - p.y) < 16 / view.s);
    if (i >= 0) drag = { kind: "cut", i };
    else drag = { kind: "pan", cx: e.clientX, cy: e.clientY, x: view.x, y: view.y };
    return;
  }
  drag = { kind: "paint", last: toPhoto(p), label: tool === "add" ? "Add" : "Erase" };
  dab(drag.last, drag.last); draw();
});
cv.addEventListener("pointermove", (e) => {
  if (!cur) return;
  const p = pointer = pt(e);
  if (drag?.kind === "pan") { view.x = drag.x + e.clientX - drag.cx; view.y = drag.y + e.clientY - drag.cy; }
  else if (drag?.kind === "paint") {
    for (const ev of (e.getCoalescedEvents?.().length ? e.getCoalescedEvents() : [e])) {
      const q = toPhoto(pt(ev)); dab(drag.last, q); drag.last = q;
    }
  }
  else if (drag?.kind === "cut") {
    const c = cur.settings.cut, f = [Math.min(1, Math.max(0, p.x / cur.w)), Math.min(1, Math.max(0, p.y / cur.h))];
    if (drag.i === 1) c[1] = [2 * f[0] - 0.5 * (c[0][0] + c[2][0]), 2 * f[1] - 0.5 * (c[0][1] + c[2][1])];
    else c[drag.i] = f;
  }
  draw();
});
function endDrag() {
  if (drag?.kind === "paint") commitMask(drag.label);
  if (drag?.kind === "cut") { pushSettings(0); record("Neckline"); }
  drag = null;
}
cv.addEventListener("pointerup", endDrag);
cv.addEventListener("pointercancel", endDrag);
cv.addEventListener("pointerleave", () => { pointer = null; draw(); });
cv.addEventListener("wheel", (e) => {
  if (!cur) return;
  e.preventDefault();
  const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  const s = Math.min(40, Math.max(0.05, view.s * Math.exp(-e.deltaY * 0.0015)));
  view.x = mx - (mx - view.x) * (s / view.s); view.y = my - (my - view.y) * (s / view.s); view.s = s;
  draw();
}, { passive: false });

function undo() { if (cur) goTo(-1, true); }
$("undo").onclick = undo;
$("close").onclick = () => select(null);
function setGrid(on) { prefs.grid = on; savePrefs(); $("gridBtn").classList.toggle("on", on); draw(); }
$("gridBtn").onclick = () => setGrid(!prefs.grid);
$("gridBtn").classList.toggle("on", prefs.grid);
function rotate(turns) {
  const it = cur;
  enqueue(async () => {
    const r = await api("POST", `/api/items/${it.id}/rotate`, { turns });
    Object.assign(it, r, { name: it.name, done: it.done }); // keep a rename that is still on its way
    if (it === cur) await openItem(it, ++loadToken); else renderList();
  });
}
$("rotL").onclick = () => rotate(-1);
$("rotR").onclick = () => rotate(1);

function setTool(t) {
  tool = t;
  if (t === "cut" && cur && !cur.settings.cut_on) { cur.settings.cut_on = true; $("cutOn").checked = true; pushSettings(0); record("Neckline on"); }
  document.querySelectorAll("[data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === tool));
  $("brush").disabled = t === "cut";
  cv.style.cursor = t === "cut" ? "grab" : "none";
  draw();
}
function setMode(m) {
  mode = m;
  document.querySelectorAll("[data-view]").forEach((b) => b.classList.toggle("on", b.dataset.view === mode));
  draw();
}
document.querySelectorAll("[data-tool]").forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
document.querySelectorAll("[data-view]").forEach((b) => (b.onclick = () => setMode(b.dataset.view)));
$("brush").addEventListener("input", draw);

addEventListener("keydown", (e) => {
  if (/^(text|number|select-one)$/.test(e.target.type || "")) return;
  if (e.key === "Escape" && !$("hist").hidden) { showHist(false); return; }
  if (e.key === "Escape" && cur) { select(null); return; }
  if (e.code === "Space") { space = true; e.preventDefault(); draw(); return; }
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && (k === "y" || (k === "z" && e.shiftKey))) { e.preventDefault(); if (cur) goTo(1, true); }
  else if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); undo(); }
  else if (e.ctrlKey || e.metaKey || e.altKey) return;
  else if (k === "g") setGrid(!prefs.grid);
  else if (k === "f") fit();
  else if (k === "b") setTool("add");
  else if (k === "e") setTool("erase");
  else if (k === "n") setTool("cut");
  else if (k === "[" || k === "]") {
    const b = $("brush");
    b.value = Math.round(+b.value * (k === "]" ? 1.25 : 0.8) + (k === "]" ? 1 : -1));
    draw();
  }
});
addEventListener("keyup", (e) => { if (e.code === "Space") { space = false; draw(); } });

/* ---------- start ---------- */
setTool("add"); setMode("overlay");
(async () => {
  try {
    items = await api("GET", "/api/items");
    fonts = await api("GET", "/api/fonts");
  } catch (e) { busy(e.message); }
  $("textRows").hidden = !fonts.length; // no usable font on this computer: no lettering
  $("font").innerHTML = fonts.map((f) => `<option value="${f.key}">${esc(f.label)}</option>`).join("");
  if (fonts.length && !fonts.some((f) => f.key === prefs.font)) prefs.font = fonts[0].key;
  renderList();
  fromHash();
})();
function fromHash() {
  const it = items.find((i) => "#" + i.id === location.hash);
  if (it !== (cur || undefined)) select(it || null);
}
addEventListener("popstate", fromHash);
