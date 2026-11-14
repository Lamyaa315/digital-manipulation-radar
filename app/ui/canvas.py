"""Two interactive canvas components (Streamlit components v2, no build step, no CDN).

* network(): the Living Network — zoom (wheel), pan (drag background), drag nodes,
  hover tooltips, click a node to light up its links and its group; the click is sent
  back to Python, so the rest of the page follows the selection.
* radar(): the Coordination Radar — the sweep is a visual metaphor only. Blips are real
  accounts placed by their held-out model score (closer to the centre = higher score);
  each sector is one detected group. Nothing is drawn until real results exist.
"""
import streamlit as st

CSS = """
.wrap { position:relative; width:100%; user-select:none; overflow:hidden; }
canvas { display:block; width:100%; cursor:grab; }
canvas.drag { cursor:grabbing; }
.tip { position:absolute; pointer-events:none; background:rgba(255,255,255,.97); border:1px solid #0E7C86; box-shadow:0 4px 14px rgba(0,0,0,.08);
  color:#1D2433; font:12px/1.5 Inter,sans-serif; padding:8px 10px; letter-spacing:.04em;
  opacity:0; transition:opacity .12s; white-space:nowrap; z-index:2; }
.tip b { color:#0E7C86; font-weight:500; } .tip .c { color:#D64550; }
.hud { position:absolute; left:10px; bottom:8px; font:12px Inter,sans-serif; color:#87919F; pointer-events:none; }
.hud i { display:inline-block; width:8px; height:8px; border-radius:50%; margin:0 5px 0 12px; vertical-align:-1px; }
"""

NET_JS = r"""
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  let wrap = parentElement.querySelector('.wrap');
  if (!wrap) {
    wrap = document.createElement('div'); wrap.className = 'wrap';
    wrap.innerHTML = '<canvas></canvas><div class="tip"></div><div class="hud"></div>';
    parentElement.appendChild(wrap);
  }
  const S = wrap.__s || (wrap.__s = {});
  if (S.raf) cancelAnimationFrame(S.raf);
  if (S.ro) S.ro.disconnect();
  const cv = wrap.querySelector('canvas'), tip = wrap.querySelector('.tip'), hud = wrap.querySelector('.hud');
  const ctx = cv.getContext('2d');
  const H = data.height || 560;
  const nodes = data.nodes, byId = {};
  nodes.forEach((n, i) => { n.i = i; n.ph = (i * 2.399) % 6.283; n.adj = []; byId[n.id] = n; });
  const links = data.links.map(([u, v, s]) => ({ a: byId[u], b: byId[v], s })).filter(l => l.a && l.b);
  links.forEach(l => { l.a.adj.push(l); l.b.adj.push(l); });
  const maxS = Math.max(3, ...links.map(l => l.s));
  const sig = nodes.length + ':' + (nodes[0] ? nodes[0].id : '');
  if (S.sig !== sig) { S.sig = sig; S.view = { k: 1, tx: 0, ty: 0 }; S.sel = data.selected_node || null;
                       S.grp = data.selected_group || null; S.moved = {}; }
  else { if (data.selected_group !== undefined) S.grp = data.selected_group || S.grp; }
  hud.innerHTML = (data.caption || '') + '<i style="background:#D64550"></i>above review threshold' +
                  '<i style="background:#7C8AA0"></i>below' + '<i style="background:#0E7C86"></i>selected links';

  // world bounds -> fit
  const qs = (a, q) => { const s = [...a].sort((p, r) => p - r); return s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))] || 0; };
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y), lo = nodes.length > 40 ? 0.02 : 0, hi = 1 - lo;
  const x0 = qs(xs, lo), x1 = qs(xs, hi), y0 = qs(ys, lo), y1 = qs(ys, hi);
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, span = Math.max(x1 - x0, y1 - y0, 0.2);
  let W = 800, fit = 1, dpr = window.devicePixelRatio || 1;
  function resize() {
    W = wrap.clientWidth || 800; cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + 'px';
    fit = Math.min(W, H) * 0.86 / span;
  }
  resize(); S.ro = new ResizeObserver(resize); S.ro.observe(wrap);
  const groups = {};
  nodes.forEach(n => { (groups[n.group] = groups[n.group] || []).push(n); });

  const V = () => S.view;
  const sx = n => (n.x - cx + (S.moved[n.id] ? S.moved[n.id][0] : 0)) * fit * V().k + W / 2 + V().tx;
  const sy = n => (n.y - cy + (S.moved[n.id] ? S.moved[n.id][1] : 0)) * fit * V().k + H / 2 + V().ty;
  const rad = n => Math.min(7, 2.2 + Math.sqrt(n.deg) * 0.32) * Math.sqrt(V().k);
  let T = 0;
  function px(n) { return [sx(n), sy(n)]; }

  function hit(mx, my) {
    let best = null, bd = 144;
    for (const n of nodes) { const [x, y] = px(n); const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < Math.max(bd, 0) && d < (rad(n) + 5) ** 2) { bd = d; best = n; } }
    return best;
  }

  function draw(t) {
    T = t; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const sel = S.sel ? byId[S.sel] : null, grp = S.grp;
    const P = nodes.map(px);
    // links
    for (const l of links) {
      const inSel = sel && (l.a === sel || l.b === sel);
      const inGrp = grp && l.a.group === grp && l.b.group === grp;
      const cross = l.a.group !== l.b.group;
      let a = cross ? 0.025 : 0.12 + 0.3 * (l.s / maxS);
      let col = '124,138,160';
      if (sel) { a = inSel ? 0.85 : 0.025; if (inSel) col = '14,124,134'; }
      else if (grp) { a = inGrp ? 0.10 + 0.4 * (l.s / maxS) : 0.025; if (inGrp) col = '14,124,134'; }
      ctx.strokeStyle = `rgba(${col},${a})`; ctx.lineWidth = inSel ? 1.4 : 0.6;
      const pa = P[l.a.i], pb = P[l.b.i];
      ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
    }
    // selected group pulse
    if (grp && groups[grp]) {
      let gx = 0, gy = 0; const g = groups[grp];
      g.forEach(n => { gx += P[n.i][0]; gy += P[n.i][1]; }); gx /= g.length; gy /= g.length;
      const ds = g.map(n => Math.hypot(P[n.i][0] - gx, P[n.i][1] - gy)).sort((p, q) => p - q);
      const r = ds[Math.floor(ds.length * 0.85)] || 0;
      ctx.strokeStyle = 'rgba(14,124,134,.35)'; ctx.setLineDash([2, 5]);
      ctx.beginPath(); ctx.arc(gx, gy, r + 10, 0, 6.283); ctx.stroke(); ctx.setLineDash([]);
    }
    // nodes
    const neigh = new Set(); if (sel) sel.adj.forEach(l => { neigh.add(l.a); neigh.add(l.b); });
    for (const n of nodes) {
      const [x, y] = P[n.i]; let r = rad(n);
      let dim = (sel && !neigh.has(n)) || (!sel && grp && n.group !== grp);
      ctx.globalAlpha = dim ? 0.18 : 1;
      ctx.fillStyle = n.flag ? '#D64550' : '#7C8AA0';
      ctx.beginPath(); ctx.arc(x, y, r, 0, 6.283); ctx.fill(); ctx.shadowBlur = 0;
      if (n === sel) { ctx.strokeStyle = '#0E7C86'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, r + 4, 0, 6.283); ctx.stroke(); }
      else if (sel && neigh.has(n)) { ctx.strokeStyle = 'rgba(14,124,134,.8)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, r + 2, 0, 6.283); ctx.stroke(); }
    }
    ctx.globalAlpha = 1;
    // group labels
    if ((data.groups || []).length > 1) {
      ctx.font = '10px JetBrains Mono, monospace'; ctx.textAlign = 'center';
      for (const g in groups) {
        let gx = 0, top = Infinity; groups[g].forEach(n => { gx += P[n.i][0]; top = Math.min(top, P[n.i][1]); });
        gx /= groups[g].length;
        ctx.fillStyle = g === grp ? '#0E7C86' : 'rgba(135,145,159,.8)';
        ctx.fillText((data.group_labels && data.group_labels[g]) || g, gx, top - 10);
      }
    }
    S.raf = requestAnimationFrame(draw);
  }
  S.raf = requestAnimationFrame(draw);

  // interaction
  let drag = null;
  const pos = e => { const b = cv.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; };
  cv.onwheel = e => {
    e.preventDefault(); const [mx, my] = pos(e); const v = V();
    const f = Math.exp(-e.deltaY * 0.0012), k = Math.min(12, Math.max(0.4, v.k * f)), r = k / v.k;
    v.tx = (v.tx + W / 2 - mx) * r + mx - W / 2; v.ty = (v.ty + H / 2 - my) * r + my - H / 2; v.k = k;
  };
  cv.onmousedown = e => { const [mx, my] = pos(e); const n = hit(mx, my);
    drag = { n, mx, my, moved: false, tx: V().tx, ty: V().ty }; cv.classList.add('drag'); };
  cv.onmousemove = e => {
    const [mx, my] = pos(e);
    if (drag) {
      const dx = mx - drag.mx, dy = my - drag.my; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (drag.n) { const m = S.moved[drag.n.id] || [0, 0]; const s = fit * V().k;
        S.moved[drag.n.id] = [m[0] + (mx - (drag.lx ?? drag.mx)) / s, m[1] + (my - (drag.ly ?? drag.my)) / s];
        drag.lx = mx; drag.ly = my; }
      else { V().tx = drag.tx + dx; V().ty = drag.ty + dy; }
      tip.style.opacity = 0; return;
    }
    const n = hit(mx, my);
    if (n) {
      tip.innerHTML = `<b>${n.label}</b><br>group&nbsp;&nbsp;${(data.group_labels && data.group_labels[n.group]) || n.group}` +
        (n.score == null ? '<br>not analysed yet' : `<br>model score&nbsp;&nbsp;${n.score.toFixed(2)}`) + (n.flag ? ' <span class="c">▲ review</span>' : '') +
        `<br>co-reply links&nbsp;&nbsp;${n.deg}`;
      tip.style.left = Math.min(mx + 14, W - 210) + 'px'; tip.style.top = (my + 14) + 'px'; tip.style.opacity = 1;
      cv.style.cursor = 'pointer';
    } else { tip.style.opacity = 0; cv.style.cursor = ''; }
  };
  const up = e => {
    if (!drag) return; cv.classList.remove('drag');
    if (!drag.moved) {
      const n = drag.n;
      if (n) { S.sel = n.id; S.grp = n.group; setTriggerValue('select', { node: n.id, group: n.group }); }
      else { S.sel = null; if (!data.lock_group) S.grp = null; }
    }
    drag = null;
  };
  cv.onmouseup = up; cv.onmouseleave = e => { up(e); tip.style.opacity = 0; };
  cv.ondblclick = () => { S.view = { k: 1, tx: 0, ty: 0 }; S.moved = {}; };
  return () => { cancelAnimationFrame(S.raf); S.ro && S.ro.disconnect(); };
}
"""

RADAR_JS = r"""
export default function(component) {
  const { data, parentElement, setTriggerValue } = component;
  let wrap = parentElement.querySelector('.wrap');
  if (!wrap) {
    wrap = document.createElement('div'); wrap.className = 'wrap';
    wrap.innerHTML = '<canvas></canvas><div class="tip"></div>';
    parentElement.appendChild(wrap);
  }
  const S = wrap.__s || (wrap.__s = {});
  if (S.raf) cancelAnimationFrame(S.raf);
  const cv = wrap.querySelector('canvas'), tip = wrap.querySelector('.tip'), ctx = cv.getContext('2d');
  const H = data.height || 520, dpr = window.devicePixelRatio || 1;
  let W = wrap.clientWidth || 600;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = H + 'px';
  const blips = data.blips || [], groups = data.groups || [];
  if (S.run !== data.run_id) { S.run = data.run_id; S.t0 = null; }
  const SPEED = 0.0011;               // rad per ms -> one turn ≈ 5.7 s
  let hoverG = null;
  const C = () => [W / 2, H / 2], R = () => Math.min(W, H) / 2 - 34;
  const polar = (a, r) => { const [x, y] = C(); return [x + Math.cos(a) * r * R(), y + Math.sin(a) * r * R()]; };

  function draw(t) {
    if (S.t0 === null) S.t0 = t;
    const el = Math.min(t - S.t0, 6.283 / SPEED), sweep = (el * SPEED) % 6.283 - 1.5708, turns = el * SPEED / 6.283;
    const done = turns >= 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const [x, y] = C(), r = R();
    // rings
    ctx.strokeStyle = 'rgba(135,145,159,.18)'; ctx.lineWidth = 1;
    [0.25, 0.5, 0.75, 1].forEach(q => { ctx.beginPath(); ctx.arc(x, y, r * q, 0, 6.283); ctx.stroke(); });
    ctx.beginPath(); ctx.moveTo(x - r, y); ctx.lineTo(x + r, y); ctx.moveTo(x, y - r); ctx.lineTo(x, y + r); ctx.stroke();
    ctx.font = '9px JetBrains Mono, monospace'; ctx.fillStyle = 'rgba(135,145,159,.7)'; ctx.textAlign = 'left';
    [[0.25, '0.75'], [0.5, '0.50'], [0.75, '0.25']].forEach(([q, s]) => ctx.fillText('score ' + s, x + 4, y - r * q - 3));
    // sectors
    groups.forEach(g => {
      const hot = g.id === hoverG;
      ctx.strokeStyle = 'rgba(135,145,159,.12)'; ctx.beginPath(); ctx.moveTo(x, y);
      ctx.lineTo(...polar(g.a0, 1.02)); ctx.stroke();
      if (hot) { ctx.fillStyle = 'rgba(14,124,134,.06)'; ctx.beginPath(); ctx.moveTo(x, y);
        ctx.arc(x, y, r, g.a0, g.a1); ctx.closePath(); ctx.fill(); }
      const am = (g.a0 + g.a1) / 2, [lx, ly] = polar(am, 1.09);
      ctx.fillStyle = hot ? '#0E7C86' : (g.score >= 0.7 ? '#D64550' : 'rgba(135,145,159,.85)');
      ctx.textAlign = 'center'; ctx.font = '10px JetBrains Mono, monospace';
      const sweptG = turns >= 1 || ((am + 1.5708 + 6.283) % 6.283) < ((sweep + 1.5708 + 6.283) % 6.283);
      if (sweptG) ctx.fillText(g.label, lx, ly + 3);
    });
    // sweep wedge
    const grad = (!done && ctx.createConicGradient) ? ctx.createConicGradient(sweep - 0.9, x, y) : null;
    if (grad) { grad.addColorStop(0, 'rgba(14,124,134,0)'); grad.addColorStop(0.143, 'rgba(14,124,134,.22)');
      grad.addColorStop(0.1431, 'rgba(14,124,134,0)'); ctx.fillStyle = grad;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, r, sweep - 0.9, sweep); ctx.closePath(); ctx.fill(); }
    if (!done) { ctx.strokeStyle = 'rgba(14,124,134,.85)'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(x, y);
      ctx.lineTo(...polar(sweep, 1)); ctx.stroke(); }
    // blips
    for (const b of blips) {
      const rel = ((b.a - sweep) % 6.283 + 6.283) % 6.283;       // angle since sweep passed (0..2pi)
      const since = 6.283 - rel;                                   // how far behind the sweep
      const revealed = turns >= 1 || ((b.a + 1.5708 + 6.283) % 6.283) <= ((sweep + 1.5708 + 6.283) % 6.283);
      if (!revealed) continue;
      const fresh = done ? 0.6 : Math.max(0, 1 - since / 2.2);
      const [bx, by] = polar(b.a, b.r);
      const g = hoverG && b.group !== hoverG;
      ctx.globalAlpha = g ? 0.15 : 0.35 + 0.65 * fresh;
      ctx.fillStyle = b.flag ? '#D64550' : '#9AA7B8';
      ctx.beginPath(); ctx.arc(bx, by, b.flag ? 2.6 : 1.8, 0, 6.283); ctx.fill(); ctx.shadowBlur = 0;
    }
    ctx.globalAlpha = 1;
    // centre readout
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(9,11,16,.82)'; ctx.beginPath(); ctx.arc(x, y, Math.min(70, r * 0.3), 0, 6.283); ctx.fill();
    ctx.strokeStyle = 'rgba(14,124,134,.25)'; ctx.stroke();
    if (!blips.length) {
      ctx.fillStyle = 'rgba(135,145,159,.9)'; ctx.font = '13px Inter, sans-serif';
      (data.idle || 'Not analysed yet').split(' — ').forEach((s, i) => ctx.fillText(s, x, y - 2 + i * 14));
    } else if (turns >= 1) {
      ctx.fillStyle = '#E8EDF3'; ctx.font = '600 22px Space Grotesk, sans-serif';
      ctx.fillText(groups.length + ' groups', x, y + 6);
      ctx.fillStyle = '#87919F'; ctx.font = '9px JetBrains Mono, monospace';
      ctx.fillText('click a slice', x, y + 22);
    } else {
      ctx.fillStyle = '#0E7C86'; ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillText('scanning…', x, y + 4);
    }
    S.raf = requestAnimationFrame(draw);
  }
  S.raf = requestAnimationFrame(draw);

  const gAt = e => {
    const b = cv.getBoundingClientRect(), mx = e.clientX - b.left, my = e.clientY - b.top, [x, y] = C();
    const d = Math.hypot(mx - x, my - y) / R(); if (d > 1.12 || d < 0.04) return [null, mx, my];
    let a = Math.atan2(my - y, mx - x); if (a < -1.5708) a += 6.283;
    const g = groups.find(g => a >= g.a0 && a < g.a1); return [g || null, mx, my];
  };
  cv.onmousemove = e => {
    const [g, mx, my] = gAt(e); hoverG = g ? g.id : null; cv.style.cursor = g ? 'pointer' : '';
    if (g) { tip.innerHTML = `<b>${g.label}</b><br>${g.size} accounts<br>mean model score ${g.score.toFixed(2)}<br>${g.title}`;
      tip.style.left = Math.min(mx + 14, W - 220) + 'px'; tip.style.top = (my + 14) + 'px'; tip.style.opacity = 1; }
    else tip.style.opacity = 0;
  };
  cv.onmouseleave = () => { hoverG = null; tip.style.opacity = 0; };
  cv.onclick = e => { const [g] = gAt(e); if (g) setTriggerValue('case', g.id); };
  return () => cancelAnimationFrame(S.raf);
}
"""

_net = st.components.v2.component("radar_network", css=CSS, js=NET_JS)
_radar = st.components.v2.component("radar_scope", css=CSS, js=RADAR_JS)


def network(payload: dict, key: str, height: int = 560, selected_group=None, caption: str = "",
            group_labels: dict | None = None, lock_group: bool = False):
    data = {**payload, "height": height, "selected_group": selected_group, "caption": caption,
            "group_labels": group_labels or {}, "lock_group": lock_group}
    res = _net(data=data, key=key, height=height, on_select_change=lambda: None)
    return res.select if res is not None else None


def radar(blips: list, groups: list, run_id: str, key: str, height: int = 520, idle: str = ""):
    res = _radar(data={"blips": blips, "groups": groups, "run_id": run_id, "height": height, "idle": idle},
                 key=key, height=height, on_case_change=lambda: None)
    return res.case if res is not None else None
