// Match Outcome Predictor: static front end. Data comes from data/*.json (written by export_site.py);
// head-to-head predictions run the exported Random Forest in the browser (js/model.js).
import { buildFeatures, predict, restDays } from "./model.js";

const CREST = (id) => `https://a.espncdn.com/combiner/i?img=/i/teamlogos/soccer/500/${id}.png&h=120&w=120`;
const LEAGUE_LOGO = (id) => `https://a.espncdn.com/i/leaguelogos/soccer/500/${id}.png`;
const LEAGUE_IDS = { "Premier League": 23, "La Liga": 15, "Bundesliga": 10, "Serie A": 12, "Ligue 1": 9, "Champions League": 2 };
const MODEL_NAMES = {
  random_forest: "Random Forest", xgboost: "XGBoost",
  "baseline: always home win (probs = class freq.)": "Always pick home win",
  "baseline: Elo-only logistic regression": "Elo-only logistic regression",
};
const HOME = "#60a5fa", AWAY = "#fb923c", DRAW = "#94a3b8";

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
const fmtNum = (v, d) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const dateFmt = (d, opts) => d.toLocaleString("en-GB", { timeZone: "UTC", ...opts });

const state = { teams: new Map(), byName: new Map(), fixtures: [], report: null, meta: null, model: null, modelPromise: null };

// ---------- loading -----------------------------------------------------------------------------
async function getJSON(name, attempts = 3) {
  for (let i = 1; ; i++) {
    try {
      const res = await fetch(`data/${name}`);
      if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
      return await res.json();
    } catch (err) {
      if (i >= attempts) throw err;
      await new Promise((r) => setTimeout(r, 400 * i));
    }
  }
}

function loadModel() {
  state.modelPromise ??= getJSON("model.json").then((m) => (state.model = m));
  return state.modelPromise;
}

// ---------- small HTML builders -----------------------------------------------------------------
const crest = (id, size = "md") => `<span class="crest crest-${size}"><img src="${CREST(id)}" alt="" loading="lazy"></span>`;
const leagueLogo = (comp) => (LEAGUE_IDS[comp] ? `<img src="${LEAGUE_LOGO(LEAGUE_IDS[comp])}" alt="">` : "");
const section = (title, sub = "") => `<div class="sec"><h2>${title}</h2>${sub ? `<p>${sub}</p>` : ""}</div>`;

function formChips(team) {
  if (!team?.recent?.length) return '<span class="conf">no matches yet</span>';
  return team.recent.map((r) => `<span class="chip ${r.result}" title="${esc(r.opp)} ${esc(r.score)}">${r.result}</span>`).join("");
}

/** m = {home_id, home, away_id, away, competition, neutral, date|null, p:[away, draw, home]} */
function matchCard(m, focusId = null) {
  const when = m.date
    ? `${dateFmt(m.date, { weekday: "short" })} ${dateFmt(m.date, { day: "2-digit", month: "short", year: "numeric" })} · ${dateFmt(m.date, { hour: "2-digit", minute: "2-digit" })} UTC`
    : "Hypothetical match · played today";
  const outcomes = [["home", `${m.home} win`, m.p[2]], ["draw", "Draw", m.p[1]], ["away", `${m.away} win`, m.p[0]]];
  const best = outcomes.reduce((a, b) => (b[2] > a[2] ? b : a));
  const p = best[2];
  const confidence = p >= 0.6 ? "Strong favourite" : p >= 0.5 ? "Clear favourite" : p >= 0.42 ? "Slight edge" : "Too close to call";
  const side = (id, name, tag) => {
    const focus = id === focusId;
    return `<div class="side${focus ? " focus" : ""}">${crest(id, "lg")}<div class="tname">${esc(name)}</div>
      <div class="ttag">${tag}${focus ? " · YOUR TEAM" : ""}</div><div class="form">${formChips(state.teams.get(id))}</div></div>`;
  };
  const nums = outcomes.map(([k, label, v]) => `<div class="pnum ${k}${k === best[0] ? " best" : ""}"><div class="pv">${pct(v)}</div><div class="pl">${esc(label)}</div></div>`).join("");
  const bar = outcomes.map(([k, , v]) => `<div class="seg ${k}" style="flex:${v.toFixed(4)}"></div>`).join("");
  return `<div class="match-card">
    <div class="mc-top"><span class="comp">${leagueLogo(m.competition)}${esc(m.competition)}</span><span>${when}</span></div>
    <div class="mc-teams">${side(m.home_id, m.home, m.neutral ? "NEUTRAL" : "HOME")}<div class="vs">VS</div>${side(m.away_id, m.away, m.neutral ? "NEUTRAL" : "AWAY")}</div>
    <div class="pnums">${nums}</div><div class="pbar">${bar}</div>
    <div class="verdict"><span class="pill">Model pick</span><b>${esc(best[1])}</b><span class="conf">· ${confidence} · ${esc(state.meta.model)}</span></div>
  </div>`;
}

const TAPE = [ // label, state key, decimals, higher is better (null = neutral), offset, suffix
  ["Elo rating", "elo", 0, true, 1200, ""],
  ["Points per game · last 5", "points_5", 2, true, 0, ""],
  ["Points per game · last 15", "points_15", 2, true, 0, ""],
  ["Goals scored per game", "gf_15", 2, true, 0, ""],
  ["Goals conceded per game", "ga_15", 2, false, 0, ""],
  ["Shots per game", "shots_for_15", 1, true, 0, ""],
  ["Shots on target per game", "sot_for_15", 1, true, 0, ""],
  ["Shots on target conceded", "sot_against_15", 1, false, 0, ""],
  ["Possession", "poss_15", 0, true, 0, "%"],
  ["Days of rest", "rest", 0, null, 0, ""],
];

function taleOfTheTape(m, date) {
  const hs = state.teams.get(m.home_id).state, as = state.teams.get(m.away_id).state;
  const val = (s, key) => (key === "rest" ? restDays(s, date) : s[key]);
  const rows = TAPE.map(([label, key, d, higher, offset, suffix]) => {
    const h = val(hs, key), a = val(as, key);
    if (h === null || a === null || Number.isNaN(h) || Number.isNaN(a)) return "";
    const hv = Math.max(h - offset, 0), av = Math.max(a - offset, 0), tot = hv + av || 1;
    let hb = "", ab = "";
    if (higher !== null && Math.abs(h - a) > 1e-9) [hb, ab] = (h > a) === higher ? [" better", ""] : ["", " better"];
    return `<div class="tape-row"><div class="tl">${label}</div><div class="tgrid">
      <span class="tv${hb}">${fmtNum(h, d)}${suffix}</span>
      <div class="tbar l${hb}"><span style="width:${(hv / tot * 100).toFixed(1)}%"></span></div>
      <div class="tbar r${ab}"><span style="width:${(av / tot * 100).toFixed(1)}%"></span></div>
      <span class="tv r${ab}">${fmtNum(a, d)}${suffix}</span></div></div>`;
  }).join("");
  return `<div class="panel"><h3>Tale of the tape</h3>
    <div class="tape-head"><span class="th">${crest(m.home_id)}${esc(m.home)}</span><span class="th">${esc(m.away)}${crest(m.away_id)}</span></div>${rows}</div>`;
}

function recentPanel(m) {
  const block = (id, name) => {
    const items = [...(state.teams.get(id)?.recent ?? [])].reverse().map((r) => `<div class="rr">
      <span class="rr-date">${dateFmt(new Date(r.date), { day: "2-digit", month: "short" })}</span><span class="chip ${r.result}">${r.result}</span>
      <span class="rr-opp"><span class="ha">${r.venue}</span>${crest(r.opp_id, "sm")}<span class="n">${esc(r.opp)}</span></span>
      <span class="rr-score">${esc(r.score)}</span></div>`).join("");
    return `<div class="rr-team">${crest(id)}${esc(name)}</div><div class="rr-list">${items}</div>`;
  };
  return `<div class="panel"><h3>Recent results</h3>${block(m.home_id, m.home)}${block(m.away_id, m.away)}</div>`;
}

function renderMatch(el, m, focusId = null) {
  el.innerHTML = matchCard(m, focusId) + `<div class="two-col">${taleOfTheTape(m, m.date ?? new Date())}${recentPanel(m)}</div>`;
}

// ---------- team picker (league filter + team select) ------------------------------------------
function teamPicker(el, key, label, defaultName, onChange) {
  const leagues = ["All leagues", ...[...new Set([...state.teams.values()].map((t) => t.league))].sort()];
  el.innerHTML = `<div class="field"><label for="${key}-lg">League</label><select id="${key}-lg">${leagues.map((l) => `<option>${esc(l)}</option>`).join("")}</select></div>
    <div class="field"><label for="${key}-team">${label}</label><select id="${key}-team"></select></div>`;
  const lg = $(`#${key}-lg`, el), team = $(`#${key}-team`, el);
  const fill = (keep) => {
    const pool = [...state.teams.values()].filter((t) => lg.value === "All leagues" || t.league === lg.value)
      .sort((a, b) => a.name.localeCompare(b.name));
    team.innerHTML = pool.map((t) => `<option value="${esc(t.id)}">${esc(t.name)}</option>`).join("");
    if (keep && pool.some((t) => t.id === keep)) team.value = keep;
  };
  fill(state.byName.get(defaultName)?.id);
  lg.addEventListener("change", () => { fill(team.value); onChange(); });
  team.addEventListener("change", onChange);
  return () => team.value;
}

function toggleGroup(el, onChange) {
  el.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    el.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    onChange();
  });
  return () => el.querySelector('[aria-pressed="true"]').dataset.value;
}

// ---------- tabs ----------------------------------------------------------------------------------
function upcoming() {
  const cutoff = Date.now() - 3 * 3600 * 1000;
  return state.fixtures.filter((f) => f.date.getTime() >= cutoff);
}

function setupNext() {
  const out = $("#next-out");
  const getTeam = teamPicker($("#next-picker"), "next", "Team", "Arsenal", render);
  function render() {
    const id = getTeam();
    const fx = upcoming().find((f) => f.home_id === id || f.away_id === id);
    if (!fx) {
      out.innerHTML = `<div class="empty">No scheduled fixture found for <b>${esc(state.teams.get(id).name)}</b> in the current data.<br>Try the <b>Head to head</b> tab to set up any matchup.</div>`;
      return;
    }
    renderMatch(out, fx, id);
  }
  render();
}

function setupH2H() {
  const out = $("#h2h-out");
  const getHome = teamPicker($("#home-picker"), "home", "Home team", "Real Madrid", render);
  const getAway = teamPicker($("#away-picker"), "away", "Away team", "Bayern Munich", render);
  const getVenue = toggleGroup($("#venue"), render);
  const getComp = toggleGroup($("#competition"), render);
  async function render() {
    const home = getHome(), away = getAway();
    if (home === away) { out.innerHTML = '<div class="empty">Pick two different teams.</div>'; return; }
    if (!state.model) { out.innerHTML = '<p class="loading">Loading the model…</p>'; await loadModel(); }
    const neutral = getVenue() === "neutral", ucl = getComp() === "ucl";
    const h = state.teams.get(home), a = state.teams.get(away);
    const p = predict(state.model, buildFeatures(h.state, a.state, { neutral, ucl, date: new Date() }));
    const competition = ucl ? "Champions League" : (h.league === "Champions League only" ? "League" : h.league);
    renderMatch(out, { home_id: home, home: h.name, away_id: away, away: a.name, competition, neutral, date: null, p });
  }
  return render;
}

function setupFixtures() {
  const out = $("#fixtures-out"), days = $("#days"), daysOut = $("#days-out"), pills = $("#comp-pills");
  const comps = [...new Set(state.fixtures.map((f) => f.competition))].sort();
  pills.innerHTML = comps.map((c) => `<button type="button" class="pill-btn" data-comp="${esc(c)}" aria-pressed="true">${leagueLogo(c)}${esc(c)}</button>`).join("");
  pills.addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    btn.setAttribute("aria-pressed", String(btn.getAttribute("aria-pressed") !== "true"));
    render();
  });
  days.addEventListener("input", render);
  function render() {
    const n = Number(days.value);
    daysOut.textContent = `${n} day${n === 1 ? "" : "s"}`;
    const chosen = new Set([...pills.querySelectorAll('[aria-pressed="true"]')].map((b) => b.dataset.comp));
    const until = Date.now() + n * 86400000;
    const list = upcoming().filter((f) => chosen.has(f.competition) && f.date.getTime() <= until);
    if (!list.length) { out.innerHTML = '<div class="empty">No fixtures in that window. Try more days or more competitions.</div>'; return; }
    let html = "", lastDay = "";
    for (const f of list) {
      const day = dateFmt(f.date, { weekday: "long", day: "2-digit", month: "long" });
      if (day !== lastDay) { html += `<div class="day">${day.replace(",", " ·")}</div>`; lastDay = day; }
      const probs = { home: f.p[2], draw: f.p[1], away: f.p[0] };
      const best = Object.keys(probs).reduce((a, b) => (probs[b] > probs[a] ? b : a));
      html += `<div class="fx"><div class="fx-time">${leagueLogo(f.competition)}${dateFmt(f.date, { hour: "2-digit", minute: "2-digit" })}</div>
        <div class="fx-team h"><span class="n">${esc(f.home)}</span>${crest(f.home_id, "sm")}</div>
        <div class="fx-mid"><div class="fx-nums">${Object.entries(probs).map(([k, v]) => `<span class="${k}${k === best ? " best" : ""}">${pct(v)}</span>`).join("")}</div>
          <div class="pbar sm">${Object.entries(probs).map(([k, v]) => `<div class="seg ${k}" style="flex:${v.toFixed(4)}"></div>`).join("")}</div></div>
        <div class="fx-team a">${crest(f.away_id, "sm")}<span class="n">${esc(f.away)}</span></div></div>`;
    }
    out.innerHTML = html + `<div class="count">${list.length} fixtures · kick-off times in UTC · <span style="color:${HOME}">home win</span> / draw / <span style="color:${AWAY}">away win</span></div>`;
  }
  render();
}

// ---------- model performance (HTML/SVG charts, no chart library) --------------------------------
function prettyFeature(f) {
  const fixed = { elo_diff: "Elo difference", home_elo: "Home Elo", away_elo: "Away Elo", home_adv: "Home advantage",
    is_ucl: "Champions League game", rest_diff: "Rest-days difference", home_rest_days: "Home rest days",
    away_rest_days: "Away rest days", home_n_prior: "Home matches in data", away_n_prior: "Away matches in data" };
  if (fixed[f]) return fixed[f];
  const stats = { gf: "goals scored", ga: "goals conceded", shots_for: "shots", shots_against: "shots conceded",
    sot_for: "shots on target", sot_against: "shots on target conceded", poss: "possession", points: "points per game" };
  const [side, ...rest] = f.split("_");
  const window = rest.pop();
  const stat = rest.join("_");
  return `${{ diff: "Δ", home: "Home", away: "Away" }[side]} ${stats[stat] ?? stat} (last ${window})`;
}

function hbars(rows, { max, fmt }) {
  const top = Math.max(...rows.map((r) => r.v));
  return `<div class="hbar">${rows.map((r) => `<span class="lbl" title="${esc(r.label)}">${esc(r.label)}</span>
    <div class="trk"><div class="fill${r.v === top ? " top" : ""}" style="width:${(r.v / max * 100).toFixed(1)}%"></div></div>
    <span class="val">${fmt(r.v)}</span>`).join("")}</div>`;
}

function confusion(cm) {
  const labels = ["Away win", "Draw", "Home win"];
  const max = Math.max(...cm.flat());
  const mix = (t) => { // surface -> teal -> emerald
    const a = [19, 27, 46], b = [15, 118, 110], c = [52, 211, 153];
    const [p, q, u] = t < 0.5 ? [a, b, t * 2] : [b, c, (t - 0.5) * 2];
    return `rgb(${p.map((x, i) => Math.round(x + (q[i] - x) * u)).join(",")})`;
  };
  let html = `<div class="cm"><span></span>${labels.map((l) => `<span class="h">${l}</span>`).join("")}`;
  cm.forEach((row, i) => {
    const total = row.reduce((s, v) => s + v, 0);
    html += `<span class="h rh">${labels[i]}</span>` + row.map((v, j) => {
      const t = v / max;
      return `<div class="cell" style="background:${mix(t)};color:${t > 0.6 ? "#06281c" : "#e6ebf5"}" title="Actual ${labels[i]}, predicted ${labels[j]}: ${v}">${v}<small>${pct(v / total)} of row</small></div>`;
    }).join("");
  });
  return html + `<span></span><span class="axis">Predicted →</span></div>`;
}

function calibration(cal) {
  const W = 420, H = 270, L = 44, R = 12, T = 12, B = 36;
  const series = [["home_win", "Home win", HOME], ["draw", "Draw", DRAW], ["away_win", "Away win", AWAY]];
  const maxV = Math.min(1, Math.ceil(Math.max(...Object.values(cal).flat().flatMap((d) => [d.predicted, d.actual])) * 10) / 10);
  const x = (v) => L + (v / maxV) * (W - L - R), y = (v) => H - B - (v / maxV) * (H - T - B);
  const ticks = []; for (let v = 0; v <= maxV + 1e-9; v += 0.2) ticks.push(+v.toFixed(1));
  let svg = `<svg class="cal" viewBox="0 0 ${W} ${H}" role="img" aria-label="Calibration chart">`;
  for (const t of ticks) {
    svg += `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" stroke="#1f2a40"/><text x="${L - 8}" y="${y(t) + 4}" text-anchor="end">${pct(t)}</text>`;
    svg += `<text x="${x(t)}" y="${H - B + 18}" text-anchor="middle">${pct(t)}</text>`;
  }
  svg += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(maxV)}" y2="${y(maxV)}" stroke="#3b4a66" stroke-dasharray="5 5"/>`;
  for (const [key, label, color] of series) {
    const pts = (cal[key] ?? []).map((d) => [x(d.predicted), y(d.actual), d]);
    if (!pts.length) continue;
    svg += `<polyline fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round" points="${pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ")}"/>`;
    svg += pts.map(([px, py, d]) => `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="4" fill="${color}"><title>${label}: predicted ${pct(d.predicted, 1)}, actual ${pct(d.actual, 1)} (${d.n} matches)</title></circle>`).join("");
  }
  svg += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle">Predicted probability</text></svg>`;
  return svg + `<div class="legend">${series.map(([, l, c]) => `<span><i style="background:${c}"></i>${l}</span>`).join("")}</div>`;
}

function renderPerformance() {
  const R = state.report;
  const bestAcc = Math.max(...R.results.map((r) => r.accuracy));
  const rows = R.results.map((r) => {
    const sel = r.model === R.selected;
    return `<tr class="${sel ? "sel" : ""}"><td>${esc(MODEL_NAMES[r.model] ?? r.model)}${sel ? '<span class="tag">deployed</span>' : ""}</td>
      <td><div class="acc"><span>${pct(r.accuracy, 1)}</span><div class="bar"><span style="width:${(r.accuracy / bestAcc * 100).toFixed(0)}%"></span></div></div></td>
      <td>${r.log_loss.toFixed(3)}</td><td>${r.brier.toFixed(3)}</td></tr>`;
  }).join("");
  const fi = Object.entries(R.feature_importance).slice(0, 10).map(([f, v]) => ({ label: prettyFeature(f), v }));
  const comp = [...R.accuracy_by_competition].sort((a, b) => b.accuracy - a.accuracy).map((c) => ({ label: c.competition, v: c.accuracy }));
  $("#perf-out").innerHTML = section("How good is it?",
    `Tuned on ${R.train_matches.toLocaleString("en")} matches (${R.train_period[0]} → ${R.train_period[1]}), then tested on ${R.test_matches.toLocaleString("en")} later matches (${R.test_period[0]} → ${R.test_period[1]}) it never saw.`) +
    `<div class="rwrap"><table class="rtable"><thead><tr><th>Model</th><th>Accuracy</th><th>Log loss ↓</th><th>Brier ↓</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="note">Log loss and Brier score grade the probabilities themselves (lower is better). Bookmakers reach roughly 53–55% accuracy on top-flight football, so the ceiling is low.</p>
    <div class="chart-grid">
      <div class="chart-card"><p class="chart-title">What drives the prediction</p><p class="chart-sub">Top 10 features by importance</p>${hbars(fi, { max: fi[0].v, fmt: (v) => pct(v, 1) })}</div>
      <div class="chart-card"><p class="chart-title">Confusion matrix</p><p class="chart-sub">Test-set predictions (columns) vs what actually happened (rows)</p>${confusion(R.confusion_matrix)}</div>
      <div class="chart-card"><p class="chart-title">Accuracy by competition</p><p class="chart-sub">Test set</p>${hbars(comp, { max: 0.7, fmt: (v) => pct(v) })}</div>
      <div class="chart-card"><p class="chart-title">Calibration</p><p class="chart-sub">Close to the dashed line = probabilities can be taken at face value</p>${calibration(R.calibration)}</div>
    </div>`;
}

function renderHow() {
  const R = state.report, M = state.meta;
  const lift = ((R.results.find((r) => r.model === R.selected).accuracy - R.results[0].accuracy) * 100).toFixed(1);
  const steps = [
    ["Collect", "<code>fetch_data.py</code> pulls every match since 2019-20 from ESPN's public API: score, possession, shots, shots on target, plus upcoming fixtures."],
    ["Engineer", `<code>features.py</code> builds ${M.features} features per match from <b>only games played before kick-off</b>: rolling 5- and 15-match form, goals, shots, possession and an Elo rating.`],
    ["Train", "<code>train.py</code> tunes a Random Forest and XGBoost with time-series cross-validation, optimising log loss. The split is by date, never shuffled."],
    ["Evaluate", `Both models are scored on ${R.test_matches.toLocaleString("en")} unseen recent matches against an always-home-win baseline and an Elo-only model.`],
    ["Serve", "<code>export_site.py</code> exports the forest's trees and each team's form to JSON. This page rebuilds the features and runs the model in your browser."],
  ];
  const insights = [
    ["Team strength rules", "Elo difference is the single most important feature, followed by the gap in shots on target, which shows chance quality better than goals do."],
    ["Beats the baseline", `${lift} accuracy points above always picking the home side. Its edge over Elo alone is small, because most of the predictable signal is team strength.`],
    ["Draws are hard", "About a quarter of matches end level, yet a draw is almost never the single most likely result. The draw probabilities themselves are still well calibrated."],
  ];
  $("#how-out").innerHTML = section("The pipeline", "From raw match data to a probability, in five steps.") +
    `<div class="steps">${steps.map(([t, d], i) => `<div class="step"><div class="n">${i + 1}</div><h4>${t}</h4><p>${d}</p></div>`).join("")}</div>` +
    section("What the model learned") +
    `<div class="insights">${insights.map(([t, d]) => `<div class="insight"><b>${t}</b><span>${d}</span></div>`).join("")}</div>` +
    section("Built with") +
    `<div class="badges">${["Python", "pandas", "NumPy", "scikit-learn", "XGBoost", "JavaScript", "Netlify", "ESPN API"].map((t) => `<span>${t}</span>`).join("")}</div>`;
}

// ---------- header ----------------------------------------------------------------------------------
function renderHeader() {
  const M = state.meta, R = state.report;
  const base = R.results.find((r) => r.model.startsWith("baseline: always"));
  document.querySelectorAll('[data-bind="model"]').forEach((el) => (el.textContent = M.model));
  document.querySelectorAll('[data-bind="matches"]').forEach((el) => (el.textContent = M.matches.toLocaleString("en")));
  $("#league-row").innerHTML = Object.entries(LEAGUE_IDS).map(([n, id]) => `<span class="league"><img src="${LEAGUE_LOGO(id)}" alt="">${n}</span>`).join("");
  const asOf = new Date(M.data_as_of);
  $("#kpis").innerHTML = `
    <div class="kpi"><div class="k-label">Matches analysed</div><div class="k-val">${M.matches.toLocaleString("en")}</div><div class="k-sub">6 competitions since 2019-20</div></div>
    <div class="kpi"><div class="k-label">Test accuracy</div><div class="k-val">${pct(M.accuracy, 1)}</div><div class="k-sub up">▲ ${((M.accuracy - base.accuracy) * 100).toFixed(1)} pts vs home-win baseline</div></div>
    <div class="kpi"><div class="k-label">Features per match</div><div class="k-val">${M.features}</div><div class="k-sub">form, shots, possession, Elo</div></div>
    <div class="kpi"><div class="k-label">Results up to</div><div class="k-val">${dateFmt(asOf, { day: "2-digit", month: "short" })}</div><div class="k-sub">${upcoming().length.toLocaleString("en")} upcoming fixtures</div></div>`;
  const ageDays = (Date.now() - asOf.getTime()) / 86400000;
  if (ageDays > 10) {
    const s = $("#stale");
    s.hidden = false;
    s.textContent = `Heads up: results in this snapshot run to ${dateFmt(asOf, { day: "numeric", month: "long", year: "numeric" })}, so recent form and fixtures may be out of date.`;
  }
}

// ---------- tabs + boot -------------------------------------------------------------------------------
function setupTabs(onShow) {
  const tabs = [...document.querySelectorAll(".tab")];
  const show = (name, push) => {
    if (!tabs.some((t) => t.dataset.tab === name)) name = "next";
    tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.dataset.tab).hidden = !on;
    });
    if (push) history.replaceState(null, "", `#${name}`);
    onShow(name);
  };
  tabs.forEach((t) => t.addEventListener("click", () => show(t.dataset.tab, true)));
  $(".tabs").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const i = tabs.findIndex((t) => t.getAttribute("aria-selected") === "true");
    const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    next.focus();
    show(next.dataset.tab, true);
  });
  window.addEventListener("hashchange", () => show(location.hash.slice(1), false));
  show(location.hash.slice(1) || "next", false);
}

async function boot() {
  try {
    const [teams, fixtures, report] = await Promise.all([getJSON("teams.json"), getJSON("fixtures.json"), getJSON("report.json")]);
    state.meta = teams.meta;
    state.report = report;
    teams.teams.forEach((t) => { state.teams.set(t.id, t); state.byName.set(t.name, t); });
    state.fixtures = fixtures.fixtures.map((f) => ({ ...f, date: new Date(f.date) }))
      .filter((f) => state.teams.has(f.home_id) && state.teams.has(f.away_id))
      .sort((a, b) => a.date - b.date);
  } catch (err) {
    $("#next-out").innerHTML = `<div class="empty">Couldn't load the match data (${esc(err.message)}).</div>`;
    return;
  }
  renderHeader();
  setupNext();
  setupFixtures();
  renderPerformance();
  renderHow();
  const renderH2H = setupH2H();
  let h2hShown = false;
  setupTabs((name) => { if (name === "h2h" && !h2hShown) { h2hShown = true; renderH2H(); } });
  // fetch the model in the background so head to head is instant when opened
  (window.requestIdleCallback ?? ((cb) => setTimeout(cb, 1500)))(() => loadModel());
}

boot();
