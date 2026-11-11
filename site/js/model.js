// Browser port of the prediction step: rebuild the model's 58 input features from two teams' current
// form (mirrors features.build_features in Python) and run the exported Random Forest.
// Used by the website and by check_parity.mjs, which checks it against Python's own probabilities.

const ELO_HFA = 60;
const WINDOWS = [5, 15];
const ROLL_STATS = ["gf", "ga", "shots_for", "shots_against", "sot_for", "sot_against", "poss", "points"];
const DAY_MS = 86400000;

const num = (v) => (v === null || v === undefined ? NaN : v);

/** Days between a team's last match and the given match date (NaN if it has none). */
export function restDays(state, date) {
  if (!state.last_match) return NaN;
  return (date.getTime() - new Date(state.last_match).getTime()) / DAY_MS;
}

/**
 * Feature dict for home vs away. `home`/`away` are team states from teams.json,
 * opts = { neutral: bool, ucl: bool, date: Date }.
 */
export function buildFeatures(home, away, { neutral = false, ucl = false, date = new Date() } = {}) {
  const f = {};
  const homeAdv = neutral ? 0 : 1;
  const hr = restDays(home, date);
  const ar = restDays(away, date);
  f.home_adv = homeAdv;
  f.is_ucl = ucl ? 1 : 0;
  f.home_elo = num(home.elo);
  f.away_elo = num(away.elo);
  f.elo_diff = f.home_elo - f.away_elo + homeAdv * ELO_HFA;
  f.rest_diff = Math.min(hr, 14) - Math.min(ar, 14);
  f.home_rest_days = Math.min(hr, 30);
  f.away_rest_days = Math.min(ar, 30);
  f.home_n_prior = num(home.n_prior) || 0;
  f.away_n_prior = num(away.n_prior) || 0;
  for (const w of WINDOWS) {
    for (const c of ROLL_STATS) {
      const h = num(home[`${c}_${w}`]);
      const a = num(away[`${c}_${w}`]);
      f[`home_${c}_${w}`] = h;
      f[`away_${c}_${w}`] = a;
      f[`diff_${c}_${w}`] = h - a;
    }
  }
  return f;
}

/** [P(away win), P(draw), P(home win)] — the average of every tree's leaf distribution, like scikit-learn. */
export function predict(model, features) {
  // scikit-learn casts inputs to float32 before comparing them with the (float64) split thresholds
  const x = model.features.map((name) => {
    const v = features[name];
    return Number.isNaN(v) ? NaN : Math.fround(v);
  });
  const out = [0, 0, 0];
  for (const tree of model.trees) {
    let node = 0;
    while (tree.l[node] !== -1) {
      const v = x[tree.f[node]];
      const goLeft = Number.isNaN(v) ? tree.m[node] === 1 : v <= tree.t[node];
      node = goLeft ? tree.l[node] : tree.r[node];
    }
    const leaf = tree.v[node];
    out[0] += leaf[0];
    out[1] += leaf[1];
    out[2] += leaf[2];
  }
  const n = model.trees.length;
  return out.map((v) => v / n);
}
