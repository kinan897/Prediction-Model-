// Checks that the browser model (js/model.js) reproduces Python's probabilities for every upcoming fixture.
// Usage: node tests/check_parity.mjs        (run after export_site.py)
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { buildFeatures, predict } from "../site/js/model.js";

const here = dirname(fileURLToPath(import.meta.url));
const load = (f) => JSON.parse(readFileSync(join(here, "..", "site", "data", f), "utf8"));
const model = load("model.json");
const teams = new Map(load("teams.json").teams.map((t) => [t.id, t]));
const { fixtures } = load("fixtures.json");

let checked = 0, skipped = 0, worst = 0, worstFixture = null;
for (const fx of fixtures) {
  const home = teams.get(fx.home_id), away = teams.get(fx.away_id);
  if (!home || !away) { skipped++; continue; }
  const f = buildFeatures(home.state, away.state, {
    neutral: fx.neutral, ucl: fx.competition === "Champions League", date: new Date(fx.date),
  });
  const p = predict(model, f);
  const err = Math.max(...p.map((v, i) => Math.abs(v - fx.p[i])));
  if (err > worst) { worst = err; worstFixture = `${fx.home} vs ${fx.away}`; }
  checked++;
}
const ok = worst < 1e-5;
console.log(`checked ${checked} fixtures (${skipped} skipped: team not active), max abs difference ${worst.toExponential(2)}` +
  (worstFixture ? ` (${worstFixture})` : ""));
console.log(ok ? "PARITY OK" : "PARITY FAILED");
process.exit(ok ? 0 : 1);
