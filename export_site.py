"""Export everything the static website (site/) needs into site/data/*.json.

The website is plain HTML/CSS/JS so it can be hosted on any static host (Netlify). It needs:
  teams.json     every active team's current form (the model's per-team inputs) + last 5 results
  fixtures.json  upcoming fixtures with model probabilities computed here in Python
  model.json     the Random Forest's trees, so the browser can predict any head-to-head matchup itself
  report.json    test-set evaluation (from reports/metrics.json) and dataset facts

Usage:  python export_site.py         (after fetch_data.py and/or train.py)
Then run `node tests/check_parity.mjs` to confirm the browser model reproduces Python's probabilities.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from features import ROLL_STATS, WINDOWS, build_features
from predictor import Predictor

ROOT = Path(__file__).parent
OUT = ROOT / "site" / "data"
STATE_KEYS = ["elo", "n_prior"] + [f"{c}_{w}" for w in WINDOWS for c in ROLL_STATS]
MODEL_NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost"}


def clean(x, digits=None):
    """JSON-safe number: NaN -> None, optional rounding."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    x = float(x)
    return round(x, digits) if digits is not None else x


def team_states(p: Predictor) -> dict:
    """Current per-team model inputs, computed by the real feature pipeline.

    One dummy, unplayed fixture per active team (the team against itself, dated now) is pushed through
    build_features; its home_* columns are exactly the team state the model would see for a match now.
    """
    now = p.now
    last_league = (pd.concat([p.matches[["date", "home_id", "league_code"]].set_axis(["date", "id", "lg"], axis=1),
                              p.matches[["date", "away_id", "league_code"]].set_axis(["date", "id", "lg"], axis=1)])
                   .sort_values("date").drop_duplicates("id", keep="last").set_index("id")["lg"])
    dummies = pd.DataFrame([{
        "event_id": f"state-{tid}", "date": now, "competition": "state", "league_code": last_league[tid],
        "season": None, "status": "state", "completed": False, "played": False, "neutral": False,
        "home_team": name, "home_id": tid, "away_team": name, "away_id": tid} for tid, name in p.teams.name.items()])
    feats = build_features(pd.concat([p.matches, dummies], ignore_index=True))
    feats = feats[feats.event_id.str.startswith("state-", na=False)].set_index("home_id")

    out = {}
    for tid, row in feats.iterrows():
        out[tid] = {k: clean(row[f"home_{k}"]) for k in STATE_KEYS}
        out[tid]["last_match"] = None
    # rest days depend on the match date, so the browser computes them from each team's last-match time
    played = p.matches[p.matches.played]
    last = pd.concat([played[["date", "home_id"]].set_axis(["date", "id"], axis=1),
                      played[["date", "away_id"]].set_axis(["date", "id"], axis=1)]).groupby("id")["date"].max()
    for tid in out:
        if tid in last.index:
            out[tid]["last_match"] = last[tid].isoformat()
    return out


def export_model(model: RandomForestClassifier, cols: list[str]) -> dict:
    """Flatten each tree. Thresholds are kept at full precision; the browser casts inputs to float32
    (as scikit-learn does) before comparing, so splits resolve identically."""
    trees = []
    for est in model.estimators_:
        t = est.tree_
        leaf = t.children_left == -1
        trees.append({
            "f": t.feature.tolist(),
            "t": [float(x) for x in t.threshold],
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "m": t.missing_go_to_left.astype(int).tolist(),
            "v": [[round(float(v), 8) for v in t.value[i, 0]] if leaf[i] else 0 for i in range(t.node_count)],
        })
    return {"type": "random_forest", "classes": ["away_win", "draw", "home_win"], "features": cols, "trees": trees}


def main():
    p = Predictor()
    if not isinstance(p.model, RandomForestClassifier):
        raise SystemExit("export_site.py exports Random Forest models only; the saved model is "
                         f"{type(p.model).__name__}. Retrain with the random forest selected.")
    report = json.loads((ROOT / "reports" / "metrics.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    states = team_states(p)
    teams = []
    for tid, t in p.teams.iterrows():
        res = p.recent_results(tid, 5)
        teams.append({
            "id": tid, "name": t["name"], "league": t["league"], "state": states.get(tid, {}),
            "recent": [{"date": r.date.strftime("%Y-%m-%d"), "opp": r.opponent, "opp_id": r.opponent_id,
                        "venue": r.venue, "score": r.score, "result": r.result} for r in res.itertuples()],
        })

    fixtures = [{
        "id": r.event_id, "date": r.date.isoformat(), "competition": r.competition, "neutral": bool(r.neutral),
        "home_id": r.home_id, "home": r.home_team, "away_id": r.away_id, "away": r.away_team,
        "p": [clean(r.p_away, 6), clean(r.p_draw, 6), clean(r.p_home, 6)],
    } for r in p.upcoming.itertuples()]

    selected = next(r for r in report["results"] if r["model"] == report["selected"])
    meta = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "data_as_of": p.data_as_of.isoformat(),
        "matches": int(p.matches.played.sum()), "features": len(p.cols), "upcoming": len(fixtures),
        "model": MODEL_NAMES.get(report["selected"], report["selected"]), "accuracy": selected["accuracy"],
        "log_loss": selected["log_loss"],
    }

    files = {
        "teams.json": {"meta": meta, "teams": teams},
        "fixtures.json": {"meta": meta, "fixtures": fixtures},
        "model.json": export_model(p.model, p.cols),
        "report.json": {"meta": meta, **report},
    }
    for name, payload in files.items():
        path = OUT / name
        path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:,.0f} KB)")
    print(f"{len(teams)} teams, {len(fixtures)} upcoming fixtures, data up to {p.data_as_of:%Y-%m-%d}")


if __name__ == "__main__":
    main()
