"""Shared prediction logic used by the web app (app.py) and the command-line tool (predict.py)."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import build_features, load_matches

ROOT = Path(__file__).parent
LEAGUE_NAMES = {
    "eng.1": "Premier League", "esp.1": "La Liga", "ger.1": "Bundesliga",
    "ita.1": "Serie A", "fra.1": "Ligue 1", "uefa.champions": "Champions League only",
}


class Predictor:
    def __init__(self, root: Path = ROOT):
        bundle = joblib.load(root / "models" / "model.joblib")
        self.model, self.cols, self.model_name = bundle["model"], bundle["features"], bundle["name"]
        self.matches = load_matches(root / "data" / "matches.csv")
        self.feats = build_features(self.matches)
        self.now = pd.Timestamp.now(tz="UTC")
        self.data_as_of = self.matches.loc[self.matches.played, "date"].max()

        # id -> latest display name, and each team's domestic league (or UCL-only for other leagues)
        long = pd.concat([
            self.matches[["date", "home_id", "home_team", "league_code"]].set_axis(["date", "id", "name", "lg"], axis=1),
            self.matches[["date", "away_id", "away_team", "league_code"]].set_axis(["date", "id", "name", "lg"], axis=1),
        ]).sort_values("date")
        self.names = long.drop_duplicates("id", keep="last").set_index("id")["name"]
        domestic = long[long.lg != "uefa.champions"].drop_duplicates("id", keep="last").set_index("id")["lg"]
        last_seen = long.groupby("id")["date"].max()
        teams = pd.DataFrame({"name": self.names, "last_seen": last_seen})
        teams["league"] = domestic.reindex(teams.index).fillna("uefa.champions").map(LEAGUE_NAMES)
        # "active" = has a match in the past ~15 months or a scheduled one
        self.teams = teams[teams.last_seen > self.now - pd.Timedelta(days=450)].sort_values("name")

        up = self.feats[~self.feats.played & (self.feats.date >= self.now - pd.Timedelta(hours=3))].copy()
        if len(up):
            p = self.model.predict_proba(up[self.cols])
            up["p_away"], up["p_draw"], up["p_home"] = p[:, 0], p[:, 1], p[:, 2]
        self.upcoming = up.sort_values("date")

    # ---- lookups -------------------------------------------------------------------------------
    def find_team(self, query: str) -> str:
        """Return a team id from a (partial, case-insensitive) name. Raises ValueError if not unique."""
        q = query.lower().strip()
        names = self.names
        exact = names[names.str.lower() == q]
        hits = exact if len(exact) else names[names.str.lower().str.contains(q, regex=False)]
        if len(hits) == 1:
            return hits.index[0]
        if not len(hits):
            raise ValueError(f"No team matching '{query}'.")
        raise ValueError(f"'{query}' is ambiguous: {', '.join(sorted(hits))}")

    def recent_results(self, team_id: str, n: int = 5) -> pd.DataFrame:
        m = self.matches[self.matches.played & ((self.matches.home_id == team_id) | (self.matches.away_id == team_id))]
        rows = []
        for r in m.tail(n).itertuples():
            home = r.home_id == team_id
            gf, ga = (r.home_goals, r.away_goals) if home else (r.away_goals, r.home_goals)
            rows.append({"date": r.date, "competition": r.competition,
                         "opponent": r.away_team if home else r.home_team,
                         "opponent_id": r.away_id if home else r.home_id,
                         "venue": "N" if r.neutral else ("H" if home else "A"),
                         "score": f"{int(gf)}-{int(ga)}",
                         "result": "W" if gf > ga else "D" if gf == ga else "L"})
        return pd.DataFrame(rows)

    # ---- predictions ---------------------------------------------------------------------------
    def next_match(self, team_id: str):
        """(feature row, team is home?) for the team's next scheduled fixture, or None."""
        m = self.upcoming[(self.upcoming.home_id == team_id) | (self.upcoming.away_id == team_id)]
        if m.empty:
            return None
        row = m.iloc[0]
        return row, row.home_id == team_id

    def custom_match(self, home_id: str, away_id: str, neutral: bool = False, ucl: bool = False) -> pd.Series:
        """Feature row + probabilities for a hypothetical match played now, with current form."""
        played_league = self.matches.loc[self.matches.home_id == home_id, "league_code"]
        league = "uefa.champions" if ucl else (played_league.iloc[-1] if len(played_league) else "eng.1")
        fixture = pd.DataFrame([{
            "event_id": "custom", "date": self.now, "competition": "Champions League" if ucl else LEAGUE_NAMES.get(league, "League"),
            "league_code": league, "season": None, "status": "hypothetical", "completed": False, "played": False,
            "neutral": neutral, "home_team": self.names[home_id], "home_id": home_id,
            "away_team": self.names[away_id], "away_id": away_id}])
        feats = build_features(pd.concat([self.matches, fixture], ignore_index=True))
        row = feats[feats.event_id == "custom"].iloc[0].copy()
        p = self.model.predict_proba(row[self.cols].to_frame().T.astype(float))[0]
        row["p_away"], row["p_draw"], row["p_home"] = p
        return row

    @staticmethod
    def team_probs(row: pd.Series, team_is_home: bool) -> dict:
        """Win/draw/loss from one team's point of view."""
        if team_is_home:
            return {"win": row.p_home, "draw": row.p_draw, "loss": row.p_away}
        return {"win": row.p_away, "draw": row.p_draw, "loss": row.p_home}

    @staticmethod
    def comparison(row: pd.Series) -> pd.DataFrame:
        """Side-by-side pre-match stats for the two teams in a feature row."""
        stats = [("Elo rating", "elo", "{:.0f}"),
                 ("Points per game (last 5)", "points_5", "{:.2f}"),
                 ("Points per game (last 15)", "points_15", "{:.2f}"),
                 ("Goals scored / game (last 15)", "gf_15", "{:.2f}"),
                 ("Goals conceded / game (last 15)", "ga_15", "{:.2f}"),
                 ("Shots / game (last 15)", "shots_for_15", "{:.1f}"),
                 ("Shots on target / game (last 15)", "sot_for_15", "{:.1f}"),
                 ("Shots on target conceded (last 15)", "sot_against_15", "{:.1f}"),
                 ("Possession % (last 15)", "poss_15", "{:.1f}"),
                 ("Days since last match", "rest_days", "{:.0f}")]
        out = {}
        for label, key, fmt in stats:
            vals = [row.get(f"{s}_{key}") for s in ("home", "away")]
            out[label] = [fmt.format(v) if v is not None and not pd.isna(v) else "–" for v in vals]
        return pd.DataFrame(out, index=[row.home_team, row.away_team]).T
