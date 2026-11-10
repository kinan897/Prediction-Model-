"""Pre-match feature engineering.

Every feature for a match is built only from matches that finished *before* kickoff, so the same
code serves training (historical matches) and prediction (upcoming fixtures) with no leakage.

Per team, across all competitions (league + Champions League):
  * rolling averages over the last 5 and last 15 matches: goals scored/conceded, shots for/against,
    shots on target for/against, possession, points per game (recent form)
  * Elo rating (team strength that carries across leagues via Champions League games)
  * days since last match, number of prior matches in the data
Per match: home-advantage flag (0 at neutral venues), Champions League flag, home-minus-away differences.
"""
import numpy as np
import pandas as pd

WINDOWS = (5, 15)
ROLL_STATS = ["gf", "ga", "shots_for", "shots_against", "sot_for", "sot_against", "poss", "points"]
ELO_START, ELO_K, ELO_HFA = 1500.0, 20.0, 60.0
LABELS = ["away_win", "draw", "home_win"]  # target encoding 0 / 1 / 2


def load_matches(path="data/matches.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], dtype={"home_id": str, "away_id": str, "event_id": str})
    df = df.sort_values(["date", "event_id"]).reset_index(drop=True)
    # treat a game as played only if it has a final score
    df["played"] = df["completed"] & df["home_goals"].notna() & df["away_goals"].notna()
    # ESPN reports missing match stats as zeros; possession that doesn't sum to ~100 marks those games
    bad_stats = (df["home_poss"] + df["away_poss"] - 100).abs() > 1
    stat_cols = [f"{s}_{c}" for s in ("home", "away") for c in ("poss", "shots", "sot", "corners", "fouls")]
    df.loc[bad_stats, stat_cols] = np.nan
    # postponed/cancelled games that never got a score are dropped unless they are still in the future
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["played"] | (df["date"] > now - pd.Timedelta(days=1))].reset_index(drop=True)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    gd = df["home_goals"] - df["away_goals"]
    df["target"] = np.select([gd < 0, gd == 0, gd > 0], [0, 1, 2], default=-1)
    df.loc[~df["played"], "target"] = -1
    return df


def compute_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-match Elo for every match. Upcoming matches get each team's current rating."""
    ratings: dict[str, float] = {}
    league_of: dict[str, str] = {}
    home_elo, away_elo = np.empty(len(df)), np.empty(len(df))

    def initial(team, league):
        # a team appearing for the first time in a domestic league after the first season is usually
        # newly promoted: start it at the league's lower quartile instead of the global average
        peers = [r for t, r in ratings.items() if league_of.get(t) == league]
        if league != "uefa.champions" and len(peers) >= 15:
            return float(np.percentile(peers, 25))
        return ELO_START

    for i, row in enumerate(df.itertuples(index=False)):
        for team in (row.home_id, row.away_id):
            if team not in ratings:
                ratings[team] = initial(team, row.league_code)
            if row.league_code != "uefa.champions":
                league_of[team] = row.league_code
        rh, ra = ratings[row.home_id], ratings[row.away_id]
        home_elo[i], away_elo[i] = rh, ra
        if not row.played:
            continue
        hfa = 0.0 if row.neutral else ELO_HFA
        expected_home = 1.0 / (1.0 + 10 ** ((ra - rh - hfa) / 400))
        gd = row.home_goals - row.away_goals
        actual_home = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        margin = 1.0 if abs(gd) <= 1 else 1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8
        delta = ELO_K * margin * (actual_home - expected_home)
        ratings[row.home_id] += delta
        ratings[row.away_id] -= delta

    df["home_elo"], df["away_elo"] = home_elo, away_elo
    return df


def team_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team) for played matches, from that team's perspective."""
    parts = []
    for side, opp in (("home", "away"), ("away", "home")):
        p = pd.DataFrame({
            "event_id": df["event_id"], "date": df["date"], "team_id": df[f"{side}_id"],
            "gf": df[f"{side}_goals"], "ga": df[f"{opp}_goals"],
            "shots_for": df[f"{side}_shots"], "shots_against": df[f"{opp}_shots"],
            "sot_for": df[f"{side}_sot"], "sot_against": df[f"{opp}_sot"],
            "poss": df[f"{side}_poss"],
        })
        parts.append(p)
    long = pd.concat(parts, ignore_index=True)
    long["points"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], default=0)
    return long.sort_values(["team_id", "date"]).reset_index(drop=True)


def team_states(df: pd.DataFrame) -> pd.DataFrame:
    """Each team's rolling state *after* each of its played matches."""
    long = team_long(df[df["played"]])
    g = long.groupby("team_id", sort=False)
    out = long[["team_id", "date"]].copy()
    for w in WINDOWS:
        rolled = g[ROLL_STATS].rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        for c in ROLL_STATS:
            out[f"{c}_{w}"] = rolled[c]
    out["n_prior"] = g.cumcount() + 1
    out["last_match"] = out["date"]
    return out


def attach_states(df: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """For each match side, take the team's latest state strictly before kickoff."""
    states = states.sort_values("date")
    for side in ("home", "away"):
        left = df[["event_id", "date", f"{side}_id"]].rename(columns={f"{side}_id": "team_id"})
        merged = pd.merge_asof(
            left.sort_values("date"), states, on="date", by="team_id", allow_exact_matches=False
        ).set_index("event_id")
        merged["rest_days"] = (merged["date"] - merged["last_match"]).dt.total_seconds() / 86400
        merged = merged.drop(columns=["team_id", "date", "last_match"])
        merged["n_prior"] = merged["n_prior"].fillna(0)
        merged.columns = [f"{side}_{c}" for c in merged.columns]
        df = df.join(merged, on="event_id")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_target(df.copy())
    df = compute_elo(df)
    df = attach_states(df, team_states(df))
    df["home_adv"] = (~df["neutral"]).astype(int)
    df["is_ucl"] = (df["league_code"] == "uefa.champions").astype(int)
    df["elo_diff"] = df["home_elo"] - df["away_elo"] + df["home_adv"] * ELO_HFA
    df["rest_diff"] = df["home_rest_days"].clip(upper=14) - df["away_rest_days"].clip(upper=14)
    df["home_rest_days"] = df["home_rest_days"].clip(upper=30)
    df["away_rest_days"] = df["away_rest_days"].clip(upper=30)
    for w in WINDOWS:
        for c in ROLL_STATS:
            df[f"diff_{c}_{w}"] = df[f"home_{c}_{w}"] - df[f"away_{c}_{w}"]
    return df


def feature_columns() -> list[str]:
    cols = ["home_adv", "is_ucl", "home_elo", "away_elo", "elo_diff", "rest_diff",
            "home_rest_days", "away_rest_days", "home_n_prior", "away_n_prior"]
    for w in WINDOWS:
        for c in ROLL_STATS:
            cols += [f"home_{c}_{w}", f"away_{c}_{w}", f"diff_{c}_{w}"]
    return cols
