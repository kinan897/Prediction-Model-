"""Download historical league + Champions League matches (with match stats) from ESPN's public API.

Writes data/matches.csv with one row per match: teams, score, possession, shots, shots on target,
corners, fouls, plus upcoming (scheduled) fixtures so predict.py can find a team's next match.

Usage:  python fetch_data.py [--seasons 2019 2026] [--workers 8]
Parsed results are cached per match day in data/cache/, so re-runs only fetch new/recent days.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
COMPETITIONS = {
    "eng.1": "Premier League",
    "esp.1": "La Liga",
    "ger.1": "Bundesliga",
    "ita.1": "Serie A",
    "fra.1": "Ligue 1",
    "uefa.champions": "Champions League",
}
STATS = {
    "possessionPct": "poss",
    "totalShots": "shots",
    "shotsOnTarget": "sot",
    "wonCorners": "corners",
    "foulsCommitted": "fouls",
}
DATA = Path(__file__).parent / "data"
CACHE = DATA / "cache"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "match-predictor/1.0"


def get(league: str, date: str) -> dict:
    for attempt in range(4):
        try:
            r = SESSION.get(BASE.format(league=league), params={"dates": date, "limit": 200}, timeout=20)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed: {league} {date}")


def season_calendar(league: str, season: int) -> list[str]:
    """Match days (YYYYMMDD) for a season, using the calendar ESPN returns with any in-season date."""
    data = get(league, f"{season}1015")
    cal = data["leagues"][0].get("calendar", [])
    days = []
    for entry in cal:
        # league calendars are plain ISO strings; UCL returns stage objects with entries
        if isinstance(entry, str):
            days.append(entry[:10].replace("-", ""))
        elif isinstance(entry, dict):
            for sub in entry.get("entries", []) or []:
                start = datetime.fromisoformat(sub["startDate"].replace("Z", "+00:00"))
                end = datetime.fromisoformat(sub["endDate"].replace("Z", "+00:00"))
                d = start
                while d <= end:
                    days.append(d.strftime("%Y%m%d"))
                    d += timedelta(days=1)
    return sorted(set(days))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse(league: str, data: dict) -> list[dict]:
    rows = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        status = comp["status"]["type"]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        if "home" not in teams or "away" not in teams:
            continue
        row = {
            "event_id": ev["id"],
            "date": ev["date"],
            "competition": COMPETITIONS[league],
            "league_code": league,
            "season": ev.get("season", {}).get("year"),
            "stage": ev.get("season", {}).get("slug"),
            "status": status.get("name"),
            "completed": bool(status.get("completed")),
            "neutral": bool(comp.get("neutralSite")),
        }
        for side in ("home", "away"):
            c = teams[side]
            row[f"{side}_team"] = c["team"]["displayName"]
            row[f"{side}_id"] = c["team"]["id"]
            row[f"{side}_goals"] = num(c.get("score")) if row["completed"] else None
            stats = {s["name"]: s.get("displayValue") for s in c.get("statistics", [])}
            for espn_name, col in STATS.items():
                row[f"{side}_{col}"] = num(stats.get(espn_name))
        rows.append(row)
    return rows


def fetch_day(league: str, day: str, today: str) -> list[dict]:
    path = CACHE / league / f"{day}.json"
    # past days are final; re-fetch anything from the last 3 days onward (results/fixtures may change)
    if path.exists() and day < today:
        return json.loads(path.read_text())
    rows = parse(league, get(league, day))
    path.parent.mkdir(parents=True, exist_ok=True)
    if day < today:
        path.write_text(json.dumps(rows))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs=2, type=int, default=[2019, 2026], metavar=("FIRST", "LAST"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    today = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y%m%d")
    jobs = []
    for league in COMPETITIONS:
        for season in range(args.seasons[0], args.seasons[1] + 1):
            days = season_calendar(league, season)
            print(f"{COMPETITIONS[league]:17s} {season}-{(season + 1) % 100:02d}: {len(days)} match days")
            jobs += [(league, d) for d in days]

    rows, done = [], 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch_day, lg, d, today) for lg, d in jobs]
        for f in as_completed(futures):
            rows += f.result()
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)} days fetched")

    df = pd.DataFrame(rows).drop_duplicates("event_id")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values("date").reset_index(drop=True)
    DATA.mkdir(exist_ok=True)
    df.to_csv(DATA / "matches.csv", index=False)

    played = df[df.completed]
    print(f"\nSaved {len(df)} matches ({len(played)} played, {len(df) - len(played)} upcoming) -> data/matches.csv")
    print("Stat coverage of played matches (share with possession recorded):")
    print(played.groupby(["competition", "season"])["home_poss"].apply(lambda s: s.notna().mean()).unstack().round(2))


if __name__ == "__main__":
    main()
