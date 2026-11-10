"""Command-line predictions (the web app in app.py offers the same things in a browser).

  python predict.py "Arsenal"                          # Arsenal's next scheduled match
  python predict.py "Real Madrid" --opponent "Bayern" --venue away --ucl   # any hypothetical match
  python predict.py --upcoming --days 7                # every fixture in the next 7 days

Run fetch_data.py first to refresh results and fixtures (the model uses form up to the latest result).
"""
import argparse
import sys

import pandas as pd

from predictor import Predictor


def show(row, team_is_home: bool, model_name: str):
    probs = Predictor.team_probs(row, team_is_home)
    team, opp = (row.home_team, row.away_team) if team_is_home else (row.away_team, row.home_team)
    venue = "neutral" if row.neutral else ("home" if team_is_home else "away")
    when = "now (hypothetical)" if row.event_id == "custom" else row.date.strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{team} vs {opp}  ({row.competition}, {venue}, {when})")
    for k, v in probs.items():
        print(f"  {k:5s} {v:6.1%}  {'#' * round(v * 40)}")
    print(f"  -> most likely: {max(probs, key=probs.get).upper()}   (model: {model_name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("team", nargs="?")
    ap.add_argument("--opponent")
    ap.add_argument("--venue", choices=["home", "away", "neutral"], default="home")
    ap.add_argument("--ucl", action="store_true", help="hypothetical match is a Champions League game")
    ap.add_argument("--upcoming", action="store_true")
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    if not args.team and not args.upcoming:
        ap.error("give a team name or --upcoming")

    p = Predictor()
    if args.upcoming:
        up = p.upcoming[p.upcoming.date <= p.now + pd.Timedelta(days=args.days)]
        if up.empty:
            sys.exit(f"No fixtures in the next {args.days} days (run fetch_data.py to refresh).")
        table = pd.DataFrame({
            "date": up.date.dt.strftime("%a %d %b %H:%M"), "competition": up.competition,
            "home": up.home_team, "away": up.away_team,
            "home_win": up.p_home.map("{:.0%}".format), "draw": up.p_draw.map("{:.0%}".format),
            "away_win": up.p_away.map("{:.0%}".format)})
        print(table.to_string(index=False))
        return

    try:
        tid = p.find_team(args.team)
        if args.opponent:
            oid = p.find_team(args.opponent)
            home, away = (oid, tid) if args.venue == "away" else (tid, oid)
            row = p.custom_match(home, away, neutral=args.venue == "neutral", ucl=args.ucl)
            show(row, home == tid, p.model_name)
            return
    except ValueError as e:
        sys.exit(str(e))

    nxt = p.next_match(tid)
    if nxt is None:
        sys.exit("No upcoming fixture found for that team. Use --opponent for a hypothetical match.")
    show(*nxt, p.model_name)


if __name__ == "__main__":
    main()
