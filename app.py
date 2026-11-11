"""Streamlit web app for the football match outcome predictor.

Run locally:  streamlit run app.py
Styling lives in assets/style.css and .streamlit/config.toml; the HTML pieces are built by the helpers below.
"""
import json
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from predictor import ROOT, Predictor

st.set_page_config(page_title="Match Outcome Predictor", page_icon="⚽", layout="wide",
                   initial_sidebar_state="collapsed")

GITHUB = "https://github.com/kinan897/Prediction-Model-"
HOME, AWAY, DRAW, ACCENT, MUTED, LINE = "#60a5fa", "#fb923c", "#94a3b8", "#34d399", "#8b97ad", "#1f2a40"
CREST = "https://a.espncdn.com/combiner/i?img=/i/teamlogos/soccer/500/{}.png&h=120&w=120"
LEAGUE_LOGO = "https://a.espncdn.com/i/leaguelogos/soccer/500/{}.png"
LEAGUE_IDS = {"Premier League": 23, "La Liga": 15, "Bundesliga": 10, "Serie A": 12, "Ligue 1": 9,
              "Champions League": 2}

st.html(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>")


@alt.theme.register("matchday", enable=True)
def _matchday_theme():
    axis = {"labelColor": MUTED, "titleColor": MUTED, "gridColor": LINE, "domainColor": LINE, "tickColor": LINE,
            "labelFont": "Inter", "titleFont": "Inter", "titleFontWeight": 500, "labelFontSize": 12}
    return alt.theme.ThemeConfig({"config": {
        "background": "transparent", "view": {"stroke": None}, "font": "Inter", "axis": axis,
        "legend": {"labelColor": "#cfd7e6", "titleColor": MUTED, "labelFont": "Inter", "orient": "bottom"},
        "text": {"color": "#e6ebf5", "font": "Inter"}}})


# ---------- data ------------------------------------------------------------------------------
@st.cache_resource(ttl=6 * 3600, show_spinner="Loading match history and model…")
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def get_report() -> dict:
    return json.loads((ROOT / "reports" / "metrics.json").read_text())


@st.cache_data(show_spinner="Crunching current form…", max_entries=200)
def custom_match(home: str, away: str, neutral: bool, ucl: bool) -> pd.Series:
    return get_predictor().custom_match(home, away, neutral, ucl)


P = get_predictor()
R = get_report()
selected = next(r for r in R["results"] if r["model"] == R["selected"])
home_baseline = next(r for r in R["results"] if r["model"].startswith("baseline: always"))
MODEL_NAMES = {"random_forest": "Random Forest", "xgboost": "XGBoost",
               "baseline: always home win (probs = class freq.)": "Always pick home win",
               "baseline: Elo-only logistic regression": "Elo-only logistic regression"}
MODEL_NAME = MODEL_NAMES.get(R["selected"], R["selected"])
N_PLAYED = int(P.matches.played.sum())


# ---------- HTML building blocks --------------------------------------------------------------
def crest(team_id: str, size: str = "md") -> str:
    return f'<span class="crest crest-{size}"><img src="{CREST.format(team_id)}" alt="" loading="lazy"></span>'


def league_logo(competition: str) -> str:
    lid = LEAGUE_IDS.get(competition)
    return f'<img src="{LEAGUE_LOGO.format(lid)}" alt="">' if lid else ""


def form_chips(team_id: str) -> str:
    res = P.recent_results(team_id, 5)
    if res.empty:
        return '<span class="conf">no matches yet</span>'
    return "".join(f'<span class="chip {r}" title="{escape(o)} {s}">{r}</span>'
                   for r, o, s in zip(res.result, res.opponent, res.score))


def section(title: str, sub: str = "") -> None:
    st.html(f'<div class="sec"><h2>{title}</h2>{f"<p>{sub}</p>" if sub else ""}</div>')


def match_card(row: pd.Series, focus_id: str | None = None) -> str:
    if row.event_id == "custom":
        when = "Hypothetical match · played today"
    else:
        when = row.date.strftime("%a %d %b %Y · %H:%M UTC")
    venue_tag = "NEUTRAL" if row.neutral else "HOME"
    outcomes = [("home", f"{row.home_team} win", row.p_home), ("draw", "Draw", row.p_draw),
                ("away", f"{row.away_team} win", row.p_away)]
    best = max(outcomes, key=lambda o: o[2])
    p = best[2]
    confidence = ("Strong favourite" if p >= .6 else "Clear favourite" if p >= .5 else
                  "Slight edge" if p >= .42 else "Too close to call")

    def side(tid, name, tag):
        focus = " focus" if tid == focus_id else ""
        tag = f"{tag} · YOUR TEAM" if focus else tag
        return (f'<div class="side{focus}">{crest(tid, "lg")}<div class="tname">{escape(name)}</div>'
                f'<div class="ttag">{tag}</div><div class="form">{form_chips(tid)}</div></div>')

    nums = "".join(f'<div class="pnum {k}{" best" if k == best[0] else ""}"><div class="pv">{v:.0%}</div>'
                   f'<div class="pl">{escape(label)}</div></div>' for k, label, v in outcomes)
    bar = "".join(f'<div class="seg {k}" style="flex:{v:.4f}"></div>' for k, _, v in outcomes)
    return (f'<div class="match-card">'
            f'<div class="mc-top"><span class="comp">{league_logo(row.competition)}{escape(row.competition)}</span>'
            f'<span>{when}</span></div>'
            f'<div class="mc-teams">{side(row.home_id, row.home_team, venue_tag)}<div class="vs">VS</div>'
            f'{side(row.away_id, row.away_team, "AWAY" if not row.neutral else "NEUTRAL")}</div>'
            f'<div class="pnums">{nums}</div><div class="pbar">{bar}</div>'
            f'<div class="verdict"><span class="pill">Model pick</span><b>{escape(best[1])}</b>'
            f'<span class="conf">· {confidence} · {MODEL_NAME}</span></div></div>')


TAPE = [  # label, feature key, format, higher is better (None = neutral), offset to make bars readable
    ("Elo rating", "elo", "{:.0f}", True, 1200),
    ("Points per game · last 5", "points_5", "{:.2f}", True, 0),
    ("Points per game · last 15", "points_15", "{:.2f}", True, 0),
    ("Goals scored per game", "gf_15", "{:.2f}", True, 0),
    ("Goals conceded per game", "ga_15", "{:.2f}", False, 0),
    ("Shots per game", "shots_for_15", "{:.1f}", True, 0),
    ("Shots on target per game", "sot_for_15", "{:.1f}", True, 0),
    ("Shots on target conceded", "sot_against_15", "{:.1f}", False, 0),
    ("Possession", "poss_15", "{:.0f}%", True, 0),
    ("Days of rest", "rest_days", "{:.0f}", None, 0),
]


def tale_of_the_tape(row: pd.Series) -> str:
    rows = []
    for label, key, fmt, higher, offset in TAPE:
        h, a = row.get(f"home_{key}"), row.get(f"away_{key}")
        if h is None or a is None or pd.isna(h) or pd.isna(a):
            continue
        hv, av = max(h - offset, 0), max(a - offset, 0)
        total = (hv + av) or 1
        hb = ab = ""
        if higher is not None and abs(h - a) > 1e-9:
            home_better = (h > a) == higher
            hb, ab = (" better", "") if home_better else ("", " better")
        rows.append(
            f'<div class="tape-row"><div class="tl">{label}</div><div class="tgrid">'
            f'<span class="tv{hb}">{fmt.format(h)}</span>'
            f'<div class="tbar l{hb}"><span style="width:{hv / total * 100:.1f}%"></span></div>'
            f'<div class="tbar r{ab}"><span style="width:{av / total * 100:.1f}%"></span></div>'
            f'<span class="tv r{ab}">{fmt.format(a)}</span></div></div>')
    head = (f'<div class="tape-head"><span class="th">{crest(row.home_id, "md")}{escape(row.home_team)}</span>'
            f'<span class="th">{escape(row.away_team)}{crest(row.away_id, "md")}</span></div>')
    return f'<div class="panel"><h3>Tale of the tape</h3>{head}{"".join(rows)}</div>'


def recent_panel(row: pd.Series) -> str:
    blocks = []
    for tid, name in ((row.home_id, row.home_team), (row.away_id, row.away_team)):
        res = P.recent_results(tid, 5).iloc[::-1]
        items = "".join(
            f'<div class="rr"><span class="rr-date">{r.date:%d %b}</span><span class="chip {r.result}">{r.result}</span>'
            f'<span class="rr-opp"><span class="ha">{r.venue}</span>{crest(r.opponent_id, "sm")}'
            f'<span class="n">{escape(r.opponent)}</span></span><span class="rr-score">{r.score}</span></div>'
            for r in res.itertuples())
        blocks.append(f'<div class="rr-team">{crest(tid, "md")}{escape(name)}</div><div class="rr-list">{items}</div>')
    return f'<div class="panel"><h3>Recent results</h3>{"".join(blocks)}</div>'


def render_match(row: pd.Series, focus_id: str | None = None) -> None:
    st.html(match_card(row, focus_id))
    left, right = st.columns([1.35, 1], gap="medium")
    left.html(tale_of_the_tape(row))
    right.html(recent_panel(row))


def fixture_list(up: pd.DataFrame) -> str:
    out = []
    for day, g in up.groupby(up.date.dt.date, sort=True):
        out.append(f'<div class="day">{pd.Timestamp(day):%A · %d %B}</div>')
        for r in g.itertuples():
            probs = {"home": r.p_home, "draw": r.p_draw, "away": r.p_away}
            best = max(probs, key=probs.get)
            nums = "".join(f'<span class="{k}{" best" if k == best else ""}">{v:.0%}</span>' for k, v in probs.items())
            bar = "".join(f'<div class="seg {k}" style="flex:{v:.4f}"></div>' for k, v in probs.items())
            out.append(
                f'<div class="fx"><div class="fx-time">{league_logo(r.competition)}{r.date:%H:%M}</div>'
                f'<div class="fx-team h"><span class="n">{escape(r.home_team)}</span>{crest(r.home_id, "sm")}</div>'
                f'<div class="fx-mid"><div class="fx-nums">{nums}</div><div class="pbar sm">{bar}</div></div>'
                f'<div class="fx-team a">{crest(r.away_id, "sm")}<span class="n">{escape(r.away_team)}</span></div></div>')
    return "".join(out)


def team_picker(key: str, default: str | None, label: str) -> str:
    leagues = ["All leagues"] + sorted(P.teams.league.unique())
    c1, c2 = st.columns([1, 1.4])
    lg = c1.selectbox("League", leagues, key=f"{key}_lg")
    pool = P.teams if lg == "All leagues" else P.teams[P.teams.league == lg]
    ids = list(pool.index)
    idx = ids.index(default) if default in ids else 0
    return c2.selectbox(label, ids, index=idx, format_func=lambda t: P.names[t], key=f"{key}_team")


def default_team(name: str) -> str | None:
    try:
        return P.find_team(name)
    except ValueError:
        return None


def pretty_feature(f: str) -> str:
    fixed = {"elo_diff": "Elo difference", "home_elo": "Home Elo", "away_elo": "Away Elo",
             "home_adv": "Home advantage", "is_ucl": "Champions League game", "rest_diff": "Rest-days difference",
             "home_rest_days": "Home rest days", "away_rest_days": "Away rest days",
             "home_n_prior": "Home matches in data", "away_n_prior": "Away matches in data"}
    if f in fixed:
        return fixed[f]
    stats = {"gf": "goals scored", "ga": "goals conceded", "shots_for": "shots", "shots_against": "shots conceded",
             "sot_for": "shots on target", "sot_against": "shots on target conceded", "poss": "possession",
             "points": "points per game"}
    side, rest = f.split("_", 1)
    stat, window = rest.rsplit("_", 1)
    prefix = {"diff": "Δ", "home": "Home", "away": "Away"}[side]
    return f"{prefix} {stats.get(stat, stat)} (last {window})"


# ---------- hero + KPIs -----------------------------------------------------------------------
leagues_html = "".join(f'<span class="league"><img src="{LEAGUE_LOGO.format(i)}" alt="">{n}</span>'
                       for n, i in LEAGUE_IDS.items())
st.html(f"""
<section class="hero">
  <span class="eyebrow"><span class="dot"></span>Machine learning · Football analytics</span>
  <h1>Match Outcome <span class="grad">Predictor</span></h1>
  <p class="lede">Win, draw and loss probabilities for Europe's top five leagues and the Champions League,
  from a {MODEL_NAME} trained on {N_PLAYED:,} matches of goals, shots, possession and form.</p>
  <div class="leagues">{leagues_html}</div>
  <div class="cta"><a class="btn" href="{GITHUB}" target="_blank" rel="noopener">View source on GitHub ↗</a></div>
</section>""")

lift = (selected["accuracy"] - home_baseline["accuracy"]) * 100
st.html(f"""
<div class="kpis">
  <div class="kpi"><div class="k-label">Matches analysed</div><div class="k-val">{N_PLAYED:,}</div>
    <div class="k-sub">6 competitions since 2019-20</div></div>
  <div class="kpi"><div class="k-label">Test accuracy</div><div class="k-val">{selected['accuracy']:.1%}</div>
    <div class="k-sub up">▲ {lift:.1f} pts vs home-win baseline</div></div>
  <div class="kpi"><div class="k-label">Features per match</div><div class="k-val">{len(P.cols)}</div>
    <div class="k-sub">form, shots, possession, Elo</div></div>
  <div class="kpi"><div class="k-label">Results up to</div><div class="k-val">{P.data_as_of:%d %b}</div>
    <div class="k-sub">{len(P.upcoming):,} upcoming fixtures</div></div>
</div>""")

tab_next, tab_custom, tab_upcoming, tab_perf, tab_how = st.tabs(
    ["Next match", "Head to head", "Fixtures", "Model performance", "How it works"])

# ---------- next match ------------------------------------------------------------------------
with tab_next:
    with st.container(border=True):
        tid = team_picker("next", default_team("Arsenal"), "Team")
    nxt = P.next_match(tid)
    if nxt is None:
        st.html(f'<div class="empty">No scheduled fixture found for <b>{escape(P.names[tid])}</b>.<br>'
                f'Try the <b>Head to head</b> tab to set up any matchup.</div>')
    else:
        render_match(nxt[0], focus_id=tid)

# ---------- custom matchup --------------------------------------------------------------------
with tab_custom:
    with st.container(border=True):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            home = team_picker("home", default_team("Real Madrid"), "Home team")
        with c2:
            away = team_picker("away", default_team("Bayern Munich"), "Away team")
        c3, c4 = st.columns(2, gap="large")
        venue = c3.segmented_control("Venue", ["Home team's stadium", "Neutral venue"],
                                     default="Home team's stadium", key="venue")
        comp = c4.segmented_control("Competition", ["League", "Champions League"], default="Champions League",
                                    key="comp")
    if home == away:
        st.html('<div class="empty">Pick two different teams.</div>')
    else:
        render_match(custom_match(home, away, venue == "Neutral venue", comp == "Champions League"))

# ---------- upcoming --------------------------------------------------------------------------
with tab_upcoming:
    with st.container(border=True):
        c1, c2 = st.columns([2, 1], gap="large")
        comps = sorted(P.upcoming.competition.unique())
        chosen = c1.pills("Competitions", comps, default=comps, selection_mode="multi", key="comps")
        days = c2.slider("Days ahead", 1, 45, 14)
    up = P.upcoming[P.upcoming.competition.isin(chosen or []) & (P.upcoming.date <= P.now + pd.Timedelta(days=days))]
    if up.empty:
        st.html('<div class="empty">No fixtures in that window. Try more days or more competitions.</div>')
    else:
        st.html(fixture_list(up) + f'<div class="count">{len(up)} fixtures · kick-off times in UTC · '
                                   f'<span style="color:{HOME}">home win</span> / draw / '
                                   f'<span style="color:{AWAY}">away win</span></div>')

# ---------- performance -----------------------------------------------------------------------
with tab_perf:
    section("How good is it?",
            f"Tuned on {R['train_matches']:,} matches ({R['train_period'][0]} → {R['train_period'][1]}), then tested "
            f"on {R['test_matches']:,} later matches ({R['test_period'][0]} → {R['test_period'][1]}) it never saw.")
    best_acc = max(r["accuracy"] for r in R["results"])
    rows = []
    for r in R["results"]:
        sel = r["model"] == R["selected"]
        name = MODEL_NAMES.get(r["model"], r["model"]) + ('<span class="tag">deployed</span>' if sel else "")
        rows.append(f'<tr class="{"sel" if sel else ""}"><td>{name}</td>'
                    f'<td><div class="acc"><span>{r["accuracy"]:.1%}</span><div class="bar">'
                    f'<span style="width:{r["accuracy"] / best_acc * 100:.0f}%"></span></div></div></td>'
                    f'<td>{r["log_loss"]:.3f}</td><td>{r["brier"]:.3f}</td></tr>')
    st.html('<div class="rwrap"><table class="rtable"><thead><tr><th>Model</th><th>Accuracy</th>'
            '<th>Log loss ↓</th><th>Brier ↓</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
            '<p class="note">Log loss and Brier score grade the probabilities themselves (lower is better). '
            'Bookmakers reach roughly 53–55% accuracy on top-flight football, so the ceiling is low.</p>')

    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        st.html('<p class="chart-title">What drives the prediction</p>'
                '<p class="chart-sub">Top 10 features by importance</p>')
        fi = pd.Series(R["feature_importance"]).head(10)
        fi = pd.DataFrame({"feature": [pretty_feature(f) for f in fi.index], "importance": fi.values})
        bars = alt.Chart(fi).mark_bar(cornerRadiusEnd=4, height=16).encode(
            x=alt.X("importance:Q", title=None, axis=alt.Axis(format="%", tickCount=4)),
            y=alt.Y("feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
            color=alt.Color("importance:Q", scale=alt.Scale(range=["#1d4ed8", ACCENT]), legend=None),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".1%")])
        st.altair_chart(bars.properties(height=330), use_container_width=True, theme=None)
    with b, st.container(border=True):
        st.html('<p class="chart-title">Confusion matrix</p>'
                '<p class="chart-sub">Test-set predictions vs what actually happened</p>')
        labels = ["Away win", "Draw", "Home win"]
        cm = pd.DataFrame([{"actual": labels[i], "predicted": labels[j], "n": v}
                           for i, row in enumerate(R["confusion_matrix"]) for j, v in enumerate(row)])
        heat = alt.Chart(cm).encode(x=alt.X("predicted:N", sort=labels, title="Predicted", axis=alt.Axis(labelAngle=0)),
                                    y=alt.Y("actual:N", sort=labels, title="Actual"))
        rect = heat.mark_rect(cornerRadius=6, stroke="#0a0f1c", strokeWidth=3).encode(
            color=alt.Color("n:Q", scale=alt.Scale(range=["#131b2e", "#0f766e", ACCENT]), legend=None),
            tooltip=["actual", "predicted", "n"])
        txt = heat.mark_text(fontSize=16, fontWeight="bold", font="Space Grotesk").encode(
            text="n:Q", color=alt.condition("datum.n > 500", alt.value("#06281c"), alt.value("#e6ebf5")))
        st.altair_chart((rect + txt).properties(height=330), use_container_width=True, theme=None)

    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        st.html('<p class="chart-title">Accuracy by competition</p><p class="chart-sub">Test set</p>')
        bc = pd.DataFrame(R["accuracy_by_competition"])
        base = alt.Chart(bc).encode(y=alt.Y("competition:N", sort="-x", title=None))
        acc_bars = base.mark_bar(cornerRadiusEnd=4, height=18, color=HOME).encode(
            x=alt.X("accuracy:Q", title=None, axis=alt.Axis(format="%", tickCount=4), scale=alt.Scale(domain=[0, .7])),
            tooltip=["competition", alt.Tooltip("accuracy:Q", format=".1%"), "matches"])
        acc_txt = base.mark_text(align="left", dx=6, fontWeight="bold").encode(
            x="accuracy:Q", text=alt.Text("accuracy:Q", format=".0%"))
        st.altair_chart((acc_bars + acc_txt).properties(height=250), use_container_width=True, theme=None)
    with b, st.container(border=True):
        st.html('<p class="chart-title">Calibration</p>'
                '<p class="chart-sub">Close to the dashed line = probabilities can be taken at face value</p>')
        cal = pd.concat([pd.DataFrame(v).assign(outcome=k.replace("_", " ")) for k, v in R["calibration"].items()])
        diag = alt.Chart(pd.DataFrame({"x": [0, .8], "y": [0, .8]})).mark_line(
            strokeDash=[5, 5], color="#3b4a66").encode(x="x:Q", y="y:Q")
        lines = alt.Chart(cal).mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2.5).encode(
            x=alt.X("predicted:Q", title="Predicted probability", axis=alt.Axis(format="%", tickCount=5)),
            y=alt.Y("actual:Q", title="Actual frequency", axis=alt.Axis(format="%", tickCount=5)),
            color=alt.Color("outcome:N", title=None,
                            scale=alt.Scale(domain=["home win", "draw", "away win"], range=[HOME, DRAW, AWAY])),
            tooltip=["outcome", alt.Tooltip("predicted:Q", format=".1%"), alt.Tooltip("actual:Q", format=".1%"), "n"])
        st.altair_chart((diag + lines).properties(height=250), use_container_width=True, theme=None)

# ---------- how it works ----------------------------------------------------------------------
with tab_how:
    section("The pipeline", "From raw match data to a probability, in five steps.")
    steps = [
        ("Collect", "<code>fetch_data.py</code> pulls every match since 2019-20 from ESPN's public API: score, "
                    "possession, shots, shots on target, plus upcoming fixtures."),
        ("Engineer", "<code>features.py</code> builds 58 features per match from <b>only games played before "
                     "kick-off</b>: rolling 5- and 15-match form, goals, shots, possession and an Elo rating."),
        ("Train", "<code>train.py</code> tunes a Random Forest and XGBoost with time-series cross-validation, "
                  "optimising log loss. The split is by date, never shuffled."),
        ("Evaluate", f"Both models are scored on {R['test_matches']:,} unseen recent matches against an "
                     "always-home-win baseline and an Elo-only model."),
        ("Serve", "This Streamlit app and a command-line tool share <code>predictor.py</code> to turn current "
                  "form into win / draw / loss probabilities."),
    ]
    st.html('<div class="steps">' + "".join(
        f'<div class="step"><div class="n">{i}</div><h4>{t}</h4><p>{d}</p></div>'
        for i, (t, d) in enumerate(steps, 1)) + "</div>")

    section("What the model learned")
    insights = [
        ("Team strength rules", "Elo difference is the single most important feature, followed by the gap in "
                                "shots on target, which shows chance quality better than goals do."),
        ("Beats the baseline", f"{lift:.1f} accuracy points above always picking the home side. Its edge over "
                               "Elo alone is small, because most of the predictable signal is team strength."),
        ("Draws are hard", "About a quarter of matches end level, yet a draw is almost never the single most "
                           "likely result. The draw probabilities themselves are still well calibrated."),
    ]
    st.html('<div class="insights">' + "".join(
        f'<div class="insight"><b>{t}</b><span>{d}</span></div>' for t, d in insights) + "</div>")

    section("Built with")
    st.html('<div class="badges">' + "".join(f"<span>{t}</span>" for t in
            ["Python", "pandas", "NumPy", "scikit-learn", "XGBoost", "Streamlit", "Altair", "ESPN API"]) + "</div>")

st.html(f'<footer class="foot"><span>Built with Python, scikit-learn, XGBoost and Streamlit · Data from ESPN</span>'
        f'<span><a href="{GITHUB}" target="_blank" rel="noopener">Source on GitHub</a> · For fun and learning, '
        f'not betting advice</span></footer>')
