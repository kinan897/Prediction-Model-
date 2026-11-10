"""Streamlit web app for the football match outcome predictor.

Run locally:  streamlit run app.py
"""
import json

import altair as alt
import pandas as pd
import streamlit as st

from predictor import ROOT, Predictor

st.set_page_config(page_title="Match Outcome Predictor", page_icon="⚽", layout="wide")

WIN, DRAW, LOSS = "#1f9d55", "#8a94a6", "#d64545"
HOME, AWAY = "#2563eb", "#e8590c"
GITHUB = "https://github.com/kinan897/Prediction-Model-"

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.chip {display:inline-block; width:2rem; height:2rem; line-height:2rem; border-radius:6px; margin-right:4px;
       text-align:center; font-weight:700; color:white; font-size:0.9rem;}
.chip.W {background:#1f9d55;} .chip.D {background:#8a94a6;} .chip.L {background:#d64545;}
.fixture {font-size:1.35rem; font-weight:700; margin-bottom:0;}
.sub {color:#6b7280; margin-top:0.1rem;}
</style>""", unsafe_allow_html=True)


# ---------- data ------------------------------------------------------------------------------
@st.cache_resource(ttl=6 * 3600, show_spinner="Loading match history and model…")
def get_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def get_report() -> dict:
    return json.loads((ROOT / "reports" / "metrics.json").read_text())


@st.cache_data(show_spinner="Computing current form…", max_entries=200)
def custom_match(home: str, away: str, neutral: bool, ucl: bool) -> pd.Series:
    return get_predictor().custom_match(home, away, neutral, ucl)


P = get_predictor()
R = get_report()
selected = next(r for r in R["results"] if r["model"] == R["selected"])
home_baseline = next(r for r in R["results"] if r["model"].startswith("baseline: always"))
team_label = lambda tid: P.names[tid]
MODEL_NAME = {"random_forest": "Random Forest", "xgboost": "XGBoost"}.get(R["selected"], R["selected"])


# ---------- shared widgets --------------------------------------------------------------------
def prob_bar(parts: list[tuple[str, float, str]]) -> alt.Chart:
    """Single horizontal stacked bar. parts = [(label, probability, colour), ...]."""
    d = pd.DataFrame(parts, columns=["outcome", "p", "color"])
    d["end"] = d.p.cumsum()
    d["start"] = d.end - d.p
    d["mid"] = (d.start + d.end) / 2
    d["text"] = [f"{o}  {p:.0%}" if p >= 0.12 else f"{p:.0%}" for o, p in zip(d.outcome, d.p)]
    base = alt.Chart(d)
    bars = base.mark_bar(height=52, cornerRadius=4).encode(
        x=alt.X("start:Q", axis=None, scale=alt.Scale(domain=[0, 1])), x2="end:Q",
        color=alt.Color("outcome:N", scale=alt.Scale(domain=list(d.outcome), range=list(d.color)), legend=None),
        tooltip=[alt.Tooltip("outcome:N"), alt.Tooltip("p:Q", format=".1%", title="probability")])
    text = base.mark_text(color="white", fontWeight="bold", fontSize=14).encode(x="mid:Q", text="text:N")
    return (bars + text).properties(height=60).configure_view(strokeWidth=0)


def form_chips(team_id: str) -> str:
    res = P.recent_results(team_id, 5)
    if res.empty:
        return "<span class='sub'>no matches yet</span>"
    return "".join(f"<span class='chip {r}' title='{o} {s}'>{r}</span>"
                   for r, o, s in zip(res.result, res.opponent, res.score))


def render_match(row: pd.Series, focus_home: bool | None = None):
    """Fixture header, probabilities and side-by-side stats. focus_home=None shows home/draw/away."""
    venue = "neutral venue" if row.neutral else f"at {row.home_team}"
    when = "played today (hypothetical)" if row.event_id == "custom" else row.date.strftime("%a %d %b %Y · %H:%M UTC")
    st.markdown(f"<p class='fixture'>{row.home_team} vs {row.away_team}</p>"
                f"<p class='sub'>{row.competition} · {venue} · {when}</p>", unsafe_allow_html=True)

    if focus_home is None:
        parts = [(f"{row.home_team} win", row.p_home, HOME), ("Draw", row.p_draw, DRAW),
                 (f"{row.away_team} win", row.p_away, AWAY)]
    else:
        pr = Predictor.team_probs(row, focus_home)
        parts = [("Win", pr["win"], WIN), ("Draw", pr["draw"], DRAW), ("Loss", pr["loss"], LOSS)]
    cols = st.columns(3)
    for c, (label, p, _) in zip(cols, parts):
        c.metric(label, f"{p:.1%}")
    st.altair_chart(prob_bar(parts), use_container_width=True)
    pick = max(parts, key=lambda x: x[1])[0]
    st.caption(f"Most likely outcome: **{pick}** · model: {MODEL_NAME}")

    left, right = st.columns([1, 2], gap="medium")
    with left:
        st.markdown("##### Recent form")
        st.caption("Last 5 matches, oldest → newest")
        for tid, name in ((row.home_id, row.home_team), (row.away_id, row.away_team)):
            st.markdown(f"**{name}**<br>{form_chips(tid)}", unsafe_allow_html=True)
            st.write("")
    with right:
        st.markdown("##### Pre-match comparison")
        st.table(Predictor.comparison(row))


def team_picker(key: str, default: str | None = None, label: str = "Team") -> str:
    leagues = ["All leagues"] + sorted(P.teams.league.unique())
    lg = st.selectbox("League", leagues, key=f"{key}_lg")
    pool = P.teams if lg == "All leagues" else P.teams[P.teams.league == lg]
    ids = list(pool.index)
    idx = ids.index(default) if default in ids else 0
    return st.selectbox(label, ids, index=idx, format_func=team_label, key=f"{key}_team")


def default_team(name: str) -> str | None:
    try:
        return P.find_team(name)
    except ValueError:
        return None


# ---------- header ----------------------------------------------------------------------------
st.title("⚽ Football Match Outcome Predictor")
st.markdown("Win / draw / loss probabilities for **Premier League, La Liga, Bundesliga, Serie A, Ligue 1** "
            "and **Champions League** matches, from a "
            f"{MODEL_NAME} trained on {P.matches.played.sum():,} matches since 2019.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Matches in dataset", f"{P.matches.played.sum():,}")
m2.metric("Test accuracy", f"{selected['accuracy']:.1%}",
          f"{(selected['accuracy'] - home_baseline['accuracy']) * 100:+.1f} pts vs home-win baseline")
m3.metric("Test log loss", f"{selected['log_loss']:.3f}", f"{selected['log_loss'] - home_baseline['log_loss']:+.3f}",
          delta_color="inverse")
m4.metric("Results up to", P.data_as_of.strftime("%d %b %Y"))

with st.sidebar:
    st.header("About")
    st.markdown(
        "A machine-learning model that predicts football match outcomes from each team's recent "
        "goals, shots, shots on target, possession, form, Elo rating and home advantage.\n\n"
        f"- **Model:** {MODEL_NAME}\n"
        f"- **Trained on:** {R['train_period'][0]} → {R['train_period'][1]}\n"
        f"- **Tested on:** {R['test_period'][0]} → {R['test_period'][1]} ({R['test_matches']:,} unseen matches)\n"
        f"- **Data:** ESPN public match data\n\n"
        f"[Source code on GitHub]({GITHUB})")
    st.caption("For fun and learning, not betting advice.")

tab_next, tab_custom, tab_upcoming, tab_perf, tab_how = st.tabs(
    ["🔮 Next match", "⚔️ Custom matchup", "📅 Upcoming fixtures", "📊 Model performance", "🛠️ How it works"])

# ---------- next match ------------------------------------------------------------------------
with tab_next:
    c1, c2 = st.columns([1, 2.6], gap="large")
    with c1:
        tid = team_picker("next", default_team("Arsenal"))
    with c2:
        nxt = P.next_match(tid)
        if nxt is None:
            st.info(f"No scheduled fixture found for {P.names[tid]}. Try the **Custom matchup** tab.")
        else:
            render_match(nxt[0], focus_home=nxt[1])

# ---------- custom matchup --------------------------------------------------------------------
with tab_custom:
    c1, c2, c3 = st.columns([1, 1, 1], gap="large")
    with c1:
        home = team_picker("home", default_team("Real Madrid"), "Home team")
    with c2:
        away = team_picker("away", default_team("Bayern Munich"), "Away team")
    with c3:
        venue = st.radio("Venue", ["Home team's stadium", "Neutral venue"])
        comp = st.radio("Competition", ["League", "Champions League"], horizontal=True)
    if home == away:
        st.warning("Pick two different teams.")
    else:
        st.divider()
        render_match(custom_match(home, away, venue == "Neutral venue", comp == "Champions League"))

# ---------- upcoming --------------------------------------------------------------------------
with tab_upcoming:
    c1, c2 = st.columns([2, 1])
    comps = sorted(P.upcoming.competition.unique())
    chosen = c1.multiselect("Competitions", comps, default=comps)
    days = c2.slider("Days ahead", 1, 45, 21)
    up = P.upcoming[P.upcoming.competition.isin(chosen) & (P.upcoming.date <= P.now + pd.Timedelta(days=days))]
    if up.empty:
        st.info("No fixtures in that window.")
    else:
        pick = up[["p_home", "p_draw", "p_away"]].idxmax(axis=1).map(
            {"p_home": "Home win", "p_draw": "Draw", "p_away": "Away win"})
        table = pd.DataFrame({
            "Kick-off (UTC)": up.date.dt.tz_localize(None), "Competition": up.competition,
            "Home": up.home_team, "Away": up.away_team,
            "Home win": up.p_home, "Draw": up.p_draw, "Away win": up.p_away, "Pick": pick})
        pc = lambda t: st.column_config.ProgressColumn(t, format="percent", min_value=0, max_value=1)
        st.dataframe(table, hide_index=True, use_container_width=True, height=min(38 * len(table) + 40, 700),
                     column_config={"Kick-off (UTC)": st.column_config.DatetimeColumn(format="ddd D MMM, HH:mm"),
                                    "Home win": pc("Home win"), "Draw": pc("Draw"), "Away win": pc("Away win")})
        st.caption(f"{len(table)} fixtures")

# ---------- performance -----------------------------------------------------------------------
with tab_perf:
    st.markdown(f"Tuned on **{R['train_matches']:,}** matches ({R['train_period'][0]} → {R['train_period'][1]}) "
                f"with time-series cross-validation, then tested on **{R['test_matches']:,}** later matches "
                f"({R['test_period'][0]} → {R['test_period'][1]}) the model never saw.")
    res = pd.DataFrame(R["results"])[["model", "accuracy", "log_loss", "brier", "macro_f1"]]
    res["model"] = res.model.replace({"random_forest": "Random Forest", "xgboost": "XGBoost",
                                      "baseline: always home win (probs = class freq.)": "Baseline: always home win",
                                      "baseline: Elo-only logistic regression": "Baseline: Elo-only logistic regression"})
    st.dataframe(res, hide_index=True, use_container_width=True, column_config={
        "model": "Model", "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
        "log_loss": st.column_config.NumberColumn("Log loss ↓", format="%.3f"),
        "brier": st.column_config.NumberColumn("Brier score ↓", format="%.3f"),
        "macro_f1": st.column_config.NumberColumn("Macro F1", format="%.3f")})
    st.caption("Log loss and Brier score measure how good the probabilities are (lower is better). "
               "Bookmaker odds typically reach about 53–55% accuracy on top-flight football, so the ceiling is low.")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Most important features")
        fi = pd.Series(R["feature_importance"]).head(12).rename_axis("feature").reset_index(name="importance")
        st.altair_chart(alt.Chart(fi).mark_bar(color=HOME, cornerRadiusEnd=3).encode(
            x=alt.X("importance:Q", title="importance"), y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")]).properties(height=340),
            use_container_width=True)
    with b:
        st.markdown("##### Confusion matrix (test set)")
        labels = ["Away win", "Draw", "Home win"]
        cm = pd.DataFrame([{"actual": labels[i], "predicted": labels[j], "n": v}
                           for i, row in enumerate(R["confusion_matrix"]) for j, v in enumerate(row)])
        heat = alt.Chart(cm).encode(x=alt.X("predicted:N", sort=labels, title="predicted"),
                                    y=alt.Y("actual:N", sort=labels, title="actual"))
        st.altair_chart((heat.mark_rect().encode(color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), legend=None))
                         + heat.mark_text(fontSize=15, fontWeight="bold").encode(
                             text="n:Q", color=alt.condition("datum.n > 400", alt.value("white"), alt.value("black"))))
                        .properties(height=340), use_container_width=True)
    st.caption("Draws are rarely the single most likely outcome, so the model almost never picks them, "
               "even though its draw probabilities are realistic (see calibration).")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("##### Accuracy by competition (test set)")
        bc = pd.DataFrame(R["accuracy_by_competition"])
        st.altair_chart(alt.Chart(bc).mark_bar(color=HOME, cornerRadiusEnd=3).encode(
            x=alt.X("accuracy:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.7])),
            y=alt.Y("competition:N", sort="-x", title=None),
            tooltip=["competition", alt.Tooltip("accuracy:Q", format=".1%"), "matches"]).properties(height=260),
            use_container_width=True)
    with b:
        st.markdown("##### Calibration: predicted vs actual frequency")
        cal = pd.concat([pd.DataFrame(v).assign(outcome=k.replace("_", " ")) for k, v in R["calibration"].items()])
        diag = alt.Chart(pd.DataFrame({"x": [0, 0.8], "y": [0, 0.8]})).mark_line(strokeDash=[4, 4], color="#9ca3af").encode(x="x", y="y")
        lines = alt.Chart(cal).mark_line(point=True).encode(
            x=alt.X("predicted:Q", title="predicted probability", axis=alt.Axis(format="%")),
            y=alt.Y("actual:Q", title="actual frequency", axis=alt.Axis(format="%")),
            color=alt.Color("outcome:N", scale=alt.Scale(domain=["away win", "draw", "home win"], range=[AWAY, DRAW, HOME])),
            tooltip=["outcome", alt.Tooltip("predicted:Q", format=".1%"), alt.Tooltip("actual:Q", format=".1%"), "n"])
        st.altair_chart((diag + lines).properties(height=260), use_container_width=True)
        st.caption("Points near the dashed line mean the probabilities can be taken at face value.")

# ---------- how it works ----------------------------------------------------------------------
with tab_how:
    st.markdown(f"""
#### Pipeline
1. **Data:** `fetch_data.py` downloads every match since 2019-20 from ESPN's public API: score, possession,
   shots, shots on target, corners and fouls, plus upcoming fixtures. It covers the top five European leagues
   and the Champions League.
2. **Features:** `features.py` builds features for each team across all its competitions, using **only matches
   finished before kickoff**, so there is no leakage from the result:
   - rolling averages over the last 5 and 15 matches of goals scored and conceded, shots for and against,
     shots on target for and against, and possession
   - recent form as points per game
   - an **Elo rating** updated after every match. Champions League games put teams from different leagues
     on one scale.
   - home advantage (switched off at neutral venues), days of rest, a Champions League flag and
     home-minus-away differences
3. **Training:** `train.py` tunes a **Random Forest** and an **XGBoost** classifier with `TimeSeriesSplit`
   cross-validation, optimising log loss. The split is by date, never shuffled. The model with the better
   cross-validation score is kept and refit on all the data.
4. **Evaluation:** both models are compared with an "always home win" baseline and an Elo-only logistic
   regression on {R['test_matches']:,} held-out recent matches.
5. **Serving:** this Streamlit app (`app.py`) and a command-line tool (`predict.py`) share `predictor.py`.

#### Findings
- Team strength (Elo) and the shots-on-target difference are by far the strongest signals.
- The tree models beat the home-win baseline by about 8 accuracy points, but only slightly beat Elo alone.
  Most of the predictable signal is team strength.
- Draws are the hardest outcome. They make up about 25% of matches but are almost never the single most
  likely result.

#### Tech stack
Python · pandas · scikit-learn · XGBoost · Streamlit · Altair

[View the code on GitHub]({GITHUB})
""")
