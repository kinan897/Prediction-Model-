# ⚽ Football Match Outcome Predictor

A machine-learning website that predicts **win / draw / loss** probabilities for football matches in the
Premier League, La Liga, Bundesliga, Serie A, Ligue 1 and the UEFA Champions League. The model is trained
on **14,000+ matches since 2019**. Its features are each team's recent goals, shots, shots on target,
possession, form, Elo rating and home advantage.

The website is fully static and hosted on Netlify. The trained Random Forest is exported to JSON and
**runs in the visitor's browser**, so any head-to-head matchup can be predicted with no backend server.

![Website screenshot](docs/screenshot.png)

## Features
- **Next match:** pick any team to see its next fixture as a match card with team crests, win/draw/loss
  probabilities, recent form, a "tale of the tape" stat comparison and the last 5 results for both sides.
- **Head to head:** any two teams, at either stadium or a neutral venue, in a league or Champions League game,
  predicted live in the browser.
- **Fixtures:** predictions for every scheduled match in the coming weeks, grouped by day and filterable by
  competition.
- **Model performance:** test-set metrics compared with baselines, feature importance, confusion matrix,
  accuracy by competition and calibration.
- **Custom design:** dark theme, works on phones, and plain HTML/CSS/JS with no frameworks or chart libraries.

## How it works

```
ESPN public API ─► fetch_data.py ─► data/matches.csv ─► features.py ─► train.py ─► models/model.joblib
                                                                                        │
                    site/ (Netlify)  ◄── site/data/*.json ◄── export_site.py ◄──────────┘
                    └ js/model.js runs the exported trees in the browser
```

1. **Data collection** (`fetch_data.py`): every match since 2019-20 with score, possession, shots, shots on
   target, corners and fouls, plus upcoming fixtures. Downloads are cached per match day and fetched in parallel.
2. **Feature engineering** (`features.py`): 58 pre-match features per match. Each one uses **only matches
   finished before kickoff**, so the result can't leak into its own prediction:
   - rolling averages over the last 5 and 15 matches of goals scored and conceded, shots for and against,
     shots on target for and against, possession, and points per game (form)
   - an **Elo rating** updated after every match. Champions League games put teams from different leagues
     on one scale.
   - home advantage (switched off at neutral venues), days of rest, a Champions League flag and
     home-minus-away differences
3. **Model training** (`train.py`): a Random Forest and XGBoost tuned with `GridSearchCV` +
   `TimeSeriesSplit`, optimising log loss. The split is chronological and never shuffled. The model with the
   better cross-validation score is kept and refit on all labelled data.
4. **Export** (`export_site.py`): writes each team's current form, the upcoming fixtures with their
   predictions, the evaluation report and the forest's 400 trees to `site/data/`.
5. **Serving** (`site/`): a static website. `js/model.js` rebuilds the 58 features from two teams' form and
   walks the trees exactly as scikit-learn does, including float32 inputs and missing-value routing.
   `tests/check_parity.mjs` confirms it matches Python on every upcoming fixture (max difference < 1e-6).

## Results
Trained on 2020-21 to 2024-25 (9,515 matches) and tested on 2025-26 plus 2026-27 to date (2,455 matches the
model never saw):

| Model | Accuracy | Log loss ↓ | Brier ↓ |
|---|---|---|---|
| Baseline: always home win | 45.0% | 1.066 | 0.644 |
| Baseline: Elo-only logistic regression | 52.8% | 0.984 | 0.585 |
| **Random Forest** (selected by CV) | 52.6% | 0.982 | 0.584 |
| XGBoost | 52.9% | 0.981 | 0.583 |

**Takeaways**
- About 53% accuracy on a three-way outcome. Bookmakers typically reach about 53–55%, so the ceiling is low.
- Elo difference and the shots-on-target difference are the strongest features.
- The tree models beat the home-win baseline by 8 points but only slightly beat Elo alone. Most of the
  predictable signal is team strength.
- Draws (25% of matches) are almost never the single most likely outcome, a known difficulty in football
  prediction. The draw probabilities themselves are well calibrated.

## Run it locally

```bash
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m http.server 8000 --directory site      # then open http://localhost:8000
```

## Update the website with new results

```bash
python fetch_data.py --seasons 2026 2026   # refresh the current season (other seasons are kept)
python export_site.py                      # rebuild site/data/*.json
node tests/check_parity.mjs                # optional: browser model == Python model
git add -A && git commit -m "Update data" && git push    # Netlify redeploys automatically
```

Retraining (`python train.py`, about 5 minutes) is only needed occasionally. Run `export_site.py` after it.

## Deploy on Netlify
`netlify.toml` already holds the settings: base directory `site`, no build command. Netlify only serves the
pre-built files.

1. In Netlify: **Add new site → Import an existing project → GitHub**, then pick this repository.
2. Leave the build settings as Netlify reads them from `netlify.toml`, and click **Deploy**.
3. Optionally, under **Domain management**, add a subdomain of your own domain (for example
   `predictor.yourdomain.com`).

## Other ways to use it
- **Command line:** `python predict.py "Arsenal"`, `python predict.py "Real Madrid" --opponent "Bayern" --venue away --ucl`,
  `python predict.py --upcoming --days 7`
- **Streamlit dashboard (local):** `streamlit run app.py`

## Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost · JavaScript · HTML/CSS · Netlify

## Project structure
| File | Purpose |
|---|---|
| `fetch_data.py` | Data collection from ESPN's public API |
| `features.py` | Leakage-free feature engineering and Elo ratings |
| `train.py` | Hyperparameter tuning, evaluation and model selection |
| `predictor.py` | Prediction logic shared by the exporter, the CLI and the Streamlit app |
| `export_site.py` | Exports team form, fixtures, report and the Random Forest to `site/data/` |
| `site/` | The static website (`index.html`, `css/style.css`, `js/app.js`, `js/model.js`) |
| `tests/check_parity.mjs` | Checks the in-browser model against Python's predictions |
| `netlify.toml` | Netlify deploy settings |
| `predict.py` | Command-line interface |
| `app.py`, `assets/`, `.streamlit/` | Optional local Streamlit dashboard |
| `reports/metrics.json` | Full evaluation results |

*Built for learning. Not betting advice.*
