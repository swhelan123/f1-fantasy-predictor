# 🏎️ F1 Fantasy Predictor

An automated ML pipeline that predicts the optimal F1 Fantasy team each race week, powered by FastF1, LightGBM, and GitHub Actions.

## How it works

1. **Data** — Pulls race, qualifying, and pit stop data via FastF1 + scrapes live fantasy prices
2. **Features** — Engineers rolling form, circuit-specific stats, PPM trends, pit stop consistency
3. **Model** — LightGBM regression predicts fantasy points per driver/constructor
4. **Optimiser** — PuLP solves the constrained $100M knapsack to pick the best team
5. **Report** — Generates a Markdown report with the optimal team, predictions, and rationale
6. **Automation** — GitHub Actions runs the full pipeline every Thursday and publishes the report

## Viewing your weekly prediction

There are **three ways** to see your predicted team each week:

### 1. GitHub Actions Job Summary (easiest)

Go to **Actions → Weekly F1 Fantasy Prediction → latest run** and scroll down to the **Job Summary**. The full report renders directly in the browser — no downloads needed.

### 2. Reports directory

Every run commits a dated Markdown report to [`/reports/`](reports/). The most recent is always available at [`reports/latest.md`](reports/latest.md).

### 3. Downloadable artifact

Each workflow run uploads a **f1-fantasy-report** artifact containing the Markdown report and raw prediction parquets. Download it from the Actions run summary page (retained for 90 days).

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/f1-fantasy-predictor
cd f1-fantasy-predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env  # Add your F1 Fantasy credentials
```

## Running locally

### Full pipeline (recommended)

```bash
# Run the complete pipeline end-to-end
python src/pipeline.py

# Specify season and number of recent races to fetch
python src/pipeline.py --season 2026 --races 5

# Skip the price scraper (use hardcoded prices)
python src/pipeline.py --skip-scraper

# Skip model training (reuse existing model)
python src/pipeline.py --skip-train

# Combine flags
python src/pipeline.py --skip-scraper --skip-train
```

### Individual steps

```bash
# 1. Fetch race data
python src/data/fetch_fastf1.py --season 2026 --races 3

# 2. Scrape fantasy prices (requires F1 Fantasy credentials in .env)
python src/data/scrape_prices.py

# 3. Engineer features
python src/features/engineer.py

# 4. Train model
python src/models/train.py            # default params
python src/models/train.py --tune     # Optuna hyperparameter optimisation

# 5. Generate predictions
python src/models/predict.py
python src/models/predict.py --wet    # force wet race prediction
python src/models/predict.py --dry    # force dry race prediction

# 6. Optimise team
python src/optimiser/team_selector.py
python src/optimiser/team_selector.py --budget 100 --turbo VER
python src/optimiser/team_selector.py --current-team VER,NOR,LEC,RUS,PIA,McLaren,Ferrari

# 7. Generate report
python src/report/generate.py
```

## GitHub Actions — automated weekly predictions

The pipeline runs automatically **twice per race week** via GitHub Actions — Thursday (pre-deadline) and Sunday (post-qualifying, pre-race). You can also trigger it manually.

### Manual trigger

1. Go to **Actions → Weekly F1 Fantasy Prediction**
2. Click **Run workflow**
3. Optionally adjust the season, or toggle skip flags
4. Click **Run workflow** to start

### Required GitHub Secrets

| Secret                | Purpose                                             |
| --------------------- | --------------------------------------------------- |
| `F1_FANTASY_EMAIL`    | Your F1 Fantasy login email (for price scraping)    |
| `F1_FANTASY_PASSWORD` | Your F1 Fantasy login password (for price scraping) |

> **Note:** If these secrets are not set, the pipeline will automatically skip the price scraper and fall back to hardcoded launch prices. The rest of the pipeline still works.

### What the workflow does

1. **Caches** pip dependencies, FastF1 data, the DuckDB database, and the trained model across runs
2. **Runs** the full pipeline (`src/pipeline.py`) — fetch data → scrape prices → engineer features → train model → predict → optimise → generate report
3. **Publishes** the report as a GitHub Actions Job Summary (viewable in the browser)
4. **Uploads** the report and prediction parquets as a downloadable artifact (90-day retention)
5. **Commits** the report back to the `reports/` directory
6. **Uploads** debug screenshots as an artifact if the scraper fails

### Adjusting the schedule

Edit the cron expressions in `.github/workflows/weekly_prediction.yml`:

```yaml
on:
  schedule:
    - cron: "0 8 * * 4" # Thursday 08:00 UTC — pre-deadline
    - cron: "0 6 * * 0" # Sunday 06:00 UTC — post-qualifying
```

Some common alternatives:

- `"0 6 * * 4"` — Thursday 06:00 UTC
- `"0 12 * * 3"` — Wednesday 12:00 UTC
- `"0 8 * * 5"` — Friday 08:00 UTC (after FP1)

## Project structure

```
f1-fantasy-predictor/
├── .github/workflows/
│   └── weekly_prediction.yml  # GitHub Actions pipeline
├── src/
│   ├── pipeline.py            # End-to-end pipeline runner
│   ├── data/
│   │   ├── fetch_fastf1.py    # FastF1 race data fetcher
│   │   ├── fetch_testing.py   # Pre-season testing data
│   │   └── scrape_prices.py   # Playwright price scraper
│   ├── features/
│   │   └── engineer.py        # Feature engineering (2026 scoring)
│   ├── models/
│   │   ├── train.py           # LightGBM training + Optuna HPO
│   │   └── predict.py         # Prediction generation
│   ├── optimiser/
│   │   └── team_selector.py   # PuLP integer linear programme
│   └── report/
│       └── generate.py        # Markdown report generator
├── data/
│   ├── f1_fantasy.duckdb      # Local DuckDB database
│   ├── raw/                   # Raw data files
│   └── processed/             # Feature parquets + predictions
├── models/
│   └── lgbm_predictor.pkl     # Saved model artifact
├── reports/                   # Auto-generated weekly team picks
│   └── latest.md              # Most recent report
├── fastf1_cache/              # FastF1 data cache (gitignored)
├── mlruns/                    # MLflow experiment tracking (gitignored)
└── requirements.txt
```

## Weekly reports

Each race week a new report is auto-committed to `/reports/` with:

- 🌤️ Weather forecast for the race weekend
- ⚡ Optimal $100M team selection with turbo driver pick
- 🏁 Predicted points per driver (ranked)
- 🏗️ Predicted points per constructor (ranked)
- 💎 Value picks (high PPM drivers outside the optimal team)
- 🎯 Turbo driver rationale

## Tech stack

| Component    | Technology                             |
| ------------ | -------------------------------------- |
| Data         | FastF1, Playwright, DuckDB, Pandas     |
| ML           | LightGBM, scikit-learn, Optuna, SHAP   |
| Optimisation | PuLP (CBC solver)                      |
| Weather      | Open-Meteo API (free, no key required) |
| Tracking     | MLflow                                 |
| Reporting    | Markdown                               |
| Automation   | GitHub Actions                         |
