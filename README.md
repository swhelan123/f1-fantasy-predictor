# 🏎️ F1 Fantasy Predictor

An automated ML pipeline that predicts the optimal F1 Fantasy team each race week, powered by FastF1, LightGBM, and GitHub Actions.

## How it works

1. **Data** — Pulls race, qualifying, and pit stop data via FastF1 + scrapes live fantasy prices
2. **Features** — Engineers rolling form, circuit-specific stats, PPM trends, pit stop consistency
3. **Model** — LightGBM regression predicts fantasy points per driver/constructor
4. **Optimiser** — PuLP solves the constrained $100M knapsack to pick the best team
5. **Automation** — GitHub Actions runs the full pipeline every Thursday before the race deadline

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

## GitHub Secrets required

| Secret | Purpose |
|---|---|
| `F1_FANTASY_EMAIL` | Your F1 Fantasy login email |
| `F1_FANTASY_PASSWORD` | Your F1 Fantasy login password |

## Project structure

```
f1-fantasy-predictor/
├── .github/workflows/weekly_prediction.yml
├── src/
│   ├── data/          # FastF1 fetcher + price scraper
│   ├── features/      # Feature engineering
│   ├── models/        # Train + predict
│   ├── optimiser/     # PuLP team selector
│   └── report/        # Report generator
├── data/
│   ├── raw/
│   └── processed/
├── models/            # Saved model artifacts
├── reports/           # Auto-generated weekly team picks
└── requirements.txt
```

## Weekly reports

Each race week a new report is auto-committed to `/reports/` with:
- Predicted points per driver and constructor
- Optimal $100M team selection
- Recommended DRS Boost driver
- Transfer suggestions (within limit)
