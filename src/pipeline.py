"""
pipeline.py

End-to-end pipeline runner for the F1 Fantasy Predictor.
Orchestrates all steps in order, with error handling and timing.

Designed to be called from GitHub Actions or locally:
    python src/pipeline.py
    python src/pipeline.py --season 2026 --skip-scraper
    python src/pipeline.py --skip-train   # reuse existing model

Steps:
    1. Fetch latest race data (FastF1)
    2. Scrape fantasy prices (Playwright)
    3. Engineer features
    4. Train / update model
    5. Generate predictions
    6. Run team optimiser
    7. Generate Markdown report

Exit codes:
    0 — success
    1 — pipeline failed (check logs)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ── Step Runner ───────────────────────────────────────────────────────────────


def run_step(name: str, func, **kwargs) -> bool:
    """Run a single pipeline step with timing and error handling."""
    log.info("┌─── %s ───", name)
    t0 = time.time()
    try:
        func(**kwargs)
        elapsed = time.time() - t0
        log.info("└─── %s ✓ (%.1fs) ───\n", name, elapsed)
        return True
    except Exception as e:
        elapsed = time.time() - t0
        log.error("└─── %s ✗ FAILED after %.1fs: %s ───\n", name, elapsed, e)
        return False


# ── Individual Step Wrappers ──────────────────────────────────────────────────


def step_fetch_data(season: int, races: int):
    from src.data.fetch_fastf1 import run as fetch_run

    fetch_run(season=season, last_n=races)


def step_scrape_prices():
    from src.data.scrape_prices import run as scrape_run

    result = scrape_run(headless=True)
    if result is None:
        raise RuntimeError(
            "Price scraper returned no data. "
            "Check F1_FANTASY_EMAIL / F1_FANTASY_PASSWORD env vars."
        )


def step_engineer_features():
    from src.features.engineer import run as engineer_run

    engineer_run()


def step_train_model():
    from src.models.train import run as train_run

    train_run(tune=False)


def step_predict(season: int):
    from src.models.predict import run as predict_run

    predict_run(season=season)


def step_optimise():
    from src.optimiser.team_selector import run as optimise_run

    optimise_run()


def step_generate_report() -> str:
    from src.report.generate import run as report_run

    return report_run()


# ── Main Pipeline ─────────────────────────────────────────────────────────────


def run(
    season: int = 2026,
    races: int = 3,
    skip_scraper: bool = False,
    skip_train: bool = False,
    skip_report: bool = False,
) -> bool:
    """
    Run the full F1 Fantasy prediction pipeline.

    Args:
        season:       F1 season year
        races:        Number of recent races to fetch
        skip_scraper: Skip price scraping (use cached/hardcoded prices)
        skip_train:   Skip model training (reuse existing model)
        skip_report:  Skip report generation

    Returns:
        True if pipeline completed successfully, False otherwise
    """
    log.info("=" * 60)
    log.info("  🏎️  F1 Fantasy Predictor — Full Pipeline")
    log.info("  Season: %d | Races: %d", season, races)
    log.info(
        "  Scraper: %s | Train: %s | Report: %s",
        "SKIP" if skip_scraper else "ON",
        "SKIP" if skip_train else "ON",
        "SKIP" if skip_report else "ON",
    )
    log.info("=" * 60 + "\n")

    pipeline_start = time.time()
    failed_steps = []

    # ── Step 1: Fetch race data ───────────────────────────────────────────────
    ok = run_step(
        "Step 1/7 — Fetch Race Data",
        step_fetch_data,
        season=season,
        races=races,
    )
    if not ok:
        failed_steps.append("fetch_data")
        # This is critical — can't continue without data
        log.error("Data fetch failed — aborting pipeline.")
        return False

    # ── Step 2: Scrape prices ─────────────────────────────────────────────────
    if skip_scraper:
        log.info("⏭️  Skipping price scraper (--skip-scraper)\n")
    else:
        ok = run_step("Step 2/7 — Scrape Fantasy Prices", step_scrape_prices)
        if not ok:
            failed_steps.append("scrape_prices")
            log.warning(
                "Price scraping failed — continuing with hardcoded/cached prices.\n"
            )

    # ── Step 3: Feature engineering ───────────────────────────────────────────
    ok = run_step("Step 3/7 — Engineer Features", step_engineer_features)
    if not ok:
        failed_steps.append("engineer_features")
        log.error("Feature engineering failed — aborting pipeline.")
        return False

    # ── Step 4: Train model ───────────────────────────────────────────────────
    model_path = ROOT / "models" / "lgbm_predictor.pkl"
    if skip_train and model_path.exists():
        log.info("⏭️  Skipping training (--skip-train, model exists)\n")
    else:
        if skip_train and not model_path.exists():
            log.warning(
                "Model not found at %s — training anyway despite --skip-train\n",
                model_path,
            )
        ok = run_step("Step 4/7 — Train Model", step_train_model)
        if not ok:
            failed_steps.append("train_model")
            if not model_path.exists():
                log.error("Training failed and no existing model — aborting pipeline.")
                return False
            else:
                log.warning("Training failed — using existing model.\n")

    # ── Step 5: Predictions ───────────────────────────────────────────────────
    ok = run_step("Step 5/7 — Generate Predictions", step_predict, season=season)
    if not ok:
        failed_steps.append("predict")
        log.error("Prediction generation failed — aborting pipeline.")
        return False

    # ── Step 6: Team optimisation ─────────────────────────────────────────────
    ok = run_step("Step 6/7 — Optimise Team", step_optimise)
    if not ok:
        failed_steps.append("optimise")
        log.warning("Optimiser failed — report will have limited data.\n")

    # ── Step 7: Generate report ───────────────────────────────────────────────
    if skip_report:
        log.info("⏭️  Skipping report generation (--skip-report)\n")
    else:
        ok = run_step("Step 7/7 — Generate Report", step_generate_report)
        if not ok:
            failed_steps.append("generate_report")
            log.warning("Report generation failed.\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    log.info("=" * 60)

    if failed_steps:
        log.warning("  ⚠️  Pipeline completed with warnings (%.0fs)", elapsed)
        log.warning("  Failed steps: %s", ", ".join(failed_steps))
    else:
        log.info("  ✅  Pipeline completed successfully (%.0fs)", elapsed)

    log.info("=" * 60)

    # Return True if critical steps (data, features, predictions) all passed
    critical_failures = {"fetch_data", "engineer_features", "predict"}
    return not critical_failures.intersection(failed_steps)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="F1 Fantasy Predictor — end-to-end pipeline"
    )
    parser.add_argument(
        "--season", type=int, default=2026, help="F1 season year (default: 2026)"
    )
    parser.add_argument(
        "--races",
        type=int,
        default=3,
        help="Number of recent races to fetch (default: 3)",
    )
    parser.add_argument(
        "--skip-scraper",
        action="store_true",
        help="Skip price scraping (use cached/hardcoded prices)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training (reuse existing model if available)",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip report generation",
    )
    args = parser.parse_args()

    success = run(
        season=args.season,
        races=args.races,
        skip_scraper=args.skip_scraper,
        skip_train=args.skip_train,
        skip_report=args.skip_report,
    )

    sys.exit(0 if success else 1)
