"""
generate.py

Generates a Markdown report summarising the weekly F1 Fantasy prediction:
  - Next race metadata & weather forecast
  - Predicted points per driver (ranked)
  - Predicted points per constructor (ranked)
  - Optimal team selection (drivers, constructors, turbo pick)
  - Budget breakdown & PPM analysis
  - DRS boost / turbo recommendation rationale

The report is saved to /reports/ with a date-stamped filename and also
written to /reports/latest.md for easy access.

Usage:
    python src/report/generate.py
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_predictions(
    season: int | None = None,
    round_number: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load driver and constructor predictions.

    If season and round are given, load the round-specific parquets.
    Otherwise fall back to the ``_latest`` files.
    """
    if season is not None and round_number is not None:
        driver_path = (
            PROCESSED / f"predictions_drivers_S{season}_R{round_number:02d}.parquet"
        )
        ctor_path = (
            PROCESSED
            / f"predictions_constructors_S{season}_R{round_number:02d}.parquet"
        )
    else:
        driver_path = PROCESSED / "predictions_drivers_latest.parquet"
        ctor_path = PROCESSED / "predictions_constructors_latest.parquet"

    if not driver_path.exists() or not ctor_path.exists():
        raise FileNotFoundError(
            "Prediction parquets not found. Run predict.py first.\n"
            f"  Expected: {driver_path}\n"
            f"  Expected: {ctor_path}"
        )

    drivers = pd.read_parquet(driver_path)
    ctors = pd.read_parquet(ctor_path)
    return drivers, ctors


def _load_prices() -> pd.DataFrame | None:
    """Load scraped prices if available."""
    prices_path = PROCESSED / "player_prices.parquet"
    if prices_path.exists():
        return pd.read_parquet(prices_path)
    return None


def _run_optimiser(
    season: int | None = None,
    round_number: int | None = None,
) -> dict:
    """Run the team selector and return the result dict."""
    import sys

    sys.path.insert(0, str(ROOT))
    from src.optimiser.team_selector import run as optimise_run

    return optimise_run(season=season, round_number=round_number)


def _get_race_metadata(drivers: pd.DataFrame) -> dict:
    """Extract race metadata from prediction dataframe."""
    meta = {}
    if "EventName" in drivers.columns and len(drivers) > 0:
        meta["event_name"] = drivers["EventName"].iloc[0]
    else:
        meta["event_name"] = "Unknown"

    if "Season" in drivers.columns:
        meta["season"] = int(drivers["Season"].iloc[0])
    else:
        meta["season"] = datetime.now().year

    if "Round" in drivers.columns:
        meta["round"] = int(drivers["Round"].iloc[0])
    else:
        meta["round"] = 0

    if "Location" in drivers.columns:
        meta["location"] = drivers["Location"].iloc[0]
    else:
        meta["location"] = "Unknown"

    # Weather info
    meta["rainfall"] = None
    if "Rainfall" in drivers.columns:
        meta["rainfall"] = bool(drivers["Rainfall"].iloc[0])

    meta["air_temp"] = None
    if "AvgAirTemp" in drivers.columns:
        meta["air_temp"] = round(float(drivers["AvgAirTemp"].iloc[0]), 1)

    meta["humidity"] = None
    if "AvgHumidity" in drivers.columns:
        meta["humidity"] = round(float(drivers["AvgHumidity"].iloc[0]), 1)

    meta["wind_speed"] = None
    if "AvgWindSpeed" in drivers.columns:
        meta["wind_speed"] = round(float(drivers["AvgWindSpeed"].iloc[0]), 1)

    return meta


def _weather_section(meta: dict) -> str:
    """Build weather forecast Markdown section."""
    lines = []
    lines.append("## 🌤️ Weather Forecast\n")

    if meta.get("air_temp") is not None:
        conditions = []
        conditions.append(f"🌡️ **Temperature:** {meta['air_temp']}°C")
        if meta.get("humidity") is not None:
            conditions.append(f"💧 **Humidity:** {meta['humidity']}%")
        if meta.get("wind_speed") is not None:
            conditions.append(f"💨 **Wind:** {meta['wind_speed']} km/h")
        if meta.get("rainfall") is not None:
            rain_icon = (
                "🌧️ **Rain expected**"
                if meta["rainfall"]
                else "☀️ **Dry conditions expected**"
            )
            conditions.append(rain_icon)

        lines.append(" | ".join(conditions))
        lines.append("")

        if meta.get("rainfall"):
            lines.append(
                "> ⚠️ **Wet race alert** — wet-weather specialists may overperform their dry-condition averages.\n"
            )
    else:
        lines.append("_Weather forecast unavailable — using historical averages._\n")

    return "\n".join(lines)


def _driver_predictions_section(drivers: pd.DataFrame) -> str:
    """Build driver predictions table."""
    lines = []
    lines.append("## 🏁 Driver Predictions\n")
    lines.append("| Rank | Driver | Constructor | Predicted Pts |")
    lines.append("|-----:|--------|-------------|-------------:|")

    display = (
        drivers.sort_values("PredictedPts", ascending=False)
        .head(22)
        .reset_index(drop=True)
    )
    for i, row in display.iterrows():
        rank = i + 1
        driver = row.get("Driver", "?")
        ctor = row.get("Constructor", "?")
        pts = round(float(row["PredictedPts"]), 1)

        # Highlight top 5
        if rank <= 3:
            medal = ["🥇", "🥈", "🥉"][rank - 1]
            lines.append(f"| {medal} {rank} | **{driver}** | {ctor} | **{pts}** |")
        elif rank <= 5:
            lines.append(f"| {rank} | **{driver}** | {ctor} | {pts} |")
        else:
            lines.append(f"| {rank} | {driver} | {ctor} | {pts} |")

    lines.append("")
    return "\n".join(lines)


def _constructor_predictions_section(ctors: pd.DataFrame) -> str:
    """Build constructor predictions table."""
    lines = []
    lines.append("## 🏗️ Constructor Predictions\n")
    lines.append("| Rank | Constructor | Predicted Pts |")
    lines.append("|-----:|-------------|-------------:|")

    display = ctors.sort_values("PredictedPts", ascending=False).reset_index(drop=True)
    for i, row in display.iterrows():
        rank = i + 1
        ctor = row.get("Constructor", "?")
        pts = round(float(row["PredictedPts"]), 1)

        if rank <= 3:
            medal = ["🥇", "🥈", "🥉"][rank - 1]
            lines.append(f"| {medal} {rank} | **{ctor}** | **{pts}** |")
        else:
            lines.append(f"| {rank} | {ctor} | {pts} |")

    lines.append("")
    return "\n".join(lines)


def _optimal_team_section(result: dict) -> str:
    """Build the optimal team selection section."""
    lines = []
    lines.append("## ⚡ Optimal Fantasy Team\n")

    # Drivers table
    lines.append("### Drivers\n")
    lines.append("| Driver | Constructor | Price | Predicted Pts | PPM | Turbo |")
    lines.append("|--------|-------------|------:|-------------:|----:|:-----:|")

    d_detail = result.get("driver_details")
    if d_detail is not None and len(d_detail) > 0:
        d_detail = d_detail.sort_values("PredictedPts", ascending=False)
        for _, row in d_detail.iterrows():
            driver = row["Driver"]
            ctor = row.get("Constructor", "?")
            price = float(row["Price"])
            pts = round(float(row["PredictedPts"]), 1)
            ppm = round(float(row.get("PPM", 0)), 2)
            is_turbo = "⚡ YES" if driver == result.get("turbo") else ""

            lines.append(
                f"| **{driver}** | {ctor} | ${price:.1f}M | {pts} | {ppm} | {is_turbo} |"
            )
    else:
        for drv in result.get("drivers", []):
            turbo_mark = "⚡ YES" if drv == result.get("turbo") else ""
            lines.append(f"| **{drv}** | — | — | — | — | {turbo_mark} |")

    lines.append("")

    # Constructors table
    lines.append("### Constructors\n")
    lines.append("| Constructor | Price | Predicted Pts | PPM |")
    lines.append("|-------------|------:|-------------:|----:|")

    c_detail = result.get("ctor_details")
    if c_detail is not None and len(c_detail) > 0:
        for _, row in c_detail.iterrows():
            ctor = row["Constructor"]
            price = float(row["Price"])
            pts = round(float(row["PredictedPts"]), 1)
            ppm = round(float(row.get("PPM", 0)), 2)
            lines.append(f"| **{ctor}** | ${price:.1f}M | {pts} | {ppm} |")
    else:
        for ctor in result.get("constructors", []):
            lines.append(f"| **{ctor}** | — | — | — |")

    lines.append("")

    # Summary box
    lines.append("### 💰 Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(
        f"| **Turbo Driver** | ⚡ {result.get('turbo', '?')} (points doubled) |"
    )
    lines.append(f"| **Total Cost** | ${result.get('total_cost', 0):.1f}M |")
    lines.append(
        f"| **Budget Remaining** | ${result.get('budget_remaining', 0):.1f}M |"
    )
    lines.append(
        f"| **Predicted Score** | {result.get('predicted_score', 0):.1f} pts |"
    )
    lines.append("")

    return "\n".join(lines)


def _value_picks_section(drivers: pd.DataFrame, result: dict) -> str:
    """Highlight value picks (high PPM drivers not in the optimal team)."""
    lines = []
    lines.append("## 💎 Value Picks (not in optimal team)\n")

    selected = list(result.get("drivers", []))

    if "PPM" in drivers.columns and "Price" in drivers.columns:
        bench = drivers[~drivers["Driver"].isin(selected)].copy()
        bench = bench.sort_values(by="PPM", ascending=False).head(5)

        if len(bench) > 0:
            lines.append("| Driver | Constructor | Price | Predicted Pts | PPM |")
            lines.append("|--------|-------------|------:|-------------:|----:|")
            for _, row in bench.iterrows():
                driver = row["Driver"]
                ctor = row.get("Constructor", "?")
                price = float(row["Price"])
                pts = round(float(row["PredictedPts"]), 1)
                ppm = round(float(row["PPM"]), 2)
                lines.append(f"| {driver} | {ctor} | ${price:.1f}M | {pts} | {ppm} |")
            lines.append("")
            lines.append(
                "_PPM = Predicted Points per $M — useful for budget-constrained swaps._\n"
            )
        else:
            lines.append("_No bench drivers to display._\n")
    else:
        lines.append(
            "_Price data not available — run scrape_prices.py to enable PPM analysis._\n"
        )

    return "\n".join(lines)


def _turbo_rationale_section(drivers: pd.DataFrame, turbo: str | None) -> str:
    """Explain why the turbo pick was chosen."""
    lines = []
    lines.append("## 🎯 Turbo Driver Rationale\n")

    if turbo is None:
        lines.append("_No turbo driver selected._\n")
        return "\n".join(lines)

    turbo_row = drivers[drivers["Driver"] == turbo]
    if len(turbo_row) == 0:
        lines.append(f"**{turbo}** selected as turbo driver.\n")
        return "\n".join(lines)

    turbo_row = turbo_row.iloc[0]
    pts = round(float(turbo_row["PredictedPts"]), 1)

    lines.append(f"**{turbo}** was selected as the turbo driver (points doubled).\n")
    lines.append(f"- **Predicted base points:** {pts}")
    lines.append(f"- **With turbo bonus:** {pts * 2} (effectively)")

    # Rank among all drivers
    rank = int(turbo_row.get("PredictedRank", 0))
    if rank > 0:
        lines.append(f"- **Prediction rank:** #{rank} overall")

    # Rolling form indicators
    if "RollingAvgFantasyPts" in turbo_row.index and pd.notna(
        turbo_row.get("RollingAvgFantasyPts")
    ):
        rolling = round(float(turbo_row["RollingAvgFantasyPts"]), 1)
        lines.append(f"- **Rolling avg fantasy pts:** {rolling}")

    if "CircuitAvgFantasyPts" in turbo_row.index and pd.notna(
        turbo_row.get("CircuitAvgFantasyPts")
    ):
        circuit = round(float(turbo_row["CircuitAvgFantasyPts"]), 1)
        lines.append(f"- **Circuit history avg pts:** {circuit}")

    lines.append("")
    lines.append(
        "> 💡 The turbo driver is chosen by the optimiser to maximise total team score. "
        "The highest-predicted driver in the team gets turbo since doubling a larger "
        "number yields the biggest gain.\n"
    )

    return "\n".join(lines)


def _footer() -> str:
    """Build report footer."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append("---\n")
    lines.append(
        f"_Generated automatically by [F1 Fantasy Predictor](https://github.com) on {now}_\n"
    )
    lines.append(
        "_Model: LightGBM · Optimiser: PuLP ILP · Data: FastF1 + Open-Meteo_\n"
    )
    return "\n".join(lines)


# ── Main Report Builder ──────────────────────────────────────────────────────


def build_report(
    drivers: pd.DataFrame,
    ctors: pd.DataFrame,
    result: dict,
) -> str:
    """
    Assemble the full Markdown report from predictions and optimiser output.

    Args:
        drivers:  Driver predictions DataFrame (with PredictedPts, optionally Price/PPM)
        ctors:    Constructor predictions DataFrame
        result:   Dict returned by team_selector.optimise_team()

    Returns:
        Markdown string
    """
    meta = _get_race_metadata(drivers)

    sections = []

    # Title
    rain_flag = " 🌧️" if meta.get("rainfall") else ""
    title = (
        f"# 🏎️ F1 Fantasy Prediction — {meta['event_name']}{rain_flag}\n\n"
        f"**Season {meta['season']} · Round {meta['round']} · {meta['location']}**\n"
    )
    sections.append(title)

    # Weather
    sections.append(_weather_section(meta))

    # Optimal team (the main event — put it high up)
    sections.append(_optimal_team_section(result))

    # Turbo rationale
    sections.append(_turbo_rationale_section(drivers, result.get("turbo")))

    # Full predictions
    sections.append(_driver_predictions_section(drivers))
    sections.append(_constructor_predictions_section(ctors))

    # Value picks
    sections.append(_value_picks_section(drivers, result))

    # Footer
    sections.append(_footer())

    return "\n".join(sections)


def save_report(report_md: str, meta: dict) -> tuple[Path, Path]:
    """
    Save the report to a dated file and a latest symlink.

    Returns:
        (dated_path, latest_path)
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    season = meta.get("season", datetime.now().year)
    round_num = meta.get("round", 0)

    dated_name = f"{today}_S{season}_R{round_num:02d}_{meta.get('event_name', 'race').replace(' ', '_')}.md"
    dated_path = REPORTS_DIR / dated_name
    latest_path = REPORTS_DIR / "latest.md"

    dated_path.write_text(report_md, encoding="utf-8")
    latest_path.write_text(report_md, encoding="utf-8")

    log.info("Report saved → %s", dated_path.name)
    log.info("Report saved → latest.md")

    return dated_path, latest_path


# ── Run ───────────────────────────────────────────────────────────────────────


def run(
    season: int | None = None,
    round_number: int | None = None,
) -> str:
    """
    Full report generation pipeline:
      1. Load predictions (for a specific season/round or latest)
      2. Run optimiser
      3. Build Markdown report
      4. Save to /reports/

    Returns the Markdown report string.
    """
    log.info("━━━ F1 Fantasy Report Generator ━━━")

    # Load predictions
    drivers, ctors = _load_predictions(season=season, round_number=round_number)
    log.info(
        "Loaded %d driver predictions, %d constructor predictions",
        len(drivers),
        len(ctors),
    )

    # Run optimiser to get team selection (same season/round as predictions).
    # The optimiser already loads, enriches with prices, and optimises in one
    # pass — it returns the enriched dataframes on the result dict so we don't
    # need to duplicate that work here.
    result = _run_optimiser(season=season, round_number=round_number)

    # Use the fully-enriched driver/ctor lists the optimiser already built
    if "all_drivers" in result and "all_ctors" in result:
        drivers = result["all_drivers"]
        ctors = result["all_ctors"]

    meta = _get_race_metadata(drivers)

    # Build report
    report_md = build_report(drivers, ctors, result)

    # Save
    dated_path, latest_path = save_report(report_md, meta)

    # Print report to stdout for CI visibility
    print("\n" + report_md)

    log.info("━━━ Done ━━━")
    return report_md


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate F1 Fantasy report")
    parser.add_argument(
        "--season", type=int, default=None, help="F1 season year (e.g. 2026)"
    )
    parser.add_argument("--round", type=int, default=None, help="Round number (e.g. 2)")
    args = parser.parse_args()

    run(season=args.season, round_number=args.round)
