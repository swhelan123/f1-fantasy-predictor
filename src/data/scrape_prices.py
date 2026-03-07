"""
scrape_prices.py

Fetches live driver and constructor prices from the official F1 Fantasy
JSON feed. No login or browser automation required.

Primary source:
    https://fantasy.formula1.com/feeds/drivers/1_en.json

Falls back to Playwright-based DOM scraping if the API feed is
unreachable (requires playwright install chromium + env credentials).

Stores results in DuckDB and outputs a prices parquet for the optimiser.

Usage:
    python src/data/scrape_prices.py
    python src/data/scrape_prices.py --source api       # default
    python src/data/scrape_prices.py --source playwright # legacy
"""

import json
import logging
import os
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "f1_fantasy.duckdb"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# Official F1 Fantasy JSON feed — no auth required
F1_FANTASY_FEED_URL = "https://fantasy.formula1.com/feeds/drivers/1_en.json"

# Playwright-based fallback config (only used with --source playwright)
F1_FANTASY_URL = "https://fantasy.formula1.com"
LOGIN_URL = (
    "https://account.formula1.com/#/en/login"
    "?redirect=https%3A%2F%2Ffantasy.formula1.com%2Fen"
    "&lead_source=web_fantasy"
)
TEAM_PICKER_URL = f"{F1_FANTASY_URL}/en/create-team"

EMAIL = os.getenv("F1_FANTASY_EMAIL")
PASSWORD = os.getenv("F1_FANTASY_PASSWORD")

# Request settings
REQUEST_TIMEOUT = 30  # seconds
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 2.0  # seconds, doubled on each retry


# ═══════════════════════════════════════════════════════════════════════════════
#  PRIMARY SOURCE — Official F1 Fantasy JSON Feed
# ═══════════════════════════════════════════════════════════════════════════════


def fetch_from_api(url: str = F1_FANTASY_FEED_URL) -> pd.DataFrame | None:
    """
    Fetch driver and constructor prices from the official F1 Fantasy
    JSON feed at fantasy.formula1.com/feeds/drivers/1_en.json.

    Returns a DataFrame with columns:
        Code, Name, Team, Type, Price, FantasyPoints,
        SelectedPct, TeamId, PlayerId

    Returns None if the feed is unreachable after retries.
    """
    log.info("Fetching prices from official F1 Fantasy API feed...")
    log.info("  URL: %s", url)

    data = _fetch_with_retries(url)
    if data is None:
        return None

    # ── Parse the JSON structure ──────────────────────────────────────────
    try:
        players = data["Data"]["Value"]
    except (KeyError, TypeError) as e:
        log.error("Unexpected API response structure: %s", e)
        log.debug(
            "Response keys: %s",
            list(data.keys()) if isinstance(data, dict) else type(data),
        )
        return None

    if not players:
        log.error("API returned empty player list")
        return None

    feed_time = data.get("Data", {}).get("FeedTime", {}).get("UTCTime", "unknown")
    log.info("Feed timestamp (UTC): %s", feed_time)
    log.info("Parsing %d entries from API...", len(players))

    rows = []
    for p in players:
        position = (p.get("PositionName") or "").upper()
        is_constructor = position == "CONSTRUCTOR"

        tla = p.get("DriverTLA", "")
        full_name = p.get("FUllName") or p.get("DisplayName") or ""
        display_name = p.get("DisplayName") or full_name
        team_name = p.get("TeamName") or ""
        price = p.get("Value", 0.0)
        overall_pts = p.get("OverallPpints", "0")
        selected_pct = p.get("SelectedPercentage", "0")
        player_id = p.get("PlayerId", "")
        team_id = p.get("TeamId", "")
        is_active = p.get("IsActive", "0")

        # Parse numeric fields that may come as strings
        try:
            price = float(price)
        except (ValueError, TypeError):
            price = 0.0

        try:
            overall_pts = float(overall_pts)
        except (ValueError, TypeError):
            overall_pts = 0.0

        try:
            selected_pct = float(selected_pct)
        except (ValueError, TypeError):
            selected_pct = 0.0

        # For constructors, the "Code" should be the full team name
        # (this is what the optimiser/team_selector expects for
        #  constructor matching in enrich_with_prices).
        # For drivers, Code = 3-letter TLA (VER, NOR, LEC, etc.)
        if is_constructor:
            code = full_name  # e.g. "Mercedes", "McLaren", "Red Bull Racing"
            entry_type = "constructor"
            team = ""  # constructors don't have a parent team
        else:
            code = tla.upper() if tla else full_name[:3].upper()
            entry_type = "driver"
            team = team_name

        rows.append(
            {
                "Code": code,
                "Name": full_name,
                "DisplayName": display_name,
                "Team": team,
                "Type": entry_type,
                "Price": price,
                "FantasyPoints": overall_pts,
                "SelectedPct": selected_pct,
                "TeamId": team_id,
                "PlayerId": player_id,
                "IsActive": str(is_active) == "1",
                "TLA": tla.upper() if tla else "",
            }
        )

    df = pd.DataFrame(rows)

    # Filter to active entries with valid prices
    df = df[df["Price"] > 0].copy()

    n_drivers = len(df[df["Type"] == "driver"])
    n_ctors = len(df[df["Type"] == "constructor"])
    log.info(
        "Parsed %d drivers and %d constructors from API feed",
        n_drivers,
        n_ctors,
    )

    if n_drivers < 10:
        log.warning("Only %d drivers found — expected ~20", n_drivers)
    if n_ctors < 5:
        log.warning("Only %d constructors found — expected ~11", n_ctors)

    return pd.DataFrame(df)


def _fetch_with_retries(url: str) -> dict | None:
    """Fetch JSON from URL with exponential backoff retries."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    backoff = REQUEST_BACKOFF
    resp = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            log.info("  Attempt %d/%d...", attempt, REQUEST_RETRIES)
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()

            data = resp.json()

            # Validate response shape
            meta = data.get("Meta", {})
            if not meta.get("Success", False):
                log.warning(
                    "API returned non-success: %s",
                    meta.get("Message", "unknown"),
                )
                # Still try to use the data if it has content
                if "Data" not in data:
                    raise ValueError("No Data key in response")

            log.info("  API responded OK (HTTP %d)", resp.status_code)
            return data

        except requests.exceptions.Timeout:
            log.warning("  Request timed out after %ds", REQUEST_TIMEOUT)
        except requests.exceptions.ConnectionError as e:
            log.warning("  Connection error: %s", e)
        except requests.exceptions.HTTPError as e:
            log.warning("  HTTP error: %s", e)
            if resp is not None and resp.status_code == 404:
                log.error("  Feed URL not found — the endpoint may have changed")
                return None  # Don't retry on 404
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("  Invalid JSON response: %s", e)

        if attempt < REQUEST_RETRIES:
            log.info("  Retrying in %.0fs...", backoff)
            time.sleep(backoff)
            backoff *= 2

    log.error("All %d attempts failed for %s", REQUEST_RETRIES, url)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  FALLBACK — Playwright-based scraping (legacy)
# ═══════════════════════════════════════════════════════════════════════════════


def fetch_from_playwright(headless: bool = True) -> pd.DataFrame | None:
    """
    Legacy Playwright-based scraper. Requires:
      - playwright install chromium
      - F1_FANTASY_EMAIL and F1_FANTASY_PASSWORD in env/.env

    This is kept as a fallback in case the JSON feed is discontinued.
    """
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.error(
            "Playwright not installed. Install with: pip install playwright && playwright install chromium"
        )
        return None

    if not EMAIL or not PASSWORD:
        log.error(
            "F1_FANTASY_EMAIL and F1_FANTASY_PASSWORD not set for Playwright fallback."
        )
        return None

    log.info("Using Playwright fallback scraper...")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        try:
            # ── Login ─────────────────────────────────────────────────────
            _playwright_login(page, EMAIL, PASSWORD)
            _playwright_navigate_to_team_picker(page)

            # ── Intercept API calls made by the SPA ───────────────────────
            prices = []

            def handle_response(response):
                url = response.url
                if "players" in url and "api" in url:
                    try:
                        data = response.json()
                        if isinstance(data, list):
                            prices.extend(data)
                        elif isinstance(data, dict) and "players" in data:
                            prices.extend(data["players"])
                        log.info(
                            "Intercepted %d players from API: %s", len(prices), url
                        )
                    except Exception:
                        pass

            page.on("response", handle_response)
            page.reload(wait_until="networkidle")
            time.sleep(4)

            if prices:
                log.info("Parsing %d intercepted API records...", len(prices))
                df = _parse_playwright_api(prices)
            else:
                log.warning("API interception returned nothing — trying DOM scraping")
                dom_data = _playwright_scrape_dom(page)
                if not dom_data:
                    log.error("DOM scraping also returned nothing.")
                    return None
                df = _parse_playwright_dom(dom_data)

            df = pd.DataFrame(df[df["Price"] > 0])
            log.info("Playwright scraped %d valid prices", len(df))

            if len(df) < 10:
                log.warning("Only %d prices found", len(df))
                page.screenshot(path=str(ROOT / "data" / "scraper_debug.png"))

            return df

        except Exception as e:
            log.error("Playwright scraper failed: %s", e)
            try:
                page.screenshot(path=str(ROOT / "data" / "scraper_debug.png"))
            except Exception:
                pass
            raise
        finally:
            browser.close()


def _playwright_login(page, email: str, password: str):
    """Log into F1 Fantasy via account.formula1.com SPA."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    log.info("Navigating to F1 login page...")
    page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30000)
    time.sleep(4)

    # Dismiss consent banner
    consent_dismissed = False
    frames_to_check = [page.main_frame] + [
        f for f in page.frames if f != page.main_frame
    ]
    for frame in frames_to_check:
        if (
            "consent" in frame.url
            or "sourcepoint" in frame.url
            or frame == page.main_frame
        ):
            for selector in [
                "button.sp_choice_type_11",
                "button[title='Accept all']",
                "button[aria-label='Accept all']",
                "button:has-text('Accept all')",
            ]:
                try:
                    frame.wait_for_selector(selector, state="visible", timeout=3000)
                    frame.click(selector, timeout=3000)
                    log.info("Dismissed consent: %s", selector)
                    consent_dismissed = True
                    time.sleep(3)
                    break
                except PlaywrightTimeout:
                    continue
            if consent_dismissed:
                break

    # Fill login form
    email_selector = "input.txtLogin, input[name='Login']"
    password_selector = "input.txtPassword, input[name='Password']"
    submit_selector = (
        "button:has-text('SIGN IN'), button:has-text('Sign In'), button[type='submit']"
    )

    log.info("Waiting for login form...")
    try:
        page.wait_for_selector(email_selector, state="visible", timeout=15000)
    except PlaywrightTimeout:
        for frame in page.frames:
            try:
                frame.wait_for_selector(email_selector, timeout=3000)
                break
            except PlaywrightTimeout:
                continue
        else:
            raise RuntimeError("Login form not found")

    page.fill(email_selector, email)
    time.sleep(0.5)
    page.fill(password_selector, password)
    time.sleep(0.5)
    page.click(submit_selector)
    log.info("Submitted login form")

    try:
        page.wait_for_url("**/fantasy.formula1.com/**", timeout=20000)
        log.info("Logged in — redirected to %s", page.url)
    except PlaywrightTimeout:
        log.warning("No redirect — current URL: %s", page.url)


def _playwright_navigate_to_team_picker(page):
    """Navigate to team picker page."""
    log.info("Navigating to team picker...")
    page.goto(TEAM_PICKER_URL, wait_until="networkidle")
    time.sleep(3)


def _parse_playwright_api(raw: list[dict]) -> pd.DataFrame:
    """Parse intercepted Playwright API response."""
    rows = []
    for p in raw:
        name = (
            p.get("displayName")
            or p.get("lastName")
            or p.get("name")
            or p.get("shortName", "")
        )
        price = p.get("price") or p.get("cost") or p.get("value", 0)
        ptype = p.get("type") or p.get("playerType") or p.get("positionType", "")
        team = p.get("teamName") or p.get("team") or p.get("constructorName", "")
        code = p.get("driverCode") or p.get("code") or p.get("abbreviation", "")

        if isinstance(price, (int, float)) and price > 500:
            price = price / 10.0

        rows.append(
            {
                "Code": code.upper() if code else name[:3].upper(),
                "Name": name,
                "Team": team,
                "Type": ptype,
                "Price": float(price),
            }
        )
    return pd.DataFrame(rows)


def _playwright_scrape_dom(page) -> list[dict]:
    """Scrape prices from DOM as last resort."""
    log.info("Falling back to DOM scraping...")
    time.sleep(3)
    return page.evaluate("""
        () => {
            const results = [];
            const selectors = [
                '[data-player-id]', '.player-card', '.player-list-item',
                '[class*="PlayerCard"]', '[class*="player-card"]',
            ];
            for (const sel of selectors) {
                const cards = document.querySelectorAll(sel);
                if (cards.length > 0) {
                    cards.forEach(card => {
                        const name  = card.querySelector('[class*="name"], .name, h3, h4')?.innerText?.trim();
                        const price = card.querySelector('[class*="price"], .price, [class*="Price"]')?.innerText?.trim();
                        const team  = card.querySelector('[class*="team"], .team, [class*="Team"]')?.innerText?.trim();
                        const type  = card.getAttribute('data-player-type') ||
                                      (card.closest('[class*="constructor"]') ? 'constructor' : 'driver');
                        if (name && price) results.push({ name, price, team, type });
                    });
                    break;
                }
            }
            return results;
        }
    """)


def _parse_playwright_dom(raw: list[dict]) -> pd.DataFrame:
    """Parse DOM-scraped price strings."""
    rows = []
    for p in raw:
        price_str = p.get("price", "0")
        price_clean = (
            price_str.replace("$", "").replace("M", "").replace("m", "").strip()
        )
        try:
            price = float(price_clean)
        except ValueError:
            price = 0.0
        rows.append(
            {
                "Code": p.get("name", "")[:3].upper(),
                "Name": p.get("name", ""),
                "Team": p.get("team", ""),
                "Type": p.get("type", "driver"),
                "Price": price,
            }
        )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  Persistence
# ═══════════════════════════════════════════════════════════════════════════════


def save_prices(df: pd.DataFrame, source: str = "api"):
    """Save prices to DuckDB and parquet."""
    df = df.copy()
    df["ScrapedAt"] = pd.Timestamp.now()
    df["Source"] = source

    # Ensure the core columns exist (Playwright fallback may lack some)
    for col in (
        "FantasyPoints",
        "SelectedPct",
        "TeamId",
        "PlayerId",
        "IsActive",
        "TLA",
        "DisplayName",
    ):
        if col not in df.columns:
            df[col] = None

    con = duckdb.connect(str(DB_PATH))
    con.execute("DROP TABLE IF EXISTS player_prices")
    con.execute("CREATE TABLE player_prices AS SELECT * FROM df")
    con.close()

    out = PROCESSED / "player_prices.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved %d prices to %s (source: %s)", len(df), out, source)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════


def run(
    headless: bool = True,
    source: str = "api",
) -> pd.DataFrame | None:
    """
    Fetch F1 Fantasy prices.

    Args:
        headless: If True, run Playwright in headless mode (only relevant for
                  source="playwright").
        source:   "api" (default) — use the official JSON feed.
                  "playwright"    — use the legacy Playwright scraper.
                  "auto"          — try API first, fall back to Playwright.

    Returns:
        DataFrame of prices, or None on failure.
    """
    log.info("━━━ F1 Fantasy Price Fetcher ━━━")
    log.info("Source: %s", source)

    df = None

    # ── Try API feed ──────────────────────────────────────────────────────
    if source in ("api", "auto"):
        df = fetch_from_api()
        if df is not None and len(df) >= 10:
            log.info("API feed succeeded — %d entries", len(df))
            save_prices(df, source="api")
            _log_price_summary(df)
            log.info("━━━ Done ━━━")
            return df
        elif source == "api":
            log.error("API feed failed or returned insufficient data.")
            return None
        else:
            log.warning("API feed unavailable — falling back to Playwright...")

    # ── Playwright fallback ───────────────────────────────────────────────
    if source in ("playwright", "auto"):
        df = fetch_from_playwright(headless=headless)
        if df is not None and len(df) >= 10:
            log.info("Playwright scraper succeeded — %d entries", len(df))
            save_prices(df, source="playwright")
            _log_price_summary(df)
            log.info("━━━ Done ━━━")
            return df
        else:
            log.error("Playwright scraper failed or returned insufficient data.")
            return None

    log.error("No valid source specified: %s", source)
    return None


def _log_price_summary(df: pd.DataFrame):
    """Print a nice summary table of fetched prices."""
    drivers = pd.DataFrame(df[df["Type"] == "driver"]).sort_values(
        by="Price", ascending=False
    )
    ctors = pd.DataFrame(df[df["Type"] == "constructor"]).sort_values(
        by="Price", ascending=False
    )

    cols = ["Code", "Name", "Team", "Price"]
    # Only show columns that exist
    dcols = [c for c in cols if c in drivers.columns]
    ccols = [c for c in ["Code", "Price"] if c in ctors.columns]

    log.info(
        "\n── Drivers (%d) ──\n%s", len(drivers), drivers[dcols].to_string(index=False)
    )
    log.info(
        "\n── Constructors (%d) ──\n%s", len(ctors), ctors[ccols].to_string(index=False)
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch F1 Fantasy driver & constructor prices"
    )
    parser.add_argument(
        "--source",
        choices=["api", "playwright", "auto"],
        default="api",
        help="Price data source (default: api)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window for Playwright (debugging)",
    )
    args = parser.parse_args()

    result = run(
        headless=not args.no_headless,
        source=args.source,
    )

    if result is None:
        log.error("Price fetch failed.")
        exit(1)
