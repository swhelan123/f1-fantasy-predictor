"""
scrape_prices.py

Scrapes live driver and constructor prices from the F1 Fantasy website
using Playwright. Stores results in DuckDB and outputs a prices parquet
for the optimiser to consume.

Requires:
    playwright install chromium
    F1_FANTASY_EMAIL and F1_FANTASY_PASSWORD in .env or environment

Usage:
    python src/data/scrape_prices.py
"""

import json
import logging
import os
import time
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

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

F1_FANTASY_URL = "https://fantasy.formula1.com"
LOGIN_URL = "https://account.formula1.com/#/en/login?redirect=https%3A%2F%2Ffantasy.formula1.com%2Fen&lead_source=web_fantasy"
TEAM_PICKER_URL = f"{F1_FANTASY_URL}/en/create-team"

EMAIL = os.getenv("F1_FANTASY_EMAIL")
PASSWORD = os.getenv("F1_FANTASY_PASSWORD")

# ── Scraper ───────────────────────────────────────────────────────────────────


def login(page, email: str, password: str):
    """Log into F1 Fantasy via account.formula1.com SPA."""
    log.info("Navigating to F1 login page...")
    page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=30000)
    time.sleep(4)

    # ── Dismiss Sourcepoint consent banner (appears on both domains) ─────────
    consent_dismissed = False

    # Check all frames including main frame
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
                    log.info(
                        "Dismissed consent: %s (frame: %s)", selector, frame.url[:60]
                    )
                    consent_dismissed = True
                    time.sleep(3)
                    break
                except PlaywrightTimeout:
                    continue
            if consent_dismissed:
                break

    if not consent_dismissed:
        log.info("No consent banner found — continuing")

    # ── Wait for the login form (SPA loads async after hash navigation) ──────
    email_selector = "input.txtLogin, input[name='Login']"
    password_selector = "input.txtPassword, input[name='Password']"
    submit_selector = (
        "button:has-text('SIGN IN'), button:has-text('Sign In'), button[type='submit']"
    )

    log.info("Waiting for login form...")
    try:
        page.wait_for_selector(email_selector, state="visible", timeout=15000)
        log.info("Login form found in main frame")
    except PlaywrightTimeout:
        found_frame = None
        for frame in page.frames:
            try:
                frame.wait_for_selector(email_selector, timeout=3000)
                found_frame = frame
                log.info("Login form found in iframe: %s", frame.url[:80])
                break
            except PlaywrightTimeout:
                continue
        if not found_frame:
            page.screenshot(path=str(ROOT / "data" / "scraper_debug.png"))
            raise RuntimeError(
                f"Login form not found. URL: {page.url}\n"
                f"Frames: {[f.url[:60] for f in page.frames]}\n"
                "Check data/scraper_debug.png"
            )

    # ── Fill credentials ──────────────────────────────────────────────────────
    log.info("Filling login form...")
    page.fill(email_selector, email)
    time.sleep(0.5)
    page.fill(password_selector, password)
    time.sleep(0.5)
    page.screenshot(path=str(ROOT / "data" / "scraper_debug_pre_submit.png"))
    page.click(submit_selector)
    log.info("Submitted login form")

    # ── Wait for redirect back to fantasy.formula1.com ────────────────────────
    try:
        page.wait_for_url("**/fantasy.formula1.com/**", timeout=20000)
        log.info("Logged in — redirected to %s", page.url)
    except PlaywrightTimeout:
        log.warning("No redirect — current URL: %s", page.url)
        page.screenshot(path=str(ROOT / "data" / "scraper_debug_post_login.png"))


def navigate_to_team_picker(page):
    """Navigate to the team creation/edit page where player prices are shown."""
    log.info("Navigating to team picker...")
    page.goto(TEAM_PICKER_URL, wait_until="networkidle")
    time.sleep(3)


def intercept_prices_from_api(page) -> list[dict] | None:
    """
    Intercept the F1 Fantasy API response that contains player prices.
    The site loads player data via XHR — we capture it directly.
    This is more reliable than scraping the DOM.
    """
    prices = []

    def handle_response(response):
        url = response.url
        # F1 Fantasy loads players from an endpoint like /api/game/v1/players
        if "players" in url and "api" in url:
            try:
                data = response.json()
                if isinstance(data, list):
                    prices.extend(data)
                elif isinstance(data, dict) and "players" in data:
                    prices.extend(data["players"])
                log.info("Intercepted %d players from API: %s", len(prices), url)
            except Exception:
                pass

    page.on("response", handle_response)

    # Reload to trigger API calls
    page.reload(wait_until="networkidle")
    time.sleep(4)

    return prices if prices else None


def scrape_prices_from_dom(page) -> list[dict]:
    """
    Fallback: scrape prices directly from the DOM if API interception fails.
    Looks for the player list cards rendered on the team picker page.
    """
    log.info("Falling back to DOM scraping...")
    time.sleep(3)

    players = page.evaluate("""
        () => {
            const results = [];

            // Try multiple possible selectors for player cards
            const selectors = [
                '[data-player-id]',
                '.player-card',
                '.player-list-item',
                '[class*="PlayerCard"]',
                '[class*="player-card"]',
            ];

            for (const sel of selectors) {
                const cards = document.querySelectorAll(sel);
                if (cards.length > 0) {
                    cards.forEach(card => {
                        const name  = card.querySelector('[class*="name"], .name, h3, h4')?.innerText?.trim();
                        const price = card.querySelector('[class*="price"], .price, [class*="Price"]')?.innerText?.trim();
                        const team  = card.querySelector('[class*="team"], .team, [class*="Team"]')?.innerText?.trim();
                        const type  = card.getAttribute('data-player-type') ||
                                      (card.closest('[class*="constructor"], [class*="Constructor"]') ? 'constructor' : 'driver');
                        if (name && price) {
                            results.push({ name, price, team, type });
                        }
                    });
                    break;
                }
            }
            return results;
        }
    """)

    log.info("DOM scraping found %d players", len(players))
    return players


def parse_api_prices(raw: list[dict]) -> pd.DataFrame:
    """Parse the JSON API response into a clean prices dataframe."""
    rows = []
    for p in raw:
        # Handle different API response shapes
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

        # Price is often in tenths of millions (e.g. 300 = $30.0M)
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


def parse_dom_prices(raw: list[dict]) -> pd.DataFrame:
    """Parse DOM-scraped price strings into a clean dataframe."""
    rows = []
    for p in raw:
        price_str = p.get("price", "0")
        # Strip "$", "M", spaces
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


def save_prices(df: pd.DataFrame):
    """Save prices to DuckDB and parquet."""
    # Add timestamp
    df["ScrapedAt"] = pd.Timestamp.now()

    con = duckdb.connect(str(DB_PATH))
    con.execute("DROP TABLE IF EXISTS player_prices")
    con.execute("CREATE TABLE player_prices AS SELECT * FROM df")
    con.close()

    out = PROCESSED / "player_prices.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved %d prices to %s", len(df), out)


# ── Main ──────────────────────────────────────────────────────────────────────


def run(headless: bool = True) -> pd.DataFrame | None:
    log.info("━━━ F1 Fantasy Price Scraper ━━━")

    if not EMAIL or not PASSWORD:
        log.error("F1_FANTASY_EMAIL and F1_FANTASY_PASSWORD not set in environment.")
        log.error("Create a .env file with these values and retry.")
        return None

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
            login(page, EMAIL, PASSWORD)
            navigate_to_team_picker(page)

            # Try API interception first
            api_data = intercept_prices_from_api(page)

            if api_data:
                log.info("Parsing %d API records...", len(api_data))
                df = parse_api_prices(api_data)
            else:
                log.warning("API interception returned nothing — trying DOM scraping")
                dom_data = scrape_prices_from_dom(page)
                if not dom_data:
                    log.error(
                        "DOM scraping also returned nothing. Site may have changed."
                    )
                    return None
                df = parse_dom_prices(dom_data)

            # Basic validation
            df = df[df["Price"] > 0].copy()
            log.info("Scraped %d valid prices", len(df))

            if len(df) < 10:
                log.warning("Only %d prices found — something may be wrong", len(df))
                # Save a screenshot for debugging
                page.screenshot(path=str(ROOT / "data" / "scraper_debug.png"))
                log.info("Debug screenshot saved to data/scraper_debug.png")

            save_prices(df)
            log.info(
                "\n%s",
                df.sort_values("Price", ascending=False)
                .head(15)
                .to_string(index=False),
            )

        except Exception as e:
            log.error("Scraper failed: %s", e)
            try:
                page.screenshot(path=str(ROOT / "data" / "scraper_debug.png"))
                log.info("Debug screenshot saved to data/scraper_debug.png")
            except Exception:
                pass
            raise

        finally:
            browser.close()

    log.info("━━━ Done ━━━")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window (useful for debugging)",
    )
    args = parser.parse_args()
    run(headless=not args.no_headless)
