"""
Scrapes Ron 95 price history from giaxanghomnay.com using Playwright.
The site renders prices via JavaScript — Playwright runs a real Chrome
browser to execute the JS before we parse the HTML.

Table structure found from debug:
  Col 0: Xăng/dầu  (fuel type)
  Col 1: Giá cũ    (old price)
  Col 2: Giá mới   (new price)  ← we want this
  Col 3: Thay đổi  (change)
  Col 4: Thời gian (date)        ← we want this

Run once for full backfill:
    python scrape_ron95.py

Scheduled by Modal on 1st, 11th, 21st of each month automatically.
"""
import asyncio
import psycopg2
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

# Load .env from project root (two levels up from backend/scripts/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
DATABASE_URL = os.getenv("DATABASE_URL")

HISTORY_URLS = [
    "https://giaxanghomnay.com/lich-su-gia-xang",
    "https://giaxanghomnay.com/lich-su-gia-xang?year=2025",
    "https://giaxanghomnay.com/lich-su-gia-xang?year=2024",
    "https://giaxanghomnay.com/lich-su-gia-xang?year=2023",
]

# Ron 95 keywords in Vietnamese — match any of these
RON95_KEYWORDS = ["RON 95", "RON95", "xăng 95", "A95", "95-III", "95-IV", "95-V"]


def is_ron95_row(fuel_text: str) -> bool:
    """Returns True if the fuel type cell refers to Ron 95."""
    fuel_upper = fuel_text.upper()
    return any(k.upper() in fuel_upper for k in RON95_KEYWORDS)


def parse_vnd_price(text: str) -> float | None:
    """
    Parses a Vietnamese price string into a float.
    Handles formats: 24.325, 24,325, 24325
    Returns None if parsing fails or value is out of range.
    """
    cleaned = re.sub(r"[^\d]", "", text.strip())
    if not cleaned or len(cleaned) < 4:
        return None
    price = float(cleaned)
    if price < 10_000 or price > 60_000:
        return None
    return price


def parse_date(text: str) -> datetime | None:
    """
    Parses a Vietnamese date string.
    Handles formats: 27/03/2026, 2026-03-27, 27-03-2026
    Also handles datetime strings by extracting just the date part.
    """
    text = text.strip()
    # Extract just the date portion if time is appended
    # e.g. "27/03/2026 00:00" → "27/03/2026"
    date_part = text.split(" ")[0].split("T")[0]

    for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%y"]:
        try:
            return datetime.strptime(date_part, fmt)
        except ValueError:
            continue
    return None


async def fetch_page_html() -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
        )

        # Intercept all API calls to find the chart data endpoint
        api_responses = []

        async def handle_response(response):
            url = response.url
            # Look for JSON responses that might contain price data
            if any(x in url for x in ["api", "data", "chart", "price", "xang", "json"]):
                try:
                    body = await response.text()
                    if any(x in body for x in ["24330", "29950", "ron95", "RON95"]):
                        print(f"\nFOUND DATA ENDPOINT: {url}")
                        print(f"Response preview: {body[:500]}")
                        api_responses.append({"url": url, "body": body})
                except Exception:
                    pass

        page = await context.new_page()
        page.on("response", handle_response)

        print(f"Opening {URL}...")
        await page.goto(URL, wait_until="networkidle", timeout=30_000)

        # Wait for all chart data to load
        print("Waiting for chart data to load...")
        await page.wait_for_timeout(5000)

        if api_responses:
            print(f"\nFound {len(api_responses)} data endpoints")
        else:
            print("No API endpoints found — trying to extract from window object")
            # Try extracting the amCharts data object directly
            data = await page.evaluate("""
                () => {
                    // amCharts stores data in chart objects
                    try {
                        const charts = am4core.registry.baseSprites;
                        if (charts && charts.length > 0) {
                            return JSON.stringify(charts[0].data);
                        }
                    } catch(e) {}
                    
                    // Try window.chartData or similar
                    const keys = Object.keys(window).filter(k => 
                        k.includes('chart') || k.includes('data') || k.includes('price')
                    );
                    return JSON.stringify({keys: keys});
                }
            """)
            print(f"Window data: {data[:500] if data else 'None'}")

        html = await page.content()
        await browser.close()
        return html


def parse_prices(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    records = []

    # Find all headings that contain dates
    # The page structure is: <h2>27/03/2026</h2> <table>...</table>
    # Try finding date headings near tables
    
    all_elements = soup.find_all(["h1","h2","h3","h4","h5","p","div","span","td","th"])
    
    # First find all dates on the page
    date_pattern = re.compile(r'\b(\d{1,2})[/\-](\d{1,2})[/\-](20\d{2})\b')
    
    print("\n=== ALL DATES FOUND ON PAGE ===")
    dates_found = []
    for el in all_elements:
        text = el.get_text(strip=True)
        match = date_pattern.search(text)
        if match and len(text) < 50:  # short elements only
            try:
                date = datetime.strptime(
                    f"{match.group(1)}/{match.group(2)}/{match.group(3)}", 
                    "%d/%m/%Y"
                )
                dates_found.append((date, el))
                print(f"  {date.strftime('%Y-%m-%d')} in <{el.name}>: {text[:40]}")
            except ValueError:
                continue

    print(f"\nTotal dates found: {len(dates_found)}")

    # Find all tables and try to associate them with nearby dates
    tables = soup.find_all("table")
    print(f"Total tables: {len(tables)}")

    for i, table in enumerate(tables):
        # Look for a date in the previous sibling elements
        current = table
        nearby_date = None
        
        for _ in range(10):  # look up to 10 elements back
            current = current.find_previous_sibling()
            if current is None:
                break
            text = current.get_text(strip=True)
            match = date_pattern.search(text)
            if match:
                try:
                    nearby_date = datetime.strptime(
                        f"{match.group(1)}/{match.group(2)}/{match.group(3)}",
                        "%d/%m/%Y"
                    )
                    break
                except ValueError:
                    continue

        # Also check parent's previous siblings
        if not nearby_date and table.parent:
            current = table.parent
            for _ in range(5):
                current = current.find_previous_sibling()
                if current is None:
                    break
                text = current.get_text(strip=True) if current else ""
                match = date_pattern.search(text)
                if match:
                    try:
                        nearby_date = datetime.strptime(
                            f"{match.group(1)}/{match.group(2)}/{match.group(3)}",
                            "%d/%m/%Y"
                        )
                        break
                    except ValueError:
                        continue

        rows = table.find_all("tr")
        print(f"\nTable {i} (date={nearby_date}): {len(rows)} rows")

        for row in rows[1:]:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            col_texts = [c.get_text(strip=True) for c in cols]
            print(f"  {col_texts}")

            fuel = cols[0].get_text(strip=True)
            if not is_ron95_row(fuel):
                continue

            price = parse_vnd_price(cols[1].get_text(strip=True))
            date = nearby_date

            if price and date:
                print(f"  ✓ Ron 95 found: {date.strftime('%Y-%m-%d')} → {price:,.0f} VND")
                records.append({
                    "effective_at": date,
                    "price_vnd": price,
                    "source_url": URL,
                })

    # Deduplicate
    by_date = {}
    for r in records:
        key = r["effective_at"].strftime("%Y-%m-%d")
        if key not in by_date or r["price_vnd"] > by_date[key]["price_vnd"]:
            by_date[key] = r

    unique = sorted(by_date.values(), key=lambda x: x["effective_at"])
    print(f"\nTotal unique Ron 95 records: {len(unique)}")
    return unique

def save_to_supabase(records: list[dict]) -> tuple[int, int]:
    """
    Inserts records into ron95_prices table.
    Skips duplicates via ON CONFLICT DO NOTHING.
    Returns (inserted, skipped) counts.
    """
    if not records:
        print("No records to save.")
        return 0, 0

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    inserted = skipped = 0

    for r in records:
        cur.execute("""
            INSERT INTO ron95_prices
                (announced_at, effective_at, price_vnd, source_url, notes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (effective_at) DO NOTHING
        """, (
            r["effective_at"],
            r["effective_at"],
            r["price_vnd"],
            r["source_url"],
            "Scraped via Playwright headless Chrome",
        ))
        if cur.rowcount > 0:
            inserted += 1
        else:
            skipped += 1

    conn.commit()
    cur.close()
    conn.close()
    return inserted, skipped


async def main():
    all_records = []

    # Scrape all history URLs to get maximum coverage
    for url in HISTORY_URLS:
        global URL
        URL = url
        print(f"\n{'='*50}")
        print(f"Scraping: {url}")

        html = await fetch_page_html()
        if not html:
            print(f"ERROR: Could not fetch {url}")
            continue

        records = parse_prices(html)
        if records:
            all_records.extend(records)
            print(f"Got {len(records)} records from {url}")
        else:
            print(f"No records from {url}")

    if not all_records:
        print("\nERROR: No Ron 95 prices found from any URL.")
        return

    # Deduplicate across all URLs
    by_date = {}
    for r in all_records:
        key = r["effective_at"].strftime("%Y-%m-%d")
        if key not in by_date or r["price_vnd"] > by_date[key]["price_vnd"]:
            by_date[key] = r

    unique = sorted(by_date.values(), key=lambda x: x["effective_at"])
    print(f"\nTotal unique records across all URLs: {len(unique)}")

    print(f"\nFirst 3 records:")
    for r in unique[:3]:
        print(f"  {r['effective_at'].strftime('%Y-%m-%d')}: {r['price_vnd']:,.0f} VND")

    print(f"\nLast 3 records:")
    for r in unique[-3:]:
        print(f"  {r['effective_at'].strftime('%Y-%m-%d')}: {r['price_vnd']:,.0f} VND")

    inserted, skipped = save_to_supabase(unique)
    print(f"\nSaved {inserted} new rows · Skipped {skipped} duplicates")


if __name__ == "__main__":
    asyncio.run(main())