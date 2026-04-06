"""
Modal scheduled job — detects geopolitical shock events using GPT-4o.
Runs every 6 hours. Fetches headlines from GNews, sends to GPT-4o for
classification, saves genuine shocks to shock_events table.

Deploy: modal deploy backend/ml/training/shock_schedule.py
Test:   modal run backend/ml/training/shock_schedule.py
"""
import modal
import time
app = modal.App("shock-event-detector")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "httpx==0.27.0",
        "psycopg2-binary==2.9.9",
        "openai==1.30.0",
    )
)


@app.function(
    image=image,
    schedule=modal.Cron("0 */6 * * *"),  # every 6 hours
    secrets=[modal.Secret.from_name("energy-forecaster-secrets")],
    timeout=120,
)
def detect_shock_events():
    import os
    import httpx
    import psycopg2
    import json
    from datetime import datetime, timezone
    from openai import OpenAI

    DATABASE_URL   = os.environ["DATABASE_URL"]
    GNEWS_API_KEY  = os.environ.get("GNEWS_API_KEY", "")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

    if not GNEWS_API_KEY:
        print("ERROR: Missing GNEWS_API_KEY")
        return

    if not OPENAI_API_KEY:
        print("ERROR: Missing OPENAI_API_KEY")
        return

    print(f"[{datetime.utcnow().isoformat()}] Starting shock detection...")

    # ── Step 1: Fetch headlines from GNews ────────────────────────────────
    headlines = []
    queries = [
        "oil price geopolitical energy",
        "Iran sanctions crude oil",
        "Federal Reserve interest rate emergency",
        "Bitcoin Ethereum crypto market crash",
        "Vietnam economy dong currency",
        "OPEC oil production cut",
        "Middle East conflict energy supply",
        "global markets crash recession",
    ]

    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(2)  # 2 second delay between requests
        try:
            resp = httpx.get(
                "https://gnews.io/api/v4/search",
                params={
                    "q":      query,
                    "lang":   "en",
                    "max":    10,   # get more per query instead
                    "apikey": GNEWS_API_KEY,
                },
                timeout=15,
            )
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            for a in articles:
                headlines.append({
                    "title":       a["title"],
                    "description": a.get("description", ""),
                    "published":   a.get("publishedAt", ""),
                    "source":      a.get("source", {}).get("name", ""),
                })
            print(f"  Query '{query}': {len(articles)} articles")
        except Exception as e:
            print(f"  GNews query '{query}' failed: {e}")

    if not headlines:
        print("No headlines fetched — skipping")
        return

    # Deduplicate by title
    seen = set()
    unique_headlines = []
    for h in headlines:
        if h["title"] not in seen:
            seen.add(h["title"])
            unique_headlines.append(h)

    print(f"Total unique headlines: {len(unique_headlines)}")
    conn_check = psycopg2.connect(DATABASE_URL, sslmode="require")
    cur_check = conn_check.cursor()
    cur_check.execute("""
        SELECT trigger_headline FROM shock_events
        WHERE event_date >= NOW() - INTERVAL '48 hours'
        AND trigger_headline IS NOT NULL
    """)
    recent_headlines = {row[0][:50] for row in cur_check.fetchall()}
    cur_check.close()
    conn_check.close()

    # Filter out already-processed headlines
    filtered = [
        h for h in unique_headlines
        if not any(h["title"][:50] in rh for rh in recent_headlines)
    ]

    print(f"After dedup against DB: {len(filtered)} new headlines to analyze")

    if not filtered:
        print("All headlines already processed — skipping GPT-4o call")
        return

    unique_headlines = filtered
    # ── Step 2: Send to GPT-4o for classification ─────────────────────────
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Format headlines for the prompt
    headlines_text = "\n".join([
        f"{i+1}. [{h['source']}] {h['title']}"
        + (f"\n   Summary: {h['description'][:150]}" if h["description"] else "")
        for i, h in enumerate(unique_headlines)
    ])

    system_prompt = (
        "You are a senior financial market analyst specializing in macro events "
        "that move crypto and energy markets. You have 20 years of experience "
        "identifying genuine market-moving events from noise. "
        "Always respond with valid JSON only — no markdown, no explanation."
    )

    user_prompt = f"""Analyze these news headlines published in the last 6 hours.
Identify ONLY genuine market-moving shock events — not routine news or analyst opinions.

HEADLINES:
{headlines_text}

WHAT COUNTS AS A SHOCK EVENT:
Direct oil supply threats (Hormuz closure, pipeline attacks, OPEC surprise cuts)
Active military conflict escalation or de-escalation (ceasefire, new strikes)
Surprise central bank decisions (emergency rate changes, unexpected holds)
Major crypto market events (large exchange collapse, unexpected ETF rejection/approval, regulatory ban)
Financial system stress (major bank failure, sovereign default, liquidity crisis)
Events directly affecting Vietnam economy (dong devaluation, capital controls, sanctions)

WHAT DOES NOT COUNT:
Analyst predictions or price forecasts ("oil could hit $150 if...")
Routine scheduled events with expected outcomes
Minor daily price movements without a clear catalyst
Old news being re-reported
Company earnings (unless the company is systemically important)
Opinion pieces or commentary

SEVERITY GUIDE:
- HIGH: Direct oil supply disruption, active armed conflict, major market crash. Expected price moves >5%.
- ELEVATED: Escalation risk, significant policy surprise, moderate market stress. Expected moves 2-5%.
- LOW: Minor escalation, small policy adjustment, limited market impact. Expected moves <2%.

SHOCK TYPE OPTIONS: oil, conflict, financial, crypto, political, vietnam

Respond with a JSON object containing a "shocks" array.
Each shock must have ALL these fields:

{{
  "shocks": [
    {{
      "headline": "exact headline text copied from above",
      "is_shock": true,
      "severity": "HIGH or ELEVATED or LOW",
      "shock_type": "oil or conflict or financial or crypto or political or vietnam",
      "oil_impact_estimate": 5.2,
      "btc_impact_estimate": -3.1,
      "is_deescalation": false,
      "description": "One clear sentence: what happened and why it moves markets.",
      "affected_assets": ["BTC", "ETH", "BRENT", "VND"]
    }}
  ]
}}

If zero genuine shocks found, return: {{"shocks": []}}"""

    print("Sending to GPT-4o for classification...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1500,
            temperature=0.1,        # low temperature = consistent, factual
            response_format={"type": "json_object"},  # guarantees valid JSON
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
        )

        raw = response.choices[0].message.content.strip()
        print(f"GPT-4o raw response: {raw[:300]}...")

        parsed = json.loads(raw)
        shocks = parsed.get("shocks", [])
        print(f"GPT-4o identified {len(shocks)} shock event(s)")

    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON from GPT-4o: {e}")
        print(f"Raw response was: {raw}")
        return
    except Exception as e:
        print(f"ERROR: OpenAI API call failed: {e}")
        return

    # ── Step 3: Filter and save genuine shocks ────────────────────────────
    if not shocks:
        print("No shock events identified — market conditions normal")
        return
    
    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cur  = conn.cursor()
    saved = 0

    for shock in shocks:
        # Only process events GPT-4o confirmed as shocks
        if not shock.get("is_shock", False):
            continue

        severity = shock.get("severity", "LOW")
        headline = shock.get("headline", "")[:500]

        # Check for duplicate — same severity event in last 12 hours
        # Prevents saving the same breaking news multiple times per day
        cur.execute("""
            SELECT id FROM shock_events
            WHERE event_date >= NOW() - INTERVAL '12 hours'
            AND severity = %s
            AND (
                description ILIKE %s
                OR trigger_headline ILIKE %s
            )
            LIMIT 1
        """, (
            severity,
            f"%{headline[:50]}%",
            f"%{headline[:50]}%",
        ))

        if cur.fetchone():
            print(f"  DUPLICATE skipped: {headline[:70]}")
            continue
        
        # Build description and notes
        description = shock.get("description", headline)
        if len(description) > 500:
            description = description[:500]

        affected = ", ".join(shock.get("affected_assets", []))
        deesc    = shock.get("is_deescalation", False)
        stype    = shock.get("shock_type", "unknown")
        oil_est  = shock.get("oil_impact_estimate", 0.0)
        btc_est  = shock.get("btc_impact_estimate", 0.0)

        notes = (
            f"Auto-detected by GPT-4o. "
            f"Type: {stype}. "
            f"Affected: {affected}. "
            f"De-escalation: {deesc}. "
            f"Oil estimate: {oil_est:+.1f}%. "
            f"BTC estimate: {btc_est:+.1f}%."
        )

        # Save to shock_events
        # verified=FALSE — requires human review before used in ML training
        # The dashboard shows it immediately for situational awareness
        cur.execute("""
            INSERT INTO shock_events
                (event_date, description, severity, oil_impact,
                 trigger_headline, verified, notes)
            VALUES (%s, %s, %s, %s, %s, FALSE, %s)
            ON CONFLICT DO NOTHING
            RETURNING id
        """, (
            datetime.now(timezone.utc),
            description,
            severity,
            oil_est,
            headline,
            notes,
        ))

        result = cur.fetchone()
        if result:
            saved += 1
            print(f"  SAVED [{severity}] {headline[:70]}")
            print(f"    Type: {stype} | Oil: {oil_est:+.1f}% | BTC: {btc_est:+.1f}%")
            print(f"    → {description[:120]}")
        else:
            print(f"  Conflict (already exists): {headline[:60]}")

    conn.commit()
    cur.close()
    conn.close()

    print(f"\n{'='*50}")
    print(f"Done. Saved {saved} new shock event(s).")
    if saved == 0:
        print("Either no new shocks or all were duplicates.")

@app.function(
    image=image,
    schedule=modal.Cron("0 1 * * *"),  # 01:00 UTC daily
    secrets=[modal.Secret.from_name("energy-forecaster-secrets")],
    timeout=60,
)
def measure_shock_impact():
    """
    Runs daily at 01:00 UTC.
    Finds shock events from yesterday with no impact data.
    Fetches BTC/ETH prices from CoinGecko and fills in the impact.
    """
    import os
    import httpx
    import psycopg2
    from datetime import datetime, timezone, timedelta

    DATABASE_URL = os.environ["DATABASE_URL"]

    print(f"[{datetime.utcnow().isoformat()}] Measuring shock impact...")

    conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    cur = conn.cursor()

    # Find events from last 48 hours with no impact measured
    # 48 hours covers events that were saved late in the day
    cur.execute("""
        SELECT id, event_date, description, severity
        FROM shock_events
        WHERE event_date >= NOW() - INTERVAL '48 hours'
        AND btc_impact_24h IS NULL
        ORDER BY event_date ASC
    """)
    events = cur.fetchall()

    if not events:
        print("No events need impact measurement")
        cur.close()
        conn.close()
        return

    print(f"Found {len(events)} event(s) needing impact measurement")

    # Fetch BTC and ETH price history from CoinGecko
    # Gets last 2 days of prices — before and after the shock
    try:
        btc_resp = httpx.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": "3", "interval": "daily"},
            timeout=15,
            headers={"Accept": "application/json"},
        )
        eth_resp = httpx.get(
            "https://api.coingecko.com/api/v3/coins/ethereum/market_chart",
            params={"vs_currency": "usd", "days": "3", "interval": "daily"},
            timeout=15,
            headers={"Accept": "application/json"},
        )
        btc_resp.raise_for_status()
        eth_resp.raise_for_status()

        btc_prices = btc_resp.json().get("prices", [])
        eth_prices = eth_resp.json().get("prices", [])

        print(f"CoinGecko: {len(btc_prices)} BTC price points")
        print(f"CoinGecko: {len(eth_prices)} ETH price points")

        if len(btc_prices) < 2 or len(eth_prices) < 2:
            print("Not enough price data to compute impact")
            cur.close()
            conn.close()
            return

    except Exception as e:
        print(f"CoinGecko fetch failed: {e}")
        cur.close()
        conn.close()
        return

    # For each event, find the closest price points
    # before and after the event date
    updated = 0

    for event_id, event_date, description, severity in events:
        event_ts = event_date.timestamp() * 1000  # convert to ms

        # Find BTC price just before and just after the event
        btc_before = None
        btc_after  = None

        for i, (ts, price) in enumerate(btc_prices):
            if ts <= event_ts:
                btc_before = price
            elif ts > event_ts and btc_after is None:
                btc_after = price

        # If no price after event yet — use latest available
        if btc_before and not btc_after:
            btc_after = btc_prices[-1][1]

        eth_before = None
        eth_after  = None

        for i, (ts, price) in enumerate(eth_prices):
            if ts <= event_ts:
                eth_before = price
            elif ts > event_ts and eth_after is None:
                eth_after = price

        if eth_before and not eth_after:
            eth_after = eth_prices[-1][1]

        if not btc_before or not btc_after:
            print(f"  Event {event_id}: insufficient price data, skipping")
            continue

        # Compute impact
        btc_impact = round(
            ((btc_after - btc_before) / btc_before) * 100, 2
        )
        eth_impact = round(
            ((eth_after - eth_before) / eth_before) * 100, 2
        )

        # Auto-verify based on actual price movement
        # If the market moved significantly → event was real
        # If market barely moved → keep as unverified for manual review
        auto_verify = False
        if severity == "HIGH" and abs(btc_impact) > 3.0:
            auto_verify = True
        elif severity == "ELEVATED" and abs(btc_impact) > 1.5:
            auto_verify = True
        # LOW severity always requires manual review

        # Save impact and verification status
        cur.execute("""
            UPDATE shock_events
            SET btc_impact_24h = %s,
                eth_impact_24h = %s,
                verified = %s
            WHERE id = %s
        """, (btc_impact, eth_impact, auto_verify, event_id))

        updated += 1
        print(
            f"  Event {event_id} [{severity}] {description[:60]}"
            f"\n    BTC: {btc_impact:+.2f}% | ETH: {eth_impact:+.2f}%"
            f"\n    Verified: {auto_verify}"
            + (" ← AUTO-VERIFIED" if auto_verify else " ← awaiting manual review")
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone. Updated {updated} event(s) with price impact.")


@app.local_entrypoint()
def main():
    detect_shock_events.remote()

@app.local_entrypoint()
def measure():
    measure_shock_impact.remote()
