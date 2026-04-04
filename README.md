# Crypto Agent

> A multi-layer AI trading platform for BTC/ETH Futures — combining XGBoost ML predictions, Gemini Flash vision analysis, macro fundamentals, and real-time technical gates into a fully automated 5-minute trading engine.

![Dashboard Preview](docs/images/dashboard.png)
<!-- SCREENSHOT NEEDED: Full dashboard view showing the main tab with ticker strip, ML forecasts, and trading overview -->

---

## Overview

Crypto Agent is a production-grade algorithmic trading system that:

- **Predicts** BTC/ETH 24-hour price direction using XGBoost models trained on macro + technical features
- **Executes** trades every 5 minutes on Binance Futures with multi-layer signal validation
- **Confirms** chart patterns using Gemini Flash (vision + language model) before entering any position
- **Monitors** macro events (FED rate, CPI, NFP), energy shocks, Fear & Greed, and funding rates
- **Displays** a real-time React dashboard with ML forecasts, trade history, and live P&L

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Internet                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │ 80 / 443
                    ┌───────▼────────┐
                    │  Nginx Proxy   │
                    └──┬─────────┬──┘
                       │         │
              ┌────────▼──┐  ┌───▼──────────┐
              │ Frontend  │  │  Go Backend  │ :8080
              │ React SPA │  │  API Gateway │
              └───────────┘  └──┬───────┬───┘
                                │       │
                    ┌───────────▼─┐  ┌──▼──────────┐
                    │ ML Service  │  │  PostgreSQL  │
                    │ FastAPI     │  │  TimescaleDB │
                    │ :8001       │  └─────────────┘
                    └─────────────┘
                                        ┌──────────┐
                    ┌───────────────┐   │  Redis   │
                    │Trading Engine │◄──┤  Cache   │
                    │ 5-min cycle   │   └──────────┘
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │ Binance       │
                    │ Futures API   │
                    └───────────────┘
```

---

## Trading Engine Pipeline

Every 5 minutes, the engine runs each symbol (BTC, ETH) through a layered gate system:

```
Gate 0 │ Open Position?     Skip symbol if already in a trade
   ↓
Gate 1 │ Macro Bias         ML direction + Fear&Greed + funding rate
   ↓
Gate 2 │ BTC Correlation    (ETH only) Don't trade against BTC trend
   ↓
Gate 3 │ Technical          H1/M15 trend alignment, score ≥ 3/5
        │ — or —
        │ Setup E            SIDEWAY: price at BB extreme + RSI extreme
   ↓
Gate 4 │ Gemini Vision      Chart image → BUY / SELL / WAIT
   ↓
       │ Execute            Market order + native SL/TP on Binance
```

![Trading Engine Logs](docs/images/engine_logs.png)
<!-- SCREENSHOT NEEDED: Terminal showing the trading engine cycle logs with gate pass/fail output -->

### Supported Setups

| Setup | Mode | Signal |
|-------|------|--------|
| **A** | UPTREND / DOWNTREND | EMA pullback + pinbar/engulfing |
| **B** | Strong trend (score ≥ 4) | Consolidation breakout/breakdown |
| **C** | UPTREND / DOWNTREND | Fake drop/pump reversal |
| **D** | VOLATILE_RANGE | Bounce off H1 S/R level |
| **E** | SIDEWAY | BB mean reversion → SMA20 target |

---

## Services

| Service | Stack | Role |
|---------|-------|------|
| `nginx` | nginx:alpine | Reverse proxy — only public-facing port |
| `frontend` | React 19 + Vite | Real-time dashboard SPA |
| `go-backend` | Go 1.26 + Gin | API gateway + WebSocket broadcast |
| `ml-service` | Python + FastAPI | XGBoost inference + model serving |
| `trading-engine` | Python | 5-min LLM trading loop |
| `scheduler` | Python | Data ingestion daemon |
| `redis` | Redis 7 | Prediction cache + session state |

---

## Dashboard

### ML Forecasts Tab
![ML Tab](docs/images/ml_tab.png)
<!-- SCREENSHOT NEEDED: ML forecasts tab showing BTC/ETH UP/DOWN predictions with confidence percentages -->

XGBoost models predict BTC and ETH 24h / 7d price direction. Features include:
- Candlestick OHLCV (30-min intervals)
- Macro indicators: FED rate, CPI, NFP, Fear & Greed
- On-chain signals: ETF net flows, funding rates
- Technical: rolling returns, volatility, price vs moving averages

Models are trained on DagsHub/MLflow and auto-promoted to `@champion` on every successful training run.

### Trading Tab
![Trading Tab](docs/images/trading_tab.png)
<!-- SCREENSHOT NEEDED: Trading tab showing open positions (with entry, SL, TP, unrealized P&L) and trade history table -->

Displays:
- Live open positions synced from Binance (entry price, SL, TP, mark price, unrealized P&L)
- Full trade history with outcome, P&L per trade, and close reason
- Aggregate stats: total P&L, win rate, total trades

### Macro Tab
![Macro Tab](docs/images/macro_tab.png)
<!-- SCREENSHOT NEEDED: Macro tab showing Fear & Greed gauge, FED rate chart, CPI, and NFP data -->

Tracks macro fundamentals that influence the ML model and macro bias gate:
- Fear & Greed Index (0–100)
- FED Funds Rate
- CPI (inflation)
- NFP (non-farm payrolls)
- Brent crude oil price

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML Model** | XGBoost, scikit-learn, MLflow, DagsHub |
| **LLM / Vision** | Gemini 2.0 Flash (Google AI) |
| **Trading API** | Binance Futures (python-binance) |
| **Backend API** | Go + Gin, Python + FastAPI |
| **Database** | PostgreSQL + TimescaleDB |
| **Cache** | Redis 7 |
| **Frontend** | React 19, Vite, lightweight-charts, Recharts |
| **Infrastructure** | Docker Compose, nginx, GitHub Actions CI/CD |
| **Data Sources** | Binance, FRED (FED/CPI/NFP), alternative.me (Fear&Greed) |

---

## Project Structure

```
crypto-agent/
├── backend/
│   ├── training/          # ML inference service (FastAPI :8001)
│   │   ├── main.py        # Prediction endpoints + model reload
│   │   ├── train.py       # XGBoost training pipeline
│   │   ├── model_store.py # MLflow model loading + Redis cache
│   │   └── feature_engineering.py
│   ├── trading/           # 5-min trading engine
│   │   ├── engine.py      # Main loop + DB persistence + monitor thread
│   │   ├── strategy_core.py  # Technical indicators + gate logic
│   │   ├── llm_analyst.py    # Gemini Flash chart analysis
│   │   ├── chart_gen.py      # M5 candlestick chart generation
│   │   ├── trade_logger.py   # Structured rotating log files
│   │   └── backtest.py       # Rule-based historical backtester
│   └── schedule/          # Data ingestion scheduler
│       ├── run_scheduler.py  # Fear&Greed, funding rates (24h/8h)
│       └── run_macro_etf_ingestor.py
├── go-backend/            # Go API gateway
│   ├── main.go
│   └── handlers/          # ml, trading, crypto, macro, websocket
├── frontend/              # React SPA
│   └── src/
│       ├── App.jsx
│       ├── MLTab.jsx
│       ├── TradingTab.jsx
│       └── AITab.jsx
├── nginx/                 # Reverse proxy config + SSL certs
├── infra/                 # SQL schema + migrations
├── .github/workflows/     # CI lint/build + CD auto-deploy
└── docker-compose.yml
```

---

## Setup

### Prerequisites

- Docker + Docker Compose
- Binance Futures account (testnet or live)
- Google AI Studio API key (Gemini Flash)
- DagsHub account (MLflow tracking)
- PostgreSQL instance (e.g. Neon, Supabase, or self-hosted)

### 1. Clone

```bash
git clone https://github.com/TuanTran1504/Crypto-Agent.git
cd Crypto-Agent
```

### 2. Environment Variables

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

```env
# Database
DATABASE_URL=postgresql://user:password@host:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379

# Binance Futures (Testnet)
BINANCE_FUTURES_API_KEY=your_key
BINANCE_FUTURES_SECRET_KEY=your_secret

# Google AI (Gemini Flash — for trading vision)
GOOGLE_API_KEY=your_google_ai_key

# MLflow / DagsHub (model registry)
MLFLOW_TRACKING_URI=https://dagshub.com/user/repo.mlflow
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token

# FRED (macro data — optional)
FRED_API_KEY=your_fred_key

# ML thresholds
BTC_UP_THRESHOLD=0.55
ETH_UP_THRESHOLD=0.50
```

### 3. Run

```bash
docker compose up -d
```

Services will be available at:
- Dashboard: `http://localhost`
- Go API: `http://localhost:8080`
- ML Service: `http://localhost:8001`

### 4. Initialize Database

```bash
psql $DATABASE_URL < infra/schema.sql
```

---

## Backtesting

Run a rule-based backtest on historical Binance data (no LLM cost):

```bash
docker compose exec trading-engine python backtest.py --symbol BTC --days 30
docker compose exec trading-engine python backtest.py --symbol ETH --days 60 --output results/eth.csv
```

Output includes: win rate, total P&L, avg win/loss, profit factor, max drawdown, Sharpe ratio, and a CSV trade log.

---

## CI/CD

Every push to `main`:

1. **CI** — ruff lint (Python), `go vet` + `go build` (Go), `npm run build` (frontend)
2. **CD** — SSH into VPS → `git pull` → `docker compose build --parallel` → `docker compose up -d`

---

## Risk Disclaimer

This project is for **educational and research purposes only**. It is currently configured to trade on **Binance Testnet** (paper trading — no real money). Never deploy to a live account without thorough testing, risk management review, and an understanding of the financial risks involved.

---

## License

MIT
