import { useState, useEffect, useRef } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

const WS_URL = "ws://localhost:8080/ws/live";
const API_URL = "http://localhost:8080";

// ── Utilities ────────────────────────────────────────────────────────────────
const fmt = (n, decimals = 2) =>
  n == null ? "—" : Number(n).toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });

const fmtVND = (n) =>
  n == null ? "—" : Number(n).toLocaleString("vi-VN");

const change = (n) => {
  if (n == null) return null;
  const pos = n >= 0;
  return { value: fmt(Math.abs(n), 2), pos };
};

const ChangeTag = ({ n, suffix = "%" }) => {
  const c = change(n);
  if (!c) return <span className="muted">—</span>;
  return (
    <span className={c.pos ? "pos" : "neg"}>
      {c.pos ? "▲" : "▼"} {c.value}{suffix}
    </span>
  );
};

// ── Shock Alert Bar ───────────────────────────────────────────────────────────
const ShockBar = ({ score, level, message }) => {
  const color =
    level === "HIGH" ? "#ff3b30" :
    level === "ELEVATED" ? "#ff9f0a" : "#30d158";

  const width = `${Math.round((score || 0) * 100)}%`;

  return (
    <div className="shock-bar">
      <div className="shock-left">
        <span className="shock-label" style={{ color }}>⬤ {level || "NORMAL"}</span>
        <span className="shock-message">{message || "No significant disruption detected"}</span>
      </div>
      <div className="shock-meter">
        <div className="shock-track">
          <div className="shock-fill" style={{ width, background: color }} />
        </div>
        <span className="shock-score">{fmt(score, 4)}</span>
      </div>
    </div>
  );
};

// ── Ticker Strip ──────────────────────────────────────────────────────────────
const TickerStrip = ({ signals }) => {
  if (!signals) return null;
  const { brent_crude, usd_vnd, crypto } = signals;

  const items = [
    { label: "BRENT", value: `$${fmt(brent_crude?.price_usd)}`, change: brent_crude?.delta_day_pct },
    { label: "USD/VND", value: fmtVND(usd_vnd?.usd_to_vnd), change: usd_vnd?.change_7day_pct },
    { label: "BTC/USDT", value: `$${fmt(crypto?.btc?.price_usd, 0)}`, change: crypto?.btc?.change_24h_pct },
    { label: "ETH/USDT", value: `$${fmt(crypto?.eth?.price_usd)}`, change: crypto?.eth?.change_24h_pct },
    { label: "BTC VOL", value: `$${fmt(crypto?.btc?.volume_24h_usd / 1e9, 2)}B`, change: null },
  ];

  return (
    <div className="ticker-strip">
      {items.map((item, i) => (
        <div key={i} className="ticker-item">
          <span className="ticker-label">{item.label}</span>
          <span className="ticker-value">{item.value}</span>
          <ChangeTag n={item.change} />
        </div>
      ))}
    </div>
  );
};

// ── Macro Tab ─────────────────────────────────────────────────────────────────
const MacroTab = ({ signals, history }) => {
  if (!signals) return <div className="loading">Awaiting data stream...</div>;
  const { brent_crude, usd_vnd } = signals;

  return (
    <div className="tab-content">
      <div className="card-grid">
        {/* Brent Crude */}
        <div className="card">
          <div className="card-header">
            <span className="card-label">BRENT CRUDE</span>
            <span className="card-sub">USD / barrel</span>
          </div>
          <div className="card-value">${fmt(brent_crude?.price_usd)}</div>
          <div className="card-meta">
            <ChangeTag n={brent_crude?.delta_day_pct} /> <span className="muted">24h</span>
          </div>
          <div className="card-note">Iran-US conflict · Hormuz closure risk</div>
        </div>

        {/* USD/VND */}
        <div className="card">
          <div className="card-header">
            <span className="card-label">USD / VND</span>
            <span className="card-sub">Exchange rate</span>
          </div>
          <div className="card-value">{fmtVND(usd_vnd?.usd_to_vnd)}</div>
          <div className="card-meta">
            <ChangeTag n={usd_vnd?.change_7day_pct} /> <span className="muted">7d</span>
            &nbsp;&nbsp;
            <ChangeTag n={usd_vnd?.change_30day_pct} /> <span className="muted">30d</span>
          </div>
          <div className="card-note">VND import cost pressure</div>
        </div>

        {/* Oil delta context */}
        <div className="card card-wide">
          <div className="card-header">
            <span className="card-label">GEOPOLITICAL CONTEXT</span>
          </div>
          <div className="context-lines">
            <div className="context-line">
              <span className="ctx-key">Strait of Hormuz</span>
              <span className="ctx-val neg">DISRUPTED</span>
            </div>
            <div className="context-line">
              <span className="ctx-key">IEA Reserve Release</span>
              <span className="ctx-val pos">400M barrels</span>
            </div>
            <div className="context-line">
              <span className="ctx-key">US-Iran talks</span>
              <span className="ctx-val">ONGOING</span>
            </div>
            <div className="context-line">
              <span className="ctx-key">Brent 30d change</span>
              <span className="ctx-val pos">+40%</span>
            </div>
          </div>
        </div>
      </div>

      {/* VND history chart */}
      {history.length > 1 && (
        <div className="chart-section">
          <div className="chart-label">USD/VND — 30 day history</div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={history}>
              <XAxis dataKey="date" tick={{ fill: "#666", fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis domain={["auto", "auto"]} tick={{ fill: "#666", fontSize: 10 }} tickLine={false} axisLine={false} width={60} tickFormatter={v => v.toLocaleString()} />
              <Tooltip
                contentStyle={{ background: "#0d0d0d", border: "1px solid #333", borderRadius: 4 }}
                labelStyle={{ color: "#888" }}
                itemStyle={{ color: "#00ff88" }}
                formatter={v => [v.toLocaleString(), "USD/VND"]}
              />
              <Line type="monotone" dataKey="rate" stroke="#00ff88" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

// ── Crypto Tab ────────────────────────────────────────────────────────────────
const CryptoTab = ({ signals, priceHistory }) => {
  if (!signals?.crypto) return <div className="loading">Awaiting crypto stream...</div>;
  const { btc, eth, usdt_to_vnd } = signals.crypto;

  return (
    <div className="tab-content">
      <div className="card-grid">
        {/* BTC */}
        <div className="card card-accent-orange">
          <div className="card-header">
            <span className="card-label">BITCOIN</span>
            <span className="card-sub">BTC / USDT · Binance</span>
          </div>
          <div className="card-value">${fmt(btc?.price_usd, 0)}</div>
          <div className="card-meta">
            <ChangeTag n={btc?.change_24h_pct} /> <span className="muted">24h</span>
          </div>
          <div className="card-note">
            ≈ {fmtVND(btc?.price_usd * usdt_to_vnd)} VND
          </div>
          <div className="card-meta" style={{ marginTop: 8 }}>
            <span className="muted">Vol 24h</span>
            &nbsp;
            <span>${fmt(btc?.volume_24h_usd / 1e9, 2)}B</span>
          </div>
        </div>

        {/* ETH */}
        <div className="card card-accent-blue">
          <div className="card-header">
            <span className="card-label">ETHEREUM</span>
            <span className="card-sub">ETH / USDT · Binance</span>
          </div>
          <div className="card-value">${fmt(eth?.price_usd)}</div>
          <div className="card-meta">
            <ChangeTag n={eth?.change_24h_pct} /> <span className="muted">24h</span>
          </div>
          <div className="card-note">
            ≈ {fmtVND(eth?.price_usd * usdt_to_vnd)} VND
          </div>
          <div className="card-meta" style={{ marginTop: 8 }}>
            <span className="muted">Vol 24h</span>
            &nbsp;
            <span>${fmt(eth?.volume_24h_usd / 1e9, 2)}B</span>
          </div>
        </div>

        {/* USDT/VND */}
        <div className="card">
          <div className="card-header">
            <span className="card-label">USDT / VND</span>
            <span className="card-sub">Derived from USD/VND</span>
          </div>
          <div className="card-value">{fmtVND(usdt_to_vnd)}</div>
          <div className="card-note">1 USDT ≈ 1 USD in VND terms</div>
        </div>
      </div>

      {/* BTC price history */}
      {priceHistory.btc.length > 1 && (
        <div className="chart-section">
          <div className="chart-label">BTC/USDT — recent sessions</div>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={priceHistory.btc}>
              <XAxis dataKey="date" tick={{ fill: "#666", fontSize: 10 }} tickLine={false} axisLine={false} />
              <YAxis domain={["auto", "auto"]} tick={{ fill: "#666", fontSize: 10 }} tickLine={false} axisLine={false} width={70} tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} />
              <Tooltip
                contentStyle={{ background: "#0d0d0d", border: "1px solid #333", borderRadius: 4 }}
                labelStyle={{ color: "#888" }}
                itemStyle={{ color: "#ff9f0a" }}
                formatter={v => [`$${v.toLocaleString()}`, "BTC"]}
              />
              <Line type="monotone" dataKey="price" stroke="#ff9f0a" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

// ── Signals Tab ───────────────────────────────────────────────────────────────
const SignalsTab = ({ signals, shockData }) => {
  const assets = [
    {
      name: "BRENT CRUDE",
      change: signals?.brent_crude?.delta_day_pct,
      signal: signals?.brent_crude?.delta_day_pct > 2 ? "SELL OIL IMPORTERS" :
              signals?.brent_crude?.delta_day_pct < -2 ? "BUY OIL IMPORTERS" : "HOLD",
      rationale: "Oil price movement drives downstream asset reactions",
      confidence: 72,
    },
    {
      name: "BTC / USDT",
      change: signals?.crypto?.btc?.change_24h_pct,
      signal: signals?.crypto?.btc?.change_24h_pct < -3 ? "WATCH SUPPORT" :
              signals?.crypto?.btc?.change_24h_pct > 3 ? "MOMENTUM UP" : "NEUTRAL",
      rationale: "Risk-off sentiment during geopolitical shocks affects crypto",
      confidence: 58,
    },
    {
      name: "ETH / USDT",
      change: signals?.crypto?.eth?.change_24h_pct,
      signal: signals?.crypto?.eth?.change_24h_pct < -3 ? "WATCH SUPPORT" :
              signals?.crypto?.eth?.change_24h_pct > 3 ? "MOMENTUM UP" : "NEUTRAL",
      rationale: "Follows BTC with higher beta during volatility",
      confidence: 54,
    },
    {
      name: "USD / VND",
      change: signals?.usd_vnd?.change_7day_pct,
      signal: signals?.usd_vnd?.change_7day_pct > 0.5 ? "VND WEAKENING" :
              signals?.usd_vnd?.change_7day_pct < -0.5 ? "VND STRENGTHENING" : "STABLE",
      rationale: "Import cost pressure from oil shock affects VND",
      confidence: 81,
    },
  ];

  const signalColor = (s) =>
    s.includes("BUY") || s.includes("UP") || s.includes("STRENGTHENING") ? "#30d158" :
    s.includes("SELL") || s.includes("WEAKENING") || s.includes("SUPPORT") ? "#ff3b30" : "#888";

  return (
    <div className="tab-content">
      <div className="signals-note">
        ⚠ Signals are model-generated indicators for research purposes only.
        Not financial advice. ML model pending training.
      </div>

      <div className="signals-table">
        <div className="signals-header">
          <span>ASSET</span>
          <span>24H CHANGE</span>
          <span>SIGNAL</span>
          <span>CONFIDENCE</span>
          <span>RATIONALE</span>
        </div>
        {assets.map((a, i) => (
          <div key={i} className="signals-row">
            <span className="sig-name">{a.name}</span>
            <span><ChangeTag n={a.change} /></span>
            <span className="sig-signal" style={{ color: signalColor(a.signal) }}>
              {a.signal}
            </span>
            <span className="sig-conf">
              <div className="conf-bar">
                <div className="conf-fill" style={{ width: `${a.confidence}%` }} />
              </div>
              <span>{a.confidence}%</span>
            </span>
            <span className="sig-rat muted">{a.rationale}</span>
          </div>
        ))}
      </div>

      {shockData && (
        <div className="shock-detail">
          <div className="chart-label">SHOCK SIGNAL COMPONENTS</div>
          <div className="shock-components">
            <div className="shock-comp">
              <span className="muted">Oil delta</span>
              <span>{fmt(shockData.oil_delta_pct, 2)}%</span>
            </div>
            <div className="shock-comp">
              <span className="muted">Headline score</span>
              <span>{fmt(shockData.headline_score, 4)}</span>
            </div>
            <div className="shock-comp">
              <span className="muted">Grid imbalance</span>
              <span>£{fmt(shockData.imbalance_price_gbp_mwh, 2)}/MWh</span>
            </div>
            <div className="shock-comp">
              <span className="muted">Alert level</span>
              <span className={shockData.alert_level === "HIGH" ? "neg" :
                              shockData.alert_level === "ELEVATED" ? "warn" : "pos"}>
                {shockData.alert_level}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("macro");
  const [signals, setSignals] = useState(null);
  const [shockData, setShockData] = useState(null);
  const [connected, setConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);
  const [clientCount, setClientCount] = useState(0);
  const [vndHistory, setVndHistory] = useState([]);
  const [btcHistory, setBtcHistory] = useState([]);
  const wsRef = useRef(null);

  // ── WebSocket connection ─────────────────────────────────────────────────
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        console.log("WebSocket connected");
      };

      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "signals_update") {
          setSignals(msg.signals);
          setClientCount(msg.client_count);
          setLastUpdate(new Date(msg.timestamp));
        }
      };

      ws.onclose = () => {
        setConnected(false);
        // Reconnect after 3 seconds
        setTimeout(connect, 3000);
      };

      ws.onerror = () => ws.close();
    };

    connect();
    return () => wsRef.current?.close();
  }, []);

  // ── Fetch shock status ───────────────────────────────────────────────────
  useEffect(() => {
    const fetchShock = async () => {
      try {
        const res = await fetch(`${API_URL}/shock/status`);
        const data = await res.json();
        setShockData(data);
      } catch (e) {
        console.error("Shock fetch failed:", e);
      }
    };
    fetchShock();
    const interval = setInterval(fetchShock, 60000);
    return () => clearInterval(interval);
  }, []);

  // ── Fetch VND history from Supabase via API ──────────────────────────────
  useEffect(() => {
    // Simulate history from current signal changes for demo
    // In production: fetch from GET /vnd/history
    if (signals?.usd_vnd) {
      const rate = signals.usd_vnd.usd_to_vnd;
      setVndHistory(prev => {
        const now = new Date().toLocaleTimeString("en", { hour: "2-digit", minute: "2-digit" });
        const next = [...prev, { date: now, rate }].slice(-20);
        return next;
      });
    }
  }, [signals]);

  // ── Fetch BTC history ────────────────────────────────────────────────────
  useEffect(() => {
    const fetchBtcHistory = async () => {
      try {
        // Binance klines — last 7 daily candles
        const res = await fetch(
          "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=7"
        );
        const data = await res.json();
        const formatted = data.map(c => ({
          date: new Date(c[0]).toLocaleDateString("en", { month: "short", day: "numeric" }),
          price: parseFloat(c[4]),
        }));
        setBtcHistory(formatted);
      } catch (e) {
        console.error("BTC history fetch failed:", e);
      }
    };
    fetchBtcHistory();
  }, []);

  const shockLevel = shockData?.alert_level || "NORMAL";
  const shockScore = shockData?.shock_score || 0;
  const shockMessage = shockData?.message || "No significant disruption detected";

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <span className="logo">VIETNAM MARKET INTELLIGENCE</span>
          <span className="logo-sub">Real-time · Macro · Crypto · Signals</span>
        </div>
        <div className="header-right">
          <span className={`conn-dot ${connected ? "connected" : "disconnected"}`} />
          <span className="conn-label">{connected ? "LIVE" : "RECONNECTING"}</span>
          {lastUpdate && (
            <span className="last-update">
              Updated {lastUpdate.toLocaleTimeString()}
            </span>
          )}
          <span className="client-count">{clientCount} viewer{clientCount !== 1 ? "s" : ""}</span>
        </div>
      </header>

      {/* ── Shock bar ── */}
      <ShockBar score={shockScore} level={shockLevel} message={shockMessage} />

      {/* ── Ticker strip ── */}
      <TickerStrip signals={signals} />

      {/* ── Tabs ── */}
      <div className="tabs">
        {["macro", "crypto", "signals"].map(t => (
          <button
            key={t}
            className={`tab-btn ${tab === t ? "active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t === "macro" ? "MACRO" : t === "crypto" ? "CRYPTO" : "SIGNALS"}
          </button>
        ))}
      </div>

      {/* ── Tab content ── */}
      <main className="main">
        {tab === "macro" && (
          <MacroTab signals={signals} history={vndHistory} />
        )}
        {tab === "crypto" && (
          <CryptoTab signals={signals} priceHistory={{ btc: btcHistory }} />
        )}
        {tab === "signals" && (
          <SignalsTab signals={signals} shockData={shockData} />
        )}
      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        <span>Sources: EIA · ExchangeRate-API · Binance · GNews</span>
        <span>For research purposes only · Not financial advice</span>
      </footer>
    </div>
  );
}
