import { useState, useEffect, useRef, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { createChart, CandlestickSeries } from "lightweight-charts";
import MLTab from "./MLTab";
import AITab from "./AITab";
import TradingTab from "./TradingTab";

const WS_URL  = "ws://localhost:8080/ws/live";
const API_URL = "http://localhost:8080";

// ── Utilities ─────────────────────────────────────────────────────────────────
const fmt = (n, decimals = 2) =>
  n == null ? "—" : Number(n).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

const fmtVND = (n) =>
  n == null ? "—" : Number(n).toLocaleString("en-US", { maximumFractionDigits: 0 });

const change = (n) => {
  if (n == null) return null;
  return { value: fmt(Math.abs(n), 2), pos: n >= 0 };
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
    level === "HIGH"     ? "#ff3b30" :
    level === "ELEVATED" ? "#ff9f0a" : "#30d158";

  return (
    <div className="shock-bar">
      <div className="shock-left">
        <span className="shock-label" style={{ color }}>⬤ {level || "NORMAL"}</span>
        <span className="shock-message">{message || "No significant disruption detected"}</span>
      </div>
      <div className="shock-meter">
        <div className="shock-track">
          <div className="shock-fill" style={{ width: `${Math.round((score || 0) * 100)}%`, background: color }} />
        </div>
        <span className="shock-score">{fmt(score, 4)}</span>
      </div>
    </div>
  );
};

// ── Ticker Strip ──────────────────────────────────────────────────────────────
const TickerStrip = ({ signals }) => {
  if (!signals) return null;
  const { brent_crude, crypto } = signals;

  const items = [
    { label: "BRENT",    value: `$${fmt(brent_crude?.price_usd)}`,                change: brent_crude?.delta_day_pct },
    { label: "BTC/USDT", value: `$${fmt(crypto?.btc?.price_usd, 0)}`,             change: crypto?.btc?.change_24h_pct },
    { label: "ETH/USDT", value: `$${fmt(crypto?.eth?.price_usd)}`,                change: crypto?.eth?.change_24h_pct },
    { label: "BTC VOL",  value: `$${fmt(crypto?.btc?.volume_24h_usd / 1e9, 2)}B`, change: null },
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
  const { brent_crude } = signals;

  return (
    <div className="tab-content">
      <div className="card-grid">
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

      {history.length > 1 && (
        <div className="chart-section">
          <div className="chart-label">USD/VND — session history</div>
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
const SELECT_STYLE = {
  background: "#1a1a1a",
  border: "1px solid #333",
  color: "#fff",
  padding: "4px 8px",
  borderRadius: 4,
  fontFamily: "var(--mono)",
  fontSize: 12,
};

const CryptoTab = ({ signals }) => {
  // ALL hooks before any conditional return
  const [cryptoHistory,    setCryptoHistory]    = useState([]);
  const [selectedInterval, setSelectedInterval] = useState("1H");
  const [selectedSymbol,   setSelectedSymbol]   = useState("BTC");
  const [loading,          setLoading]           = useState(false);

  const chartRef = useRef(null);

  // Format candles for lightweight-charts — no date filtering, chart handles zoom natively
  const chartData = useMemo(() => {
    return cryptoHistory
      .map(c => ({
        time:  Math.floor(new Date(c.timestamp).getTime() / 1000),
        open:  parseFloat(c.open)  || 0,
        high:  parseFloat(c.high)  || 0,
        low:   parseFloat(c.low)   || 0,
        close: parseFloat(c.close) || 0,
      }))
      .filter(c => c.open > 0)
      .sort((a, b) => a.time - b.time);
  }, [cryptoHistory]);

  // FETCH — re-runs when symbol or interval changes
  useEffect(() => {
    let cancelled = false;
    const fetchHistory = async () => {
      setLoading(true);
      try {
        const res  = await fetch(
          `${API_URL}/crypto/history?symbol=${selectedSymbol}&interval=${selectedInterval}&days=365`
        );
        const data = await res.json();
        if (!cancelled) setCryptoHistory(data.candles || []);
      } catch (e) {
        console.error("Crypto history fetch failed:", e);
      }
      if (!cancelled) setLoading(false);
    };
    fetchHistory();
    return () => { cancelled = true; };
  }, [selectedInterval, selectedSymbol]);

  // CHART — initialises lightweight-charts when data arrives
  // lightweight-charts handles scroll/zoom/pan natively at 60fps — no React state needed
  useEffect(() => {
    if (!chartRef.current || chartData.length === 0) return;

    const container = chartRef.current;
    let chart = null;

    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect?.width;
      if (!width || chart) return;
      observer.disconnect();

      try {
        chart = createChart(container, {
          layout:          { background: { color: "#0d0d0d" }, textColor: "#888", fontSize: 12 },
          width,
          height:          400,
          timeScale:       {
            timeVisible:    true,
            secondsVisible: false,
            rightOffset:    5,
            barSpacing:     6,
            minBarSpacing:  2,
          },
          rightPriceScale: { scaleMargins: { top: 0.1, bottom: 0.1 } },
          crosshair:       { mode: 1 },
          // Native scroll/zoom — silky smooth, no React re-renders
          handleScroll:    { mouseWheel: true, pressedMouseMove: true },
          handleScale:     { mouseWheel: true, pinch: true },
        });

        const series = chart.addSeries(CandlestickSeries, {
          upColor:         "#26a69a",
          downColor:       "#ef5350",
          borderUpColor:   "#26a69a",
          borderDownColor: "#ef5350",
          wickUpColor:     "#26a69a",
          wickDownColor:   "#ef5350",
        });

        series.setData(chartData);
        chart.timeScale().fitContent();

        const onResize = () => {
          if (container.clientWidth > 0)
            chart.applyOptions({ width: container.clientWidth });
        };
        window.addEventListener("resize", onResize);
        container._chartCleanup = () => {
          window.removeEventListener("resize", onResize);
          try { chart.remove(); } catch (_) {}
        };
      } catch (e) {
        console.error("Chart init error:", e);
      }
    });

    observer.observe(container);

    return () => {
      observer.disconnect();
      container._chartCleanup?.();
      container._chartCleanup = null;
    };
  }, [chartData]);

  // Conditional return AFTER all hooks
  if (!signals?.crypto) return <div className="loading">Awaiting crypto stream...</div>;
  const { btc, eth, usdt_to_vnd } = signals.crypto;
  const vndRate = usdt_to_vnd ?? signals?.usd_vnd?.usd_to_vnd;

  return (
    <div className="tab-content">

      {/* Controls */}
      <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 20, flexWrap: "wrap" }}>
        <div>
          <label style={{ fontSize: 11, color: "#666", marginRight: 8, letterSpacing: "0.08em" }}>SYMBOL</label>
          <select value={selectedSymbol} onChange={e => setSelectedSymbol(e.target.value)} style={SELECT_STYLE}>
            <option value="BTC">BTC</option>
            <option value="ETH">ETH</option>
          </select>
        </div>
        <div>
          <label style={{ fontSize: 11, color: "#666", marginRight: 8, letterSpacing: "0.08em" }}>INTERVAL</label>
          <select value={selectedInterval} onChange={e => setSelectedInterval(e.target.value)} style={SELECT_STYLE}>
            <option value="30m">30m</option>
            <option value="1H">1H</option>
            <option value="4H">4H</option>
            <option value="1D">1D</option>
            <option value="1W">1W</option>
            <option value="1M">1M</option>
          </select>
        </div>
        <span style={{ color: "#444", fontSize: 10, marginLeft: "auto", letterSpacing: "0.06em" }}>
          SCROLL TO ZOOM · DRAG TO PAN
        </span>
      </div>

      {/* Price cards */}
      <div className="card-grid">
        <div className="card card-accent-orange">
          <div className="card-header">
            <span className="card-label">BITCOIN</span>
            <span className="card-sub">BTC / USDT · Binance</span>
          </div>
          <div className="card-value">${fmt(btc?.price_usd, 0)}</div>
          <div className="card-meta">
            <ChangeTag n={btc?.change_24h_pct} /> <span className="muted">24h</span>
          </div>
          <div className="card-note">≈ {fmtVND(btc?.price_usd * vndRate)} VND</div>
          <div className="card-meta" style={{ marginTop: 8 }}>
            <span className="muted">Vol 24h</span>&nbsp;
            <span>${fmt(btc?.volume_24h_usd / 1e9, 2)}B</span>
          </div>
        </div>

        <div className="card card-accent-blue">
          <div className="card-header">
            <span className="card-label">ETHEREUM</span>
            <span className="card-sub">ETH / USDT · Binance</span>
          </div>
          <div className="card-value">${fmt(eth?.price_usd)}</div>
          <div className="card-meta">
            <ChangeTag n={eth?.change_24h_pct} /> <span className="muted">24h</span>
          </div>
          <div className="card-note">≈ {fmtVND(eth?.price_usd * vndRate)} VND</div>
          <div className="card-meta" style={{ marginTop: 8 }}>
            <span className="muted">Vol 24h</span>&nbsp;
            <span>${fmt(eth?.volume_24h_usd / 1e9, 2)}B</span>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-label">USDT / VND</span>
            <span className="card-sub">Derived from USD/VND</span>
          </div>
          <div className="card-value">{fmtVND(vndRate)}</div>
          <div className="card-note">1 USDT ≈ 1 USD in VND terms</div>
        </div>
      </div>

      {/* Candlestick chart */}
      <div className="chart-section">
        <div className="chart-label">
          {selectedSymbol}/USDT — {selectedInterval} · {chartData.length} candles
          {loading && <span style={{ marginLeft: 8, color: "#444" }}>Loading...</span>}
        </div>
        <div
          ref={chartRef}
          style={{
            width:        "100%",
            height:       chartData.length > 0 ? 400 : 160,
            background:   "#0d0d0d",
            borderRadius: 3,
            border:       "1px solid #1e1e1e",
            position:     "relative",
          }}
        >
          {!chartData.length && (
            <div style={{
              position: "absolute", inset: 0,
              display: "flex", alignItems: "center", justifyContent: "center",
              color: "#333", fontSize: 11, letterSpacing: "0.08em",
            }}>
              {loading ? "LOADING..." : "NO DATA — check /crypto/history endpoint"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ── Signals Tab ───────────────────────────────────────────────────────────────
const SignalsTab = ({ signals, shockData }) => {
  const assets = [
    {
      name:       "BRENT CRUDE",
      change:     signals?.brent_crude?.delta_day_pct,
      signal:     signals?.brent_crude?.delta_day_pct > 2  ? "SELL OIL IMPORTERS" :
                  signals?.brent_crude?.delta_day_pct < -2 ? "BUY OIL IMPORTERS"  : "HOLD",
      rationale:  "Oil price movement drives downstream asset reactions",
      confidence: 72,
    },
    {
      name:       "BTC / USDT",
      change:     signals?.crypto?.btc?.change_24h_pct,
      signal:     signals?.crypto?.btc?.change_24h_pct < -3 ? "WATCH SUPPORT" :
                  signals?.crypto?.btc?.change_24h_pct > 3  ? "MOMENTUM UP"   : "NEUTRAL",
      rationale:  "Risk-off sentiment during geopolitical shocks affects crypto",
      confidence: 58,
    },
    {
      name:       "ETH / USDT",
      change:     signals?.crypto?.eth?.change_24h_pct,
      signal:     signals?.crypto?.eth?.change_24h_pct < -3 ? "WATCH SUPPORT" :
                  signals?.crypto?.eth?.change_24h_pct > 3  ? "MOMENTUM UP"   : "NEUTRAL",
      rationale:  "Follows BTC with higher beta during volatility",
      confidence: 54,
    },
    {
      name:       "USD / VND",
      change:     signals?.usd_vnd?.change_7day_pct,
      signal:     signals?.usd_vnd?.change_7day_pct > 0.5  ? "VND WEAKENING"     :
                  signals?.usd_vnd?.change_7day_pct < -0.5 ? "VND STRENGTHENING" : "STABLE",
      rationale:  "Import cost pressure from oil shock affects VND",
      confidence: 81,
    },
  ];

  const signalColor = (s) =>
    s.includes("BUY")  || s.includes("UP")   || s.includes("STRENGTHENING") ? "#30d158" :
    s.includes("SELL") || s.includes("WEAK") || s.includes("SUPPORT")       ? "#ff3b30" : "#888";

  return (
    <div className="tab-content">
      <div className="signals-note">
        ⚠ Signals are model-generated indicators for research purposes only. Not financial advice.
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
            <span className="sig-signal" style={{ color: signalColor(a.signal) }}>{a.signal}</span>
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
              <span className="muted">Alert level</span>
              <span className={
                shockData.alert_level === "HIGH"     ? "neg"  :
                shockData.alert_level === "ELEVATED" ? "warn" : "pos"
              }>
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
  const [tab,           setTab]           = useState("macro");
  const [signals,       setSignals]       = useState(null);
  const [shockData,     setShockData]     = useState(null);
  const [connected,     setConnected]     = useState(false);
  const [lastUpdate,    setLastUpdate]    = useState(null);
  const [clientCount,   setClientCount]   = useState(0);
  const [vndHistory,    setVndHistory]    = useState([]);
  const [mlPredictions, setMlPredictions] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen  = () => setConnected(true);
      ws.onclose = () => { setConnected(false); setTimeout(connect, 3000); };
      ws.onerror = () => ws.close();
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "signals_update") {
          setSignals(msg.signals);
          setClientCount(msg.client_count);
          setLastUpdate(new Date(msg.timestamp));
          if (msg.ml_predictions) setMlPredictions(msg.ml_predictions);
        }
      };
    };
    connect();
    return () => wsRef.current?.close();
  }, []);

  useEffect(() => {
    const fetchShock = async () => {
      try {
        const res  = await fetch(`${API_URL}/shock/status`);
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

  useEffect(() => {
    if (!signals?.usd_vnd) return;
    const rate = signals.usd_vnd.usd_to_vnd;
    setVndHistory(prev => {
      const now = new Date().toLocaleTimeString("en", { hour: "2-digit", minute: "2-digit" });
      return [...prev, { date: now, rate }].slice(-20);
    });
  }, [signals]);

  const shockLevel   = shockData?.alert_level || "NORMAL";
  const shockScore   = shockData?.shock_score  || 0;
  const shockMessage = shockData?.message       || "No significant disruption detected";

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <span className="logo">VIETNAM MARKET INTELLIGENCE</span>
          <span className="logo-sub">Real-time · Macro · Crypto · Signals · ML</span>
        </div>
        <div className="header-right">
          <span className={`conn-dot ${connected ? "connected" : "disconnected"}`} />
          <span className="conn-label">{connected ? "LIVE" : "RECONNECTING"}</span>
          {lastUpdate && <span className="last-update">Updated {lastUpdate.toLocaleTimeString()}</span>}
          <span className="client-count">{clientCount} viewer{clientCount !== 1 ? "s" : ""}</span>
        </div>
      </header>

      <ShockBar score={shockScore} level={shockLevel} message={shockMessage} />
      <TickerStrip signals={signals} />

      <div className="tabs">
        {["macro", "crypto", "signals", "ml", "ai", "trading"].map(t => (
          <button
            key={t}
            className={`tab-btn ${tab === t ? "active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t === "macro" ? "MACRO" : t === "crypto" ? "CRYPTO" : t === "signals" ? "SIGNALS" : t === "ml" ? "ML MODEL" : t === "ai" ? "AI ANALYST" : "TRADING"}
          </button>
        ))}
      </div>

      <main className="main">
        {tab === "macro"   && <MacroTab   signals={signals} history={vndHistory} />}
        {tab === "crypto"  && <CryptoTab  signals={signals} />}
        {tab === "signals" && <SignalsTab  signals={signals} shockData={shockData} />}
        {tab === "ml"      && <MLTab      mlPredictions={mlPredictions} />}
        {tab === "ai"      && <AITab />}
        {tab === "trading" && <TradingTab livePrices={{
          BTC: signals?.crypto?.btc?.price_usd,
          ETH: signals?.crypto?.eth?.price_usd,
        }} />}
      </main>

      <footer className="footer">
        <span>Sources: EIA · ExchangeRate-API · Binance · GNews</span>
        <span>For research purposes only · Not financial advice</span>
      </footer>
    </div>
  );
}
