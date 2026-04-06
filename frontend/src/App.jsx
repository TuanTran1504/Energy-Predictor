import { useState, useEffect, useRef, useMemo } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { createChart, CandlestickSeries } from "lightweight-charts";
import MLTab from "./MLTab";
import AITab from "./AITab";
import TradingTab from "./TradingTab";
import LiveTradingTab from "./LiveTradingTab";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8080";
const WS_URL  = import.meta.env.VITE_WS_URL ||
  `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/live`;

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
const fgColor = (v) =>
  v <= 25 ? "#ff3b30" : v <= 45 ? "#ff9f0a" : v <= 55 ? "#aeaeb2" : v <= 75 ? "#30d158" : "#00ff88";

const eventTypeLabel = (t) =>
  t === "FED_RATE" ? "FED RATE" : t === "CPI" ? "CPI" : t === "NFP" ? "NFP" : t;

const eventTypeColor = (t) =>
  t === "FED_RATE" ? "#0a84ff" : t === "CPI" ? "#ff9f0a" : t === "NFP" ? "#30d158" : "#aeaeb2";

const MacroTab = ({ signals }) => {
  const [macro, setMacro] = useState(null);
  const [macroLoading, setMacroLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${API_URL}/macro/indicators`);
        setMacro(await res.json());
      } catch (e) {
        console.error("Macro fetch failed:", e);
      } finally {
        setMacroLoading(false);
      }
    };
    load();
    const id = setInterval(load, 5 * 60 * 1000);
    return () => clearInterval(id);
  }, []);

  const { brent_crude } = signals || {};
  const fg = macro?.fear_greed;
  const fgVal = fg?.value ?? 0;

  return (
    <div className="tab-content">

      {/* ── Key indicators ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, marginBottom: 20 }}>

        {/* Brent Crude */}
        <div className="card" style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: 10, color: "var(--muted2)", letterSpacing: "0.08em", marginBottom: 4 }}>BRENT CRUDE</div>
          <div style={{ fontSize: 22, fontWeight: 300, fontFamily: "var(--mono)" }}>${fmt(brent_crude?.price_usd)}</div>
          <div style={{ fontSize: 11, marginTop: 4 }}><ChangeTag n={brent_crude?.delta_day_pct} /> <span className="muted">24h</span></div>
        </div>

        {/* Fed Rate */}
        <div className="card" style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: 10, color: "var(--muted2)", letterSpacing: "0.08em", marginBottom: 4 }}>FED RATE</div>
          {macroLoading ? <div style={{ color: "var(--muted2)", fontSize: 12 }}>Loading...</div> : macro?.fed_rate ? (
            <>
              <div style={{ fontSize: 22, fontWeight: 300, fontFamily: "var(--mono)", color: "#0a84ff" }}>
                {macro.fed_rate.actual}
              </div>
              <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>{macro.fed_rate.event_date}</div>
              <div style={{ fontSize: 10, color: "var(--muted2)", marginTop: 2 }}>{macro.fed_rate.description}</div>
            </>
          ) : <div style={{ color: "var(--muted2)", fontSize: 11 }}>No data</div>}
        </div>

        {/* CPI */}
        <div className="card" style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: 10, color: "var(--muted2)", letterSpacing: "0.08em", marginBottom: 4 }}>CPI (MoM)</div>
          {macroLoading ? <div style={{ color: "var(--muted2)", fontSize: 12 }}>Loading...</div> : macro?.cpi ? (
            <>
              <div style={{ fontSize: 22, fontWeight: 300, fontFamily: "var(--mono)", color: macro.cpi.actual?.startsWith("+") ? "#ff9f0a" : "#30d158" }}>
                {macro.cpi.actual}
              </div>
              <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>{macro.cpi.event_date}</div>
              <div style={{ fontSize: 10, color: "var(--muted2)", marginTop: 2 }}>US Consumer Price Index</div>
            </>
          ) : <div style={{ color: "var(--muted2)", fontSize: 11 }}>No data</div>}
        </div>

        {/* NFP */}
        <div className="card" style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: 10, color: "var(--muted2)", letterSpacing: "0.08em", marginBottom: 4 }}>NFP JOBS</div>
          {macroLoading ? <div style={{ color: "var(--muted2)", fontSize: 12 }}>Loading...</div> : macro?.nfp ? (
            <>
              <div style={{ fontSize: 22, fontWeight: 300, fontFamily: "var(--mono)", color: macro.nfp.actual?.startsWith("+") ? "#30d158" : "#ff3b30" }}>
                {macro.nfp.actual}
              </div>
              <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>{macro.nfp.event_date}</div>
              <div style={{ fontSize: 10, color: "var(--muted2)", marginTop: 2 }}>Non-Farm Payrolls</div>
            </>
          ) : <div style={{ color: "var(--muted2)", fontSize: 11 }}>No data</div>}
        </div>

        {/* Fear & Greed */}
        <div className="card" style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: 10, color: "var(--muted2)", letterSpacing: "0.08em", marginBottom: 4 }}>FEAR & GREED</div>
          {macroLoading ? <div style={{ color: "var(--muted2)", fontSize: 12 }}>Loading...</div> : fg ? (
            <>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                <div style={{ fontSize: 32, fontWeight: 300, fontFamily: "var(--mono)", color: fgColor(fgVal) }}>{fgVal}</div>
                <div style={{ fontSize: 11, color: fgColor(fgVal) }}>{fg.classification}</div>
              </div>
              <div style={{ marginTop: 8, height: 6, background: "var(--border)", borderRadius: 3 }}>
                <div style={{ height: "100%", width: `${fgVal}%`, background: fgColor(fgVal), borderRadius: 3, transition: "width 0.5s" }} />
              </div>
              <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>{fg.date}</div>
            </>
          ) : <div style={{ color: "var(--muted2)", fontSize: 11 }}>No data</div>}
        </div>

        {/* BTC Funding Rate */}
        <div className="card" style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: 10, color: "var(--muted2)", letterSpacing: "0.08em", marginBottom: 4 }}>BTC FUNDING RATE</div>
          {macroLoading ? <div style={{ color: "var(--muted2)", fontSize: 12 }}>Loading...</div> : macro?.funding_btc ? (
            <>
              <div style={{ fontSize: 22, fontWeight: 300, fontFamily: "var(--mono)", color: macro.funding_btc.rate_avg > 0 ? "#30d158" : "#ff3b30" }}>
                {(macro.funding_btc.rate_avg * 100).toFixed(4)}%
              </div>
              <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>{macro.funding_btc.date} · 8h avg</div>
              <div style={{ fontSize: 10, color: "var(--muted2)", marginTop: 2 }}>
                {macro.funding_btc.rate_avg > 0.0001 ? "Longs paying shorts" : macro.funding_btc.rate_avg < -0.0001 ? "Shorts paying longs" : "Neutral"}
              </div>
            </>
          ) : <div style={{ color: "var(--muted2)", fontSize: 11 }}>No data</div>}
        </div>

        {/* ETH Funding Rate */}
        <div className="card" style={{ padding: "12px 16px" }}>
          <div style={{ fontSize: 10, color: "var(--muted2)", letterSpacing: "0.08em", marginBottom: 4 }}>ETH FUNDING RATE</div>
          {macroLoading ? <div style={{ color: "var(--muted2)", fontSize: 12 }}>Loading...</div> : macro?.funding_eth ? (
            <>
              <div style={{ fontSize: 22, fontWeight: 300, fontFamily: "var(--mono)", color: macro.funding_eth.rate_avg > 0 ? "#30d158" : "#ff3b30" }}>
                {(macro.funding_eth.rate_avg * 100).toFixed(4)}%
              </div>
              <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>{macro.funding_eth.date} · 8h avg</div>
              <div style={{ fontSize: 10, color: "var(--muted2)", marginTop: 2 }}>
                {macro.funding_eth.rate_avg > 0.0001 ? "Longs paying shorts" : macro.funding_eth.rate_avg < -0.0001 ? "Shorts paying longs" : "Neutral"}
              </div>
            </>
          ) : <div style={{ color: "var(--muted2)", fontSize: 11 }}>No data</div>}
        </div>
      </div>

      {/* ── Fear & Greed 30-day chart ── */}
      {macro?.fear_greed_30d?.length > 1 && (
        <div className="chart-section" style={{ marginBottom: 20 }}>
          <div className="chart-label">FEAR & GREED — 30 DAYS</div>
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={[...macro.fear_greed_30d].reverse()}>
              <XAxis dataKey="date" tick={{ fill: "#444", fontSize: 9 }} tickLine={false} axisLine={false}
                tickFormatter={d => d.slice(5)} interval="preserveStartEnd" />
              <YAxis domain={[0, 100]} tick={{ fill: "#444", fontSize: 9 }} tickLine={false} axisLine={false} width={28} />
              <Tooltip
                contentStyle={{ background: "#0d0d0d", border: "1px solid #333", borderRadius: 4 }}
                labelStyle={{ color: "#888", fontSize: 10 }}
                itemStyle={{ color: "#aeaeb2", fontSize: 10 }}
                formatter={(v, _, p) => [`${v} — ${p.payload.classification}`, "Fear & Greed"]}
              />
              <Line type="monotone" dataKey="value" stroke="#0a84ff" strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Recent macro events timeline ── */}
      {!macroLoading && macro?.recent_events?.length > 0 && (
        <div>
          <div className="chart-label" style={{ marginBottom: 10 }}>MACRO EVENTS TIMELINE</div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--mono)", fontSize: 11 }}>
              <thead>
                <tr style={{ color: "var(--muted2)", borderBottom: "1px solid var(--border)" }}>
                  {["DATE", "TYPE", "EVENT", "ACTUAL", "BTC 24H", "ETH 24H"].map(h => (
                    <th key={h} style={{ padding: "6px 8px", textAlign: "left", fontWeight: 400, letterSpacing: "0.06em" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {macro.recent_events.map((ev, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
                    <td style={{ padding: "6px 8px", color: "var(--muted)" }}>{ev.event_date}</td>
                    <td style={{ padding: "6px 8px" }}>
                      <span style={{
                        padding: "2px 6px", borderRadius: 3, fontSize: 10,
                        background: eventTypeColor(ev.event_type) + "22",
                        color: eventTypeColor(ev.event_type),
                      }}>{eventTypeLabel(ev.event_type)}</span>
                    </td>
                    <td style={{ padding: "6px 8px", color: "var(--text)", maxWidth: 280 }}>{ev.description}</td>
                    <td style={{ padding: "6px 8px", fontWeight: 600, color: "var(--text)" }}>{ev.actual || "—"}</td>
                    <td style={{ padding: "6px 8px", color: ev.btc_impact_24h == null ? "var(--muted2)" : ev.btc_impact_24h >= 0 ? "#30d158" : "#ff3b30" }}>
                      {ev.btc_impact_24h != null ? `${ev.btc_impact_24h >= 0 ? "+" : ""}${ev.btc_impact_24h.toFixed(2)}%` : "—"}
                    </td>
                    <td style={{ padding: "6px 8px", color: ev.eth_impact_24h == null ? "var(--muted2)" : ev.eth_impact_24h >= 0 ? "#30d158" : "#ff3b30" }}>
                      {ev.eth_impact_24h != null ? `${ev.eth_impact_24h >= 0 ? "+" : ""}${ev.eth_impact_24h.toFixed(2)}%` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: 10, color: "var(--muted2)", marginTop: 8 }}>
            Sources: FRED (Federal Reserve) · BTC/ETH impact measured 24h after event
            · Updated daily at 06:00 UTC
          </div>
        </div>
      )}

      {macroLoading && (
        <div style={{ color: "var(--muted2)", fontSize: 11, padding: "20px 0" }}>Loading macro data...</div>
      )}
      {!macroLoading && !macro?.recent_events?.length && (
        <div className="card" style={{ color: "var(--muted2)", fontSize: 11, padding: 16 }}>
          No macro events yet — scheduler will populate data on first run.
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


  const shockLevel   = shockData?.alert_level || "NORMAL";
  const shockScore   = shockData?.shock_score  || 0;
  const shockMessage = shockData?.message       || "No significant disruption detected";

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <span className="logo">CRYPTO AGENT</span>
          <span className="logo-sub">Real-time · Macro · Crypto · Signals · ML · AI Trading</span>
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
        {["macro", "crypto", "signals", "ml", "ai", "trading", "live"].map(t => (
          <button
            key={t}
            className={`tab-btn ${tab === t ? "active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t === "macro" ? "MACRO" : t === "crypto" ? "CRYPTO" : t === "signals" ? "SIGNALS" : t === "ml" ? "ML MODEL" : t === "ai" ? "AI ANALYST" : t === "trading" ? "TRADING" : "LIVE"}
          </button>
        ))}
      </div>

      <main className="main">
        {tab === "macro"   && <MacroTab   signals={signals} />}
        {tab === "crypto"  && <CryptoTab  signals={signals} />}
        {tab === "signals" && <SignalsTab  signals={signals} shockData={shockData} />}
        {tab === "ml"      && <MLTab      mlPredictions={mlPredictions} />}
        {tab === "ai"      && <AITab />}
        {tab === "trading" && <TradingTab livePrices={{
          BTC: signals?.crypto?.btc?.price_usd,
          ETH: signals?.crypto?.eth?.price_usd,
          SOL: signals?.crypto?.sol?.price_usd,
          XRP: signals?.crypto?.xrp?.price_usd,
        }} />}
        {tab === "live" && <LiveTradingTab livePrices={{
          BTC: signals?.crypto?.btc?.price_usd,
          ETH: signals?.crypto?.eth?.price_usd,
          SOL: signals?.crypto?.sol?.price_usd,
          XRP: signals?.crypto?.xrp?.price_usd,
        }} />}
      </main>

      <footer className="footer">
        <span>Sources: EIA · ExchangeRate-API · Binance · GNews</span>
        <span>For research purposes only · Not financial advice</span>
      </footer>
    </div>
  );
}
