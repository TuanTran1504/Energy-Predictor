import { useState, useEffect, useCallback } from "react";
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const API_URL = "http://localhost:8080";

const fmt = (n, d = 2) =>
  n == null ? "—" : Number(n).toLocaleString("en-US", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });

const pct = (n) => (n == null ? "—" : `${(n * 100).toFixed(1)}%`);

// ── Live prediction card ───────────────────────────────────────────────────────
const PredictionCard = ({ symbol, label = "24H FORECAST", prediction }) => {
  if (!prediction) {
    return (
      <div className="card" style={{ opacity: 0.5 }}>
        <div className="card-header">
          <span className="card-label">{symbol} · {label}</span>
        </div>
        <div className="loading" style={{ padding: "20px 0" }}>
          Awaiting prediction...
        </div>
      </div>
    );
  }

  const isUp      = prediction.direction === "UP";
  const dirColor  = isUp ? "#30d158" : "#ff3b30";
  const confPct   = (prediction.confidence * 100).toFixed(1);
  const accentClass = symbol === "BTC" ? "card-accent-orange" : "card-accent-blue";
  const accentColor = symbol === "BTC" ? "#ff9f0a" : "#0a84ff";

  return (
    <div className={`card ${accentClass}`}>
      <div className="card-header">
        <span className="card-label">{symbol} / USDT · {label}</span>
        <span className="card-sub">XGBoost · updated {new Date(prediction.predicted_at).toLocaleTimeString()}</span>
      </div>

      {/* Direction */}
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 10 }}>
        <span style={{ fontSize: 32, fontWeight: 300, color: dirColor, fontFamily: "var(--mono)" }}>
          {isUp ? "▲" : "▼"} {prediction.direction}
        </span>
      </div>

      {/* UP probability bar */}
      <div style={{ marginBottom: 6 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 10, color: "var(--muted2)" }}>
          <span>UP {pct(prediction.up_prob)}</span>
          <span>DOWN {pct(prediction.down_prob)}</span>
        </div>
        <div style={{ height: 4, background: "var(--border2)", borderRadius: 2, overflow: "hidden" }}>
          <div style={{
            height: "100%",
            width: pct(prediction.up_prob),
            background: "#30d158",
            borderRadius: 2,
            transition: "width 0.4s ease",
          }} />
        </div>
      </div>

      {/* Confidence */}
      <div className="card-meta" style={{ marginTop: 8 }}>
        <span className="muted">Confidence</span>
        <div className="conf-bar" style={{ flex: 1, maxWidth: 100 }}>
          <div
            className="conf-fill"
            style={{
              width: `${prediction.confidence * 100}%`,
              background: parseFloat(confPct) > 65 ? dirColor : "#555",
            }}
          />
        </div>
        <span style={{ color: parseFloat(confPct) > 65 ? dirColor : "var(--muted2)" }}>
          {confPct}%
        </span>
      </div>

      <div className="card-note" style={{ marginTop: 10 }}>
        Not financial advice · Research purposes only
      </div>
    </div>
  );
};

// ── Custom chart dot ───────────────────────────────────────────────────────────
const PredDot = ({ cx, cy, payload }) => {
  if (!payload?.predicted_dir) return null;
  const isUp    = payload.predicted_dir === "UP";
  const correct = payload.correct;
  const fill    = correct ? (isUp ? "#30d158" : "#ff3b30") : "#444";
  return (
    <g>
      <circle cx={cx} cy={cy} r={4} fill={fill} stroke="#080808" strokeWidth={1.5} />
      <text x={cx} y={cy - 8} textAnchor="middle" fontSize={8} fill={fill}>
        {isUp ? "▲" : "▼"}
      </text>
    </g>
  );
};

// ── Chart tooltip ──────────────────────────────────────────────────────────────
const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  const isUp    = d.predicted_dir === "UP";
  const correct = d.correct;
  return (
    <div style={{
      background: "#0d0d0d",
      border: "1px solid #1e1e1e",
      borderRadius: 3,
      padding: "8px 12px",
      fontSize: 11,
      fontFamily: "var(--mono)",
    }}>
      <div style={{ color: "#555", marginBottom: 4 }}>{label}</div>
      <div style={{ color: "#e0e0e0" }}>${fmt(d.actual_price, 0)}</div>
      <div style={{ color: isUp ? "#30d158" : "#ff3b30" }}>
        Predicted: {d.predicted_dir} ({pct(d.up_prob)} UP)
      </div>
      <div style={{ color: correct ? "#30d158" : "#ff3b30" }}>
        {correct ? "✓ Correct" : "✗ Wrong"}
      </div>
    </div>
  );
};

// ── Accuracy badge ─────────────────────────────────────────────────────────────
const AccuracyBadge = ({ accuracy, days }) => {
  const pctVal  = (accuracy * 100).toFixed(1);
  const color   = accuracy > 0.55 ? "#30d158" : accuracy > 0.50 ? "#ff9f0a" : "#ff3b30";
  return (
    <div className="card" style={{ display: "inline-flex", alignItems: "center", gap: 16, padding: "10px 16px" }}>
      <div>
        <span style={{ fontSize: 26, fontWeight: 300, color, fontFamily: "var(--mono)" }}>{pctVal}%</span>
        <span className="muted" style={{ fontSize: 11, marginLeft: 6 }}>accuracy</span>
      </div>
      <div style={{ fontSize: 10, color: "var(--muted2)" }}>
        over last {days} days · vs ~50% baseline
      </div>
    </div>
  );
};

// ── Fear & Greed label ────────────────────────────────────────────────────────
const fgLabel = (v) => {
  if (v == null) return { label: "—", color: "#555" };
  if (v <= 20)  return { label: "Extreme Fear", color: "#ff3b30" };
  if (v <= 40)  return { label: "Fear",         color: "#ff9f0a" };
  if (v <= 60)  return { label: "Neutral",      color: "#aeaeb2" };
  if (v <= 80)  return { label: "Greed",        color: "#30d158" };
  return              { label: "Extreme Greed", color: "#34c759" };
};

// ── Market Signals section ─────────────────────────────────────────────────────
const MarketSignals = ({ signals }) => {
  if (!signals) return null;

  const fg    = signals.fear_greed;
  const fgVal = fg?.value ?? null;
  const { label: fgLbl, color: fgColor } = fgLabel(fgVal);

  const btcFund = signals.funding_rates?.BTC?.rate_avg ?? null;
  const ethFund = signals.funding_rates?.ETH?.rate_avg ?? null;
  const ratio   = signals.ratio_7d_change ?? null;

  const fundColor = (r) => {
    if (r == null) return "#555";
    if (r > 0.0005)  return "#ff3b30";   // extreme long
    if (r < -0.0005) return "#0a84ff";   // extreme short
    return "#aeaeb2";
  };

  const ratioColor = ratio == null ? "#555" : ratio > 0 ? "#30d158" : "#ff3b30";

  return (
    <div className="card" style={{ marginTop: 0 }}>
      <div className="card-header" style={{ marginBottom: 14 }}>
        <span className="card-label">MARKET SIGNALS</span>
        <span className="card-sub">Fear &amp; Greed · Funding · BTC/ETH Ratio</span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16 }}>

        {/* Fear & Greed */}
        <div>
          <div style={{ fontSize: 10, color: "var(--muted2)", letterSpacing: "0.08em", marginBottom: 4 }}>
            FEAR &amp; GREED
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
            <span style={{ fontSize: 28, fontWeight: 300, color: fgColor, fontFamily: "var(--mono)" }}>
              {fgVal != null ? Math.round(fgVal) : "—"}
            </span>
            <span style={{ fontSize: 11, color: fgColor }}>{fgLbl}</span>
          </div>
          {/* Gauge bar */}
          <div style={{ height: 3, background: "var(--border2)", borderRadius: 2, marginTop: 6, overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${fgVal ?? 0}%`, background: fgColor, borderRadius: 2, transition: "width 0.4s" }} />
          </div>
          {fg?.date && <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>{fg.date}</div>}
        </div>

        {/* BTC Funding Rate */}
        <div>
          <div style={{ fontSize: 10, color: "var(--muted2)", letterSpacing: "0.08em", marginBottom: 4 }}>
            BTC FUNDING (daily avg)
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
            <span style={{ fontSize: 22, fontWeight: 300, color: fundColor(btcFund), fontFamily: "var(--mono)" }}>
              {btcFund != null ? `${(btcFund * 100).toFixed(4)}%` : "—"}
            </span>
          </div>
          <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>
            {btcFund == null ? "" : btcFund > 0.0005 ? "Longs overextended" : btcFund < -0.0005 ? "Shorts overextended" : "Neutral"}
          </div>
        </div>

        {/* ETH Funding Rate */}
        <div>
          <div style={{ fontSize: 10, color: "var(--muted2)", letterSpacing: "0.08em", marginBottom: 4 }}>
            ETH FUNDING (daily avg)
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
            <span style={{ fontSize: 22, fontWeight: 300, color: fundColor(ethFund), fontFamily: "var(--mono)" }}>
              {ethFund != null ? `${(ethFund * 100).toFixed(4)}%` : "—"}
            </span>
          </div>
          <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>
            {ethFund == null ? "" : ethFund > 0.0005 ? "Longs overextended" : ethFund < -0.0005 ? "Shorts overextended" : "Neutral"}
          </div>
        </div>

        {/* BTC/ETH Ratio Change */}
        <div>
          <div style={{ fontSize: 10, color: "var(--muted2)", letterSpacing: "0.08em", marginBottom: 4 }}>
            BTC/ETH RATIO (7D)
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
            <span style={{ fontSize: 22, fontWeight: 300, color: ratioColor, fontFamily: "var(--mono)" }}>
              {ratio != null ? `${ratio > 0 ? "+" : ""}${ratio}%` : "—"}
            </span>
          </div>
          <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>
            {ratio == null ? "" : ratio > 2 ? "BTC dominance rising" : ratio < -2 ? "ETH outperforming" : "Ratio stable"}
          </div>
        </div>

      </div>
    </div>
  );
};

// ── Main ML Tab ────────────────────────────────────────────────────────────────
export default function MLTab({ mlPredictions }) {
  const [symbol,    setSymbol]    = useState("BTC");
  const [lookahead, setLookahead] = useState(1);
  const [days,      setDays]      = useState(30);
  const [backtest,  setBacktest]  = useState(null);
  const [btLoading, setBtLoading] = useState(false);
  const [btError,   setBtError]   = useState(null);
  const [signals,   setSignals]   = useState(null);

  const fetchBacktest = useCallback(async () => {
    setBtLoading(true);
    setBtError(null);
    try {
      const res = await fetch(`${API_URL}/ml/backtest?symbol=${symbol}&days=${days}&lookahead=${lookahead}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setBacktest(await res.json());
    } catch (e) {
      setBtError(e.message);
    } finally {
      setBtLoading(false);
    }
  }, [symbol, days, lookahead]);

  useEffect(() => { fetchBacktest(); }, [fetchBacktest]);

  useEffect(() => {
    fetch(`${API_URL}/ml/market-signals`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setSignals(data); })
      .catch(() => {});
  }, []);

  const chartData = backtest?.points ?? [];
  const prices    = chartData.map(p => p.actual_price).filter(Boolean);
  const minP      = prices.length ? Math.min(...prices) : 0;
  const maxP      = prices.length ? Math.max(...prices) : 1;
  const pad       = (maxP - minP) * 0.03;
  const lineColor = symbol === "BTC" ? "#ff9f0a" : "#0a84ff";

  return (
    <div className="tab-content">

      {/* ── Disclaimer ── */}
      <div className="signals-note">
        ⚠ ML predictions are research tools only · XGBoost trained on daily OHLCV + macro features · Not financial advice
      </div>

      {/* ── Controls ── */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        {/* Symbol toggle */}
        <div style={{ display: "flex", border: "1px solid var(--border)", borderRadius: 3, overflow: "hidden" }}>
          {["BTC", "ETH"].map(s => (
            <button
              key={s}
              onClick={() => setSymbol(s)}
              style={{
                padding: "6px 16px",
                background: symbol === s ? "var(--accent)" : "transparent",
                color: symbol === s ? "#000" : "var(--muted2)",
                border: "none",
                fontFamily: "var(--mono)",
                fontSize: 11,
                fontWeight: 600,
                letterSpacing: "0.1em",
                cursor: "pointer",
              }}
            >
              {s}
            </button>
          ))}
        </div>

        {/* Days toggle */}
        <div style={{ display: "flex", border: "1px solid var(--border)", borderRadius: 3, overflow: "hidden" }}>
          {[7, 14, 30, 60].map(d => (
            <button
              key={d}
              onClick={() => setDays(d)}
              style={{
                padding: "6px 12px",
                background: days === d ? "var(--border2)" : "transparent",
                color: days === d ? "var(--text)" : "var(--muted2)",
                border: "none",
                fontFamily: "var(--mono)",
                fontSize: 11,
                cursor: "pointer",
              }}
            >
              {d}D
            </button>
          ))}
        </div>

        {/* Lookahead toggle */}
        <div style={{ display: "flex", border: "1px solid var(--border)", borderRadius: 3, overflow: "hidden" }}>
          {[{ v: 1, label: "1D" }, { v: 7, label: "7D" }].map(({ v, label }) => (
            <button
              key={v}
              onClick={() => setLookahead(v)}
              style={{
                padding: "6px 12px",
                background: lookahead === v ? "var(--border2)" : "transparent",
                color: lookahead === v ? "var(--text)" : "var(--muted2)",
                border: "none",
                fontFamily: "var(--mono)",
                fontSize: 11,
                cursor: "pointer",
              }}
            >
              {label}
            </button>
          ))}
        </div>

        <button
          onClick={fetchBacktest}
          style={{
            padding: "6px 12px",
            background: "transparent",
            border: "1px solid var(--border)",
            borderRadius: 3,
            color: "var(--muted2)",
            fontFamily: "var(--mono)",
            fontSize: 11,
            cursor: "pointer",
          }}
        >
          ↻ REFRESH
        </button>
      </div>

      {/* ── Market signals ── */}
      <MarketSignals signals={signals} />

      {/* ── Prediction cards — both horizons for selected symbol ── */}
      <div style={{ fontSize: 10, color: "var(--muted2)", letterSpacing: "0.1em", marginTop: 4 }}>
        {symbol} FORECASTS
      </div>
      <div className="card-grid">
        <PredictionCard symbol={symbol} label="24H FORECAST"  prediction={mlPredictions?.[`${symbol}_1d`]} />
        <PredictionCard symbol={symbol} label="7D FORECAST"   prediction={mlPredictions?.[`${symbol}_7d`]} />
      </div>

      {/* ── Accuracy badge ── */}
      {backtest && <AccuracyBadge accuracy={backtest.accuracy} days={days} />}

      {/* ── Predicted vs Actual chart ── */}
      <div className="chart-section">
        <div className="chart-label">
          {symbol}/USDT · BACKTEST · {lookahead === 1 ? "24H" : `${lookahead}D`} HORIZON · LAST {days}D
        </div>
        <div style={{ fontSize: 10, color: "var(--muted)", marginBottom: 12 }}>
          <span style={{ color: "#30d158" }}>▲ UP</span>
          {" / "}
          <span style={{ color: "#ff3b30" }}>▼ DOWN</span>
          {" "}— dot color: <span style={{ color: "#30d158" }}>correct</span>
          {", "}
          <span style={{ color: "#444" }}>wrong</span>
        </div>

        {btLoading && <div className="loading">Loading backtest data...</div>}
        {btError   && <div style={{ color: "#ff3b30", fontSize: 11, padding: "20px 0" }}>Error: {btError}</div>}

        {!btLoading && chartData.length > 0 && (
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={chartData} margin={{ top: 20, right: 8, bottom: 0, left: 8 }}>
              <XAxis
                dataKey="date"
                tick={{ fill: "#555", fontSize: 10, fontFamily: "var(--mono)" }}
                tickLine={false}
                axisLine={false}
                interval={Math.floor(chartData.length / 6)}
              />
              <YAxis
                domain={[minP - pad, maxP + pad]}
                tick={{ fill: "#555", fontSize: 10, fontFamily: "var(--mono)" }}
                tickLine={false}
                axisLine={false}
                width={70}
                tickFormatter={v =>
                  symbol === "BTC"
                    ? `$${(v / 1000).toFixed(0)}k`
                    : `$${v.toLocaleString()}`
                }
              />
              <Tooltip content={<ChartTooltip />} />
              <Line
                type="monotone"
                dataKey="actual_price"
                stroke={lineColor}
                strokeWidth={1.5}
                dot={<PredDot />}
                activeDot={{ r: 5, fill: lineColor }}
                name="Price"
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}

        {!btLoading && chartData.length === 0 && !btError && (
          <div className="loading">No backtest data available</div>
        )}
      </div>
    </div>
  );
}
