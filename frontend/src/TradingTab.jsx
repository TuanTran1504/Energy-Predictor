import { useState, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8080";

const fmt  = (n, d = 2) => n == null ? "—" : Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct  = (n) => n == null ? "—" : `${Number(n).toFixed(2)}%`;
const clr  = (n) => n == null ? "#aeaeb2" : n >= 0 ? "#30d158" : "#ff3b30";
const side_color = (s) => s === "BUY" ? "#30d158" : "#ff3b30";

const Badge = ({ text, color }) => (
  <span style={{
    padding: "2px 8px", borderRadius: 3,
    background: color + "22", color,
    fontFamily: "var(--mono)", fontSize: 10, fontWeight: 600,
  }}>{text}</span>
);

const StatCard = ({ label, value, color, sub }) => (
  <div className="card" style={{ padding: "12px 16px" }}>
    <div style={{ fontSize: 10, color: "var(--muted2)", letterSpacing: "0.08em", marginBottom: 6 }}>{label}</div>
    <div style={{ fontSize: 24, fontWeight: 300, color: color || "var(--text)", fontFamily: "var(--mono)" }}>{value}</div>
    {sub && <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 4 }}>{sub}</div>}
  </div>
);

function calcUnrealizedPnl(trade, currentPrice) {
  if (!currentPrice || !trade.entry_price) return null;
  const leverage = trade.leverage || 5;
  const qty      = trade.quantity;
  const entry    = trade.entry_price;
  let pnl_pct;
  if (trade.side === "BUY") {
    pnl_pct = (currentPrice - entry) / entry * leverage;
  } else {
    pnl_pct = (entry - currentPrice) / entry * leverage;
  }
  const margin   = qty * entry / leverage;
  const pnl_usdt = pnl_pct * margin;
  return { pnl_pct: pnl_pct * 100, pnl_usdt, currentPrice };
}

export default function TradingTab({ livePrices = {} }) {
  const [status,    setStatus]    = useState(null);
  const [history,   setHistory]   = useState([]);
  const [binancePos, setBinancePos] = useState({});
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);

  const fetchData = async () => {
    try {
      const [s, h] = await Promise.all([
        fetch(`${API_URL}/trading/status`).then(r => r.json()),
        fetch(`${API_URL}/trading/history`).then(r => r.json()),
      ]);
      setStatus(s);
      setHistory(h.trades || []);
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchBinancePositions = async () => {
    try {
      const data = await fetch(`${API_URL}/trading/positions/sync`).then(r => r.json());
      if (data && typeof data === "object" && !data.error) {
        setBinancePos(data);
      }
    } catch {
      // silent — Binance sync is best-effort
    }
  };

  useEffect(() => {
    fetchData();
    fetchBinancePositions();
    const id1 = setInterval(fetchData, 30_000);
    const id2 = setInterval(fetchBinancePositions, 10_000);
    return () => { clearInterval(id1); clearInterval(id2); };
  }, []);

  if (loading) return <div className="tab-content"><div className="loading">Loading trading data...</div></div>;
  if (error)   return <div className="tab-content"><div style={{ color: "#ff3b30", fontSize: 12 }}>Error: {error}</div></div>;

  const pnlColor = clr(status?.total_pnl);

  return (
    <div className="tab-content">

      <div className="signals-note">
        ⚠ Binance Testnet · Paper trading only · Not real money
      </div>

      {/* ── Summary stats ── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12 }}>
        <StatCard
          label="TOTAL P&L"
          value={`${status?.total_pnl >= 0 ? "+" : ""}${fmt(status?.total_pnl)} USDT`}
          color={pnlColor}
        />
        <StatCard
          label="WIN RATE"
          value={pct(status?.win_rate)}
          color={status?.win_rate >= 50 ? "#30d158" : "#ff3b30"}
          sub={`${status?.wins}W / ${(status?.total_closed - status?.wins) || 0}L — ${status?.total_closed} total`}
        />
        <StatCard
          label="OPEN POSITIONS"
          value={status?.open_trades?.length ?? 0}
          color="var(--text)"
          sub="BTC + ETH max 2"
        />
      </div>

      {/* ── LLM last decision ── */}
      {status?.last_decision && (() => {
        try {
          const d = JSON.parse(status.last_decision);
          return (
            <div className="card">
              <div className="card-header" style={{ marginBottom: 10 }}>
                <span className="card-label">LAST LLM DECISION</span>
                <span className="card-sub">{status.last_decided_at ? new Date(status.last_decided_at).toLocaleString() : ""}</span>
              </div>
              {d.market_summary && (
                <div style={{ fontSize: 12, color: "var(--muted2)", marginBottom: 12, fontStyle: "italic" }}>
                  "{d.market_summary}"
                </div>
              )}
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {(d.decisions || []).map((dec, i) => (
                  <div key={i} style={{ display: "flex", gap: 12, alignItems: "flex-start", padding: "8px 0", borderTop: i > 0 ? "1px solid var(--border)" : "none" }}>
                    <Badge text={dec.symbol} color="#aeaeb2" />
                    <Badge
                      text={dec.action.replace("_", " ").toUpperCase()}
                      color={dec.action === "open_long" ? "#30d158" : dec.action === "open_short" ? "#ff3b30" : dec.action === "close" ? "#ff9f0a" : "#555"}
                    />
                    {dec.action !== "hold" && dec.action !== "close" && (
                      <Badge text={`${dec.position_size_multiplier}x size`} color="#0a84ff" />
                    )}
                    <span style={{ fontSize: 11, color: "var(--muted)", flex: 1 }}>{dec.reasoning}</span>
                    <span style={{ fontSize: 10, color: "var(--muted2)", whiteSpace: "nowrap" }}>
                      {(dec.confidence_score * 100).toFixed(0)}% conf
                    </span>
                  </div>
                ))}
              </div>
            </div>
          );
        } catch { return null; }
      })()}

      {/* ── Open positions ── */}
      <div>
        <div className="chart-label" style={{ marginBottom: 10 }}>OPEN POSITIONS</div>
        {status?.open_trades?.length === 0 ? (
          <div className="card" style={{ color: "var(--muted2)", fontSize: 11, padding: "16px" }}>
            No open positions — waiting for next signal cycle
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {status.open_trades.map(t => (
              <div key={t.id} className="card" style={{ padding: "12px 16px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontFamily: "var(--mono)", fontWeight: 600 }}>{t.symbol}/USDT</span>
                    <Badge text={t.side === "BUY" ? "LONG" : "SHORT"} color={side_color(t.side)} />
                    <Badge text={`${t.leverage}x`} color="#aeaeb2" />
                    <Badge text={`${t.horizon}D MODEL`} color="#0a84ff" />
                  </div>
                  <span style={{ fontSize: 10, color: "var(--muted)" }}>
                    {new Date(t.opened_at).toLocaleString()}
                  </span>
                </div>

                {(() => {
                  // Prefer Binance live data; fall back to local WebSocket price calc
                  const bn   = binancePos[t.symbol];
                  const mark = bn?.mark_price ?? livePrices[t.symbol] ?? null;
                  const upnl = bn
                    ? { pnl_usdt: bn.unrealized_pnl, currentPrice: bn.mark_price,
                        pnl_pct: bn.mark_price && bn.entry_price
                          ? (t.side === "BUY"
                              ? (bn.mark_price - bn.entry_price) / bn.entry_price * (bn.leverage || 5) * 100
                              : (bn.entry_price - bn.mark_price) / bn.entry_price * (bn.leverage || 5) * 100)
                          : null }
                    : calcUnrealizedPnl(t, mark);
                  const qty   = bn ? Math.abs(bn.position_amt) : t.quantity;
                  const entry = bn ? bn.entry_price : t.entry_price;
                  return (
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 12 }}>
                      <div>
                        <div style={{ fontSize: 10, color: "var(--muted2)" }}>ENTRY</div>
                        <div style={{ fontFamily: "var(--mono)", fontSize: 13 }}>${fmt(entry, 2)}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 10, color: "var(--muted2)" }}>QUANTITY</div>
                        <div style={{ fontFamily: "var(--mono)", fontSize: 13 }}>{qty} {t.symbol}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 10, color: "#ff3b30" }}>STOP LOSS</div>
                        <div style={{ fontFamily: "var(--mono)", fontSize: 13, color: "#ff3b30" }}>${fmt(t.stop_loss, 2)}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 10, color: "#30d158" }}>TAKE PROFIT</div>
                        <div style={{ fontFamily: "var(--mono)", fontSize: 13, color: "#30d158" }}>${fmt(t.take_profit, 2)}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 10, color: "var(--muted2)" }}>
                          {bn ? "MARK PRICE" : "CURRENT"}
                        </div>
                        <div style={{ fontFamily: "var(--mono)", fontSize: 13 }}>
                          {mark ? `$${fmt(mark, 2)}` : "—"}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 10, color: "var(--muted2)" }}>UNREAL. P&L</div>
                        <div style={{ fontFamily: "var(--mono)", fontSize: 13, color: clr(upnl?.pnl_usdt) }}>
                          {upnl?.pnl_usdt != null
                            ? `${upnl.pnl_usdt >= 0 ? "+" : ""}${fmt(upnl.pnl_usdt)} (${upnl.pnl_pct >= 0 ? "+" : ""}${fmt(upnl.pnl_pct)}%)`
                            : "—"}
                        </div>
                      </div>
                    </div>
                  );
                })()}

                <div style={{ marginTop: 8, fontSize: 10, color: "var(--muted)" }}>
                  Confidence: {pct(t.confidence * 100)} · Order: {t.binance_order_id || "—"}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Trade history ── */}
      <div>
        <div className="chart-label" style={{ marginBottom: 10 }}>TRADE HISTORY</div>
        {history.length === 0 ? (
          <div className="card" style={{ color: "var(--muted2)", fontSize: 11, padding: "16px" }}>
            No closed trades yet
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--mono)", fontSize: 11 }}>
              <thead>
                <tr style={{ color: "var(--muted2)", borderBottom: "1px solid var(--border)" }}>
                  {["#", "SYMBOL", "SIDE", "ENTRY", "EXIT", "QTY", "P&L", "P&L %", "REASON", "DATE"].map(h => (
                    <th key={h} style={{ padding: "6px 8px", textAlign: "left", fontWeight: 400, letterSpacing: "0.06em" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.map(t => (
                  <tr key={t.id} style={{ borderBottom: "1px solid var(--border)", opacity: t.status === "OPEN" ? 1 : 0.85 }}>
                    <td style={{ padding: "6px 8px", color: "var(--muted)" }}>{t.id}</td>
                    <td style={{ padding: "6px 8px" }}>{t.symbol}</td>
                    <td style={{ padding: "6px 8px" }}>
                      <Badge text={t.side === "BUY" ? "LONG" : "SHORT"} color={side_color(t.side)} />
                    </td>
                    <td style={{ padding: "6px 8px" }}>${fmt(t.entry_price)}</td>
                    <td style={{ padding: "6px 8px" }}>{t.exit_price ? `$${fmt(t.exit_price)}` : "OPEN"}</td>
                    <td style={{ padding: "6px 8px" }}>{t.quantity}</td>
                    <td style={{ padding: "6px 8px", color: clr(t.pnl_usdt) }}>
                      {t.pnl_usdt != null ? `${t.pnl_usdt >= 0 ? "+" : ""}${fmt(t.pnl_usdt)}` : "—"}
                    </td>
                    <td style={{ padding: "6px 8px", color: clr(t.pnl_pct) }}>
                      {t.pnl_pct != null ? `${t.pnl_pct >= 0 ? "+" : ""}${fmt(t.pnl_pct)}%` : "—"}
                    </td>
                    <td style={{ padding: "6px 8px", color: "var(--muted)" }}>{t.close_reason || "open"}</td>
                    <td style={{ padding: "6px 8px", color: "var(--muted)" }}>
                      {new Date(t.opened_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
