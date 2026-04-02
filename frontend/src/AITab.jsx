import { useState, useRef, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8080";

const SUGGESTED = [
  "Analyse the current BTC market — should I go long?",
  "Open a long on BTC now",
  "Open a short on ETH",
  "What does Fear & Greed tell us today?",
  "Are funding rates signaling any risk right now?",
  "Compare BTC and ETH — which looks stronger?",
];

// Parse ---PROPOSED_ORDER--- block out of streamed text
function extractOrder(text) {
  const match = text.match(/---PROPOSED_ORDER---\s*(\{[\s\S]*?\})\s*---END_ORDER---/);
  if (!match) return null;
  try { return JSON.parse(match[1]); } catch { return null; }
}

// Strip the order block from displayed text
function stripOrderBlock(text) {
  return text.replace(/\n*---PROPOSED_ORDER---[\s\S]*?---END_ORDER---\n*/g, "").trim();
}

const Message = ({ msg, onConfirm, onCancel }) => {
  const isUser = msg.role === "user";

  return (
    <div style={{
      display: "flex",
      justifyContent: isUser ? "flex-end" : "flex-start",
      marginBottom: 16,
    }}>
      <div style={{ maxWidth: "85%", display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{
          padding: "10px 14px",
          borderRadius: 6,
          background: isUser ? "var(--accent)" : "var(--card-bg)",
          border: isUser ? "none" : "1px solid var(--border)",
          color: isUser ? "#000" : "var(--text)",
          fontFamily: "var(--mono)",
          fontSize: 12,
          lineHeight: 1.8,
          whiteSpace: "pre-wrap",
        }}>
          {msg.displayContent || msg.content}
          {msg.streaming && (
            <span style={{ opacity: 0.5, animation: "blink 1s infinite" }}>▋</span>
          )}
        </div>

        {/* Pending order confirmation card */}
        {msg.pendingOrder && !msg.executed && (
          <div style={{
            background: "#0d1a0d",
            border: "1px solid #30d158",
            borderRadius: 6,
            padding: "12px 16px",
            fontFamily: "var(--mono)",
            fontSize: 11,
          }}>
            <div style={{ color: "#30d158", fontWeight: 600, marginBottom: 8, letterSpacing: "0.08em" }}>
              PROPOSED ORDER
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginBottom: 12 }}>
              <div>
                <div style={{ color: "#555", fontSize: 10 }}>SYMBOL</div>
                <div style={{ color: "#e0e0e0" }}>{msg.pendingOrder.symbol}/USDT</div>
              </div>
              <div>
                <div style={{ color: "#555", fontSize: 10 }}>SIDE</div>
                <div style={{ color: msg.pendingOrder.side === "BUY" ? "#30d158" : "#ff3b30", fontWeight: 600 }}>
                  {msg.pendingOrder.side === "BUY" ? "▲ LONG" : "▼ SHORT"}
                </div>
              </div>
              <div>
                <div style={{ color: "#555", fontSize: 10 }}>SIZE</div>
                <div style={{ color: "#e0e0e0" }}>{msg.pendingOrder.size_multiplier}x</div>
              </div>
              <div>
                <div style={{ color: "#ff3b30", fontSize: 10 }}>STOP LOSS</div>
                <div style={{ color: "#ff3b30" }}>-{(msg.pendingOrder.stop_loss_pct * 100).toFixed(1)}%</div>
              </div>
              <div>
                <div style={{ color: "#30d158", fontSize: 10 }}>TAKE PROFIT</div>
                <div style={{ color: "#30d158" }}>+{(msg.pendingOrder.take_profit_pct * 100).toFixed(1)}%</div>
              </div>
              <div>
                <div style={{ color: "#555", fontSize: 10 }}>TYPE</div>
                <div style={{ color: "#e0e0e0" }}>MARKET</div>
              </div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                onClick={() => onConfirm(msg.pendingOrder)}
                style={{
                  flex: 1, padding: "7px", background: "#30d158", border: "none",
                  borderRadius: 4, color: "#000", fontFamily: "var(--mono)",
                  fontSize: 11, fontWeight: 700, cursor: "pointer",
                }}
              >
                ✓ CONFIRM &amp; PLACE ORDER
              </button>
              <button
                onClick={onCancel}
                style={{
                  padding: "7px 16px", background: "transparent",
                  border: "1px solid var(--border)", borderRadius: 4,
                  color: "var(--muted2)", fontFamily: "var(--mono)",
                  fontSize: 11, cursor: "pointer",
                }}
              >
                ✕ CANCEL
              </button>
            </div>
          </div>
        )}

        {msg.executed && (
          <div style={{
            background: "#0d1a0d", border: "1px solid #30d158",
            borderRadius: 6, padding: "10px 14px",
            fontFamily: "var(--mono)", fontSize: 11, color: "#30d158",
          }}>
            ✓ Order placed — {msg.executed.side === "BUY" ? "LONG" : "SHORT"} {msg.executed.quantity} {msg.executed.symbol} @ ${msg.executed.entry_price?.toLocaleString()}
            <br/>SL ${msg.executed.stop_loss?.toLocaleString()} · TP ${msg.executed.take_profit?.toLocaleString()}
            · Order #{msg.executed.order_id}
          </div>
        )}

        {msg.cancelledOrder && (
          <div style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--mono)", paddingLeft: 4 }}>
            Order cancelled.
          </div>
        )}
      </div>
    </div>
  );
};

export default function AITab() {
  const [messages, setMessages] = useState([]);
  const [input,    setInput]    = useState("");
  const [loading,  setLoading]  = useState(false);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async (question) => {
    if (!question.trim() || loading) return;

    const userMsg = { role: "user", content: question };
    const history = messages
      .filter(m => !m.streaming)
      .map(m => ({ role: m.role, content: m.content }))
      .slice(-12);

    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    setMessages(prev => [...prev, { role: "assistant", content: "", streaming: true }]);

    try {
      const res = await fetch(`${API_URL}/ml/trade/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question, history }),
      });

      if (!res.ok) {
        const err = await res.text();
        setMessages(prev => {
          const msgs = [...prev];
          msgs[msgs.length - 1] = { role: "assistant", content: `Error: ${err}` };
          return msgs;
        });
        return;
      }

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let full = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        full += decoder.decode(value, { stream: true });
        const display = stripOrderBlock(full);
        setMessages(prev => {
          const msgs = [...prev];
          msgs[msgs.length - 1] = {
            role: "assistant", content: full,
            displayContent: display, streaming: true,
          };
          return msgs;
        });
      }

      const pendingOrder   = extractOrder(full);
      const displayContent = stripOrderBlock(full);

      setMessages(prev => {
        const msgs = [...prev];
        msgs[msgs.length - 1] = {
          role: "assistant", content: full,
          displayContent, streaming: false,
          pendingOrder: pendingOrder || null,
        };
        return msgs;
      });

    } catch (e) {
      setMessages(prev => {
        const msgs = [...prev];
        msgs[msgs.length - 1] = { role: "assistant", content: `Connection error: ${e.message}` };
        return msgs;
      });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleConfirm = async (order) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/ml/trade/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ order }),
      });
      const result = await res.json();
      if (!res.ok) throw new Error(result.detail || "Execution failed");

      // Mark the message as executed and remove pending order
      setMessages(prev => prev.map(m =>
        m.pendingOrder === order
          ? { ...m, pendingOrder: null, executed: result }
          : m
      ));

      // Add confirmation message
      setMessages(prev => [...prev, {
        role: "assistant",
        content: `Order executed successfully. Trade ID: ${result.trade_id}. The position is now being monitored for SL/TP.`,
        displayContent: `Order executed successfully. Trade ID: ${result.trade_id}. The position is now being monitored for SL/TP.`,
      }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: `Failed to execute order: ${e.message}`,
        displayContent: `Failed to execute order: ${e.message}`,
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setMessages(prev => prev.map(m =>
      m.pendingOrder ? { ...m, pendingOrder: null, cancelledOrder: true } : m
    ));
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };

  return (
    <div className="tab-content" style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 160px)" }}>

      <div className="signals-note" style={{ marginBottom: 12 }}>
        ⚠ AI can place real testnet orders · Review carefully before confirming · Not financial advice
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "8px 0", marginBottom: 12 }}>
        {messages.length === 0 && (
          <div>
            <div style={{ color: "var(--muted2)", fontSize: 11, marginBottom: 16, fontFamily: "var(--mono)" }}>
              Ask me to analyse the market or place a trade. I'll show you a full analysis before executing anything.
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {SUGGESTED.map(s => (
                <button key={s} onClick={() => send(s)} style={{
                  padding: "6px 12px", background: "var(--card-bg)",
                  border: "1px solid var(--border)", borderRadius: 4,
                  color: "var(--muted2)", fontFamily: "var(--mono)",
                  fontSize: 11, cursor: "pointer", textAlign: "left",
                }}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <Message
            key={i} msg={msg}
            onConfirm={handleConfirm}
            onCancel={handleCancel}
          />
        ))}
        <div ref={bottomRef} />
      </div>

      <div style={{ display: "flex", gap: 8, alignItems: "flex-end" }}>
        <textarea
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder='Ask anything or say "open a long on BTC" to place a trade... (Enter to send)'
          disabled={loading}
          rows={2}
          style={{
            flex: 1, background: "var(--card-bg)",
            border: "1px solid var(--border)", borderRadius: 4,
            color: "var(--text)", fontFamily: "var(--mono)",
            fontSize: 12, padding: "8px 12px", resize: "none", outline: "none",
          }}
        />
        <button
          onClick={() => send(input)}
          disabled={loading || !input.trim()}
          style={{
            padding: "8px 16px",
            background: loading ? "var(--border2)" : "var(--accent)",
            border: "none", borderRadius: 4,
            color: loading ? "var(--muted2)" : "#000",
            fontFamily: "var(--mono)", fontSize: 11, fontWeight: 600,
            cursor: loading ? "not-allowed" : "pointer", whiteSpace: "nowrap",
          }}
        >
          {loading ? "..." : "SEND →"}
        </button>
      </div>

      <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }`}</style>
    </div>
  );
}
