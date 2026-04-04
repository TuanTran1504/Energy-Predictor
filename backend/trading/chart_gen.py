"""
chart_gen.py — Candlestick chart generator for the LLM analyst.

Produces a dark-themed M5 chart (last 80 candles) with:
  - Candlestick bodies + wicks
  - EMA 34 (green) + EMA 89 (orange)
  - Volume panel + Volume MA 20 (blue)
  - H1 Support (green dashed) + Resistance (red dashed)
  - Bollinger Bands (white, semi-transparent)
  - Title bar: symbol, mode, score, ADX, RSI, ATR, funding

Returns base64-encoded PNG string ready for Gemini vision API.
"""

import base64
import io
from datetime import datetime, UTC

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server/threads
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG         = "#0D1117"
GRID       = "#21262D"
SPINE      = "#30363D"
TEXT       = "#C9D1D9"
BULL_BODY  = "#26A69A"
BEAR_BODY  = "#EF5350"
BULL_WICK  = "#4DB6AC"
BEAR_WICK  = "#E57373"
EMA34_C    = "#00E676"
EMA89_C    = "#FF9800"
VOLMA_C    = "#2196F3"
RESIST_C   = "#FF5252"
SUPPORT_C  = "#69F0AE"
BB_C       = "#FFFFFF"


def _apply_dark_style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
    ax.grid(color=GRID, linestyle="--", linewidth=0.4, alpha=0.7)
    ax.yaxis.label.set_color(TEXT)


def generate_chart(df_m5: pd.DataFrame, context: dict) -> str:
    """
    df_m5   — raw OHLCV DataFrame with columns [timestamp, open, high, low, close, volume].
               'timestamp' should be epoch-ms (Binance format).
    context — output of strategy_core.compute_indicators() merged with sr, score, etc.

    Returns base64 PNG string or empty string on failure.
    """
    try:
        df = df_m5.tail(80).copy().reset_index(drop=True)

        df["ema34"]  = df["close"].ewm(span=34, adjust=False).mean()
        df["ema89"]  = df["close"].ewm(span=89, adjust=False).mean()
        df["vol_ma"] = df["volume"].rolling(20).mean()

        sma20        = df["close"].rolling(20).mean()
        std20        = df["close"].rolling(20).std()
        df["bb_up"]  = sma20 + 2 * std20
        df["bb_dn"]  = sma20 - 2 * std20

        sr          = context.get("sr", {})
        score       = context.get("score", 0)
        h1_trend    = context.get("h1_trend", "?")
        market_mode = context.get("market_mode", h1_trend)
        adx         = context.get("adx", 0)
        rsi         = context.get("rsi", 0)
        atr         = context.get("atr_m15", 0)
        funding     = context.get("funding_rate", 0)
        symbol      = context.get("symbol", "BTC/USDT")
        ml_dir      = context.get("ml_direction", "—")
        ml_conf     = context.get("ml_confidence", 0)

        fig = plt.figure(figsize=(22, 12), facecolor=BG)
        gs  = gridspec.GridSpec(
            2, 1, height_ratios=[3, 1], hspace=0.04,
            left=0.04, right=0.97, top=0.93, bottom=0.05,
        )
        ax_price  = fig.add_subplot(gs[0])
        ax_volume = fig.add_subplot(gs[1], sharex=ax_price)

        for ax in [ax_price, ax_volume]:
            _apply_dark_style(ax)

        x = range(len(df))

        price_range = df["high"].max() - df["low"].min()
        min_body    = price_range * 0.001

        for i in x:
            o = float(df["open"].iloc[i])
            c = float(df["close"].iloc[i])
            h = float(df["high"].iloc[i])
            lo = float(df["low"].iloc[i])
            is_bull = c >= o
            bc = BULL_BODY  if is_bull else BEAR_BODY
            wc = BULL_WICK  if is_bull else BEAR_WICK

            ax_price.plot([i, i], [lo, h], color=wc, linewidth=0.8, zorder=2)
            body_lo = min(o, c)
            body_hi = max(o, c)
            body_h  = max(body_hi - body_lo, min_body)
            rect = mpatches.FancyBboxPatch(
                (i - 0.36, body_lo), 0.72, body_h,
                boxstyle="square,pad=0", linewidth=0,
                facecolor=bc, zorder=3,
            )
            ax_price.add_patch(rect)

        ax_price.plot(x, df["ema34"], color=EMA34_C, linewidth=1.6,
                      label="EMA 34", zorder=4)
        ax_price.plot(x, df["ema89"], color=EMA89_C, linewidth=1.6,
                      label="EMA 89", zorder=4)

        valid = df["bb_up"].notna()
        x_arr = np.array(list(x))
        ax_price.plot(x_arr[valid], df["bb_up"][valid], color=BB_C,
                      linewidth=0.7, alpha=0.35, linestyle="--", zorder=3)
        ax_price.plot(x_arr[valid], df["bb_dn"][valid], color=BB_C,
                      linewidth=0.7, alpha=0.35, linestyle="--", zorder=3)
        ax_price.fill_between(
            x_arr[valid],
            df["bb_up"][valid], df["bb_dn"][valid],
            color=BB_C, alpha=0.04, zorder=2,
        )

        if sr.get("resistance"):
            ax_price.axhline(sr["resistance"], color=RESIST_C, linewidth=1.3,
                             linestyle="--", alpha=0.85, zorder=5)
            ax_price.text(len(df) - 1, sr["resistance"],
                          f"  R {sr['resistance']}", color=RESIST_C,
                          fontsize=8, va="bottom", ha="right")

        if sr.get("support"):
            ax_price.axhline(sr["support"], color=SUPPORT_C, linewidth=1.3,
                             linestyle="--", alpha=0.85, zorder=5)
            ax_price.text(len(df) - 1, sr["support"],
                          f"  S {sr['support']}", color=SUPPORT_C,
                          fontsize=8, va="top", ha="right")

        ax_price.legend(loc="upper left", fontsize=8,
                        facecolor="#161B22", edgecolor=SPINE, labelcolor=TEXT)
        ax_price.set_ylabel("Price (USDT)", color=TEXT, fontsize=9)
        ax_price.set_xlim(-1, len(df))

        ax_r = ax_price.twinx()
        ax_r.set_facecolor(BG)
        ax_r.set_ylim(ax_price.get_ylim())
        ax_r.tick_params(colors=TEXT, labelsize=8)
        for spine in ax_r.spines.values():
            spine.set_color(SPINE)

        vol_colors = [
            BULL_BODY if float(df["close"].iloc[i]) >= float(df["open"].iloc[i])
            else BEAR_BODY
            for i in x
        ]
        ax_volume.bar(x, df["volume"], color=vol_colors, alpha=0.75,
                      width=0.7, zorder=2)
        ax_volume.plot(x, df["vol_ma"], color=VOLMA_C, linewidth=1.4,
                       label="Vol MA 20", zorder=3)
        ax_volume.set_ylabel("Volume", color=TEXT, fontsize=9)
        ax_volume.legend(loc="upper left", fontsize=8,
                         facecolor="#161B22", edgecolor=SPINE, labelcolor=TEXT)

        if "timestamp" in df.columns:
            step = max(1, len(df) // 10)
            ticks = list(range(0, len(df), step))
            labels = []
            for i in ticks:
                ts = df["timestamp"].iloc[i]
                try:
                    dt = datetime.fromtimestamp(int(ts) / 1000, tz=UTC)
                    labels.append(dt.strftime("%m/%d %H:%M"))
                except Exception:
                    labels.append(str(i))
            ax_volume.set_xticks(ticks)
            ax_volume.set_xticklabels(labels, color=TEXT, fontsize=7, rotation=20)

        plt.setp(ax_price.get_xticklabels(), visible=False)

        mode_color = {
            "UPTREND":        EMA34_C,
            "DOWNTREND":      RESIST_C,
            "VOLATILE_RANGE": EMA89_C,
            "SIDEWAY":        "#FFC107",
        }.get(market_mode, TEXT)

        title = (
            f"{symbol} · M5 (80 candles)  |  "
            f"Mode: {market_mode}  |  H1: {h1_trend}  |  Score: {score}/5  |  "
            f"ADX: {adx:.1f}  RSI: {rsi:.1f}  ATR: {atr:.2f}  "
            f"Funding: {funding:+.4f}%  |  ML: {ml_dir} ({ml_conf:.0%})"
        )
        fig.suptitle(title, color=mode_color, fontsize=10,
                     fontweight="bold", y=0.975)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                    facecolor=BG)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        import traceback
        print(f"[chart_gen] Failed: {e}\n{traceback.format_exc()}")
        try:
            plt.close("all")
        except Exception:
            pass
        return ""
