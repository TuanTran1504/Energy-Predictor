"""
chart_gen.py - Candlestick chart generator for the LLM analyst.

Produces a dark-themed dual-view M5 chart with:
  - Wide context pane (last 160 candles)
  - Close-up execution pane (last 80 candles)
  - Candlestick bodies + wicks
  - EMA 34 (green) + EMA 89 (orange)
  - SMA 20 (white dashed, BB midline)
  - Bollinger Bands (semi-transparent fill)
  - H1 Support (green dashed) + Resistance (red dashed)
  - Current price line (yellow dotted)
  - RSI panel with 30/70 zones
  - Volume panel + Volume MA 20 (blue)
  - Title bar: symbol, mode, score, ADX, RSI, ATR, funding

Returns base64-encoded PNG string ready for Gemini vision API.
"""

import base64
import gc
import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd

UTC = timezone.utc

BG = "#0D1117"
GRID = "#21262D"
SPINE = "#30363D"
TEXT = "#C9D1D9"
BULL_BODY = "#26A69A"
BEAR_BODY = "#EF5350"
BULL_WICK = "#4DB6AC"
BEAR_WICK = "#E57373"
EMA34_C = "#00E676"
EMA89_C = "#FF9800"
SMA20_C = "#FFFFFF"
VOLMA_C = "#2196F3"
RESIST_C = "#FF5252"
SUPPORT_C = "#69F0AE"
BB_C = "#90CAF9"
RSI_C = "#CE93D8"
RSI_OB_C = "#EF5350"
RSI_OS_C = "#26A69A"
CUR_PRICE_C = "#FFD600"

CLOSE_WINDOW = 80
WIDE_WINDOW = 160


def _apply_dark_style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
    ax.grid(color=GRID, linestyle="--", linewidth=0.4, alpha=0.7)
    ax.yaxis.label.set_color(TEXT)


def _prepare_chart_df(df_m5: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df_m5.tail(window).copy().reset_index(drop=True)
    df["ema34"] = df["close"].ewm(span=34, adjust=False).mean()
    df["ema89"] = df["close"].ewm(span=89, adjust=False).mean()
    df["sma20"] = df["close"].rolling(20).mean()
    df["vol_ma"] = df["volume"].rolling(20).mean()

    std20 = df["close"].rolling(20).std()
    df["bb_up"] = df["sma20"] + 2 * std20
    df["bb_dn"] = df["sma20"] - 2 * std20

    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    return df


def _format_time_axis(ax, df: pd.DataFrame, max_labels: int = 10):
    if "timestamp" not in df.columns:
        return

    step = max(1, len(df) // max_labels)
    ticks = list(range(0, len(df), step))
    labels = []
    for i in ticks:
        ts = df["timestamp"].iloc[i]
        try:
            dt = datetime.fromtimestamp(int(ts) / 1000, tz=UTC)
            labels.append(dt.strftime("%m/%d %H:%M"))
        except Exception:
            labels.append(str(i))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, color=TEXT, fontsize=7, rotation=20)


def _draw_price_panel(ax_price, df: pd.DataFrame, sr: dict, current_price: float, *,
                      title: str = "", show_legend: bool = True,
                      annotate_levels: bool = True, last_marker: bool = True):
    import matplotlib.patches as mpatches

    x = range(len(df))
    last_x = len(df) - 1
    x_arr = np.array(list(x))
    price_range = max(float(df["high"].max() - df["low"].min()), 1e-9)
    min_body = price_range * 0.001

    for i in x:
        o = float(df["open"].iloc[i])
        c = float(df["close"].iloc[i])
        h = float(df["high"].iloc[i])
        lo = float(df["low"].iloc[i])
        is_bull = c >= o
        bc = BULL_BODY if is_bull else BEAR_BODY
        wc = BULL_WICK if is_bull else BEAR_WICK

        ax_price.plot([i, i], [lo, h], color=wc, linewidth=0.8, zorder=2)
        body_lo = min(o, c)
        body_hi = max(o, c)
        body_h = max(body_hi - body_lo, min_body)
        rect = mpatches.FancyBboxPatch(
            (i - 0.36, body_lo), 0.72, body_h,
            boxstyle="square,pad=0", linewidth=0,
            facecolor=bc, zorder=3,
        )
        ax_price.add_patch(rect)

    ax_price.plot(x, df["ema34"], color=EMA34_C, linewidth=1.6, label="EMA 34", zorder=4)
    ax_price.plot(x, df["ema89"], color=EMA89_C, linewidth=1.6, label="EMA 89", zorder=4)

    valid_sma = df["sma20"].notna()
    ax_price.plot(
        x_arr[valid_sma], df["sma20"][valid_sma],
        color=SMA20_C, linewidth=0.8, alpha=0.5,
        linestyle="--", label="SMA 20", zorder=3,
    )

    valid_bb = df["bb_up"].notna()
    ax_price.plot(x_arr[valid_bb], df["bb_up"][valid_bb], color=BB_C, linewidth=1.0, alpha=0.6, zorder=3)
    ax_price.plot(x_arr[valid_bb], df["bb_dn"][valid_bb], color=BB_C, linewidth=1.0, alpha=0.6, zorder=3)
    ax_price.fill_between(
        x_arr[valid_bb],
        df["bb_up"][valid_bb], df["bb_dn"][valid_bb],
        color=BB_C, alpha=0.05, zorder=2,
    )

    if sr.get("resistance"):
        ax_price.axhline(sr["resistance"], color=RESIST_C, linewidth=1.1, linestyle="--", alpha=0.8, zorder=5)
        if annotate_levels:
            ax_price.text(last_x, sr["resistance"], f"  R {sr['resistance']:.2f}", color=RESIST_C,
                          fontsize=8, va="bottom", ha="right")

    if sr.get("support"):
        ax_price.axhline(sr["support"], color=SUPPORT_C, linewidth=1.1, linestyle="--", alpha=0.8, zorder=5)
        if annotate_levels:
            ax_price.text(last_x, sr["support"], f"  S {sr['support']:.2f}", color=SUPPORT_C,
                          fontsize=8, va="top", ha="right")

    ax_price.axhline(current_price, color=CUR_PRICE_C, linewidth=1.0, linestyle=":", alpha=0.9, zorder=6)
    ax_price.text(0, current_price, f"{current_price:.2f} ", color=CUR_PRICE_C, fontsize=8,
                  va="bottom", ha="left", fontweight="bold", zorder=7)

    if last_marker:
        ax_price.axvline(last_x, color=CUR_PRICE_C, linewidth=0.8, linestyle=":", alpha=0.45, zorder=5)

    if show_legend:
        ax_price.legend(loc="upper left", fontsize=8, facecolor="#161B22", edgecolor=SPINE, labelcolor=TEXT)

    if title:
        ax_price.set_title(title, color=TEXT, fontsize=9, loc="left", pad=6)

    ax_price.set_ylabel("Price (USDT)", color=TEXT, fontsize=9)
    ax_price.set_xlim(-1, len(df))


def generate_chart(df_m5: pd.DataFrame, context: dict) -> str:
    """
    df_m5   - raw OHLCV DataFrame with columns [timestamp, open, high, low, close, volume].
               'timestamp' should be epoch-ms (Binance format).
    context - output of strategy_core.compute_indicators() merged with sr, score, etc.

    Returns base64 PNG string or empty string on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        df_close = _prepare_chart_df(df_m5, CLOSE_WINDOW)
        df_wide = _prepare_chart_df(df_m5, WIDE_WINDOW)

        sr = context.get("sr", {})
        score = context.get("score", 0)
        h1_trend = context.get("h1_trend", "?")
        market_mode = context.get("market_mode", h1_trend)
        adx = context.get("adx", 0)
        rsi_val = context.get("rsi", 0)
        atr = context.get("atr_m15", 0)
        funding = context.get("funding_rate", 0)
        symbol = context.get("symbol", "BTC/USDT")
        ml_dir = context.get("ml_direction", "-")
        ml_conf = context.get("ml_confidence", 0)
        current_price = float(df_close["close"].iloc[-1])

        fig = plt.figure(figsize=(14, 11), facecolor=BG)
        gs = gridspec.GridSpec(
            4, 1,
            height_ratios=[1.5, 3, 1, 1],
            hspace=0.08,
            left=0.04, right=0.97, top=0.93, bottom=0.05,
        )
        ax_wide = fig.add_subplot(gs[0])
        ax_price = fig.add_subplot(gs[1])
        ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)
        ax_volume = fig.add_subplot(gs[3], sharex=ax_price)

        for ax in [ax_wide, ax_price, ax_rsi, ax_volume]:
            _apply_dark_style(ax)

        _draw_price_panel(
            ax_wide, df_wide, sr, current_price,
            title="Wide M5 Context (160 candles) - use for range, structure, and nearby resistance/support",
            show_legend=False, annotate_levels=False, last_marker=False,
        )
        _format_time_axis(ax_wide, df_wide, max_labels=12)

        _draw_price_panel(
            ax_price, df_close, sr, current_price,
            title="Close M5 Execution View (80 candles) - use for signal candle and setup confirmation",
            show_legend=True, annotate_levels=True, last_marker=True,
        )

        x = range(len(df_close))
        last_x = len(df_close) - 1
        x_arr = np.array(list(x))

        ax_r = ax_price.twinx()
        ax_r.set_facecolor(BG)
        ax_r.set_ylim(ax_price.get_ylim())
        ax_r.tick_params(colors=TEXT, labelsize=8)
        for spine in ax_r.spines.values():
            spine.set_color(SPINE)

        valid_rsi = df_close["rsi"].notna()
        ax_rsi.plot(x_arr[valid_rsi], df_close["rsi"][valid_rsi], color=RSI_C, linewidth=1.4, zorder=3)
        ax_rsi.axhline(70, color=RSI_OB_C, linewidth=0.8, linestyle="--", alpha=0.8)
        ax_rsi.axhline(30, color=RSI_OS_C, linewidth=0.8, linestyle="--", alpha=0.8)
        ax_rsi.axhline(50, color=TEXT, linewidth=0.5, linestyle="--", alpha=0.3)
        ax_rsi.fill_between(x_arr[valid_rsi], df_close["rsi"][valid_rsi], 70,
                            where=df_close["rsi"][valid_rsi] >= 70,
                            color=RSI_OB_C, alpha=0.15, zorder=2)
        ax_rsi.fill_between(x_arr[valid_rsi], df_close["rsi"][valid_rsi], 30,
                            where=df_close["rsi"][valid_rsi] <= 30,
                            color=RSI_OS_C, alpha=0.15, zorder=2)
        ax_rsi.axvline(last_x, color=CUR_PRICE_C, linewidth=0.8, linestyle=":", alpha=0.5, zorder=5)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_yticks([30, 50, 70])
        ax_rsi.set_ylabel("RSI", color=TEXT, fontsize=9)
        cur_rsi = float(df_close["rsi"].iloc[-1]) if valid_rsi.any() else rsi_val
        rsi_color = RSI_OB_C if cur_rsi >= 70 else (RSI_OS_C if cur_rsi <= 30 else TEXT)
        ax_rsi.text(last_x + 0.3, cur_rsi, f"{cur_rsi:.1f}", color=rsi_color, fontsize=8,
                    va="center", fontweight="bold")

        vol_colors = [
            BULL_BODY if float(df_close["close"].iloc[i]) >= float(df_close["open"].iloc[i]) else BEAR_BODY
            for i in x
        ]
        ax_volume.bar(x, df_close["volume"], color=vol_colors, alpha=0.75, width=0.7, zorder=2)
        ax_volume.plot(x, df_close["vol_ma"], color=VOLMA_C, linewidth=1.4, label="Vol MA 20", zorder=3)
        ax_volume.set_ylabel("Volume", color=TEXT, fontsize=9)
        ax_volume.legend(loc="upper left", fontsize=8, facecolor="#161B22", edgecolor=SPINE, labelcolor=TEXT)
        _format_time_axis(ax_volume, df_close)

        plt.setp(ax_price.get_xticklabels(), visible=False)
        plt.setp(ax_rsi.get_xticklabels(), visible=False)

        mode_color = {
            "UPTREND": EMA34_C,
            "DOWNTREND": RESIST_C,
            "VOLATILE_RANGE": EMA89_C,
            "SIDEWAY": "#FFC107",
        }.get(market_mode, TEXT)

        title = (
            f"{symbol} - M5 dual view (160/80 candles) | "
            f"Mode: {market_mode} | H1: {h1_trend} | Score: {score}/5 | "
            f"ADX: {adx:.1f} RSI: {rsi_val:.1f} ATR: {atr:.2f} "
            f"Funding: {funding:+.4f}% | ML: {ml_dir} ({ml_conf:.0%})"
        )
        fig.suptitle(title, color=mode_color, fontsize=10, fontweight="bold", y=0.975)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        plt.close("all")
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode("utf-8")
        gc.collect()
        return data

    except Exception as e:
        import traceback
        print(f"[chart_gen] Failed: {e}\n{traceback.format_exc()}")
        try:
            import matplotlib.pyplot as _plt
            _plt.close("all")
        except Exception:
            pass
        gc.collect()
        return ""
