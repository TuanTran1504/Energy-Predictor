"""
chart_gen.py - Candlestick chart generator for the LLM analyst.

Produces a dark-themed execution+H1 chart with:
  - configurable execution pane
  - H1 context pane with decision box
  - Candlestick bodies + wicks
  - EMA 34 (green) + EMA 89 (orange)
  - H1 Support (green dashed) + Resistance (red dashed)
  - Current price line (yellow dotted)
  - RSI panel with 30/70 zones
  - Volume panel + Volume MA 20 (blue)
  - Title bar: symbol, mode, score, RSI, ATR, funding

Returns base64-encoded PNG string ready for Gemini vision API.
"""

import base64
import gc
import io
import os
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
VOLMA_C = "#2196F3"
RESIST_C = "#FF5252"
SUPPORT_C = "#69F0AE"
RSI_C = "#CE93D8"
RSI_OB_C = "#EF5350"
RSI_OS_C = "#26A69A"
CUR_PRICE_C = "#FFD600"
BOX_C = "#00BCD4"
BOX_EDGE_RATIO = 0.18
BSL_C = "#FF8A80"
SSL_C = "#80CBC4"

def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, val))


def _env_float(name: str, default: float, min_value: float, max_value: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, val))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


M5_WINDOW = _env_int("CHART_M5_WINDOW", _env_int("CHART_CLOSE_WINDOW", 100, 60, 400), 60, 500)
M15_WINDOW = _env_int("CHART_M15_WINDOW", 220, 120, 500)
H1_WINDOW = _env_int("CHART_H1_WINDOW", 300, 120, 800)
SHOW_LIQUIDITY = _env_bool("CHART_SHOW_LIQUIDITY", True)
LIQ_LOOKBACK_M5 = _env_int("CHART_LIQUIDITY_LOOKBACK_M5", 90, 30, 300)
LIQ_LOOKBACK_H1 = _env_int("CHART_LIQUIDITY_LOOKBACK_H1", 120, 40, 400)
LIQ_MAX_LEVELS = _env_int("CHART_LIQUIDITY_MAX_LEVELS", 2, 1, 5)
LIQ_TOL_PCT = _env_float("CHART_LIQUIDITY_TOL_PCT", 0.12, 0.03, 0.5)


def _exec_window_for_tf(exec_tf: str) -> int:
    tf = str(exec_tf).strip().lower()
    if tf == "15m":
        return M15_WINDOW
    return M5_WINDOW


def _apply_dark_style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
    ax.grid(color=GRID, linestyle="--", linewidth=0.4, alpha=0.7)
    ax.yaxis.label.set_color(TEXT)


def _prepare_chart_df(df_exec: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df_exec.tail(window).copy().reset_index(drop=True)
    df["ema34"] = df["close"].ewm(span=34, adjust=False).mean()
    df["ema89"] = df["close"].ewm(span=89, adjust=False).mean()
    df["vol_ma"] = df["volume"].rolling(20).mean()

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


def _swing_levels(series: pd.Series, swing_window: int, mode: str) -> list[float]:
    vals = series.astype(float).to_numpy()
    n = len(vals)
    if n < swing_window * 2 + 1:
        return []

    levels: list[float] = []
    for i in range(swing_window, n - swing_window):
        left = vals[i - swing_window:i]
        right = vals[i + 1:i + swing_window + 1]
        cur = vals[i]
        if mode == "high":
            if cur >= left.max() and cur >= right.max():
                levels.append(float(cur))
        else:
            if cur <= left.min() and cur <= right.min():
                levels.append(float(cur))
    return levels


def _cluster_levels(levels: list[float], tolerance_pct: float) -> list[dict]:
    if not levels:
        return []
    sorted_levels = sorted(levels)
    clusters: list[list[float]] = [[sorted_levels[0]]]
    for lv in sorted_levels[1:]:
        prev = clusters[-1][-1]
        rel = abs(lv - prev) / max(abs(prev), 1e-9) * 100
        if rel <= tolerance_pct:
            clusters[-1].append(lv)
        else:
            clusters.append([lv])
    out = []
    for c in clusters:
        out.append({"price": float(np.mean(c)), "touches": len(c)})
    return out


def _detect_liquidity_areas(df: pd.DataFrame, current_price: float, lookback: int) -> list[dict]:
    source = df.tail(lookback).copy()
    swing_window = 2
    highs = _swing_levels(source["high"], swing_window=swing_window, mode="high")
    lows = _swing_levels(source["low"], swing_window=swing_window, mode="low")

    high_clusters = _cluster_levels(highs, LIQ_TOL_PCT)
    low_clusters = _cluster_levels(lows, LIQ_TOL_PCT)

    areas: list[dict] = []
    for c in high_clusters:
        if c["touches"] >= 2 and c["price"] > current_price:
            areas.append({"side": "BSL", "price": c["price"], "touches": c["touches"]})
    for c in low_clusters:
        if c["touches"] >= 2 and c["price"] < current_price:
            areas.append({"side": "SSL", "price": c["price"], "touches": c["touches"]})

    areas.sort(key=lambda x: abs(x["price"] - current_price))
    bsl = [a for a in areas if a["side"] == "BSL"][:LIQ_MAX_LEVELS]
    ssl = [a for a in areas if a["side"] == "SSL"][:LIQ_MAX_LEVELS]
    return bsl + ssl


def _draw_liquidity_areas(ax_price, areas: list[dict], price_range: float):
    if not areas:
        return
    band_h = max(price_range * 0.003, 1e-9)
    bsl_labeled = False
    ssl_labeled = False
    for area in areas:
        side = area["side"]
        price = float(area["price"])
        touches = int(area["touches"])
        color = BSL_C if side == "BSL" else SSL_C
        label = None
        if side == "BSL" and not bsl_labeled:
            label = "BSL zone"
            bsl_labeled = True
        if side == "SSL" and not ssl_labeled:
            label = "SSL zone"
            ssl_labeled = True
        ax_price.axhspan(price - band_h, price + band_h, color=color, alpha=0.10, zorder=1.25)
        ax_price.axhline(price, color=color, linewidth=1.0, linestyle=":", alpha=0.9, zorder=4.6, label=label)
        ax_price.text(
            0.0, price, f" {side} x{touches}",
            color=color, fontsize=7, va="center", ha="left", zorder=6,
        )


def _draw_price_panel(ax_price, df: pd.DataFrame, sr: dict, current_price: float, *,
                      title: str = "", show_legend: bool = True,
                      annotate_levels: bool = True, last_marker: bool = True,
                      show_current_price: bool = True,
                      box_mode: str = "soft", draw_diagonals: bool = False,
                      box_bounds: tuple[float, float] | None = None,
                      liquidity_areas: list[dict] | None = None):
    import matplotlib.patches as mpatches

    x = range(len(df))
    last_x = len(df) - 1
    x_arr = np.array(list(x))
    price_range = max(float(df["high"].max() - df["low"].min()), 1e-9)
    min_body = price_range * 0.001

    support = sr.get("support")
    resistance = sr.get("resistance")
    box_bottom = None
    box_top = None
    if box_bounds is not None:
        b0, b1 = float(box_bounds[0]), float(box_bounds[1])
        if b1 > b0:
            box_bottom, box_top = b0, b1
    elif support and resistance and float(resistance) > float(support):
        box_bottom, box_top = float(support), float(resistance)

    if box_bottom is not None and box_top is not None:
        top = box_top
        bottom = box_bottom
        box_h = max(top - bottom, 1e-9)
        edge_h = box_h * BOX_EDGE_RATIO
        if box_mode == "tv":
            import matplotlib.patches as mpatches
            x0 = -0.5
            x1 = len(df) - 0.5
            rect = mpatches.Rectangle(
                (x0, bottom), x1 - x0, top - bottom,
                linewidth=1.6, edgecolor=BOX_C, facecolor=(0, 188 / 255, 212 / 255, 0.06),
                linestyle="-", zorder=1.5,
            )
            ax_price.add_patch(rect)
            if draw_diagonals:
                ax_price.plot([x0, x1], [top, bottom], color=BOX_C, linestyle=":", linewidth=1.2, alpha=0.7, zorder=2)
                ax_price.plot([x0, x1], [bottom, top], color=BOX_C, linestyle=":", linewidth=1.2, alpha=0.7, zorder=2)
        else:
            ax_price.axhspan(bottom, top, color=BOX_C, alpha=0.06, zorder=1)
            # Bias zones inside the box: top edge and bottom edge.
            ax_price.axhspan(max(bottom, top - edge_h), top, color=RESIST_C, alpha=0.08, zorder=1)
            ax_price.axhspan(bottom, min(top, bottom + edge_h), color=SUPPORT_C, alpha=0.08, zorder=1)

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
    _draw_liquidity_areas(ax_price, liquidity_areas or [], price_range)

    if resistance:
        ax_price.axhline(resistance, color=RESIST_C, linewidth=1.1, linestyle="--", alpha=0.8, zorder=5)
        if annotate_levels:
            ax_price.text(last_x, resistance, f"  R {resistance:.2f}", color=RESIST_C,
                          fontsize=8, va="bottom", ha="right")

    if support:
        ax_price.axhline(support, color=SUPPORT_C, linewidth=1.1, linestyle="--", alpha=0.8, zorder=5)
        if annotate_levels:
            ax_price.text(last_x, support, f"  S {support:.2f}", color=SUPPORT_C,
                          fontsize=8, va="top", ha="right")
            if box_bottom is not None and box_top is not None:
                ax_price.text(
                    last_x,
                    (box_bottom + box_top) / 2.0,
                    "  Decision Box",
                    color=BOX_C,
                    fontsize=8,
                    va="center",
                    ha="right",
                )

    if show_current_price:
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


def _resample_m5_to_h1(df_m5: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df_m5.columns:
        return df_m5.copy()
    df = df_m5.copy()
    dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.assign(_dt=dt).dropna(subset=["_dt"]).set_index("_dt")
    if df.empty:
        return df_m5.copy()
    agg = df.resample("1H").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    if agg.empty:
        return df_m5.copy()
    out = agg.reset_index()
    out["timestamp"] = (out["_dt"].astype("int64") // 10**6).astype("int64")
    return out[["timestamp", "open", "high", "low", "close", "volume"]]


def generate_chart(df_exec: pd.DataFrame, context: dict, df_h1: pd.DataFrame | None = None) -> str:
    """
    df_exec - raw OHLCV DataFrame with columns [timestamp, open, high, low, close, volume].
               'timestamp' should be epoch-ms (Binance format).
    context - output of strategy_core.compute_indicators() merged with sr, score, etc.

    Returns base64 PNG string or empty string on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        exec_tf = str(context.get("llm_exec_tf", "5m")).upper()
        exec_window = _exec_window_for_tf(exec_tf)
        df_exec_view = _prepare_chart_df(df_exec, exec_window)
        h1_source = df_h1 if (df_h1 is not None and not df_h1.empty) else _resample_m5_to_h1(df_exec)
        df_h1_view = _prepare_chart_df(h1_source, H1_WINDOW)

        sr = context.get("sr", {})
        score = context.get("score", 0)
        h1_trend = context.get("h1_trend", "?")
        market_mode = context.get("market_mode", h1_trend)
        rsi_val = context.get("rsi", 0)
        atr = context.get("atr_m15", 0)
        funding = context.get("funding_rate", 0)
        symbol = context.get("symbol", "BTC/USDT")
        current_price = float(context.get("current_price") or df_exec_view["close"].iloc[-1])

        fig = plt.figure(figsize=(15.5, 11), facecolor=BG)
        gs = gridspec.GridSpec(
            4, 1,
            height_ratios=[3.2, 2.0, 1.0, 1.1],
            hspace=0.08,
            left=0.04, right=0.97, top=0.93, bottom=0.05,
        )
        ax_m5 = fig.add_subplot(gs[0])
        ax_h1 = fig.add_subplot(gs[1])
        ax_rsi = fig.add_subplot(gs[2], sharex=ax_m5)
        ax_volume = fig.add_subplot(gs[3], sharex=ax_m5)

        for ax in [ax_m5, ax_h1, ax_rsi, ax_volume]:
            _apply_dark_style(ax)

        _draw_price_panel(
            ax_m5, df_exec_view, sr, current_price,
            title=f"{exec_tf} Execution View ({exec_window} candles) - use for signal candle and setup confirmation",
            show_legend=True, annotate_levels=True, last_marker=True,
            show_current_price=True, box_mode="soft",
            liquidity_areas=(
                _detect_liquidity_areas(df_exec_view, current_price, LIQ_LOOKBACK_M5)
                if SHOW_LIQUIDITY else []
            ),
        )
        h1_low = float(df_h1_view["low"].min())
        h1_high = float(df_h1_view["high"].max())
        sr_h1 = {"support": h1_low, "resistance": h1_high}

        _draw_price_panel(
            ax_h1, df_h1_view, sr_h1, current_price,
            title=f"H1 Context + Decision Box ({H1_WINDOW} candles) - use for range structure and edge behavior",
            show_legend=False, annotate_levels=True, last_marker=False,
            show_current_price=True, box_mode="tv", draw_diagonals=True,
            box_bounds=(h1_low, h1_high),
            liquidity_areas=(
                _detect_liquidity_areas(df_h1_view, current_price, LIQ_LOOKBACK_H1)
                if SHOW_LIQUIDITY else []
            ),
        )
        _format_time_axis(ax_h1, df_h1_view, max_labels=12)

        x = range(len(df_exec_view))
        last_x = len(df_exec_view) - 1
        x_arr = np.array(list(x))

        ax_r = ax_m5.twinx()
        ax_r.set_facecolor(BG)
        ax_r.set_ylim(ax_m5.get_ylim())
        ax_r.tick_params(colors=TEXT, labelsize=8)
        for spine in ax_r.spines.values():
            spine.set_color(SPINE)

        valid_rsi = df_exec_view["rsi"].notna()
        ax_rsi.plot(x_arr[valid_rsi], df_exec_view["rsi"][valid_rsi], color=RSI_C, linewidth=1.4, zorder=3)
        ax_rsi.axhline(70, color=RSI_OB_C, linewidth=0.8, linestyle="--", alpha=0.8)
        ax_rsi.axhline(30, color=RSI_OS_C, linewidth=0.8, linestyle="--", alpha=0.8)
        ax_rsi.axhline(50, color=TEXT, linewidth=0.5, linestyle="--", alpha=0.3)
        ax_rsi.fill_between(x_arr[valid_rsi], df_exec_view["rsi"][valid_rsi], 70,
                            where=df_exec_view["rsi"][valid_rsi] >= 70,
                            color=RSI_OB_C, alpha=0.15, zorder=2)
        ax_rsi.fill_between(x_arr[valid_rsi], df_exec_view["rsi"][valid_rsi], 30,
                            where=df_exec_view["rsi"][valid_rsi] <= 30,
                            color=RSI_OS_C, alpha=0.15, zorder=2)
        ax_rsi.axvline(last_x, color=CUR_PRICE_C, linewidth=0.8, linestyle=":", alpha=0.5, zorder=5)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_yticks([30, 50, 70])
        ax_rsi.set_ylabel("RSI", color=TEXT, fontsize=9)
        cur_rsi = float(df_exec_view["rsi"].iloc[-1]) if valid_rsi.any() else rsi_val
        rsi_color = RSI_OB_C if cur_rsi >= 70 else (RSI_OS_C if cur_rsi <= 30 else TEXT)
        ax_rsi.text(last_x + 0.3, cur_rsi, f"{cur_rsi:.1f}", color=rsi_color, fontsize=8,
                    va="center", fontweight="bold")

        vol_colors = [
            BULL_BODY if float(df_exec_view["close"].iloc[i]) >= float(df_exec_view["open"].iloc[i]) else BEAR_BODY
            for i in x
        ]
        ax_volume.bar(x, df_exec_view["volume"], color=vol_colors, alpha=0.75, width=0.7, zorder=2)
        ax_volume.plot(x, df_exec_view["vol_ma"], color=VOLMA_C, linewidth=1.4, label="Vol MA 20", zorder=3)
        ax_volume.set_ylabel("Volume", color=TEXT, fontsize=9)
        ax_volume.legend(loc="upper left", fontsize=8, facecolor="#161B22", edgecolor=SPINE, labelcolor=TEXT)
        _format_time_axis(ax_volume, df_exec_view)

        plt.setp(ax_m5.get_xticklabels(), visible=False)
        plt.setp(ax_h1.get_xticklabels(), visible=False)
        plt.setp(ax_rsi.get_xticklabels(), visible=False)

        mode_color = {
            "UPTREND": EMA34_C,
            "DOWNTREND": RESIST_C,
            "VOLATILE_RANGE": EMA89_C,
            "SIDEWAY": "#FFC107",
        }.get(market_mode, TEXT)

        title = (
            f"{symbol} - {exec_tf} + H1 view ({exec_window}/{H1_WINDOW} candles) | "
            f"Mode: {market_mode} | H1: {h1_trend} | Score: {score}/5 | "
            f"RSI: {rsi_val:.1f} ATR: {atr:.2f} "
            f"Funding: {funding:+.4f}%"
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
