import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import Counter
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SwingHunter V7.8 - Portfolio Defaults", layout="wide")

APP_VERSION = "V7.8"

# ==========================================================
# 1. Security + Settings
# ==========================================================
# For quick local testing the fallback password is 1234.
# For Streamlit Cloud, define APP_PASSWORD in Secrets and it will override this.
LOCAL_TEST_PASSWORD = "Pk0105Ak2701"

try:
    APP_PASSWORD = st.secrets.get("APP_PASSWORD", LOCAL_TEST_PASSWORD)
    MY_EMAIL = st.secrets.get("MY_EMAIL", "")
except Exception:
    APP_PASSWORD = os.getenv("APP_PASSWORD", LOCAL_TEST_PASSWORD)
    MY_EMAIL = os.getenv("MY_EMAIL", "")

WATCHLIST = [
    'AAPL','MSFT','NVDA','TSLA','AMZN','META','GOOGL','NFLX','AMD','AVGO','TSM','QCOM',
    'CRWD','PANW','PLTR','SNOW','DDOG','NET','SMCI','COIN','MSTR','HOOD','SOFI','SQ',
    'PYPL','AFRM','SHOP','BABA','MELI','WMT','TGT','COST','HD','UBER','ABNB','SPOT',
    'DKNG','DIS','NKE','SBUX','MCD','JPM','BAC','GS','MS','V','MA','LLY','NVO','UNH','CAT','BA','MRNA'
]

MOMENTUM_TICKERS = {
    'AMD','NVDA','TSLA','DDOG','NET','QCOM','CRWD','PANW','AVGO','AMZN',
    'MSTR','COIN','SMCI','PLTR','ARM','MU','MRVL','TSM','META','GOOGL'
}

HISTORY_FILE = "swinghunter_history.csv"

# ==========================================================
# 2. Strategy Parameters
# ==========================================================
@dataclass
class StrategyParams:
    mode: str
    armed_threshold: int
    actionable_threshold: int
    building_threshold: int
    min_rr: float
    max_risk_pct: float
    allow_bear_market_orders: bool
    bear_market_min_edge: int
    rs5_min: float
    rs20_min: float
    overbought_rsi: float
    max_20d_run: float
    stop_atr_breakout: float
    stop_atr_pullback: float
    tp1_pct: float
    tp1_fraction: float
    runner_target_pct: float
    use_trailing_runner: bool
    min_atr_pct: float
    momentum_bonus: int
    allow_leader_recovery: bool
    leader_recovery_rs20_min: float
    leader_recovery_20d_min: float
    auto_trade_momentum_only: bool
    risk_budget_default: float
    runner_trail_close_only: bool
    max_positions_default: int


def get_params(mode: str) -> StrategyParams:
    if mode == "Conservative":
        return StrategyParams(
            mode="Conservative",
            armed_threshold=85,
            actionable_threshold=75,
            building_threshold=50,
            min_rr=1.60,
            max_risk_pct=6.5,
            allow_bear_market_orders=False,
            bear_market_min_edge=65,
            rs5_min=2.5,
            rs20_min=6.0,
            overbought_rsi=76,
            max_20d_run=25,
            stop_atr_breakout=1.9,
            stop_atr_pullback=1.6,
            tp1_pct=0.07,
            tp1_fraction=0.50,
            runner_target_pct=0.20,  # display only; runner is mainly trailing-based
            use_trailing_runner=True,
            min_atr_pct=1.8,
            momentum_bonus=5,
            allow_leader_recovery=False,
            leader_recovery_rs20_min=12.0,
            leader_recovery_20d_min=10.0,
            auto_trade_momentum_only=True,
            risk_budget_default=120.0,
            runner_trail_close_only=True,
            max_positions_default=4,
        )

    if mode == "Aggressive":
        return StrategyParams(
            mode="Aggressive",
            armed_threshold=62,
            actionable_threshold=55,
            building_threshold=32,
            min_rr=1.05,
            max_risk_pct=11.0,
            allow_bear_market_orders=True,
            bear_market_min_edge=40,
            rs5_min=0.8,
            rs20_min=2.5,
            overbought_rsi=84,
            max_20d_run=55,
            stop_atr_breakout=2.6,
            stop_atr_pullback=2.1,
            tp1_pct=0.08,
            tp1_fraction=0.50,
            runner_target_pct=0.30,  # display only; runner is mainly trailing-based
            use_trailing_runner=True,
            min_atr_pct=1.5,
            momentum_bonus=12,
            allow_leader_recovery=True,
            leader_recovery_rs20_min=6.0,
            leader_recovery_20d_min=5.0,
            auto_trade_momentum_only=True,
            risk_budget_default=150.0,
            runner_trail_close_only=True,
            max_positions_default=7,
        )

    # V7.5 Power Swing: auto-orders only for Momentum/Growth, SMA200 hard gate for orders, TP1 8%, runner trails by EMA21 close, fixed-risk sizing.
    return StrategyParams(
        mode="Balanced",
        armed_threshold=68,
        actionable_threshold=60,
        building_threshold=38,
        min_rr=1.45,
        max_risk_pct=11.0,
        allow_bear_market_orders=False,  # V7.5: no automatic orders when SPY is weak
        bear_market_min_edge=60,
        rs5_min=1.5,
        rs20_min=4.0,
        overbought_rsi=82,
        max_20d_run=50,
        stop_atr_breakout=2.7,
        stop_atr_pullback=2.3,
        tp1_pct=0.08,
        tp1_fraction=0.50,
        runner_target_pct=0.25,  # display only; runner is mainly trailing-based
        use_trailing_runner=True,
        min_atr_pct=1.8,
        momentum_bonus=10,
        allow_leader_recovery=True,
        leader_recovery_rs20_min=8.0,
        leader_recovery_20d_min=6.0,
        auto_trade_momentum_only=True,
        risk_budget_default=200.0,
        runner_trail_close_only=True,
        max_positions_default=5,
    )


def weighted_reward_pct(params: StrategyParams) -> float:
    """Expected reward percentage based on partial TP1 + runner target."""
    return params.tp1_pct * params.tp1_fraction + params.runner_target_pct * (1 - params.tp1_fraction)


def calc_rr_from_weighted_reward(entry: float, stop: float, params: StrategyParams) -> float:
    risk = entry - stop
    if risk <= 0:
        return 0.0
    return (entry * weighted_reward_pct(params)) / risk

# ==========================================================
# 3. Utility Functions
# ==========================================================
def flatten_download(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels[1]) == 1:
            df.columns = df.columns.get_level_values(0)
    return df


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


@st.cache_data(ttl=3600)
def download_single(ticker: str, period: str = "300d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        df = flatten_download(df)
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_spy_context():
    spy = download_single("SPY", "300d")
    if spy.empty or "Close" not in spy:
        return None, "UNKNOWN"
    spy_close = spy["Close"].dropna()
    if len(spy_close) < 50:
        return None, "UNKNOWN"
    sma20 = spy_close.rolling(20).mean().iloc[-1]
    trend = "BULL" if spy_close.iloc[-1] > sma20 else "BEAR"
    return spy_close, trend


def get_earnings_status(ticker: str, days_ahead: int = 3):
    try:
        tkr = yf.Ticker(ticker)
        cal = tkr.get_earnings_dates(limit=5)
        if cal is not None and not cal.empty:
            now = pd.Timestamp.now(tz="UTC")
            future = cal.index[cal.index > now]
            if not future.empty:
                days = int((future[0] - now).days)
                if 0 <= days <= days_ahead:
                    return "DANGER", days
    except Exception:
        pass
    return "CLEAR", 0


def save_scan_history(df_scan: pd.DataFrame):
    required_cols = {"מניה", "החלטה", "ציון_כולל"}
    if df_scan.empty or not required_cols.issubset(df_scan.columns):
        return
    df_save = df_scan[["מניה", "החלטה", "ציון_כולל"]].copy()
    df_save["scan_date"] = datetime.now().strftime("%Y-%m-%d")
    try:
        old = pd.read_csv(HISTORY_FILE)
        combined = pd.concat([old, df_save], ignore_index=True)
        combined = combined.drop_duplicates(subset=["מניה", "scan_date"], keep="last")
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        combined = combined[combined["scan_date"] >= cutoff]
        combined.to_csv(HISTORY_FILE, index=False)
    except Exception:
        df_save.to_csv(HISTORY_FILE, index=False)


def get_setup_persistence(ticker: str):
    try:
        hist = pd.read_csv(HISTORY_FILE)
        recent = hist[hist["מניה"] == ticker].tail(5)
        if recent.empty:
            return 0, 0.0
        watch_days = recent["החלטה"].astype(str).str.contains(
            "Building Pressure|ARMED|Actionable|למעקב|פעיל",
            regex=True
        ).sum()
        score_trend = recent["ציון_כולל"].diff().fillna(0).sum()
        return int(watch_days), float(score_trend)
    except Exception:
        return 0, 0.0

# ==========================================================
# 4. Indicators + Edge Modules
# ==========================================================
def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compression_score(df: pd.DataFrame):
    try:
        high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]
        range_5 = ((high.tail(5).max() - low.tail(5).min()) / close.iloc[-1]) * 100
        avg_range_20 = (((high - low) / close).rolling(20).mean().iloc[-1]) * 100
        vol_5 = vol.tail(5).mean()
        vol_20 = vol.rolling(20).mean().iloc[-1]
        recent_lows = low.tail(5).values
        higher_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] >= recent_lows[i - 1])
        score = 0
        if range_5 < avg_range_20 * 0.75:
            score += 20
        if vol_5 < vol_20 * 0.85:
            score += 10
        if higher_lows >= 3:
            score += 15
        return score, round(range_5, 2), higher_lows
    except Exception:
        return 0, np.nan, 0


def relative_strength_vs_spy(close: pd.Series, spy_close: pd.Series):
    try:
        common = close.index.intersection(spy_close.index)
        close = close.loc[common]
        spy_close = spy_close.loc[common]
        if len(close) < 22:
            return 0.0, 0.0
        stock_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
        stock_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100
        spy_5d = (spy_close.iloc[-1] / spy_close.iloc[-6] - 1) * 100
        spy_20d = (spy_close.iloc[-1] / spy_close.iloc[-21] - 1) * 100
        return stock_5d - spy_5d, stock_20d - spy_20d
    except Exception:
        return 0.0, 0.0


def failed_breakdown_recovery(df: pd.DataFrame):
    try:
        close = df["Close"]
        low = df["Low"]
        sma5 = close.rolling(5).mean().iloc[-1]
        support_20 = float(low.iloc[-21:-5].min())
        broke_support = float(low.iloc[-5:-1].min()) < support_20 * 0.99
        reclaimed = float(close.iloc[-1]) > support_20 * 1.005 and float(close.iloc[-1]) > sma5
        if broke_support and reclaimed:
            return True, support_20
    except Exception:
        pass
    return False, 0.0


def post_event_drift(df: pd.DataFrame):
    try:
        close, open_p, high, low, vol = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
        avg_vol = vol.rolling(20).mean().iloc[-1]
        recent_vol = vol.tail(10)
        if recent_vol.max() <= avg_vol * 2.5:
            return False, 0.0, 0.0
        event_idx = recent_vol.idxmax()
        event_pos = df.index.get_loc(event_idx)
        if event_pos >= len(df) - 1:
            return False, 0.0, 0.0
        event_open = float(open_p.iloc[event_pos])
        event_close = float(close.iloc[event_pos])
        if (event_close / event_open - 1) * 100 < -5.0:
            return False, 0.0, 0.0
        event_high = float(high.iloc[event_pos])
        event_low = float(low.iloc[event_pos])
        if float(close.iloc[-1]) > event_low and low.tail(3).is_monotonic_increasing:
            return True, event_high, event_low
    except Exception:
        pass
    return False, 0.0, 0.0

# ==========================================================
# 5. Live Edge Engine
# ==========================================================
def analyze_edge(ticker: str, spy_close: pd.Series, market_trend: str, investment_budget: float, risk_budget: float, params: StrategyParams):
    df = download_single(ticker, "300d")
    if df.empty or len(df) < 220:
        return None
    close, highs, lows = df["Close"], df["High"], df["Low"]
    last_price = safe_float(close.iloc[-1])
    if not np.isfinite(last_price) or last_price <= 5:
        return None

    sma200 = close.rolling(200).mean().iloc[-1]
    ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    res_20d = float(highs.iloc[-21:-1].max())
    prev_high = float(highs.iloc[-2])
    dist_to_res = (res_20d / last_price - 1) * 100
    atr_val = calc_atr(df, 14).iloc[-1]
    rsi_val = calc_rsi(close, 14).iloc[-1]
    change_20d = (last_price / close.iloc[-21] - 1) * 100

    comp, range_5, higher_lows = compression_score(df)
    rs_5d, rs_20d = relative_strength_vs_spy(close, spy_close)
    failed_break, reclaimed_level = failed_breakdown_recovery(df)
    drift, event_high, event_low = post_event_drift(df)
    watch_days, score_trend = get_setup_persistence(ticker)
    earn_status, earn_days = get_earnings_status(ticker)

    atr_pct = (atr_val / last_price) * 100 if last_price else np.nan

    leader_recovery = (
        params.allow_leader_recovery
        and last_price < sma200
        and rs_20d > params.leader_recovery_rs20_min
        and change_20d > params.leader_recovery_20d_min
        and last_price > ema8
        and last_price > ema21
    )

    setup_score = 0
    if last_price > sma200:
        setup_score += 15
    elif leader_recovery:
        setup_score += 12  # recovery leaders are allowed even below SMA200
    if last_price > ema21:
        setup_score += 10
    if dist_to_res < 4.5:
        setup_score += 15
    if ticker in MOMENTUM_TICKERS:
        setup_score += params.momentum_bonus

    edge_score = 0
    edge_notes = []
    reject_reason = ""

    if comp >= 25 and (dist_to_res < 4.5 or rs_5d > 0 or rs_20d > 0):
        edge_score += comp
        edge_notes.append("Compression")
    if rs_5d > params.rs5_min or rs_20d > params.rs20_min:
        edge_score += 20
        edge_notes.append("RS Leader")
    if failed_break:
        edge_score += 35
        edge_notes.append("Failed Breakdown")
    if drift:
        edge_score += 25
        edge_notes.append("Post-Event Drift")
    if watch_days >= 2 and score_trend > 0:
        edge_score += 15
        edge_notes.append("Building Pressure")
    if leader_recovery:
        edge_score += 25
        edge_notes.append("Leader Recovery")

    final_rank = setup_score + edge_score
    setup_type = "No Setup"

    if earn_status == "DANGER":
        icon, decision, reject_reason, final_rank = "⚠️", "DANGER", f"דוח קרוב בעוד {earn_days} ימים", 0
    elif (rsi_val > params.overbought_rsi or change_20d > params.max_20d_run) and not leader_recovery:
        icon, decision, reject_reason, final_rank = "🔥", "חם מדי", "מתוח מדי / סכנת רדיפה", min(final_rank, 30)
    elif market_trend == "BEAR" and (not params.allow_bear_market_orders or edge_score < params.bear_market_min_edge):
        icon, decision, reject_reason, final_rank = "🔴", "Dormant", "Market BEAR + Edge לא מספיק", min(final_rank, 40)
    elif atr_pct < params.min_atr_pct and ticker not in MOMENTUM_TICKERS and edge_score < 35:
        icon, decision, reject_reason, final_rank = "🔴", "Dormant", f"ATR% נמוך מדי ({atr_pct:.1f}%)", min(final_rank, 40)
    elif final_rank >= params.armed_threshold and edge_notes:
        icon, decision = "🟢", "ARMED"
    elif final_rank >= params.actionable_threshold and edge_notes:
        icon, decision = "🟢", "Actionable"
    elif final_rank >= params.building_threshold:
        icon, decision = "🟡", "Building Pressure"
    else:
        icon, decision, reject_reason = "🔴", "Dormant", "ציון נמוך מדי"

    if "Compression" in edge_notes and dist_to_res < 3.5:
        setup_type = "Pre-Breakout Compression"
    elif failed_break:
        setup_type = "Failed Breakdown Recovery"
    elif drift:
        setup_type = "Post-Event Drift"
    elif leader_recovery:
        setup_type = "Leader Recovery"
    elif "RS Leader" in edge_notes and last_price > ema21:
        setup_type = "RS Pullback"
    elif last_price > res_20d:
        setup_type = "Momentum Breakout"

    order_data = None
    can_create_order = decision in ["ARMED", "Actionable"]

    # V7.5 safety rails:
    # 1) automatic orders only in the momentum/growth universe
    # 2) hard SMA200 gate for actual orders
    # 3) Leader Recovery is radar-only, not auto-trade
    if ticker not in MOMENTUM_TICKERS and params.auto_trade_momentum_only:
        can_create_order = False
        if not reject_reason:
            reject_reason = "לא Momentum/Growth — רדאר בלבד"
    if last_price < sma200:
        can_create_order = False
        if leader_recovery:
            reject_reason = "Leader Recovery מתחת SMA200 — רדאר בלבד"
        elif not reject_reason:
            reject_reason = "מתחת SMA200 — אין פקודה אוטומטית"
    if setup_type == "Leader Recovery":
        can_create_order = False
        reject_reason = "Leader Recovery — Watchlist only"

    if can_create_order:
        p_type = None
        entry = 0.0
        e_disp = ""

        if setup_type in ["Pre-Breakout Compression", "Post-Event Drift", "Momentum Breakout"]:
            trigger = res_20d if setup_type != "Post-Event Drift" else event_high
            entry = round(trigger * 1.002, 2)
            if last_price < entry:
                p_type = "BUY STOP LIMIT"
                e_disp = f"Stop {entry} / Lmt {round(entry * 1.008, 2)}"
            else:
                reject_reason = "כבר פרצה. לא רודפים."
                icon, decision = "🟡", "Building Pressure"

        elif setup_type == "Leader Recovery":
            # Recovery leaders are allowed below SMA200 if RS is exceptional.
            # Use shallow EMA8 pullback; if too extended, use a next-day stop above yesterday's high.
            shallow_entry = round(min(ema8 * 1.003, last_price * 0.99), 2)
            momentum_entry = round(prev_high * 1.002, 2)
            if last_price > shallow_entry and (last_price / ema8 - 1) < 0.055:
                entry = shallow_entry
                p_type = "BUY LIMIT"
                e_disp = f"{entry} (Leader Recovery EMA8 pullback)"
            elif last_price < momentum_entry:
                entry = momentum_entry
                p_type = "BUY STOP LIMIT"
                e_disp = f"Recovery stop {entry} / Lmt {round(entry * 1.006, 2)}"
            else:
                reject_reason = "Leader Recovery מתוחה מדי ללא נקודת כניסה נקייה"

        elif setup_type in ["Failed Breakdown Recovery", "RS Pullback"]:
            # Dynamic pullback: high-edge RS leaders can use EMA8 shallow pullback.
            if edge_score >= 55 and "RS Leader" in edge_notes:
                pullback_base = ema8
                entry_label = "EMA8 shallow pullback"
            else:
                pullback_base = ema21
                entry_label = "EMA21 pullback"
            entry = min(round(pullback_base * 1.003, 2), round(last_price * 0.995, 2))
            p_type = "BUY LIMIT"
            e_disp = f"{entry} ({entry_label})"

        # Extra V7.1 trigger plan for Building Pressure that is not yet orderable
        trigger_plan = ""
        if setup_type == "RS Pullback":
            trigger_plan = f"Buy Limit around EMA8/EMA21 or Buy Stop above yesterday high {round(prev_high * 1.002, 2)}"
        elif setup_type in ["Pre-Breakout Compression", "Post-Event Drift"]:
            trigger_plan = f"Buy Stop above trigger {round((res_20d if setup_type != 'Post-Event Drift' else event_high) * 1.002, 2)}"

        if p_type and entry > 0:
            if setup_type in ["Pre-Breakout Compression", "Post-Event Drift", "Momentum Breakout", "Leader Recovery"]:
                stop = round(entry - params.stop_atr_breakout * atr_val, 2)
            elif setup_type == "Failed Breakdown Recovery":
                stop = round(reclaimed_level * 0.985, 2)
            else:
                stop = round(entry - params.stop_atr_pullback * atr_val, 2)

            risk_per_share = entry - stop
            if risk_per_share > 0:
                risk_pct = risk_per_share / entry * 100
                rr = calc_rr_from_weighted_reward(entry, stop, params)
                if risk_pct <= params.max_risk_pct and rr >= params.min_rr:
                    shares_by_cash = int(investment_budget / entry)
                    shares_by_risk = int(risk_budget / risk_per_share)
                    shares = min(shares_by_cash, shares_by_risk)
                    if shares > 0:
                        actual_risk = round(shares * risk_per_share, 2)
                        order_data = {
                            "מניה": ticker,
                            "פעולה": p_type,
                            "Edge": " + ".join(edge_notes),
                            "כניסה": e_disp,
                            "כמות": shares,
                            "השקעה $": f"${round(shares * entry, 2)}",
                            "סיכון $": f"${actual_risk}",
                            "יעד ראשון": round(entry * (1 + params.tp1_pct), 2),
                            "יעד ראנר": "אין יעד קבוע — trailing EMA21",
                            "סטופ": stop,
                            "סיכון %": f"{round(risk_pct, 1)}%",
                            "R/R משוקלל": round(rr, 2),
                            "ניהול יציאה": f"Sell {int(params.tp1_fraction * 100)}% at TP1, trail rest by EMA21 close",
                            "תוקף": "DAY ONLY"
                        }
                else:
                    reject_reason = f"R/R או סיכון לא עומד בסף ({round(rr, 2)} / {round(risk_pct, 1)}%)"
    else:
        trigger_plan = ""
        if decision == "Building Pressure":
            if "RS Leader" in edge_notes:
                trigger_plan = f"אם עובר את high של אתמול: {round(prev_high * 1.002, 2)} — לבדוק Buy Stop"
            elif dist_to_res < 4.5:
                trigger_plan = f"אם פורץ התנגדות: {round(res_20d * 1.002, 2)} — לבדוק Buy Stop"

    return {
        "scanner": {
            "מניה": ticker,
            "החלטה": f"{icon} {decision}",
            "ציון_כולל": int(final_rank),
            "Setup Score": int(setup_score),
            "Edge Score": int(edge_score),
            "תבנית": setup_type,
            "Market": market_trend,
            "Watch Days": watch_days,
            "Comp. Score": comp,
            "Range 5D %": range_5,
            "Higher Lows": higher_lows,
            "RS (5D)": f"{round(rs_5d, 1)}%",
            "RS (20D)": f"{round(rs_20d, 1)}%",
            "RSI": int(rsi_val) if np.isfinite(rsi_val) else None,
            "ATR%": f"{round(atr_pct, 1)}%",
            "20D Run": f"{round(change_20d, 1)}%",
            "Trigger Plan": trigger_plan,
            "Edge Notes": " + ".join(edge_notes) if edge_notes else "None",
            "סיבת פסילה": reject_reason
        },
        "order": order_data
    }

# ==========================================================
# 6. Backtester V7.1 - Active TP1 + Runner + Missed Winners
# ==========================================================
@st.cache_data(show_spinner=False)
def fetch_backtest_data(months: int):
    # Fix: previous code did not download enough warmup history.
    # Need about 220 trading days warmup + requested test window.
    end = datetime.now()
    start = end - timedelta(days=430 + months * 45)
    return yf.download(WATCHLIST + ["SPY", "QQQ"], start=start, end=end, progress=False, auto_adjust=False)


def get_panel(data: pd.DataFrame, field: str) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        return data[field]
    raise ValueError("Expected MultiIndex data from yfinance for multi-ticker backtest")


def diagnose_missed_ticker(prices, highs, lows, ticker: str, spy_series: pd.Series, start_idx: int, end_idx: int, params: StrategyParams):
    """Post-hoc diagnostic for top movers: why did the strategy probably miss them?"""
    blockers = Counter()
    orders_possible = 0
    leader_days = 0
    leader_recovery_days = 0
    near_resistance_days = 0
    already_triggered_days = 0
    days_above_ema8 = 0
    days_above_ema21 = 0
    max_rs5 = -999.0
    max_rs20 = -999.0
    max_rsi = 0.0
    max_20d_run = -999.0

    # We scan each historical day in the tested window, using the same rough logic as the backtest order generator.
    for i in range(max(220, start_idx), end_idx):
        try:
            c = prices[ticker].iloc[i - 220:i + 1].dropna()
            h = highs[ticker].iloc[i - 220:i + 1].dropna()
            l = lows[ticker].iloc[i - 220:i + 1].dropna()
            spy_slice = spy_series.iloc[i - 220:i + 1].dropna()

            if len(c) < 220 or len(h) < 220 or len(l) < 220 or len(spy_slice) < 220:
                blockers["Insufficient data"] += 1
                continue

            last_p = float(c.iloc[-1])
            sma200 = float(c.rolling(200).mean().iloc[-1])
            ema8 = float(c.ewm(span=8, adjust=False).mean().iloc[-1])
            ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
            prev_hi = float(h.iloc[-2])
            res20 = float(h.iloc[-21:-1].max())

            atr = float(pd.concat([
                h - l,
                (h - c.shift()).abs(),
                (l - c.shift()).abs()
            ], axis=1).max(axis=1).rolling(14).mean().iloc[-1])

            rsi = float(calc_rsi(c).iloc[-1])
            chg20 = (last_p / c.iloc[-21] - 1) * 100
            max_rsi = max(max_rsi, rsi if np.isfinite(rsi) else 0)
            max_20d_run = max(max_20d_run, chg20 if np.isfinite(chg20) else -999)

            if last_p > ema8:
                days_above_ema8 += 1
            if last_p > ema21:
                days_above_ema21 += 1

            below_sma200 = last_p < sma200

            spy_sma20 = spy_slice.rolling(20).mean().iloc[-1]
            market_bull = float(spy_slice.iloc[-1]) > float(spy_sma20)
            if not market_bull and not params.allow_bear_market_orders:
                blockers["Market BEAR"] += 1
                continue

            common = c.index.intersection(spy_slice.index)
            if len(common) < 21:
                rs5, rs20 = 0.0, 0.0
            else:
                rs5, rs20 = relative_strength_vs_spy(c.loc[common], spy_slice.loc[common])

            max_rs5 = max(max_rs5, rs5)
            max_rs20 = max(max_rs20, rs20)
            edge_ok = rs5 > params.rs5_min or rs20 > params.rs20_min

            strong_leader = (
                rs20 > params.rs20_min + 2
                and chg20 > 10
                and last_p > ema8
                and last_p > ema21
            )
            leader_recovery = (
                params.allow_leader_recovery
                and below_sma200
                and rs20 > params.leader_recovery_rs20_min
                and chg20 > params.leader_recovery_20d_min
                and last_p > ema8
                and last_p > ema21
            )
            if strong_leader:
                leader_days += 1
            if leader_recovery:
                leader_recovery_days += 1

            if below_sma200 and not leader_recovery:
                blockers["Below SMA200"] += 1
                continue

            if (rsi > params.overbought_rsi or chg20 > params.max_20d_run) and not (strong_leader or leader_recovery):
                blockers["Overextended filter"] += 1
                continue

            if not edge_ok and not (strong_leader or leader_recovery):
                blockers["Weak relative strength"] += 1
                continue

            dist_to_res = (res20 / last_p - 1)
            possible = False

            if 0 < dist_to_res < 0.045:
                near_resistance_days += 1
                entry = round(res20 * 1.002, 2)
                if last_p < entry:
                    stop = round(entry - params.stop_atr_breakout * atr, 2)
                    possible = True
                else:
                    already_triggered_days += 1
                    blockers["Already above breakout trigger"] += 1

            elif leader_recovery:
                # Watchlist only in V7.5.
                blockers["Leader Recovery watchlist only"] += 1

            elif strong_leader:
                # Leader Mode: shallow EMA8 pullback or stop above yesterday high.
                shallow_entry = round(min(ema8 * 1.003, last_p * 0.995), 2)
                momentum_entry = round(prev_hi * 1.002, 2)
                if last_p > shallow_entry and (last_p / ema8 - 1) < 0.05:
                    entry = shallow_entry
                    stop = round(entry - params.stop_atr_pullback * atr, 2)
                    possible = True
                elif last_p < momentum_entry:
                    entry = momentum_entry
                    stop = round(entry - params.stop_atr_breakout * atr, 2)
                    possible = True
                else:
                    blockers["Leader too extended / no clean entry"] += 1

            elif last_p > ema21 and (last_p / ema8 - 1) < 0.035:
                base = ema8 if rs5 > params.rs5_min else ema21
                entry = min(round(base * 1.003, 2), round(last_p * 0.995, 2))
                if last_p > entry:
                    stop = round(entry - params.stop_atr_pullback * atr, 2)
                    possible = True
                else:
                    blockers["No pullback entry"] += 1
            else:
                blockers["No valid setup"] += 1

            if possible:
                if entry <= stop:
                    blockers["Invalid risk structure"] += 1
                    continue
                risk = entry - stop
                risk_pct = risk / entry * 100
                rr = calc_rr_from_weighted_reward(entry, stop, params)
                if risk_pct > params.max_risk_pct:
                    blockers[f"Risk too high"] += 1
                    continue
                if rr < params.min_rr:
                    blockers[f"R/R too low"] += 1
                    continue
                orders_possible += 1

        except Exception:
            blockers["Data/Calc error"] += 1
            continue

    likely_reason = blockers.most_common(1)[0][0] if blockers else "No blocker identified"
    return {
        "Orders Possible": orders_possible,
        "Likely Miss Reason": likely_reason,
        "Top Blockers": "; ".join([f"{k} ({v})" for k, v in blockers.most_common(3)]),
        "Max RS 5D": round(max_rs5, 1) if max_rs5 > -900 else None,
        "Max RS 20D": round(max_rs20, 1) if max_rs20 > -900 else None,
        "Max RSI": round(max_rsi, 1),
        "Max 20D Run": round(max_20d_run, 1) if max_20d_run > -900 else None,
        "Leader Days": leader_days,
        "Leader Recovery Days": leader_recovery_days,
        "Near Resistance Days": near_resistance_days,
        "Already Triggered Days": already_triggered_days,
        "Days > EMA8": days_above_ema8,
        "Days > EMA21": days_above_ema21,
    }



def run_backtest_simulation(
    data: pd.DataFrame,
    investment_per_trade: float,
    risk_budget: float,
    params: StrategyParams,
    months: int,
    starting_capital: float = 10000.0,
    max_positions: int = 6,
    diagnostics_top_n: int = 20,
    slippage_pct: float = 0.001,
    commission_per_fill: float = 0.0,
    sizing_mode: str = "Fixed Amount",
    position_pct: float = 0.20,
    max_exposure_pct: float = 0.80,
    cooldown_after_exit_days: int = 10,
    cooldown_after_loss_days: int = 15,
    double_loss_freeze_days: int = 30
):
    """
    V7.6 Backtest Integrity Fix:
    1) DAY orders are generated at close of day i and can execute only on day i+1.
    2) A position cannot be stopped/take-profit on the same day it was opened.
       This avoids false stop-outs caused by not knowing intraday sequence.
    3) BUY STOP LIMIT is simulated with both stop and limit:
       if the stock gaps above the limit, the order is not filled.
    4) Open positions are liquidated at the final close and counted in PnL.
    5) TradeID is used so Win Rate / Avg Win / Avg Loss are calculated per full trade,
       not per partial exit.
    6) Basic slippage and commission are included.
    """
    prices = get_panel(data, "Close")
    highs = get_panel(data, "High")
    lows = get_panel(data, "Low")
    opens = get_panel(data, "Open")

    cash = starting_capital
    positions = {}
    pending_orders = {}

    equity_curve, dates, trade_log = [], [], []
    total_invested = 0.0
    exposure_days = 0
    max_equity = starting_capital
    max_drawdown = 0.0
    orders_created = 0
    orders_executed = 0
    orders_expired = 0
    orders_gap_rejected = 0
    next_trade_id = 1

    # Anti-churn state: prevents repeated trading in the same noisy ticker.
    cooldown_until = {}
    recent_loss_dates = {}

    requested_test_days = int(months * 21.5)
    start_idx = max(220, len(prices) - requested_test_days)
    test_start_idx = start_idx

    def apply_buy_slippage(price: float) -> float:
        return price * (1 + slippage_pct)

    def apply_sell_slippage(price: float) -> float:
        return price * (1 - slippage_pct)

    def close_shares(ticker, pos, sell_shares, sell_price_raw, exit_label, date_str):
        nonlocal cash
        sell_shares = int(min(sell_shares, pos["remaining"]))
        if sell_shares <= 0:
            return 0.0

        sell_price = apply_sell_slippage(float(sell_price_raw))
        gross_revenue = sell_shares * sell_price
        exit_commission = commission_per_fill

        entry_commission_alloc = pos["entry_commission_remaining"] * (sell_shares / pos["remaining"])
        pos["entry_commission_remaining"] -= entry_commission_alloc

        cash += gross_revenue - exit_commission

        pnl = (
            gross_revenue
            - (pos["entry"] * sell_shares)
            - entry_commission_alloc
            - exit_commission
        )

        trade_log.append({
            "TradeID": pos["trade_id"],
            "Date": date_str,
            "EntryDate": pos.get("entry_date", ""),
            "Ticker": ticker,
            "Exit": exit_label,
            "Setup": pos.get("setup", ""),
            "Shares": sell_shares,
            "Entry": round(pos["entry"], 2),
            "ExitPrice": round(sell_price, 2),
            "PnL": round(pnl, 2)
        })

        pos["remaining"] -= sell_shares
        return pnl

    for i in range(start_idx, len(prices) - 1):
        today = prices.index[i]
        today_str = today.strftime("%Y-%m-%d")

        # --------------------------------------------------
        # 1. Execute pending DAY orders generated yesterday
        # --------------------------------------------------
        for ticker, order in list(pending_orders.items()):
            if ticker in positions or len(positions) >= max_positions:
                continue

            try:
                today_open = safe_float(opens[ticker].iloc[i])
                today_high = safe_float(highs[ticker].iloc[i])
                today_low = safe_float(lows[ticker].iloc[i])
            except Exception:
                continue

            if not np.isfinite(today_open):
                continue

            executed = False
            raw_exec_price = 0.0

            if order["type"] == "BUY LIMIT":
                if today_low <= order["price"]:
                    # Conservative: assume fill at limit, not better.
                    raw_exec_price = order["price"]
                    executed = True

            elif order["type"] == "BUY STOP LIMIT":
                stop_price = order["price"]
                limit_price = order.get("limit", stop_price * 1.006)

                # If opens above limit, a real stop-limit would probably not fill.
                if today_open > limit_price:
                    orders_gap_rejected += 1
                    continue

                if today_high >= stop_price:
                    raw_exec_price = max(today_open, stop_price)
                    if raw_exec_price <= limit_price:
                        executed = True
                    else:
                        orders_gap_rejected += 1
                        continue

            if executed:
                exec_price = apply_buy_slippage(raw_exec_price)
                shares = int(order["shares"])
                total_cost = exec_price * shares + commission_per_fill

                if shares > 0 and cash >= total_cost:
                    trade_id = next_trade_id
                    next_trade_id += 1

                    cash -= total_cost
                    total_invested += exec_price * shares
                    orders_executed += 1

                    positions[ticker] = {
                        "trade_id": trade_id,
                        "shares": shares,
                        "remaining": shares,
                        "entry": exec_price,
                        "entry_commission_remaining": commission_per_fill,
                        "stop": order["stop"],
                        "tp1": exec_price * (1 + params.tp1_pct),
                        "tp1_done": False,
                        "entry_date": today_str,
                        "entry_index": i,
                        "setup": order.get("setup", "")
                    }

        # DAY orders that did not fill are cancelled.
        # Count only today's unfilled pending orders, not cumulative executed orders.
        orders_expired += len(pending_orders)
        pending_orders.clear()

        # --------------------------------------------------
        # 2. Manage open positions
        # --------------------------------------------------
        closed = []

        for ticker, pos in list(positions.items()):
            # Do not allow exits on the same day the trade was opened.
            # With daily OHLC, we do not know whether the stop/target happened before or after entry.
            if pos.get("entry_index") == i:
                continue

            try:
                today_high = safe_float(highs[ticker].iloc[i])
                today_low = safe_float(lows[ticker].iloc[i])
                today_close = safe_float(prices[ticker].iloc[i])
                close_history = prices[ticker].iloc[max(0, i - 25):i + 1]
                ema21_now = close_history.ewm(span=21, adjust=False).mean().iloc[-1]
            except Exception:
                continue

            # Initial stop before TP1 is allowed intraday.
            if not pos["tp1_done"] and today_low <= pos["stop"]:
                close_shares(ticker, pos, pos["remaining"], pos["stop"], "STOP_LOSS", today_str)
                closed.append(ticker)
                continue

            # Partial TP1.
            if (not pos["tp1_done"]) and today_high >= pos["tp1"]:
                sell_shares = max(1, int(pos["shares"] * params.tp1_fraction))
                close_shares(ticker, pos, sell_shares, pos["tp1"], "TP1_PARTIAL", today_str)
                pos["tp1_done"] = True

                # Do not immediately trail intraday. Move emergency stop only to breakeven.
                pos["stop"] = max(pos["stop"], pos["entry"])

            if pos["remaining"] <= 0:
                closed.append(ticker)
                continue

            # Runner management:
            # after TP1, exit only if DAILY CLOSE is below EMA21 trail level.
            # This avoids intraday wick shakeouts.
            if pos["tp1_done"] and params.use_trailing_runner:
                trail_level = float(ema21_now) * 0.995

                if np.isfinite(today_close) and today_close < trail_level:
                    close_shares(ticker, pos, pos["remaining"], today_close, "TRAIL_EMA21_CLOSE", today_str)
                    closed.append(ticker)

        for ticker in closed:
            closed_pos = positions.pop(ticker, None)
            if closed_pos is not None:
                # Apply cooldown after every exit.
                cd = cooldown_after_exit_days

                # If total PnL for this trade is negative, extend cooldown.
                try:
                    trade_pnl = sum(
                        row["PnL"] for row in trade_log
                        if row.get("TradeID") == closed_pos.get("trade_id")
                    )
                except Exception:
                    trade_pnl = 0

                if trade_pnl < 0:
                    cd = max(cd, cooldown_after_loss_days)
                    recent_loss_dates.setdefault(ticker, []).append(i)
                    recent_loss_dates[ticker] = [
                        d for d in recent_loss_dates[ticker]
                        if i - d <= 30
                    ]
                    if len(recent_loss_dates[ticker]) >= 2:
                        cd = max(cd, double_loss_freeze_days)

                cooldown_until[ticker] = max(cooldown_until.get(ticker, -1), i + cd)

        # --------------------------------------------------
        # 3. Equity curve
        # --------------------------------------------------
        portfolio_value = cash

        if positions:
            exposure_days += 1

        for ticker, pos in positions.items():
            ticker_close = prices[ticker].iloc[i]

            if not pd.isna(ticker_close):
                portfolio_value += pos["remaining"] * float(ticker_close)

        max_equity = max(max_equity, portfolio_value)

        if max_equity > 0:
            drawdown = (portfolio_value / max_equity - 1) * 100
            max_drawdown = min(max_drawdown, drawdown)

        equity_curve.append(portfolio_value)
        dates.append(today_str)

        # --------------------------------------------------
        # 4. Generate DAY orders for next day
        # --------------------------------------------------
        try:
            market_slice = prices["QQQ"].iloc[i - 220:i + 1] if "QQQ" in prices.columns else prices["SPY"].iloc[i - 220:i + 1]
            if len(market_slice) < 220:
                continue
            market_bull = float(market_slice.iloc[-1]) > float(market_slice.rolling(20).mean().iloc[-1])

            # SPY is still used for relative-strength diagnostics to keep continuity.
            spy_slice = prices["SPY"].iloc[i - 220:i + 1]
        except Exception:
            continue

        if not market_bull and not params.allow_bear_market_orders:
            continue

        for ticker in WATCHLIST:
            if ticker in positions or ticker in pending_orders or len(positions) + len(pending_orders) >= max_positions:
                continue

            if i < cooldown_until.get(ticker, -1):
                continue

            if params.auto_trade_momentum_only and ticker not in MOMENTUM_TICKERS:
                continue

            try:
                c = prices[ticker].iloc[i - 220:i + 1].dropna()
                h = highs[ticker].iloc[i - 220:i + 1].dropna()
                l = lows[ticker].iloc[i - 220:i + 1].dropna()

                if len(c) < 220 or len(h) < 220 or len(l) < 220:
                    continue

                last_p = float(c.iloc[-1])
                sma200 = float(c.rolling(200).mean().iloc[-1])
                ema8 = float(c.ewm(span=8, adjust=False).mean().iloc[-1])
                ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
                res20 = float(h.iloc[-21:-1].max())
                prev_hi = float(h.iloc[-2])

                atr = float(pd.concat([
                    h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()
                ], axis=1).max(axis=1).rolling(14).mean().iloc[-1])

                rsi = float(calc_rsi(c).iloc[-1])
                chg20 = (last_p / c.iloc[-21] - 1) * 100

                common = c.index.intersection(spy_slice.index)

                if len(common) < 21:
                    rs5, rs20 = 0.0, 0.0
                else:
                    rs5, rs20 = relative_strength_vs_spy(c.loc[common], spy_slice.loc[common])

                edge_ok = rs5 > params.rs5_min or rs20 > params.rs20_min

                strong_leader = (
                    rs20 > params.rs20_min + 2
                    and chg20 > 10
                    and last_p > ema8
                    and last_p > ema21
                    and last_p > sma200
                )

                # V7.6: automatic orders require SMA200.
                if last_p < sma200:
                    continue

                if (rsi > params.overbought_rsi or chg20 > params.max_20d_run) and not strong_leader:
                    continue

                atr_pct = atr / last_p * 100
                if atr_pct < params.min_atr_pct and ticker not in MOMENTUM_TICKERS and not strong_leader:
                    continue

                dist_to_res = (res20 / last_p - 1)

                order_type = None
                entry_price = 0.0
                stop = 0.0
                setup_name = ""

                # Pre-breakout / near resistance.
                if edge_ok and 0 < dist_to_res < 0.045:
                    entry_price = round(res20 * 1.002, 2)
                    if last_p < entry_price:
                        order_type = "BUY STOP LIMIT"
                        stop = round(entry_price - params.stop_atr_breakout * atr, 2)
                        setup_name = "Near Resistance Breakout"

                # Dynamic RS pullback shallow entry.
                elif edge_ok and last_p > ema21 and (last_p / ema8 - 1) < 0.035:
                    base = ema8 if rs5 > params.rs5_min else ema21
                    entry_price = min(round(base * 1.003, 2), round(last_p * 0.995, 2))
                    if last_p > entry_price:
                        order_type = "BUY LIMIT"
                        stop = round(entry_price - params.stop_atr_pullback * atr, 2)
                        setup_name = "RS Pullback"

                # Momentum continuation with controlled buy stop above yesterday high.
                elif edge_ok and last_p > ema21 and (prev_hi / last_p - 1) < 0.025:
                    entry_price = round(prev_hi * 1.002, 2)
                    if last_p < entry_price:
                        order_type = "BUY STOP LIMIT"
                        stop = round(entry_price - params.stop_atr_breakout * atr, 2)
                        setup_name = "Momentum Continuation"

                if order_type and entry_price > stop:
                    risk_per_share = entry_price - stop
                    risk_pct = risk_per_share / entry_price * 100
                    rr = calc_rr_from_weighted_reward(entry_price, stop, params)

                    # Position sizing:
                    # Fixed Amount = original behavior.
                    # Percent of Equity = portfolio mode, e.g. 20% of current equity per trade.
                    current_equity = portfolio_value
                    current_exposure = max(0.0, current_equity - cash)
                    max_total_exposure = current_equity * max_exposure_pct
                    available_exposure_budget = max(0.0, max_total_exposure - current_exposure)

                    if sizing_mode == "Percent of Equity":
                        cash_budget_for_trade = min(current_equity * position_pct, available_exposure_budget, cash)
                    else:
                        cash_budget_for_trade = min(investment_per_trade, available_exposure_budget if max_exposure_pct < 0.999 else investment_per_trade, cash)

                    shares_by_cash = int(cash_budget_for_trade / entry_price)
                    shares_by_risk = int(risk_budget / risk_per_share)
                    shares = min(shares_by_cash, shares_by_risk)

                    if shares > 0 and risk_pct <= params.max_risk_pct and rr >= params.min_rr:
                        estimated_cost = shares * entry_price * (1 + slippage_pct) + commission_per_fill
                        if estimated_cost <= cash:
                            limit_price = round(entry_price * 1.006, 2) if order_type == "BUY STOP LIMIT" else entry_price
                            pending_orders[ticker] = {
                                "type": order_type,
                                "price": entry_price,
                                "limit": limit_price,
                                "stop": stop,
                                "shares": shares,
                                "setup": setup_name
                            }
                            orders_created += 1

            except Exception:
                continue

    # ------------------------------------------------------
    # 5. Final liquidation of open positions
    # ------------------------------------------------------
    final_idx = len(prices) - 1
    final_date = prices.index[final_idx].strftime("%Y-%m-%d")

    for ticker, pos in list(positions.items()):
        try:
            final_close = safe_float(prices[ticker].iloc[final_idx])
            if np.isfinite(final_close):
                close_shares(ticker, pos, pos["remaining"], final_close, "FINAL_CLOSE", final_date)
        except Exception:
            pass

    positions.clear()

    final_value = cash

    # Add final equity point after liquidation.
    equity_curve.append(final_value)
    dates.append(final_date)

    df_equity = pd.DataFrame({"Date": dates, "Portfolio Value": equity_curve}).set_index("Date")
    df_trades = pd.DataFrame(trade_log)

    total_days = max(1, len(dates))
    exposure_pct = exposure_days / total_days * 100

    # Full-trade metrics by TradeID.
    if not df_trades.empty and "TradeID" in df_trades.columns:
        trade_summary = (
            df_trades.groupby(["TradeID", "Ticker"], as_index=False)
            .agg(
                TotalPnL=("PnL", "sum"),
                EntryDate=("EntryDate", "min"),
                FirstExitDate=("Date", "min"),
                LastExitDate=("Date", "max"),
                Exits=("Exit", lambda x: ", ".join(map(str, x)))
            )
        )
        try:
            trade_summary["HoldingDays"] = (
                pd.to_datetime(trade_summary["LastExitDate"]) - pd.to_datetime(trade_summary["EntryDate"])
            ).dt.days
        except Exception:
            trade_summary["HoldingDays"] = None
    else:
        trade_summary = pd.DataFrame(columns=["TradeID", "Ticker", "TotalPnL", "EntryDate", "FirstExitDate", "LastExitDate", "HoldingDays", "Exits"])

    gross_profit = float(trade_summary.loc[trade_summary["TotalPnL"] > 0, "TotalPnL"].sum()) if not trade_summary.empty else 0.0
    gross_loss = float(-trade_summary.loc[trade_summary["TotalPnL"] < 0, "TotalPnL"].sum()) if not trade_summary.empty else 0.0
    wins = int((trade_summary["TotalPnL"] > 0).sum()) if not trade_summary.empty else 0
    losses = int((trade_summary["TotalPnL"] < 0).sum()) if not trade_summary.empty else 0

    # Benchmarks for exact tested window.
    benchmarks = {}
    try:
        for bench in ["SPY", "QQQ"]:
            b = prices[bench].iloc[test_start_idx:len(prices)-1].dropna()
            if len(b) > 1:
                benchmarks[bench] = (float(b.iloc[-1]) / float(b.iloc[0]) - 1) * 100
    except Exception:
        benchmarks = {}

    # Missed Winner Diagnostics.
    missed_rows = []
    traded_tickers = set(df_trades["Ticker"].unique()) if not df_trades.empty and "Ticker" in df_trades else set()

    try:
        spy_series = prices["SPY"]
        end_idx = len(prices) - 1

        for ticker in WATCHLIST:
            s = prices[ticker].iloc[test_start_idx:end_idx].dropna()

            if len(s) > 1:
                ret = (float(s.iloc[-1]) / float(s.iloc[0]) - 1) * 100
                diag = diagnose_missed_ticker(prices, highs, lows, ticker, spy_series, test_start_idx, end_idx, params)

                row = {
                    "Ticker": ticker,
                    "Return %": round(ret, 1),
                    "Traded?": "YES" if ticker in traded_tickers else "NO",
                }

                row.update(diag)

                if row["Traded?"] == "YES":
                    row["Likely Miss Reason"] = "Traded"

                missed_rows.append(row)

        df_missed = pd.DataFrame(missed_rows).sort_values(by="Return %", ascending=False).head(diagnostics_top_n)

    except Exception:
        df_missed = pd.DataFrame()

    return (
        df_equity,
        df_trades,
        trade_summary,
        wins,
        losses,
        total_invested,
        gross_profit,
        gross_loss,
        max_drawdown,
        exposure_pct,
        orders_created,
        orders_executed,
        orders_expired,
        orders_gap_rejected,
        benchmarks,
        df_missed
    )


def build_backtest_excel_report(
    df_eq: pd.DataFrame,
    df_trades: pd.DataFrame,
    trade_summary: pd.DataFrame,
    df_missed: pd.DataFrame,
    metrics: dict,
    benchmarks: dict
) -> bytes:
    """Build one Excel file with all backtest outputs in separate sheets."""
    output = BytesIO()

    metrics_df = pd.DataFrame([
        {"Metric": k, "Value": v}
        for k, v in metrics.items()
    ])

    benchmarks_df = pd.DataFrame([
        {"Benchmark": k, "Return %": round(v, 2) if isinstance(v, (int, float, np.floating)) else v}
        for k, v in benchmarks.items()
    ])

    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            metrics_df.to_excel(writer, index=False, sheet_name="Summary")
            benchmarks_df.to_excel(writer, index=False, sheet_name="Benchmarks")
            df_eq.reset_index().to_excel(writer, index=False, sheet_name="Equity Curve")
            trade_summary.to_excel(writer, index=False, sheet_name="Trade Summary")
            df_trades.to_excel(writer, index=False, sheet_name="Partial Exits")
            df_missed.to_excel(writer, index=False, sheet_name="Missed Winners")
    except Exception:
        # Fallback to xlsxwriter if available.
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            metrics_df.to_excel(writer, index=False, sheet_name="Summary")
            benchmarks_df.to_excel(writer, index=False, sheet_name="Benchmarks")
            df_eq.reset_index().to_excel(writer, index=False, sheet_name="Equity Curve")
            trade_summary.to_excel(writer, index=False, sheet_name="Trade Summary")
            df_trades.to_excel(writer, index=False, sheet_name="Partial Exits")
            df_missed.to_excel(writer, index=False, sheet_name="Missed Winners")

    output.seek(0)
    return output.getvalue()


def build_backtest_csv_report(
    df_eq: pd.DataFrame,
    df_trades: pd.DataFrame,
    trade_summary: pd.DataFrame,
    df_missed: pd.DataFrame,
    metrics: dict,
    benchmarks: dict
) -> bytes:
    """Plain CSV fallback: one file with sections."""
    parts = []

    def add_section(name, df):
        parts.append(f"===== {name} =====\n")
        parts.append(df.to_csv(index=False))
        parts.append("\n\n")

    metrics_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()])
    benchmarks_df = pd.DataFrame([{"Benchmark": k, "Return %": v} for k, v in benchmarks.items()])

    add_section("Summary", metrics_df)
    add_section("Benchmarks", benchmarks_df)
    add_section("Equity Curve", df_eq.reset_index())
    add_section("Trade Summary", trade_summary)
    add_section("Partial Exits", df_trades)
    add_section("Missed Winners", df_missed)

    return "".join(parts).encode("utf-8-sig")


# ==========================================================
# 7. UI Dashboard
# ==========================================================
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 Terminal Access")
    pwd = st.text_input("סיסמה:", type="password")
    if st.button("כניסה"):
        if APP_PASSWORD and pwd == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("סיסמה שגויה. אם לא הגדרת Secrets, ברירת המחדל היא 1234")
else:
    st.markdown("<h1 style='text-align: right;'>🎯 SwingHunter V7.8 - Portfolio Defaults</h1>", unsafe_allow_html=True)
    st.info("V7.8: ברירות המחדל כבר מכוונות למה שאנחנו רוצים לבדוק — Portfolio Mode, 20% מהתיק לפוזיציה, 80% חשיפה מקסימלית, סיכון $200 לעסקה ב-Balanced, Anti-Churn, ו-Export אחד מלא.")

    st.sidebar.header("ניהול כספי")

    # Defaults are intentionally set to the portfolio-style configuration we actually want to test.
    mode = st.sidebar.selectbox("מצב אסטרטגיה", ["Conservative", "Balanced", "Aggressive"], index=1)
    params = get_params(mode)

    months = st.sidebar.slider("תקופת בדיקה היסטורית (חודשים)", 3, 12, 12)

    sizing_mode = st.sidebar.selectbox(
        "שיטת גודל פוזיציה",
        ["Fixed Amount", "Percent of Equity"],
        index=1
    )

    investment_amount = st.sidebar.number_input(
        "תקרת השקעה בכל עסקה במצב Fixed ($)",
        value=2500,
        step=100
    )

    risk_budget = st.sidebar.number_input(
        "סיכון מקסימלי לעסקה ($)",
        value=int(params.risk_budget_default),
        step=25
    )

    position_pct = st.sidebar.slider("אחוז מהתיק לפוזיציה", 5, 35, 20) / 100
    max_exposure_pct = st.sidebar.slider("מקסימום חשיפה כוללת", 30, 100, 80) / 100
    max_positions = st.sidebar.slider("מקסימום פוזיציות פתוחות", 1, 8, params.max_positions_default)

    st.sidebar.markdown("### Anti-Churn")
    cooldown_after_exit_days = st.sidebar.slider("Cooldown אחרי כל יציאה (ימי מסחר)", 0, 30, 10)
    cooldown_after_loss_days = st.sidebar.slider("Cooldown אחרי הפסד (ימי מסחר)", 0, 45, 15)
    double_loss_freeze_days = st.sidebar.slider("Freeze אחרי 2 הפסדים ב-30 יום", 0, 60, 30)
    diagnostics_top_n = st.sidebar.slider("כמה מניות להציג ב-Missed Winners", 10, 50, 20)
    slippage_pct = st.sidebar.number_input("החלקה משוערת לכל פעולה (%)", value=0.10, min_value=0.0, max_value=2.0, step=0.05) / 100
    commission_per_fill = st.sidebar.number_input("עמלה לכל פעולה ($)", value=0.0, min_value=0.0, step=0.5)

    st.sidebar.caption(
        f"Mode={mode} | ARMED≥{params.armed_threshold} | Actionable≥{params.actionable_threshold} | "
        f"TP1={params.tp1_pct*100:.0f}% | Runner=EMA21 close trail | R/R≥{params.min_rr}"
    )

    tab_daily, tab_backtest = st.tabs(["🚀 מסך עבודה יומי", "🔬 מעבדת סימולציות"])

    with tab_daily:
        if st.button("⚡ הפק תוכנית עבודה להיום", use_container_width=True):
            with st.spinner("מנתח את השוק ומייצר פקודות... לוקח רגע"):
                spy_close, market_trend = get_spy_context()
                if spy_close is not None:
                    raw_results = [analyze_edge(t, spy_close, market_trend, investment_amount, risk_budget, params) for t in WATCHLIST]
                    raw_results = [r for r in raw_results if r is not None]
                    order_list = [r["order"] for r in raw_results if r["order"] is not None]
                    st.markdown(f"### 📝 פקודות יומיות — {mode} — תקרת השקעה: {investment_amount}$")
                    if order_list:
                        df_orders = pd.DataFrame(order_list).sort_values(by="R/R משוקלל", ascending=False).head(3)
                        st.dataframe(df_orders.style.hide(axis="index"), use_container_width=True)
                    else:
                        st.info("אין היום פקודות שעברו את כל שומרי הסף. אפשר לעיין ברדאר למטה.")
                    st.markdown("---")
                    st.markdown("### 🔍 רדאר שוק")
                    scanner_list = [r["scanner"] for r in raw_results]
                    df_scan = pd.DataFrame(scanner_list).sort_values(by="ציון_כולל", ascending=False)
                    save_scan_history(df_scan)
                    def color_logic(row):
                        val = str(row["החלטה"])
                        if "ARMED" in val or "Actionable" in val:
                            return ["background-color: rgba(46, 204, 113, 0.2)"] * len(row)
                        if "Building Pressure" in val:
                            return ["background-color: rgba(241, 196, 15, 0.1)"] * len(row)
                        if "DANGER" in val:
                            return ["background-color: rgba(231, 76, 60, 0.1)"] * len(row)
                        return [""] * len(row)
                    st.dataframe(df_scan.style.apply(color_logic, axis=1), use_container_width=True)
                    st.write(f"📈 **מצב שוק:** {market_trend}")
                else:
                    st.error("שגיאה במשיכת נתוני השוק.")

    with tab_backtest:
        st.markdown(f"### 🧪 Backtest — {months} חודשים — {mode} — {sizing_mode} — Risk ${risk_budget}")
        if st.button("⚙️ הרץ בדיקה היסטורית", type="primary"):
            with st.spinner("מריץ סימולציה. זה עשוי לקחת קצת זמן..."):
                data = fetch_backtest_data(months)
                result = run_backtest_simulation(
                    data,
                    investment_amount,
                    risk_budget,
                    params,
                    months,
                    10000.0,
                    max_positions,
                    diagnostics_top_n,
                    slippage_pct,
                    commission_per_fill,
                    sizing_mode,
                    position_pct,
                    max_exposure_pct,
                    cooldown_after_exit_days,
                    cooldown_after_loss_days,
                    double_loss_freeze_days
                )
                (df_eq, df_trades, trade_summary, wins, losses, total_invested, gross_profit, gross_loss,
                 max_drawdown, exposure_pct, orders_created, orders_executed, orders_expired,
                 orders_gap_rejected, benchmarks, df_missed) = result

                net_pnl = gross_profit - gross_loss
                final_value = df_eq["Portfolio Value"].iloc[-1] if not df_eq.empty else 10000.0
                roi = ((final_value / 10000.0) - 1) * 100
                total_trades = wins + losses
                win_rate = (wins / total_trades * 100) if total_trades else 0
                profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
                avg_win = gross_profit / wins if wins else 0
                avg_loss = gross_loss / losses if losses else 0

                st.markdown("#### 💰 שורה תחתונה")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("שווי תיק סופי", f"${final_value:,.0f}", delta=f"{roi:.1f}%")
                c2.metric("Net PnL", f"${net_pnl:,.0f}")
                c3.metric("Profit Factor", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "∞")
                c4.metric("Max Drawdown", f"{max_drawdown:.1f}%")
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Win Rate לפי עסקה", f"{win_rate:.1f}%")
                c6.metric("Avg Win / Avg Loss", f"${avg_win:.0f} / ${avg_loss:.0f}")
                c7.metric("חשיפה לשוק", f"{exposure_pct:.1f}%")
                c8.metric("השקעה מצטברת", f"${total_invested:,.0f}")
                c9, c10, c11, c12 = st.columns(4)
                c9.metric("פקודות נוצרו", f"{orders_created}")
                c10.metric("פקודות נתפסו", f"{orders_executed}")
                c10.caption(f"Expired: {orders_expired} | Gap rejected: {orders_gap_rejected}")
                c11.metric("SPY", f"{benchmarks.get('SPY', 0):.1f}%")
                c12.metric("QQQ", f"{benchmarks.get('QQQ', 0):.1f}%")

                export_metrics = {
                    "App Version": APP_VERSION,
                    "Mode": mode,
                    "Months": months,
                    "Sizing Mode": sizing_mode,
                    "Investment Cap Per Trade": investment_amount,
                    "Risk Budget Per Trade": risk_budget,
                    "Position %": position_pct,
                    "Max Exposure %": max_exposure_pct,
                    "Max Positions": max_positions,
                    "Slippage %": slippage_pct * 100,
                    "Commission Per Fill": commission_per_fill,
                    "Final Portfolio Value": final_value,
                    "ROI %": roi,
                    "Net PnL": net_pnl,
                    "Profit Factor": profit_factor if np.isfinite(profit_factor) else "∞",
                    "Max Drawdown %": max_drawdown,
                    "Win Rate %": win_rate,
                    "Avg Win": avg_win,
                    "Avg Loss": avg_loss,
                    "Exposure %": exposure_pct,
                    "Orders Created": orders_created,
                    "Orders Executed": orders_executed,
                    "Orders Expired": orders_expired,
                    "Orders Gap Rejected": orders_gap_rejected,
                    "Cooldown After Exit Days": cooldown_after_exit_days,
                    "Cooldown After Loss Days": cooldown_after_loss_days,
                    "Double Loss Freeze Days": double_loss_freeze_days,
                }

                st.markdown("#### 📦 יצוא קובץ אחד")
                try:
                    excel_bytes = build_backtest_excel_report(
                        df_eq, df_trades, trade_summary, df_missed, export_metrics, benchmarks
                    )
                    st.download_button(
                        "⬇️ הורד קובץ Excel אחד עם כל הנתונים",
                        data=excel_bytes,
                        file_name=f"swinghunter_{APP_VERSION}_{mode}_{months}m_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except Exception as e:
                    csv_bytes = build_backtest_csv_report(
                        df_eq, df_trades, trade_summary, df_missed, export_metrics, benchmarks
                    )
                    st.download_button(
                        "⬇️ הורד קובץ CSV אחד עם כל הנתונים",
                        data=csv_bytes,
                        file_name=f"swinghunter_{APP_VERSION}_{mode}_{months}m_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    st.caption(f"Excel export failed, CSV fallback used: {e}")

                st.markdown("#### 📈 Equity Curve")
                st.line_chart(df_eq)

                if not df_missed.empty:
                    with st.expander("🏃 Missed Winner Diagnostics — מי עלו הכי הרבה ולמה פספסנו"):
                        st.dataframe(df_missed, use_container_width=True)

                if not trade_summary.empty:
                    with st.expander("📌 סיכום לפי TradeID — מדידת עסקה מלאה"):
                        st.dataframe(trade_summary, use_container_width=True)

                if not df_trades.empty:
                    with st.expander("📝 יומן יציאות חלקיות"):
                        st.dataframe(df_trades, use_container_width=True)
