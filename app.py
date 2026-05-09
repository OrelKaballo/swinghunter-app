import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")
st.set_page_config(page_title="SwingHunter V7 - Edge Engine", layout="wide")

# ==========================================================
# 1. Security + Settings
# ==========================================================
# IMPORTANT:
# Best practice: define APP_PASSWORD and MY_EMAIL in Streamlit Cloud Secrets.
# For quick testing, this file has a local fallback password below.
# Change LOCAL_TEST_PASSWORD before using the app seriously.
#
# Streamlit Secrets example:
# APP_PASSWORD = "your_password_here"
# MY_EMAIL = "your_email_here"
LOCAL_TEST_PASSWORD = "Pk0105Ak2701"
LOCAL_TEST_EMAIL = "orel@peleg-eng.com"

try:
    APP_PASSWORD = st.secrets.get("APP_PASSWORD", LOCAL_TEST_PASSWORD)
    MY_EMAIL = st.secrets.get("MY_EMAIL", LOCAL_TEST_EMAIL)
except Exception:
    APP_PASSWORD = os.getenv("APP_PASSWORD", LOCAL_TEST_PASSWORD)
    MY_EMAIL = os.getenv("MY_EMAIL", LOCAL_TEST_EMAIL)

WATCHLIST = [
    'AAPL','MSFT','NVDA','TSLA','AMZN','META','GOOGL','NFLX','AMD','AVGO','TSM','QCOM',
    'CRWD','PANW','PLTR','SNOW','DDOG','NET','SMCI','COIN','MSTR','HOOD','SOFI','SQ',
    'PYPL','AFRM','SHOP','BABA','MELI','WMT','TGT','COST','HD','UBER','ABNB','SPOT',
    'DKNG','DIS','NKE','SBUX','MCD','JPM','BAC','GS','MS','V','MA','LLY','NVO','UNH','CAT','BA','MRNA'
]

HISTORY_FILE = "swinghunter_history.csv"

# ==========================================================
# 2. Strategy Parameters
# ==========================================================
@dataclass
class StrategyParams:
    mode: str
    armed_threshold: int
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


def get_params(mode: str) -> StrategyParams:
    if mode == "Conservative":
        return StrategyParams(mode, 85, 50, 1.7, 6.0, False, 60, 2.5, 6.0, 76, 25, 1.8, 1.5, 0.10, 0.50, 0.20, True)
    if mode == "Aggressive":
        return StrategyParams(mode, 68, 38, 1.25, 9.0, True, 45, 1.0, 3.0, 82, 45, 2.5, 2.0, 0.08, 0.50, 0.25, True)
    # Balanced default
    return StrategyParams(mode, 75, 45, 1.45, 7.5, True, 55, 1.5, 4.0, 80, 35, 2.2, 1.8, 0.10, 0.50, 0.22, True)

# ==========================================================
# 3. Utility Functions
# ==========================================================
def flatten_download(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        # For single ticker downloads, yfinance can return MultiIndex. Keep first level.
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
def download_single(ticker: str, period: str = "250d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        df = flatten_download(df)
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_spy_context():
    spy = download_single("SPY", "250d")
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
            now = pd.Timestamp.now(tz='UTC')
            future = cal.index[cal.index > now]
            if not future.empty:
                days = int((future[0] - now).days)
                if 0 <= days <= days_ahead:
                    return "DANGER", days
    except Exception:
        pass
    return "CLEAR", 0


def save_scan_history(df_scan: pd.DataFrame):
    if df_scan.empty or not {'מניה', 'החלטה', 'ציון_כולל'}.issubset(df_scan.columns):
        return
    df_save = df_scan[['מניה', 'החלטה', 'ציון_כולל']].copy()
    df_save["scan_date"] = datetime.now().strftime('%Y-%m-%d')
    try:
        old = pd.read_csv(HISTORY_FILE)
        combined = pd.concat([old, df_save], ignore_index=True)
        combined = combined.drop_duplicates(subset=['מניה', 'scan_date'], keep='last')
        # keep recent 30 calendar days
        cutoff = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
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
        watch_days = recent["החלטה"].astype(str).str.contains("Building Pressure|ARMED|למעקב|פעיל", regex=True).sum()
        score_trend = recent["ציון_כולל"].diff().fillna(0).sum()
        return int(watch_days), float(score_trend)
    except Exception:
        return 0, 0.0

# ==========================================================
# 4. Indicators + Edge Modules
# ==========================================================
def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
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
        s_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
        s_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100
        spy_5d = (spy_close.iloc[-1] / spy_close.iloc[-6] - 1) * 100
        spy_20d = (spy_close.iloc[-1] / spy_close.iloc[-21] - 1) * 100
        return s_5d - spy_5d, s_20d - spy_20d
    except Exception:
        return 0.0, 0.0


def failed_breakdown_recovery(df: pd.DataFrame):
    try:
        close, low = df["Close"], df["Low"]
        sma5 = close.rolling(5).mean().iloc[-1]
        support_20 = float(low.iloc[-21:-5].min())
        broke_support = float(low.iloc[-5:-1].min()) < support_20 * 0.99
        reclaimed = float(close.iloc[-1]) > support_20 * 1.005 and float(close.iloc[-1]) > sma5
        return (True, support_20) if broke_support and reclaimed else (False, support_20)
    except Exception:
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
def analyze_edge(ticker: str, spy_close: pd.Series, market_trend: str, investment_budget: float, params: StrategyParams):
    df = download_single(ticker, "250d")
    if df.empty or len(df) < 220:
        return None

    close, highs, lows = df['Close'], df['High'], df['Low']
    last_price = safe_float(close.iloc[-1])
    if not np.isfinite(last_price) or last_price <= 5:
        return None

    sma200 = close.rolling(200).mean().iloc[-1]
    ema8 = close.ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
    res_20d = float(highs.iloc[-21:-1].max())
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

    setup_score = 0
    if last_price > sma200:
        setup_score += 15
    if last_price > ema21:
        setup_score += 10
    if dist_to_res < 4.0:
        setup_score += 15

    edge_score = 0
    edge_notes = []
    reject_reason = ""

    if comp >= 25 and (dist_to_res < 4.0 or rs_5d > 0 or rs_20d > 0):
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

    final_rank = setup_score + edge_score
    setup_type = "No Setup"

    if earn_status == "DANGER":
        icon, decision, reject_reason, final_rank = "⚠️", "DANGER", f"דוח קרוב בעוד {earn_days} ימים", 0
    elif rsi_val > params.overbought_rsi or change_20d > params.max_20d_run:
        icon, decision, reject_reason, final_rank = "🔥", "חם מדי", "מתוח מדי / סכנת רדיפה", min(final_rank, 30)
    elif market_trend == "BEAR" and (not params.allow_bear_market_orders or edge_score < params.bear_market_min_edge):
        icon, decision, reject_reason, final_rank = "🔴", "Dormant", "Market BEAR + Edge לא מספיק", min(final_rank, 40)
    elif final_rank >= params.armed_threshold and edge_notes:
        icon, decision = "🟢", "ARMED"
    elif final_rank >= params.building_threshold:
        icon, decision = "🟡", "Building Pressure"
    else:
        icon, decision, reject_reason = "🔴", "Dormant", "ציון נמוך מדי"

    if "Compression" in edge_notes and dist_to_res < 3.0:
        setup_type = "Pre-Breakout Compression"
    elif failed_break:
        setup_type = "Failed Breakdown Recovery"
    elif drift:
        setup_type = "Post-Event Drift"
    elif "RS Leader" in edge_notes and last_price > ema21:
        setup_type = "RS Pullback"
    elif last_price > res_20d:
        setup_type = "Momentum Breakout"

    order_data = None
    if decision == "ARMED":
        p_type, entry, e_disp = None, 0.0, ""
        if setup_type in ["Pre-Breakout Compression", "Post-Event Drift", "Momentum Breakout"]:
            trigger = res_20d if setup_type != "Post-Event Drift" else event_high
            entry = round(trigger * 1.002, 2)
            if last_price < entry:
                p_type = "BUY STOP LIMIT"
                e_disp = f"Stop {entry} / Lmt {round(entry * 1.008, 2)}"
            else:
                reject_reason, icon, decision = "כבר פרצה. לא רודפים.", "🟡", "Building Pressure"
        elif setup_type in ["Failed Breakdown Recovery", "RS Pullback"]:
            # Dynamic pullback: high-edge RS leaders get a shallower EMA8/10-style entry.
            if edge_score >= 60 and "RS Leader" in edge_notes:
                pullback_base = ema8
                entry_label = "EMA8 shallow pullback"
            else:
                pullback_base = ema21
                entry_label = "EMA21 pullback"
            entry = min(round(pullback_base * 1.003, 2), round(last_price * 0.995, 2))
            p_type, e_disp = "BUY LIMIT", f"{entry} ({entry_label})"

        if p_type and entry > 0:
            if setup_type in ["Pre-Breakout Compression", "Post-Event Drift", "Momentum Breakout"]:
                stop = round(entry - params.stop_atr_breakout * atr_val, 2)
            elif setup_type == "Failed Breakdown Recovery":
                stop = round(reclaimed_level * 0.985, 2)
            else:
                stop = round(entry - params.stop_atr_pullback * atr_val, 2)

            risk_ps = entry - stop
            if risk_ps > 0:
                risk_pct = risk_ps / entry * 100
                rr = (entry * params.tp1_pct) / risk_ps
                if risk_pct <= params.max_risk_pct and rr >= params.min_rr:
                    shares = int(investment_budget / entry)
                    if shares > 0:
                        order_data = {
                            'מניה': ticker,
                            'פעולה': p_type,
                            'Edge': " + ".join(edge_notes),
                            'כניסה': e_disp,
                            'כמות': shares,
                            'השקעה $': f"${round(shares * entry, 2)}",
                            'יעד ראשון': round(entry * (1 + params.tp1_pct), 2),
                            'יעד ראנר': round(entry * (1 + params.runner_target_pct), 2),
                            'סטופ': stop,
                            'סיכון %': f"{round(risk_pct, 1)}%",
                            'R/R': round(rr, 2),
                            'ניהול יציאה': f"Sell {int(params.tp1_fraction*100)}% at TP1, trail rest by EMA21/ATR",
                            'תוקף': 'DAY ONLY'
                        }
                else:
                    reject_reason = f"R/R או סיכון לא עומד בסף ({round(rr,2)} / {round(risk_pct,1)}%)"

    return {
        'scanner': {
            'מניה': ticker,
            'החלטה': f"{icon} {decision}",
            'ציון_כולל': int(final_rank),
            'Setup Score': int(setup_score),
            'Edge Score': int(edge_score),
            'תבנית': setup_type,
            'Market': market_trend,
            'Watch Days': watch_days,
            'Comp. Score': comp,
            'Range 5D %': range_5,
            'Higher Lows': higher_lows,
            'RS (5D)': f"{round(rs_5d, 1)}%",
            'RS (20D)': f"{round(rs_20d, 1)}%",
            'RSI': int(rsi_val) if np.isfinite(rsi_val) else None,
            'Edge Notes': " + ".join(edge_notes) if edge_notes else "None",
            'סיבת פסילה': reject_reason
        },
        'order': order_data
    }

# ==========================================================
# 6. Backtester V7 - Same Logic Direction + Partial Exits
# ==========================================================
@st.cache_data(show_spinner=False)
def fetch_backtest_data(months: int):
    end = datetime.now()
    start = end - timedelta(days=months * 30 + 260)
    return yf.download(WATCHLIST + ['SPY'], start=start, end=end, progress=False, auto_adjust=False)


def get_panel(data: pd.DataFrame, field: str) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        return data[field]
    raise ValueError("Expected MultiIndex data from yfinance for multi-ticker backtest")


def run_backtest_simulation(data: pd.DataFrame, investment_per_trade: float, params: StrategyParams, months: int, starting_capital: float = 10000.0, max_positions: int = 5):
    prices, highs, lows, opens = get_panel(data, 'Close'), get_panel(data, 'High'), get_panel(data, 'Low'), get_panel(data, 'Open')

    cash = starting_capital
    positions = {}
    pending_orders = {}
    equity_curve, dates, trade_log = [], [], []
    wins, losses = 0, 0
    total_invested, gross_profit, gross_loss = 0.0, 0.0, 0.0
    exposure_days = 0
    max_equity = starting_capital
    max_dd = 0.0

    start_idx = max(220, len(prices) - months * 23)
    for i in range(start_idx, len(prices) - 1):
        today = prices.index[i]
        today_str = today.strftime('%Y-%m-%d')

        # Execute pending DAY orders from previous scan.
        for ticker, order in list(pending_orders.items()):
            if ticker in positions or len(positions) >= max_positions:
                continue
            try:
                t_open = safe_float(opens[ticker].iloc[i])
                t_high = safe_float(highs[ticker].iloc[i])
                t_low = safe_float(lows[ticker].iloc[i])
            except Exception:
                continue
            if not np.isfinite(t_open):
                continue
            executed, exec_price = False, 0.0
            if order['type'] == 'BUY LIMIT' and t_low <= order['price']:
                exec_price, executed = min(order['price'], t_open), True
            elif order['type'] == 'BUY STOP LIMIT' and t_high >= order['price']:
                exec_price, executed = max(t_open, order['price']), True
            if executed:
                cost = exec_price * order['shares']
                if cash >= cost:
                    cash -= cost
                    total_invested += cost
                    positions[ticker] = {
                        'shares': order['shares'],
                        'remaining': order['shares'],
                        'entry': exec_price,
                        'stop': order['stop'],
                        'tp1': exec_price * (1 + params.tp1_pct),
                        'runner_target': exec_price * (1 + params.runner_target_pct),
                        'tp1_done': False,
                        'entry_date': today_str
                    }
        pending_orders.clear()

        # Manage open positions: stop, partial TP, runner.
        closed = []
        for ticker, pos in positions.items():
            try:
                t_high = safe_float(highs[ticker].iloc[i])
                t_low = safe_float(lows[ticker].iloc[i])
                c_hist = prices[ticker].iloc[max(0, i-25):i+1]
                ema21_now = c_hist.ewm(span=21, adjust=False).mean().iloc[-1]
            except Exception:
                continue

            # Stop first: conservative assumption if both stop and target touched intraday.
            if t_low <= pos['stop']:
                sell_shares = pos['remaining']
                revenue = sell_shares * pos['stop']
                cash += revenue
                pnl = (pos['stop'] - pos['entry']) * sell_shares
                if pnl >= 0: gross_profit += pnl
                else: gross_loss += abs(pnl)
                losses += 1 if not pos['tp1_done'] else 0
                trade_log.append({'Date': today_str, 'Ticker': ticker, 'Exit': 'STOP', 'Shares': sell_shares, 'Entry': round(pos['entry'],2), 'ExitPrice': round(pos['stop'],2), 'PnL': round(pnl, 2)})
                closed.append(ticker)
                continue

            if (not pos['tp1_done']) and t_high >= pos['tp1']:
                sell_shares = max(1, int(pos['shares'] * params.tp1_fraction))
                sell_shares = min(sell_shares, pos['remaining'])
                revenue = sell_shares * pos['tp1']
                cash += revenue
                pnl = (pos['tp1'] - pos['entry']) * sell_shares
                gross_profit += pnl
                wins += 1
                pos['remaining'] -= sell_shares
                pos['tp1_done'] = True
                # Move stop to breakeven after TP1.
                pos['stop'] = max(pos['stop'], pos['entry'])
                trade_log.append({'Date': today_str, 'Ticker': ticker, 'Exit': 'TP1_PARTIAL', 'Shares': sell_shares, 'Entry': round(pos['entry'],2), 'ExitPrice': round(pos['tp1'],2), 'PnL': round(pnl, 2)})

            if pos['remaining'] <= 0:
                closed.append(ticker)
                continue

            # Runner: either final target or trailing stop below EMA21.
            if pos['tp1_done']:
                if t_high >= pos['runner_target']:
                    sell_price = pos['runner_target']
                    sell_shares = pos['remaining']
                    pnl = (sell_price - pos['entry']) * sell_shares
                    cash += sell_shares * sell_price
                    gross_profit += max(0, pnl)
                    trade_log.append({'Date': today_str, 'Ticker': ticker, 'Exit': 'RUNNER_TARGET', 'Shares': sell_shares, 'Entry': round(pos['entry'],2), 'ExitPrice': round(sell_price,2), 'PnL': round(pnl, 2)})
                    closed.append(ticker)
                elif params.use_trailing_runner:
                    trail_stop = max(pos['stop'], float(ema21_now) * 0.995)
                    pos['stop'] = trail_stop

        for t in closed:
            positions.pop(t, None)

        # Equity curve.
        portfolio_value = cash
        if positions:
            exposure_days += 1
        for ticker, pos in positions.items():
            t_close = prices[ticker].iloc[i]
            if not pd.isna(t_close):
                portfolio_value += pos['remaining'] * float(t_close)
        max_equity = max(max_equity, portfolio_value)
        if max_equity > 0:
            dd = (portfolio_value / max_equity - 1) * 100
            max_dd = min(max_dd, dd)
        equity_curve.append(portfolio_value)
        dates.append(today_str)

        # Generate orders for next day using a historically-available simplified Edge logic.
        try:
            spy_slice = prices['SPY'].iloc[i-220:i+1]
            if len(spy_slice) < 220:
                continue
            spy_sma20 = spy_slice.rolling(20).mean().iloc[-1]
            market_bull = float(spy_slice.iloc[-1]) > float(spy_sma20)
        except Exception:
            continue

        if not market_bull and not params.allow_bear_market_orders:
            continue

        for ticker in WATCHLIST:
            if ticker in positions or ticker in pending_orders or len(positions) + len(pending_orders) >= max_positions:
                continue
            try:
                c = prices[ticker].iloc[i-220:i+1].dropna()
                h = highs[ticker].iloc[i-220:i+1].dropna()
                l = lows[ticker].iloc[i-220:i+1].dropna()
                if len(c) < 220 or len(h) < 220 or len(l) < 220:
                    continue
                last_p = float(c.iloc[-1])
                sma200 = float(c.rolling(200).mean().iloc[-1])
                ema8 = float(c.ewm(span=8, adjust=False).mean().iloc[-1])
                ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
                res20 = float(h.iloc[-21:-1].max())
                atr = float(pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1).rolling(14).mean().iloc[-1])
                rsi = float(calc_rsi(c).iloc[-1])
                chg20 = (last_p / c.iloc[-21] - 1) * 100
                if last_p < sma200 or rsi > params.overbought_rsi or chg20 > params.max_20d_run:
                    continue

                # Relative strength.
                spy_aligned = spy_slice.loc[c.index.intersection(spy_slice.index)]
                if len(spy_aligned) < 21:
                    rs5, rs20 = 0, 0
                else:
                    rs5, rs20 = relative_strength_vs_spy(c.loc[spy_aligned.index], spy_aligned)

                dist_to_res = (res20 / last_p - 1)
                edge_ok = (rs5 > params.rs5_min or rs20 > params.rs20_min)
                order_type, entry_price, stop = None, 0.0, 0.0

                # Pre-breakout or near resistance.
                if 0 < dist_to_res < 0.035 and edge_ok:
                    entry_price = round(res20 * 1.002, 2)
                    if last_p < entry_price:
                        order_type = 'BUY STOP LIMIT'
                        stop = round(entry_price - params.stop_atr_breakout * atr, 2)
                # Dynamic RS pullback.
                elif edge_ok and last_p > ema21 and (last_p / ema8 - 1) < 0.025:
                    base = ema8 if (rs5 > params.rs5_min and rs20 > 0) else ema21
                    entry_price = min(round(base * 1.003, 2), round(last_p * 0.995, 2))
                    if last_p > entry_price:
                        order_type = 'BUY LIMIT'
                        stop = round(entry_price - params.stop_atr_pullback * atr, 2)

                if order_type and entry_price > stop:
                    risk_ps = entry_price - stop
                    risk_pct = risk_ps / entry_price * 100
                    rr = (entry_price * params.tp1_pct) / risk_ps
                    shares = int(investment_per_trade / entry_price)
                    if shares > 0 and risk_pct <= params.max_risk_pct and rr >= params.min_rr and shares * entry_price <= cash:
                        pending_orders[ticker] = {'type': order_type, 'price': entry_price, 'stop': stop, 'shares': shares}
            except Exception:
                continue

    df_equity = pd.DataFrame({'Date': dates, 'Portfolio Value': equity_curve}).set_index('Date')
    df_trades = pd.DataFrame(trade_log)
    total_days = max(1, len(dates))
    exposure_pct = exposure_days / total_days * 100
    return df_equity, df_trades, wins, losses, total_invested, gross_profit, gross_loss, max_dd, exposure_pct

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
            st.error("סיסמה שגויה או שלא הוגדרה APP_PASSWORD ב-Secrets")
else:
    st.markdown("<h1 style='text-align: right;'>🎯 SwingHunter V7 - Edge + Runner Edition</h1>", unsafe_allow_html=True)

    st.sidebar.header("ניהול כספי")
    investment_amount = st.sidebar.number_input("סכום קבוע להשקעה בכל עסקה ($)", value=1000, step=100)
    mode = st.sidebar.selectbox("מצב אסטרטגיה", ["Conservative", "Balanced", "Aggressive"], index=1)
    months = st.sidebar.slider("תקופת בדיקה היסטורית (חודשים)", 3, 12, 3)
    max_positions = st.sidebar.slider("מקסימום פוזיציות פתוחות", 1, 8, 5)
    params = get_params(mode)

    st.sidebar.caption(f"Mode={mode} | ARMED≥{params.armed_threshold} | R/R≥{params.min_rr} | Max risk={params.max_risk_pct}%")

    tab_daily, tab_backtest = st.tabs(["🚀 מסך עבודה יומי", "🔬 מעבדת סימולציות"])

    with tab_daily:
        if st.button("⚡ הפק תוכנית עבודה להיום", use_container_width=True):
            with st.spinner("מנתח את השוק ומייצר פקודות... לוקח רגע"):
                spy_close, market_trend = get_spy_context()
                if spy_close is not None:
                    raw_results = [analyze_edge(t, spy_close, market_trend, investment_amount, params) for t in WATCHLIST]
                    raw_results = [r for r in raw_results if r is not None]

                    order_list = [r['order'] for r in raw_results if r['order'] is not None]
                    st.markdown(f"### 📝 פקודות יומיות — {mode} — השקעה מבוקשת: {investment_amount}$")
                    if order_list:
                        df_orders = pd.DataFrame(order_list).sort_values(by="R/R", ascending=False).head(3)
                        st.dataframe(df_orders.style.hide(axis="index"), use_container_width=True)
                    else:
                        st.info("אין היום פקודות שעברו את כל שומרי הסף. אפשר לעיין ברדאר למטה.")

                    st.markdown("---")
                    st.markdown("### 🔍 רדאר שוק")
                    scanner_list = [r['scanner'] for r in raw_results]
                    df_scan = pd.DataFrame(scanner_list).sort_values(by="ציון_כולל", ascending=False)
                    save_scan_history(df_scan)

                    def color_logic(row):
                        val = str(row['החלטה'])
                        if "ARMED" in val: return ['background-color: rgba(46, 204, 113, 0.2)'] * len(row)
                        if "Building Pressure" in val: return ['background-color: rgba(241, 196, 15, 0.1)'] * len(row)
                        if "DANGER" in val: return ['background-color: rgba(231, 76, 60, 0.1)'] * len(row)
                        return [''] * len(row)

                    st.dataframe(df_scan.style.apply(color_logic, axis=1), use_container_width=True)
                    st.write(f"📈 **מצב שוק:** {market_trend}")
                else:
                    st.error("שגיאה במשיכת נתוני השוק.")

    with tab_backtest:
        st.markdown(f"### 🧪 Backtest — {months} חודשים — {mode} — {investment_amount}$ לעסקה")
        if st.button("⚙️ הרץ בדיקה היסטורית", type="primary"):
            with st.spinner("מריץ סימולציה. זה עשוי לקחת קצת זמן..."):
                data = fetch_backtest_data(months=months)
                result = run_backtest_simulation(data, investment_amount, params, months, 10000.0, max_positions)
                df_eq, df_trades, wins, losses, tot_invested, gp, gl, max_dd, exposure_pct = result

                net_pnl = gp - gl
                final_val = df_eq['Portfolio Value'].iloc[-1] if not df_eq.empty else 10000.0
                roi = ((final_val / 10000.0) - 1) * 100
                total_exits = wins + losses
                win_rate = (wins / total_exits * 100) if total_exits else 0
                profit_factor = (gp / gl) if gl > 0 else np.inf
                avg_win = gp / wins if wins else 0
                avg_loss = gl / losses if losses else 0

                st.markdown("#### 💰 שורה תחתונה")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("שווי תיק סופי", f"${final_val:,.0f}", delta=f"{roi:.1f}%")
                c2.metric("Net PnL", f"${net_pnl:,.0f}")
                c3.metric("Profit Factor", f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "∞")
                c4.metric("Max Drawdown", f"{max_dd:.1f}%")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Win Rate", f"{win_rate:.1f}%")
                c6.metric("Avg Win / Avg Loss", f"${avg_win:.0f} / ${avg_loss:.0f}")
                c7.metric("חשיפה לשוק", f"{exposure_pct:.1f}%")
                c8.metric("השקעה מצטברת", f"${tot_invested:,.0f}")

                st.markdown("#### 📈 Equity Curve")
                st.line_chart(df_eq)

                if not df_trades.empty:
                    with st.expander("📝 יומן עסקאות"):
                        st.dataframe(df_trades, use_container_width=True)
