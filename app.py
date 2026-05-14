import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SwingHunter V12.3 - Unified Portfolio Ledger", layout="wide")
APP_VERSION = "V12.3"

# ==========================================================
# 1. Security
# ==========================================================
LOCAL_TEST_PASSWORD = "1234"

try:
    APP_PASSWORD = st.secrets.get("APP_PASSWORD", LOCAL_TEST_PASSWORD)
except Exception:
    APP_PASSWORD = os.getenv("APP_PASSWORD", LOCAL_TEST_PASSWORD)


# ==========================================================
# 2. Universe
# ==========================================================
WATCHLIST = [
    'AAPL','MSFT','NVDA','TSLA','AMZN','META','GOOGL','NFLX',
    'AMD','AVGO','TSM','INTC','QCOM','MU','MRVL','ASML','ARM',
    'CRWD','PANW','PLTR','SNOW','DDOG','NET','ZS','FTNT','MDB',
    'SMCI','DELL','HPQ','IBM','COIN','MSTR','HOOD','SOFI','SQ',
    'PYPL','AFRM','MARA','RIOT','SHOP','BABA','MELI','WMT','TGT',
    'COST','HD','UBER','ABNB','BKNG','EXPE','DAL','UAL','SPOT',
    'ROKU','DKNG','DIS','NKE','SBUX','MCD','JPM','BAC','GS','MS',
    'V','MA','AXP','LLY','NVO','JNJ','UNH','PFE','MRNA','CAT',
    'BA','XOM','CVX','GE',
    # V11.0 additions: important QQQ / Nasdaq-100 names we were missing
    'LRCX','AMAT','KLAC','TXN','CSCO','TMUS','LIN','PEP'
]

MOMENTUM_TICKERS = {
    'AMD','NVDA','TSLA','DDOG','NET','QCOM','CRWD','PANW','AVGO','AMZN',
    'MSTR','COIN','SMCI','PLTR','ARM','MU','MRVL','TSM','META','GOOGL',
    'HOOD','AFRM','SHOP','NFLX','SNOW','ZS','MDB','SOFI','SQ','PYPL',
    'MARA','RIOT','ASML',
    # V11.0 additions: semiconductor / chip-equipment momentum candidates
    'LRCX','AMAT','KLAC','TXN'
}

NOTIONAL_PER_TRADE = 1000.0
DEFAULT_STARTING_BANK = 50000.0
LEDGER_DIR = Path(os.getenv("SWINGHUNTER_LEDGER_DIR", ".swinghunter_ledgers"))
DEFAULT_POSITION_PCT = 0.10


# ==========================================================
# 3. Parameters
# ==========================================================
@dataclass
class StrategyParams:
    max_risk_pct: float = 9.5
    target_pct: float = 0.0  # V10.2: no fixed profit target
    position_pct: float = DEFAULT_POSITION_PCT
    initial_stop_buffer: float = 0.995
    min_score_for_trade: float = 35.0
    min_rr: float = 0.00
    min_atr_pct: float = 1.4
    overbought_rsi: float = 86.0
    max_20d_run: float = 45.0  # V11.1: above 45% in 20d is too extended for auto trade
    stop_atr_breakout: float = 2.4
    stop_atr_pullback: float = 2.0
    rs5_min: float = 0.5
    rs20_min: float = 2.5
    max_holding_days: int = 35


# ==========================================================
# 4. Data + Indicators
# ==========================================================
def flatten_download(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return (Price, Ticker) or (Ticker, Price). Keep simple for single ticker.
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        elif "Close" in df.columns.get_level_values(1):
            df.columns = df.columns.get_level_values(1)
    return df


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def download_single(ticker: str, period: str = "370d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        return flatten_download(df).dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_backtest_data(months: int):
    end = datetime.now()
    start = end - timedelta(days=months * 31 + 340)
    return yf.download(
        WATCHLIST + ["SPY", "QQQ"],
        start=start,
        end=end,
        progress=False,
        auto_adjust=False
    )


def get_panel(data: pd.DataFrame, field: str) -> pd.DataFrame:
    if isinstance(data.columns, pd.MultiIndex):
        return data[field]
    raise ValueError("Expected MultiIndex data from yfinance")


def calc_atr_from_series(high, low, close, period=14):
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


def relative_strength_vs(ref_close: pd.Series, stock_close: pd.Series):
    try:
        common = stock_close.index.intersection(ref_close.index)
        s = stock_close.loc[common]
        r = ref_close.loc[common]
        if len(s) < 21 or len(r) < 21:
            return 0.0, 0.0

        stock_5 = (s.iloc[-1] / s.iloc[-6] - 1) * 100
        stock_20 = (s.iloc[-1] / s.iloc[-21] - 1) * 100
        ref_5 = (r.iloc[-1] / r.iloc[-6] - 1) * 100
        ref_20 = (r.iloc[-1] / r.iloc[-21] - 1) * 100
        return stock_5 - ref_5, stock_20 - ref_20
    except Exception:
        return 0.0, 0.0


def market_regime(market_slice: pd.Series) -> str:
    try:
        if len(market_slice) < 60:
            return "UNKNOWN"
        last = float(market_slice.iloc[-1])
        sma20 = float(market_slice.rolling(20).mean().iloc[-1])
        sma50 = float(market_slice.rolling(50).mean().iloc[-1])
        if last > sma20 and last > sma50:
            return "BULL_STRONG"
        if last > sma20 and last <= sma50:
            return "BULL_WEAK"
        if last <= sma20 and last > sma50:
            return "PULLBACK"
        return "BEAR"
    except Exception:
        return "UNKNOWN"


# ==========================================================
# 5. Signal Engine
# ==========================================================
def score_candidate(rs5, rs20, run20, rr, risk_pct, setup, exceptional, momentum_name):
    return (
        rs20 * 1.20
        + rs5 * 0.70
        + min(max(run20, -15), 65) * 0.25
        + rr * 10
        - risk_pct * 0.90
        + (12 if exceptional else 0)
        + (8 if setup == "Near Resistance Breakout" else 0)
        + (5 if setup == "RS Pullback" else 0)
        + (4 if momentum_name else 0)
    )


def build_reason_list(**checks):
    return " | ".join([label for label, failed in checks.items() if failed]) or ""


def evaluate_ticker(
    ticker: str,
    c: pd.Series,
    h: pd.Series,
    l: pd.Series,
    qqq_slice: pd.Series,
    params: StrategyParams
):
    """
    Returns a full row for both action table and radar.
    This function never hides the important numbers.
    """
    base = {
        "Ticker": ticker,
        "Status": "ERROR",
        "Decision": "NO DATA",
        "Score": np.nan,
        "Setup": "",
        "Current": np.nan,
        "Entry": np.nan,
        "Distance to Entry %": np.nan,
        "Stop": np.nan,
        "Target": "No fixed target",
        "Exit Rule": "Daily close below EMA21 trail",
        "Risk %": np.nan,
        "Target %": "Trend",
        "R/R": "Trend",
        "Order": "",
        "Action Now": "",
        "Regime": "",
        "EMA21": np.nan,
        "Exit Close Level": np.nan,
        "Current Protection Stop": np.nan,
        "Profit Checkpoint": np.nan,
        "RS5": np.nan,
        "RS20": np.nan,
        "RSI": np.nan,
        "ATR%": np.nan,
        "20D Run": np.nan,
        "Run Zone": "",
        "Setup Type": "",
        "Action State": "WAIT",
        "Pullback Watch Price": np.nan,
        "Pullback Deep Price": np.nan,
        "Breakout Watch Price": np.nan,
        "Reason": ""
    }

    try:
        if len(c) < 220:
            base.update(Status="REJECT", Decision="SKIP", Reason="אין מספיק היסטוריה")
            return base

        last_p = float(c.iloc[-1])
        sma200 = float(c.rolling(200).mean().iloc[-1])
        ema8 = float(c.ewm(span=8, adjust=False).mean().iloc[-1])
        ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
        res20 = float(h.iloc[-21:-1].max())
        prev_hi = float(h.iloc[-2])

        atr = float(calc_atr_from_series(h, l, c, 14).iloc[-1])
        atr_pct = atr / last_p * 100 if last_p else np.nan
        rsi = float(calc_rsi(c, 14).iloc[-1])

        # V12.3: energy / squeeze metrics.
        atr_5 = float(calc_atr_from_series(h, l, c, 5).iloc[-1])
        atr_20 = float(calc_atr_from_series(h, l, c, 20).iloc[-1])
        atr_pinch = atr_5 / atr_20 if atr_20 > 0 else 1.0

        try:
            v_15 = v.iloc[-15:]
            c_15 = c.iloc[-15:]
            c_15_prev = c.iloc[-16:-1].values
            up_vol = v_15[c_15 > c_15_prev].sum()
            down_vol = v_15[c_15 < c_15_prev].sum()
            var_15 = up_vol / down_vol if down_vol > 0 else 1.0
        except Exception:
            var_15 = 1.0

        try:
            inside_day = bool((h.iloc[-2] <= h.iloc[-3]) and (l.iloc[-2] >= l.iloc[-3]))
            tight_day = bool(((h.iloc[-2] - l.iloc[-2]) / c.iloc[-2] * 100) < (atr_pct * 0.70))
        except Exception:
            inside_day = False
            tight_day = False

        run20 = (last_p / c.iloc[-21] - 1) * 100
        rs5, rs20 = relative_strength_vs(qqq_slice, c)
        regime = market_regime(qqq_slice)

        if run20 <= 15:
            run_zone = "NORMAL"
        elif run20 <= 30:
            run_zone = "HOT"
        elif run20 <= 45:
            run_zone = "EXTENDED"
        else:
            run_zone = "TOO_EXTENDED"

        base.update({
            "Current": round(last_p, 2),
            "Regime": regime,
            "EMA21": round(ema21, 2),
            "Exit Close Level": round(ema21 * 0.995, 2),
            "RS5": round(rs5, 1),
            "RS20": round(rs20, 1),
            "RSI": round(rsi, 1),
            "ATR%": round(atr_pct, 1) if np.isfinite(atr_pct) else np.nan,
            "20D Run": round(run20, 1),
            "Run Zone": run_zone,
            "Breakout Watch Price": round(res20 * 1.002, 2)
        })

        radar_score = round(score_candidate(
            rs5, rs20, run20, 1.0, params.max_risk_pct, "RADAR",
            False, ticker in MOMENTUM_TICKERS
        ), 2)
        base["Score"] = radar_score

        edge_ok = rs5 > params.rs5_min or rs20 > params.rs20_min
        momentum_name = ticker in MOMENTUM_TICKERS

        exceptional = (
            momentum_name
            and rs20 > 20
            and rs5 > 3
            and last_p > sma200
            and last_p > ema8
            and last_p > ema21
            and atr_pct >= params.min_atr_pct
        )

        # V11.6: route the stock instead of judging everything only by RS vs QQQ.
        rs_improving = rs5 > rs20 + 3
        base_candidate = (
            last_p > sma200
            and last_p >= ema21 * 0.99
            and rs_improving
            and 35 <= rsi <= 72
            and run20 <= 25
        )
        turnaround_candidate = (
            last_p > sma200
            and rs20 < 0
            and rs_improving
            and last_p >= ema21 * 0.97
            and 30 <= rsi <= 65
        )

        if last_p < sma200:
            setup_type = "Weak / Ignore"
        elif edge_ok and last_p > ema21:
            setup_type = "Momentum Leader"
        elif base_candidate:
            setup_type = "Base Breakout"
        elif turnaround_candidate:
            setup_type = "Turnaround Watch"
        else:
            setup_type = "Weak / Ignore"

        base["Setup Type"] = setup_type

        # Hard rejects, but still with full radar info.
        if last_p < sma200:
            base.update(Status="REJECT", Decision="NO TRADE", **{"Action State": "NO TRADE"}, Reason="מתחת SMA200")
            return base

        if not np.isfinite(atr_pct) or atr_pct < params.min_atr_pct:
            base.update(Status="REJECT", Decision="NO TRADE", **{"Action State": "NO TRADE"}, Reason=f"ATR% נמוך מדי ({atr_pct:.1f}%)")
            return base

        # Regime filter: normal trades only in BULL_STRONG. Exceptional leaders can pass in weaker regimes.
        if regime != "BULL_STRONG" and not exceptional:
            base.update(Status="WATCH", Decision="WAIT", **{"Action State": "WAIT"}, Reason=f"שוק לא מספיק חזק ({regime})")
            return base

        if rsi > params.overbought_rsi and not exceptional:
            base.update(Status="WATCH", Decision="WAIT", **{"Action State": "WAIT"}, Reason="RSI גבוה מדי / המניה מתוחה")
            return base

        # V11.1 overextension gate:
        # A stock that already ran hard can stay on the radar, but auto-entry is restricted.
        if run20 > params.max_20d_run:
            base.update(
                Status="WATCH",
                Decision="WAIT",
                **{
                    "Action State": "WAIT FOR PULLBACK",
                    "Pullback Watch Price": round(ema8 * 1.003, 2),
                    "Pullback Deep Price": round(ema21 * 1.005, 2),
                },
                Reason=f"ריצה חזקה מדי ב-20 יום ({run20:.1f}%) — לא קונים עכשיו; ממתינים לפולבק לאזור EMA8/EMA21"
            )
            return base

        if not edge_ok and not exceptional and not base_candidate and not turnaround_candidate:
            base.update(Status="WATCH", Decision="WAIT", **{"Action State": "WAIT"}, Reason="RS חלש מול QQQ")
            return base

        order_type = None
        entry = 0.0
        setup = ""

        dist_to_res = (res20 / last_p - 1)

        # V11.1: do not chase breakouts after a 30%+ 20-day run.
        # If a name already ran 30%-45%, it may only enter via a quality pullback.
        allow_breakout = run20 <= 30 or (exceptional and run20 <= 35 and rs20 > 30 and rs5 > 5)
        allow_pullback = run20 <= 45

        # 1) Breakout near resistance for momentum leaders.
        if allow_breakout and 0 < dist_to_res < 0.045 and edge_ok:
            entry = round(res20 * 1.002, 2)
            if last_p < entry:
                order_type = "BUY STOP LIMIT"
                setup = "Near Resistance Breakout"

        # 1b) Base Breakout / Turnaround route:
        # Does not require positive RS20 yet, but does require improving short-term RS and recovery above EMA21 area.
        if order_type is None and allow_breakout and base_candidate and 0 < dist_to_res < 0.055:
            entry = round(res20 * 1.002, 2)
            if last_p < entry:
                order_type = "BUY STOP LIMIT"
                setup = "Base Breakout"

        # 2) Quality pullback.
        higher_low_2 = False
        try:
            higher_low_2 = bool(l.iloc[-1] >= l.iloc[-2])
        except Exception:
            higher_low_2 = False

        pullback_quality_ok = (
            last_p > ema21
            and ema8 > ema21
            and rs20 > max(params.rs20_min, 4.0)
            and rs5 > -2.0
            and 42 <= rsi <= 78
            and run20 <= 38
            and higher_low_2
        )

        if order_type is None and allow_pullback and pullback_quality_ok and (last_p / ema8 - 1) < 0.035:
            entry = min(round(ema8 * 1.003, 2), round(last_p * 0.995, 2))
            if last_p > entry:
                order_type = "BUY LIMIT"
                setup = "RS Pullback"

        # 3) Momentum continuation is WATCH only unless exceptional.
        if order_type is None and allow_breakout and exceptional and last_p > ema21 and (prev_hi / last_p - 1) < 0.025:
            entry = round(prev_hi * 1.002, 2)
            if last_p < entry:
                order_type = "BUY STOP LIMIT"
                setup = "Exceptional Momentum Continuation"

        if order_type is None:
            reason = "אין נקודת כניסה נקייה כרגע"
            action_state = "WAIT"
            extra = {}

            if run20 > 30:
                reason = "רצה מעל 30% ב-20 יום — ממתינים לפולבק איכותי, לא קונים פריצה"
                action_state = "WAIT FOR PULLBACK"
                extra = {
                    "Pullback Watch Price": round(ema8 * 1.003, 2),
                    "Pullback Deep Price": round(ema21 * 1.005, 2),
                }
            elif setup_type == "Turnaround Watch":
                reason = "שיפור ראשוני, אבל עדיין לא פריצה; לעקוב אחרי Breakout Watch Price"
                action_state = "TURNAROUND WATCH"
            elif setup_type == "Base Breakout":
                reason = "בסיס מעניין, אבל אין טריגר כניסה נקי כרגע"
                action_state = "WAIT FOR BREAKOUT"

            base.update(
                Status="WATCH",
                Decision="WAIT",
                **{"Action State": action_state},
                **extra,
                Score=round(score_candidate(rs5, rs20, run20, 1.0, params.max_risk_pct, "WATCH", exceptional, momentum_name), 2),
                Reason=reason
            )
            return base

        # Stop calculation.
        raw_stop = entry - (params.stop_atr_breakout if "Breakout" in setup or "Momentum" in setup else params.stop_atr_pullback) * atr

        # Cap by user's max risk percentage.
        stop = max(raw_stop, entry * (1 - params.max_risk_pct / 100))
        stop = round(stop, 2)

        risk_pct = (entry - stop) / entry * 100

        # V10.2: no fixed profit target. Score is based on trend quality.
        rr_proxy = max(1.0, (rs20 + max(run20, 0)) / max(risk_pct, 1.0) / 5)
        score = score_candidate(rs5, rs20, run20, rr_proxy, risk_pct, setup, exceptional, momentum_name)

        if risk_pct > params.max_risk_pct + 0.01:
            base.update(
                Status="WATCH",
                Decision="WAIT",
                Setup=setup,
                Entry=round(entry, 2),
                Stop=stop,
                Target="No fixed target",
                **{"Risk %": round(risk_pct, 2)},
                Score=round(score, 2),
                **{"Action State": "WAIT"},
                Reason=f"סיכון גבוה מדי ({risk_pct:.1f}%)"
            )
            return base

        if score < params.min_score_for_trade:
            base.update(
                Status="WATCH",
                Decision="WAIT",
                Setup=setup,
                Entry=round(entry, 2),
                Stop=stop,
                Target="No fixed target",
                **{"Risk %": round(risk_pct, 2)},
                Score=round(score, 2),
                **{"Action State": "WAIT"},
                Reason=f"Score נמוך מדי לביצוע ({score:.1f})"
            )
            return base

        distance = (entry / last_p - 1) * 100

        if order_type == "BUY LIMIT":
            action = "PLACE LIMIT" if distance > -3.0 else "WAIT - רחוקה מעל הלימיט"
        else:
            if distance < 0:
                action = "MISSED / WAIT RESET"
            elif distance <= 3.0:
                action = "PLACE STOP LIMIT"
            else:
                action = "WAIT - טריגר רחוק"

        base.update({
            "Status": "SIGNAL",
            "Decision": "ACTION",
            "Action State": "PLACE ORDER",
            "Score": round(score, 2),
            "Setup": setup,
            "Setup Type": setup_type,
            "Entry": round(entry, 2),
            "Distance to Entry %": round(distance, 2),
            "Stop": stop,
            "Current Protection Stop": stop,
            "Profit Checkpoint": round(entry * 1.15, 2),
            "Target": "No fixed target",
            "Exit Rule": "Initial stop, then EMA21 trailing exit",
            "Risk %": round(risk_pct, 2),
            "R/R": "Trend",
            "Order": order_type,
            "Action Now": action,
            "Reason": ""
        })

        return base

    except Exception as e:
        base.update(Status="ERROR", Decision="ERROR", Reason=str(e)[:120])
        return base


def execute_pending_order(order, open_p, high_p, low_p):
    if order["Order"] == "BUY LIMIT":
        return order["Entry"] if low_p <= order["Entry"] else None

    if order["Order"] == "BUY STOP LIMIT":
        stop_price = order["Entry"]
        limit_price = round(stop_price * 1.006, 2)
        if open_p > limit_price:
            return None
        if high_p >= stop_price:
            fill = max(open_p, stop_price)
            return fill if fill <= limit_price else None

    return None


# ==========================================================
# 6. Backtest - Banked, full exit, re-entry allowed after close
# ==========================================================
def run_banked_backtest(data, months, params, starting_bank=DEFAULT_STARTING_BANK):
    """
    V10.2 Trend Bank Engine:
    - Each trade allocates position_pct of current equity, not a fixed $1,000.
    - No fixed profit target.
    - Full exit only, when daily close/trailing stop breaks the EMA21 trend.
    - No second entry in a ticker while it is open.
    """
    prices = get_panel(data, "Close")
    highs = get_panel(data, "High")
    lows = get_panel(data, "Low")
    opens = get_panel(data, "Open")

    requested_days = int(months * 21.5)
    start_idx = max(220, len(prices) - requested_days)
    end_idx = len(prices) - 1

    cash_bank = float(starting_bank)
    open_positions = {}
    pending_orders = {}

    trades = []
    equity_rows = []
    pending_created = 0
    pending_filled = 0
    pending_expired = 0
    pending_no_cash = 0
    turnover = 0.0
    next_trade_id = 1
    max_open_positions = 0

    def open_value(i):
        value = 0.0
        for ticker, pos in open_positions.items():
            try:
                close_p = float(prices[ticker].iloc[i])
                value += pos["Capital"] * (close_p / pos["Entry"])
            except Exception:
                value += pos["Capital"]
        return value

    for i in range(start_idx, end_idx):
        date_str = prices.index[i].strftime("%Y-%m-%d")

        # 1) Execute yesterday's pending orders.
        processed = set()
        for ticker, order in list(pending_orders.items()):
            if ticker in open_positions:
                processed.add(ticker)
                continue

            current_equity_before_fill = cash_bank + open_value(i)
            desired_capital = current_equity_before_fill * params.position_pct
            capital = min(desired_capital, cash_bank)

            if capital < 100:
                pending_no_cash += 1
                processed.add(ticker)
                continue

            try:
                open_p = safe_float(opens[ticker].iloc[i])
                high_p = safe_float(highs[ticker].iloc[i])
                low_p = safe_float(lows[ticker].iloc[i])
            except Exception:
                processed.add(ticker)
                continue

            fill = execute_pending_order(order, open_p, high_p, low_p)

            if fill is not None:
                cash_bank -= capital
                turnover += capital

                open_positions[ticker] = {
                    "TradeID": next_trade_id,
                    "Ticker": ticker,
                    "EntryDate": date_str,
                    "EntryIndex": i,
                    "Entry": fill,
                    "Capital": capital,
                    "Stop": order["Stop"],
                    "TrailStop": order["Stop"],
                    "Setup": order["Setup"],
                    "Score": order["Score"],
                    "Risk %": (fill - order["Stop"]) / fill * 100,
                }
                next_trade_id += 1
                pending_filled += 1
                processed.add(ticker)

        pending_expired += len([t for t in pending_orders if t not in processed])
        pending_orders.clear()

        # 2) Manage positions: full exit only, trend-following.
        closed = []
        for ticker, pos in list(open_positions.items()):
            if pos["EntryIndex"] == i:
                continue

            try:
                low_p = safe_float(lows[ticker].iloc[i])
                close_p = safe_float(prices[ticker].iloc[i])
                hist = prices[ticker].iloc[max(0, i - 25):i + 1].dropna()
                ema21 = float(hist.ewm(span=21, adjust=False).mean().iloc[-1])
            except Exception:
                continue

            # Trail stop only moves upward.
            ema_trail = ema21 * params.initial_stop_buffer
            pos["TrailStop"] = max(pos["TrailStop"], ema_trail)

            exit_reason = None
            exit_price = None

            if low_p <= pos["TrailStop"]:
                exit_reason = "TRAIL_STOP"
                exit_price = pos["TrailStop"]
            elif np.isfinite(close_p) and close_p < ema21 * params.initial_stop_buffer:
                exit_reason = "EMA21_CLOSE"
                exit_price = close_p
            elif i - pos["EntryIndex"] >= params.max_holding_days:
                exit_reason = "TIME_EXIT"
                exit_price = close_p

            if exit_reason:
                ret_pct = (exit_price / pos["Entry"] - 1) * 100
                pnl = pos["Capital"] * ret_pct / 100
                cash_bank += pos["Capital"] + pnl

                try:
                    q_entry = float(prices["QQQ"].loc[pd.to_datetime(pos["EntryDate"])])
                    q_exit = float(prices["QQQ"].iloc[i])
                    q_ret = (q_exit / q_entry - 1) * 100
                    q_pnl = pos["Capital"] * q_ret / 100
                except Exception:
                    q_ret = np.nan
                    q_pnl = np.nan

                trades.append({
                    "TradeID": pos["TradeID"],
                    "Ticker": ticker,
                    "EntryDate": pos["EntryDate"],
                    "ExitDate": date_str,
                    "HoldingDays": i - pos["EntryIndex"],
                    "Setup": pos["Setup"],
                    "Capital": round(pos["Capital"], 2),
                    "Entry": round(pos["Entry"], 2),
                    "Exit": round(exit_price, 2),
                    "ExitReason": exit_reason,
                    "Return %": round(ret_pct, 2),
                    "PnL": round(pnl, 2),
                    "Risk %": round(pos["Risk %"], 2),
                    "Score": round(pos["Score"], 2),
                    "QQQ Same Window %": round(q_ret, 2) if np.isfinite(q_ret) else np.nan,
                    "QQQ Same Window PnL": round(q_pnl, 2) if np.isfinite(q_pnl) else np.nan,
                })
                closed.append(ticker)

        for ticker in closed:
            open_positions.pop(ticker, None)

        # 3) Equity row.
        ov = open_value(i)
        total_equity = cash_bank + ov
        max_open_positions = max(max_open_positions, len(open_positions))
        equity_rows.append({
            "Date": date_str,
            "Cash Bank": cash_bank,
            "Open Value": ov,
            "Total Equity": total_equity,
            "Open Positions": len(open_positions),
            "Exposure %": ov / total_equity * 100 if total_equity else 0
        })

        # 4) Generate next-day orders.
        try:
            qqq_slice = prices["QQQ"].iloc[i - 220:i + 1].dropna()
        except Exception:
            continue

        candidates = []
        for ticker in WATCHLIST:
            if ticker in open_positions or ticker in pending_orders:
                continue

            try:
                c = prices[ticker].iloc[i - 220:i + 1].dropna()
                h = highs[ticker].iloc[i - 220:i + 1].dropna()
                l = lows[ticker].iloc[i - 220:i + 1].dropna()
                row = evaluate_ticker(ticker, c, h, l, qqq_slice, params)
                if row["Status"] == "SIGNAL":
                    candidates.append(row)
            except Exception:
                continue

        candidates = sorted(candidates, key=lambda x: x["Score"], reverse=True)

        # Avoid creating more pending orders than cash can support.
        slot_size = max(100, (cash_bank + ov) * params.position_pct)
        available_slots = int(cash_bank // slot_size)
        if available_slots > 0:
            for cand in candidates[:available_slots]:
                pending_orders[cand["Ticker"]] = cand
                pending_created += 1

    # Final close.
    final_date = prices.index[end_idx].strftime("%Y-%m-%d")
    for ticker, pos in list(open_positions.items()):
        try:
            final_close = float(prices[ticker].iloc[end_idx])
            ret_pct = (final_close / pos["Entry"] - 1) * 100
            pnl = pos["Capital"] * ret_pct / 100
            cash_bank += pos["Capital"] + pnl

            trades.append({
                "TradeID": pos["TradeID"],
                "Ticker": ticker,
                "EntryDate": pos["EntryDate"],
                "ExitDate": final_date,
                "HoldingDays": end_idx - pos["EntryIndex"],
                "Setup": pos["Setup"],
                "Capital": round(pos["Capital"], 2),
                "Entry": round(pos["Entry"], 2),
                "Exit": round(final_close, 2),
                "ExitReason": "FINAL_CLOSE",
                "Return %": round(ret_pct, 2),
                "PnL": round(pnl, 2),
                "Risk %": round(pos["Risk %"], 2),
                "Score": round(pos["Score"], 2),
            })
        except Exception:
            pass

    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame(equity_rows).set_index("Date") if equity_rows else pd.DataFrame()

    if not df_equity.empty:
        df_equity.loc[final_date, "Cash Bank"] = cash_bank
        df_equity.loc[final_date, "Open Value"] = 0.0
        df_equity.loc[final_date, "Total Equity"] = cash_bank
        df_equity.loc[final_date, "Open Positions"] = 0
        df_equity.loc[final_date, "Exposure %"] = 0.0

    if not df_trades.empty:
        wins = int((df_trades["PnL"] > 0).sum())
        losses = int((df_trades["PnL"] < 0).sum())
        gross_profit = float(df_trades.loc[df_trades["PnL"] > 0, "PnL"].sum())
        gross_loss = float(-df_trades.loc[df_trades["PnL"] < 0, "PnL"].sum())
        total_pnl = float(df_trades["PnL"].sum())
        avg_ret = float(df_trades["Return %"].mean())
        avg_win = float(df_trades.loc[df_trades["PnL"] > 0, "Return %"].mean()) if wins else 0.0
        avg_loss = float(df_trades.loc[df_trades["PnL"] < 0, "Return %"].mean()) if losses else 0.0
        qqq_same_pnl = float(df_trades["QQQ Same Window PnL"].sum(skipna=True)) if "QQQ Same Window PnL" in df_trades else 0.0
    else:
        wins = losses = 0
        gross_profit = gross_loss = total_pnl = avg_ret = avg_win = avg_loss = qqq_same_pnl = 0.0

    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    win_rate = wins / max(1, wins + losses) * 100
    roi = (cash_bank / starting_bank - 1) * 100

    if not df_equity.empty:
        running_max = df_equity["Total Equity"].cummax()
        dd = (df_equity["Total Equity"] / running_max - 1) * 100
        max_dd = float(dd.min())
        avg_exposure = float(df_equity["Exposure %"].mean())
    else:
        max_dd = avg_exposure = 0.0

    try:
        q = prices["QQQ"].iloc[start_idx:end_idx].dropna()
        qqq_bh_pct = (float(q.iloc[-1]) / float(q.iloc[0]) - 1) * 100 if len(q) > 1 else np.nan
        qqq_bh_pnl = starting_bank * qqq_bh_pct / 100
    except Exception:
        qqq_bh_pct = np.nan
        qqq_bh_pnl = np.nan

    if not df_trades.empty:
        ticker_summary = (
            df_trades.groupby("Ticker", as_index=False)
            .agg(
                Trades=("TradeID", "count"),
                PnL=("PnL", "sum"),
                AvgReturn=("Return %", "mean"),
                Wins=("PnL", lambda x: int((x > 0).sum())),
                Losses=("PnL", lambda x: int((x < 0).sum())),
            )
            .sort_values("PnL", ascending=False)
        )
        ticker_summary["WinRate %"] = (ticker_summary["Wins"] / ticker_summary["Trades"] * 100).round(1)
        ticker_summary["PnL"] = ticker_summary["PnL"].round(2)
        ticker_summary["AvgReturn"] = ticker_summary["AvgReturn"].round(2)
    else:
        ticker_summary = pd.DataFrame()

    metrics = {
        "App Version": APP_VERSION,
        "Starting Bank": round(starting_bank, 2),
        "Ending Bank": round(cash_bank, 2),
        "Bank ROI %": round(roi, 2),
        "Position % of Bank": round(params.position_pct * 100, 2),
        "Max Risk %": params.max_risk_pct,
        "Exit Rule": "No fixed target; full exit on EMA21/trailing stop",
        "Trades": len(df_trades),
        "Wins": wins,
        "Losses": losses,
        "Win Rate %": round(win_rate, 2),
        "Total PnL": round(total_pnl, 2),
        "Average Trade Return %": round(avg_ret, 2),
        "Average Win %": round(avg_win, 2),
        "Average Loss %": round(avg_loss, 2),
        "Profit Factor": round(pf, 2) if np.isfinite(pf) else "∞",
        "Max Drawdown %": round(max_dd, 2),
        "Average Exposure %": round(avg_exposure, 2),
        "Max Open Positions": max_open_positions,
        "Turnover": round(turnover, 2),
        "QQQ Buy & Hold %": round(qqq_bh_pct, 2) if np.isfinite(qqq_bh_pct) else "",
        "QQQ Buy & Hold PnL": round(qqq_bh_pnl, 2) if np.isfinite(qqq_bh_pnl) else "",
        "QQQ Same Windows PnL": round(qqq_same_pnl, 2),
        "Pending Orders Created": pending_created,
        "Pending Orders Filled": pending_filled,
        "Pending Orders Expired": pending_expired,
        "Pending No Cash": pending_no_cash,
    }

    return df_equity, df_trades, ticker_summary, metrics
# ==========================================================
# 7. Live Daily Dashboard
# ==========================================================

# ==========================================================
# 7A. V12 Coiled Breakout Cheat Entry
# ==========================================================
def calc_breakout_score(
    current,
    breakout_dist,
    risk_pct,
    rr8,
    atr_pct,
    rsi,
    run20,
    range10,
    rs5,
    rs20,
    setup_type,
    atr_pinch=1.0,
    var_15=1.0,
):
    """
    Practical score: not academic momentum score.
    Rewards: close trigger, manageable risk, ability to move 8%-12%, improving RS, tight base.
    Penalizes: overextension, far trigger, wide range, high risk.
    """
    score = 50.0

    # Trigger proximity
    if 0 <= breakout_dist <= 1.0:
        score += 18
    elif 1.0 < breakout_dist <= 3.0:
        score += 12
    elif 3.0 < breakout_dist <= 6.0:
        score += 4
    else:
        score -= 12

    # Risk / reward
    score += min(rr8, 3.0) * 8
    score -= max(risk_pct - 6.0, 0) * 4

    # Volatility: we need movement, not dead money
    if 1.5 <= atr_pct <= 6.5:
        score += 8
    elif atr_pct < 1.2:
        score -= 8
    elif atr_pct > 9.0:
        score -= 6

    # RSI zone
    if 45 <= rsi <= 68:
        score += 8
    elif 68 < rsi <= 75:
        score += 2
    elif rsi > 80:
        score -= 12
    elif rsi < 35:
        score -= 8

    # Overextension
    if run20 <= 15:
        score += 8
    elif run20 <= 25:
        score += 2
    elif run20 <= 30:
        score -= 6
    else:
        score -= 20

    # Consolidation quality
    if range10 <= 7:
        score += 8
    elif range10 <= 12:
        score += 3
    else:
        score -= 5

    # Relative strength: useful, but not the only thing
    if rs20 > 3:
        score += 6
    if rs5 > rs20 + 3:
        score += 6
    elif rs5 > 0:
        score += 3
    if rs5 < -5 and rs20 < -5:
        score -= 10

    # V12.3 Energy / squeeze additions
    if atr_pinch < 0.65:
        score += 10
    elif atr_pinch < 0.75:
        score += 5

    if var_15 > 1.5:
        score += 5
    elif var_15 > 1.25:
        score += 2

    # Route
    if setup_type == "Momentum Breakout":
        score += 6
    elif setup_type == "Coiled Breakout":
        score += 10
    elif setup_type == "Base Breakout":
        score += 4
    elif setup_type == "Turnaround Watch":
        score -= 2

    return round(score, 2)



def classify_action_quality(setup_type, rs5, rs20, atr_pct, risk_pct, run20, rsi, rr8, score, trigger_dist, range10):
    """
    V12.2 gate:
    BUY SETUP READY should mean: close breakout + enough fuel for 8%-12% + not a FOMO/exhaustion setup.
    Hot names with RSI/large range/wide stop move to NEAR READY or WAIT FOR PULLBACK.
    """
    notes = []
    exhaustion_notes = []

    # If trigger is already behind us, this is not a fresh order.
    if trigger_dist < -0.25:
        return "MISSED / WAIT RESET", "המחיר כבר מעל טריגר הקנייה — לא רודפים אחרי פקודה שברחה", "MISSED"

    has_fuel = (
        setup_type == "Momentum Breakout"
        or rs20 > 0
        or rs5 >= 3
        or atr_pct >= 2.5
    )

    if not has_fuel:
        notes.append("חסר דלק: RS/ATR לא מספיקים ליעד 8%-12%")

    if setup_type == "Base Breakout" and rs20 < -5 and rs5 < 3:
        notes.append("Base Breakout עם RS20 שלילי מדי")

    if atr_pct < 2.5:
        notes.append("ATR% נמוך לפקודת ביצוע")

    if risk_pct > 7.5:
        notes.append("סיכון רחב יחסית")

    if run20 > 25:
        notes.append("ריצה קודמת גבוהה יחסית")

    if rsi > 74:
        notes.append("RSI גבוה / מעט מתוחה")

    if rr8 < 1.0:
        notes.append("R/R ל-8% נמוך מ-1")

    # V12.2 exhaustion / FOMO rules
    if rsi >= 80:
        exhaustion_notes.append("RSI מעל 80")
    if range10 >= 15:
        exhaustion_notes.append("טווח 10 ימים רחב מאוד")
    if risk_pct >= 7.5:
        exhaustion_notes.append("סטופ רחב")
    if run20 >= 15:
        exhaustion_notes.append("כבר רצה מעל 15% ב-20 יום")
    if trigger_dist < 0.30 and setup_type != "Coiled Breakout":
        exhaustion_notes.append("הטריגר צמוד מדי למחיר — חשש FOMO")

    high_exhaustion = (
        rsi >= 80
        and (
            risk_pct >= 7.0
            or range10 >= 15
            or run20 >= 15
        )
    )

    if high_exhaustion:
        combined = " | ".join(notes + exhaustion_notes)
        return "NEAR READY", combined or "חמה מדי לפקודת ביצוע", "HIGH"

    # A very wide 10-day range means the setup is not a clean base.
    if range10 >= 20:
        combined = " | ".join(notes + exhaustion_notes)
        return "NEAR READY", combined or "טווח 10 ימים רחב מדי לפריצה נקייה", "MEDIUM"

    # READY only if the setup has fuel and no major blocking notes.
    note_blob = " ".join(notes)
    major_block = any(
        key in note_blob
        for key in ["חסר דלק", "RS20 שלילי מדי", "ATR% נמוך", "R/R"]
    )

    if score >= 70 and has_fuel and not major_block and risk_pct <= 7.5:
        return "READY", " | ".join(notes) or "סטאפ איכותי: טריגר קרוב + דלק מספיק", "LOW"
    if score >= 60 and has_fuel and not major_block and not exhaustion_notes:
        return "READY", " | ".join(notes) or "סטאפ פעולה סביר", "LOW"
    if score >= 50:
        return "NEAR READY", " | ".join(notes + exhaustion_notes) or "קרוב, אבל חסרה איכות פעולה", "MEDIUM" if exhaustion_notes else "LOW"
    return "WATCH", " | ".join(notes + exhaustion_notes) or "לא מספיק איכותי לפעולה", "MEDIUM" if exhaustion_notes else "LOW"


def evaluate_breakout_action_plan(
    ticker: str,
    c: pd.Series,
    h: pd.Series,
    l: pd.Series,
    v: pd.Series,
    qqq_slice: pd.Series,
    params: StrategyParams,
):
    """
    V12 practical output:
    - BUY SETUP READY: actionable now/near now
    - WAIT FOR BREAKOUT: has a trigger price
    - WAIT FOR PULLBACK: too extended; has pullback prices
    - TURNAROUND WATCH: improving, not ready
    - IGNORE: hidden by default
    """
    base = {
        "Ticker": ticker,
        "State": "IGNORE",
        "Setup Type": "Weak / Ignore",
        "Current": np.nan,
        "Buy Trigger": np.nan,
        "Distance to Trigger %": np.nan,
        "Next Action Price": np.nan,
        "Distance to Action %": np.nan,
        "Stop": np.nan,
        "Risk %": np.nan,
        "Target 8%": np.nan,
        "Target 12%": np.nan,
        "RR 8%": np.nan,
        "RR 12%": np.nan,
        "Breakout Score": np.nan,
        "Action Quality": "",
        "Quality Notes": "",
        "Exhaustion Risk": "",
        "What We Need": "",
        "Why": "",
        "EMA21": np.nan,
        "SMA200": np.nan,
        "RS5": np.nan,
        "RS20": np.nan,
        "RSI": np.nan,
        "ATR%": np.nan,
        "20D Run": np.nan,
        "10D Range %": np.nan,
        "ATR Pinch": np.nan,
        "VAR 15d": np.nan,
        "Inside Day": False,
        "Tight Day": False,
        "Run Zone": "",
        "Pullback Watch Price": np.nan,
        "Pullback Deep Price": np.nan,
        "Breakout Watch Price": np.nan,
        "Regime": "",
    }

    try:
        if len(c) < 220:
            base.update(State="IGNORE", **{"What We Need": "More Data"}, Why="אין מספיק היסטוריה")
            return base

        last_p = float(c.iloc[-1])
        sma200 = float(c.rolling(200).mean().iloc[-1])
        sma50 = float(c.rolling(50).mean().iloc[-1])
        ema8 = float(c.ewm(span=8, adjust=False).mean().iloc[-1])
        ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
        high20 = float(h.iloc[-21:-1].max())
        high50 = float(h.iloc[-51:-1].max())
        breakout_level = max(high20, high50 if high50 <= last_p * 1.08 else high20)
        breakout_trigger = round(breakout_level * 1.002, 2)

        atr = float(calc_atr_from_series(h, l, c, 14).iloc[-1])
        atr_pct = atr / last_p * 100 if last_p else np.nan
        rsi = float(calc_rsi(c, 14).iloc[-1])
        run20 = (last_p / c.iloc[-21] - 1) * 100
        rs5, rs20 = relative_strength_vs(qqq_slice, c)
        regime = market_regime(qqq_slice)

        high10 = float(h.iloc[-10:].max())
        low10 = float(l.iloc[-10:].min())
        range10 = (high10 / low10 - 1) * 100 if low10 > 0 else np.nan

        if run20 <= 15:
            run_zone = "NORMAL"
        elif run20 <= 25:
            run_zone = "WARM"
        elif run20 <= 30:
            run_zone = "HOT"
        elif run20 <= 45:
            run_zone = "EXTENDED"
        else:
            run_zone = "TOO_EXTENDED"

        trigger_dist = (breakout_trigger / last_p - 1) * 100
        pullback_watch = round(ema8 * 1.003, 2)
        pullback_deep = round(ema21 * 1.005, 2)

        base.update({
            "Current": round(last_p, 2),
            "Buy Trigger": breakout_trigger,
            "Next Action Price": breakout_trigger,
            "Distance to Trigger %": round(trigger_dist, 2),
            "Distance to Action %": round(trigger_dist, 2),
            "Breakout Watch Price": breakout_trigger,
            "EMA21": round(ema21, 2),
            "SMA200": round(sma200, 2),
            "RS5": round(rs5, 1),
            "RS20": round(rs20, 1),
            "RSI": round(rsi, 1),
            "ATR%": round(atr_pct, 1) if np.isfinite(atr_pct) else np.nan,
            "20D Run": round(run20, 1),
            "10D Range %": round(range10, 1) if np.isfinite(range10) else np.nan,
            "ATR Pinch": round(atr_pinch, 2),
            "VAR 15d": round(var_15, 2),
            "Inside Day": inside_day,
            "Tight Day": tight_day,
            "Run Zone": run_zone,
            "Regime": regime,
        })

        if last_p < sma200:
            base.update(State="IGNORE", **{"What We Need": "Above SMA200"}, Why="מתחת SMA200 — לא מחפשים פריצה בכיוון לונג")
            return base

        if not np.isfinite(atr_pct) or atr_pct < 1.2:
            base.update(State="IGNORE", **{"What We Need": "More Volatility"}, Why="ATR% נמוך מדי — פחות מתאים לסווינג של 8%-12%")
            return base

        # Too extended: not an entry plan, but still very useful.
        if run20 > 30:
            base.update({
                "State": "WAIT FOR PULLBACK",
                "Setup Type": "Extended Leader",
                "Next Action Price": pullback_watch,
                "Distance to Action %": round((pullback_watch / last_p - 1) * 100, 2),
                "Pullback Watch Price": pullback_watch,
                "Pullback Deep Price": pullback_deep,
                "What We Need": "Pullback to EMA8/EMA21",
                "Why": "המניה חזקה אבל כבר רצה מעל 30% ב-20 יום — לא רודפים; מחכים לפולבק"
            })
            return base

        # Route classification.
        rs_improving = rs5 > rs20 + 3
        price_recovered = last_p > ema21 and ema21 >= sma50 * 0.92
        momentum_leader = (rs20 > 3 or rs5 > 3) and last_p > ema21
        base_breakout = price_recovered and run20 <= 25 and 40 <= rsi <= 72 and (rs5 > 0 or rs_improving)
        turnaround_watch = (rs20 < 0) and rs_improving and last_p >= ema21 * 0.97 and 32 <= rsi <= 65

        # V12.3: Coiled Breakout / Cheat Entry route.
        # Uses last CLOSED candle only, not the still-forming intraday candle.
        is_coiled = (
            atr_pinch < 0.75
            and np.isfinite(range10)
            and range10 <= 12
            and (inside_day or tight_day)
            and last_p > sma200
            and last_p >= ema21 * 0.98
            and run20 <= 25
            and trigger_dist <= 2.5
            and trigger_dist >= -0.25
        )

        if is_coiled:
            setup_type = "Coiled Breakout"
            entry = round(float(h.iloc[-2]) * 1.002, 2)
            raw_stop = min(float(l.iloc[-2]) * 0.99, ema8 * 0.995)
        elif momentum_leader:
            setup_type = "Momentum Breakout"
            entry = breakout_trigger
            raw_stop = max(entry - 2.2 * atr, ema21 * 0.995, entry * (1 - 0.08))
        elif base_breakout:
            setup_type = "Base Breakout"
            entry = breakout_trigger
            raw_stop = max(entry - 2.2 * atr, ema21 * 0.995, entry * (1 - 0.08))
        elif turnaround_watch:
            setup_type = "Turnaround Watch"
            entry = breakout_trigger
            raw_stop = max(entry - 2.2 * atr, ema21 * 0.995, entry * (1 - 0.08))
        else:
            setup_type = "Weak / Ignore"
            entry = breakout_trigger
            raw_stop = max(entry - 2.2 * atr, ema21 * 0.995, entry * (1 - 0.08))

        base["Setup Type"] = setup_type

        # Initial stop and target plan: relevant only when we have a buy trigger.
        # Stop = the tighter of technical protection and max risk cap, but not above entry.
        stop = min(raw_stop, entry * 0.995)
        stop = round(stop, 2)
        risk_pct = (entry - stop) / entry * 100
        target8 = round(entry * 1.08, 2)
        target12 = round(entry * 1.12, 2)
        rr8 = 8 / risk_pct if risk_pct > 0 else np.nan
        rr12 = 12 / risk_pct if risk_pct > 0 else np.nan

        # V12.3: Strict R/R enforcement for Cheat Entry.
        # If the early entry is not actually tight, revert to standard trigger.
        if setup_type == "Coiled Breakout" and (risk_pct > 5.5 or rr8 < 1.5):
            if momentum_leader:
                setup_type = "Momentum Breakout"
            elif base_breakout:
                setup_type = "Base Breakout"
            else:
                setup_type = "Turnaround Watch"

            entry = breakout_trigger
            raw_stop = max(entry - 2.2 * atr, ema21 * 0.995, entry * (1 - 0.08))
            stop = min(raw_stop, entry * 0.995)
            stop = round(stop, 2)
            risk_pct = (entry - stop) / entry * 100
            target8 = round(entry * 1.08, 2)
            target12 = round(entry * 1.12, 2)
            rr8 = 8 / risk_pct if risk_pct > 0 else np.nan
            rr12 = 12 / risk_pct if risk_pct > 0 else np.nan

        base["Setup Type"] = setup_type
        trigger_dist = (entry / last_p - 1) * 100

        base.update({
            "Buy Trigger": round(entry, 2),
            "Next Action Price": round(entry, 2),
            "Distance to Trigger %": round(trigger_dist, 2),
            "Distance to Action %": round(trigger_dist, 2),
            "Breakout Watch Price": round(entry, 2),
        })

        score = calc_breakout_score(
            last_p, trigger_dist, risk_pct, rr8, atr_pct, rsi, run20, range10,
            rs5, rs20, setup_type, atr_pinch, var_15
        )

        base.update({
            "Stop": stop,
            "Risk %": round(risk_pct, 2),
            "Target 8%": target8,
            "Target 12%": target12,
            "RR 8%": round(rr8, 2) if np.isfinite(rr8) else np.nan,
            "RR 12%": round(rr12, 2) if np.isfinite(rr12) else np.nan,
            "Breakout Score": score,
        })

        if setup_type == "Weak / Ignore":
            base.update(State="IGNORE", **{"What We Need": "Stronger setup"}, Why="אין כרגע סימני פריצה/התאוששות מספיקים")
            return base

        # Far from trigger: watch, not action.
        if trigger_dist > 6:
            base.update(
                State="TURNAROUND WATCH" if setup_type == "Turnaround Watch" else "WAIT FOR BREAKOUT",
                **{"What We Need": "Move closer to breakout trigger"},
                Why="יש כיוון מעניין אבל הטריגר עדיין רחוק מדי לפקודה"
            )
            return base

        # Good watch candidate but not actionable yet.
        if trigger_dist > 3:
            base.update(
                State="WAIT FOR BREAKOUT",
                **{"What We Need": "Breakout closer than 3%"},
                Why="מועמדת מעניינת, אבל הפריצה לא מספיק קרובה"
            )
            return base

        # Too high risk for target 8%-12%.
        if risk_pct > 8:
            base.update(
                State="WAIT FOR BREAKOUT",
                **{"What We Need": "Tighter risk / closer base"},
                Why=f"הטריגר קרוב, אבל הסטופ רחב מדי ({risk_pct:.1f}%)"
            )
            return base

        # Score threshold for action.
        if score < 60:
            state = "TURNAROUND WATCH" if setup_type == "Turnaround Watch" else "WAIT FOR BREAKOUT"
            base.update(
                State=state,
                **{"What We Need": "Higher readiness score"},
                Why=f"קרובה לטריגר, אבל איכות הסטאפ עדיין לא מספקת ({score:.1f})"
            )
            return base

        # V12.1: Action quality gate.
        quality, quality_notes, exhaustion_risk = classify_action_quality(
            setup_type, rs5, rs20, atr_pct, risk_pct, run20, rsi, rr8, score, trigger_dist, range10
        )

        base["Action Quality"] = quality
        base["Quality Notes"] = quality_notes
        base["Exhaustion Risk"] = exhaustion_risk

        if quality == "MISSED / WAIT RESET":
            base.update(
                State="MISSED / WAIT RESET",
                **{"What We Need": "Wait for new base / pullback"},
                Why=quality_notes
            )
            return base

        if quality == "READY":
            base.update(
                State="BUY SETUP READY",
                **{"What We Need": "Place stop-limit order"},
                Why="טריגר קרוב + איכות פעולה מספקת ליעד 8%-12%"
            )
            return base

        if quality == "NEAR READY":
            base.update(
                State="NEAR READY",
                **{"What We Need": "עוד אישור מומנטום / RS / ATR לפני פקודה"},
                Why=f"פריצה קרובה אבל לא מספיק חזקה לפקודת ביצוע: {quality_notes}"
            )
            return base

        base.update(
            State="WAIT FOR BREAKOUT",
            **{"What We Need": "שיפור איכות לפני פעולה"},
            Why=f"טריגר קרוב אך איכות פעולה נמוכה: {quality_notes}"
        )
        return base

    except Exception as e:
        base.update(State="ERROR", Why=str(e)[:120])
        return base


def get_today_breakout_action_plan(params: StrategyParams):
    qqq = download_single("QQQ", "370d")
    if qqq.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    qqq_slice = qqq["Close"].dropna()
    action_rows = []
    watch_rows = []
    ignore_rows = []

    for ticker in WATCHLIST:
        df = download_single(ticker, "370d")
        if df.empty or len(df) < 220:
            continue

        try:
            c, h, l, v = df["Close"].dropna(), df["High"].dropna(), df["Low"].dropna(), df["Volume"].dropna()
            row = evaluate_breakout_action_plan(ticker, c, h, l, v, qqq_slice, params)

            state = row.get("State", "IGNORE")
            if state == "BUY SETUP READY":
                action_rows.append(row)
            elif state in ["NEAR READY", "MISSED / WAIT RESET", "WAIT FOR BREAKOUT", "WAIT FOR PULLBACK", "TURNAROUND WATCH"]:
                watch_rows.append(row)
            else:
                ignore_rows.append(row)
        except Exception:
            continue

    df_action = pd.DataFrame(action_rows)
    df_watch = pd.DataFrame(watch_rows)
    df_ignore = pd.DataFrame(ignore_rows)

    if not df_action.empty:
        df_action = df_action.sort_values("Breakout Score", ascending=False)
    if not df_watch.empty:
        state_rank = {
            "NEAR READY": 0,
            "MISSED / WAIT RESET": 1,
            "WAIT FOR BREAKOUT": 2,
            "WAIT FOR PULLBACK": 3,
            "TURNAROUND WATCH": 4,
        }
        df_watch["_StateRank"] = df_watch["State"].map(state_rank).fillna(9)
        df_watch = df_watch.sort_values(["_StateRank", "Breakout Score"], ascending=[True, False], na_position="last").drop(columns=["_StateRank"])
    if not df_ignore.empty:
        df_ignore = df_ignore.sort_values(["State", "Ticker"], ascending=[True, True])

    return df_action, df_watch, df_ignore


def get_today_actions(params: StrategyParams):
    qqq = download_single("QQQ", "370d")
    if qqq.empty:
        return pd.DataFrame(), pd.DataFrame()

    qqq_slice = qqq["Close"].dropna()

    rows = []
    radar = []

    for ticker in WATCHLIST:
        df = download_single(ticker, "370d")
        if df.empty or len(df) < 220:
            continue

        c, h, l = df["Close"], df["High"], df["Low"]
        row = evaluate_ticker(ticker, c, h, l, qqq_slice, params)

        if row["Status"] == "SIGNAL":
            rows.append(row)
        else:
            radar.append(row)

    df_orders = pd.DataFrame(rows)
    if not df_orders.empty:
        df_orders = df_orders.sort_values("Score", ascending=False)

    df_radar = pd.DataFrame(radar)
    if not df_radar.empty:
        df_radar = df_radar.sort_values("Score", ascending=False)

    return df_orders, df_radar



# ==========================================================
# 8. Exit / Position Management Dashboard
# ==========================================================
def analyze_open_position(ticker: str, entry_price: float, entry_date, initial_stop: float = np.nan):
    """
    Reconstructs the current exit status for a real open position.
    Required:
    - ticker
    - entry price
    - entry date
    Optional:
    - initial stop. If missing, uses entry * (1 - max_risk_pct) is NOT available here,
      so caller should pass one when possible.
    """
    df = download_single(ticker, "370d")
    if df.empty or len(df) < 30:
        return {
            "Ticker": ticker,
            "Status": "ERROR",
            "Action": "NO DATA",
            "Reason": "אין מספיק נתונים",
        }

    try:
        entry_date = pd.to_datetime(entry_date)
    except Exception:
        entry_date = df.index[-30]

    c = df["Close"].dropna()
    h = df["High"].dropna()
    l = df["Low"].dropna()

    if c.empty:
        return {
            "Ticker": ticker,
            "Status": "ERROR",
            "Action": "NO DATA",
            "Reason": "אין נתוני Close",
        }

    last_close = float(c.iloc[-1])
    last_low = float(l.iloc[-1])
    last_date = c.index[-1].strftime("%Y-%m-%d")

    # EMA21 trail is reconstructed only from dates after entry.
    hist_from_entry = c[c.index >= entry_date]
    if len(hist_from_entry) < 2:
        hist_from_entry = c.iloc[-25:]

    ema21_series = c.ewm(span=21, adjust=False).mean()
    ema21_now = float(ema21_series.iloc[-1])

    # Initial stop fallback if user did not enter one.
    if not np.isfinite(initial_stop) or initial_stop <= 0:
        initial_stop = entry_price * 0.905  # fallback = 9.5% risk

    # Reconstruct trailing stop from entry date onward:
    # trail never moves down, it is max(initial_stop, EMA21*0.995 since entry).
    ema_after_entry = ema21_series[ema21_series.index >= entry_date]
    if len(ema_after_entry) == 0:
        ema_after_entry = ema21_series.iloc[-25:]

    trail_series = (ema_after_entry * 0.995).cummax()
    trail_stop = max(float(initial_stop), float(trail_series.iloc[-1]))

    pnl_pct = (last_close / entry_price - 1) * 100

    # Two exit signals:
    # 1. Hard/trailing stop touched intraday.
    # 2. Daily close below EMA21*0.995.
    close_exit_level = ema21_now * 0.995

    if last_low <= trail_stop:
        action = "SELL / STOP HIT"
        status = "EXIT"
        reason = "המחיר היומי נגע בסטופ/Trailing Stop"
    elif last_close < close_exit_level:
        action = "SELL AT CLOSE / NEXT OPEN"
        status = "EXIT"
        reason = "הסגירה מתחת EMA21"
    else:
        action = "HOLD"
        status = "HOLD"
        reason = "המגמה עדיין לא נשברה"

    distance_to_exit = (last_close / trail_stop - 1) * 100 if trail_stop > 0 else np.nan

    return {
        "Ticker": ticker,
        "Status": status,
        "Action": action,
        "Reason": reason,
        "Last Date": last_date,
        "Entry": round(entry_price, 2),
        "Current": round(last_close, 2),
        "PnL %": round(pnl_pct, 2),
        "Initial Stop": round(initial_stop, 2),
        "EMA21": round(ema21_now, 2),
        "Exit Close Level": round(close_exit_level, 2),
        "Trailing Stop": round(trail_stop, 2),
        "Current Protection Stop": round(max(trail_stop, close_exit_level), 2),
        "Profit Checkpoint": round(max(entry_price * 1.15, last_close * 1.05), 2),
        "Distance to Trail %": round(distance_to_exit, 2),
    }


def build_positions_template():
    return pd.DataFrame([
        {"Ticker": "AMD", "Entry": 0.0, "Entry Date": "2026-05-01", "Initial Stop": 0.0},
    ])


def build_virtual_portfolio_template():
    return pd.DataFrame([
        {
            "Ticker": "AMD",
            "Quantity": 0.0,
            "Avg Entry": 0.0,
            "Entry Date": "2026-05-01",
            "Initial Stop": 0.0,
        }
    ])


def analyze_virtual_portfolio(df_positions: pd.DataFrame):
    rows = []

    for _, row in df_positions.iterrows():
        ticker = str(row.get("Ticker", "")).strip().upper()
        qty = safe_float(row.get("Quantity", np.nan))
        avg_entry = safe_float(row.get("Avg Entry", np.nan))
        entry_date = row.get("Entry Date", "")
        initial_stop = safe_float(row.get("Initial Stop", np.nan))

        if not ticker or not np.isfinite(qty) or qty <= 0 or not np.isfinite(avg_entry) or avg_entry <= 0:
            continue

        status = analyze_open_position(ticker, avg_entry, entry_date, initial_stop)

        current = safe_float(status.get("Current", np.nan))
        cost = qty * avg_entry
        market_value = qty * current if np.isfinite(current) else np.nan
        pnl_dollar = market_value - cost if np.isfinite(market_value) else np.nan
        pnl_pct = (current / avg_entry - 1) * 100 if np.isfinite(current) and avg_entry else np.nan

        rows.append({
            "Ticker": ticker,
            "Quantity": qty,
            "Avg Entry": round(avg_entry, 2),
            "Current": round(current, 2) if np.isfinite(current) else np.nan,
            "Cost": round(cost, 2),
            "Market Value": round(market_value, 2) if np.isfinite(market_value) else np.nan,
            "PnL $": round(pnl_dollar, 2) if np.isfinite(pnl_dollar) else np.nan,
            "PnL %": round(pnl_pct, 2) if np.isfinite(pnl_pct) else np.nan,
            "Action": status.get("Action", ""),
            "Reason": status.get("Reason", ""),
            "Trailing Stop": status.get("Trailing Stop", np.nan),
            "Current Protection Stop": status.get("Current Protection Stop", np.nan),
            "Profit Checkpoint": status.get("Profit Checkpoint", np.nan),
            "Distance to Trail %": status.get("Distance to Trail %", np.nan),
            "EMA21": status.get("EMA21", np.nan),
            "Exit Close Level": status.get("Exit Close Level", np.nan),
            "Initial Stop": status.get("Initial Stop", initial_stop),
            "Entry Date": entry_date,
            "Last Date": status.get("Last Date", ""),
        })

    return pd.DataFrame(rows)

# ==========================================================
# 8. Export
# ==========================================================
def build_zip_report(df_equity, df_trades, ticker_summary, metrics, daily_orders=None, daily_radar=None):
    output = BytesIO()
    metrics_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()])

    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.csv", metrics_df.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("bank_equity.csv", df_equity.reset_index().to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("trades.csv", df_trades.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("ticker_summary.csv", ticker_summary.to_csv(index=False).encode("utf-8-sig"))
        if daily_orders is not None:
            zf.writestr("daily_orders.csv", daily_orders.to_csv(index=False).encode("utf-8-sig"))
        if daily_radar is not None:
            zf.writestr("daily_radar.csv", daily_radar.to_csv(index=False).encode("utf-8-sig"))

    output.seek(0)
    return output.getvalue()



def get_column_config():
    return {
        "Ticker": st.column_config.TextColumn("Ticker", help="סימול המניה בבורסה, למשל AMD או TSLA."),
        "Action Now": st.column_config.TextColumn("מה לעשות עכשיו", help="הוראת פעולה יומית: PLACE LIMIT, PLACE STOP LIMIT, WAIT או SELL/HOLD."),
        "Order": st.column_config.TextColumn("סוג פקודה", help="BUY LIMIT = קנייה בירידה למחיר מסוים. BUY STOP LIMIT = קנייה רק אם המחיר פורץ למעלה לרמת הכניסה."),
        "Current": st.column_config.NumberColumn("מחיר נוכחי", help="המחיר האחרון שהמערכת משכה מ-Yahoo Finance.", format="%.2f"),
        "Entry": st.column_config.NumberColumn("שער כניסה", help="השער שבו המודל רוצה להיכנס. לא בהכרח השער הנוכחי.", format="%.2f"),
        "Distance to Entry %": st.column_config.NumberColumn("מרחק לכניסה %", help="כמה אחוזים המחיר צריך לעלות/לרדת כדי להגיע לשער הכניסה.", format="%.2f%%"),
        "Stop": st.column_config.NumberColumn("סטופ", help="שער הגנה התחלתי. אם המחיר מגיע אליו, המודל יוצא מהעסקה.", format="%.2f"),
        "Trailing Stop": st.column_config.NumberColumn("Trailing Stop", help="סטופ עוקב שעולה עם המגמה ולא יורד. מיועד לשמור על רווחים כשהמניה עולה.", format="%.2f"),
        "Current Protection Stop": st.column_config.NumberColumn("שער הגנה נוכחי", help="זה לא יעד רווח. זה שער ההגנה הנוכחי: סטופ התחלתי / Trailing Stop / רמת יציאה לפי EMA21. אם המחיר נשבר אליו — יוצאים.", format="%.2f"),
        "Profit Checkpoint": st.column_config.NumberColumn("יעד רווח לבדיקה", help="יעד רווח אינדיקטיבי מעל המחיר: בכניסה חדשה כ-15% מעל שער הכניסה; בפוזיציה קיימת הגבוה מבין 15% מעל הכניסה או 5% מעל המחיר הנוכחי. לא פקודת מכירה אוטומטית.", format="%.2f"),
        "Exit Close Level": st.column_config.NumberColumn("רמת יציאה בסגירה", help="אם המניה סוגרת יום מתחת לרמה הזו, המודל מסמן יציאה. מחושב כ-EMA21 × 0.995.", format="%.2f"),
        "EMA21": st.column_config.NumberColumn("EMA21", help="ממוצע נע אקספוננציאלי של 21 ימי מסחר. משמש למדידת המגמה ולכללי יציאה.", format="%.2f"),
        "Exit Rule": st.column_config.TextColumn("כלל יציאה", help="בגרסת Trend אין יעד רווח קשיח. יוצאים כשהמניה שוברת EMA21/Trailing Stop."),
        "Risk %": st.column_config.NumberColumn("סיכון %", help="המרחק באחוזים בין שער הכניסה לסטופ ההתחלתי.", format="%.2f%%"),
        "Target": st.column_config.TextColumn("יעד", help="בגרסת Trend אין יעד רווח קשיח; הרווח רץ כל עוד המגמה חיה."),
        "Target %": st.column_config.TextColumn("יעד %", help="בגרסת Trend לא משתמשים ביעד אחוזי קבוע."),
        "R/R": st.column_config.TextColumn("R/R", help="Risk/Reward. בגרסת Trend אין יעד קשיח ולכן זה מסומן כ-Trend."),
        "Setup": st.column_config.TextColumn("תבנית", help="סוג האיתות: פריצה, פולבק איכותי, או המשך מומנטום חריג."),
        "Score": st.column_config.NumberColumn("ניקוד", help="ציון איכות של האיתות לפי חוזק יחסי, מומנטום, סיכון, תבנית ומצב שוק.", format="%.2f"),
        "Regime": st.column_config.TextColumn("מצב שוק", help="מצב QQQ: BULL_STRONG, BULL_WEAK, PULLBACK או BEAR."),
        "RS5": st.column_config.NumberColumn("RS5", help="חוזק יחסי של המניה מול QQQ ב-5 ימי מסחר. חיובי = המניה חזקה מ-QQQ.", format="%.1f"),
        "RS20": st.column_config.NumberColumn("RS20", help="חוזק יחסי של המניה מול QQQ ב-20 ימי מסחר. חיובי = המניה מובילה את QQQ.", format="%.1f"),
        "RSI": st.column_config.NumberColumn("RSI", help="מדד מומנטום 0–100. גבוה מאוד יכול להעיד שהמניה מתוחה.", format="%.1f"),
        "ATR%": st.column_config.NumberColumn("ATR%", help="תנודתיות יומית ממוצעת כאחוז מהמחיר. עוזר להבין אם המניה זזה מספיק לסווינג.", format="%.1f%%"),
        "20D Run": st.column_config.NumberColumn("ריצה 20 יום", help="כמה המניה עלתה/ירדה ב-20 ימי המסחר האחרונים.", format="%.1f%%"),
        "Run Zone": st.column_config.TextColumn("אזור ריצה", help="NORMAL עד 15%, HOT עד 30%, EXTENDED עד 45%, TOO_EXTENDED מעל 45%. מעל 30% לא קונים פריצה אלא מחכים לפולבק."),
        "Setup Type": st.column_config.TextColumn("סוג מסלול", help="Momentum Leader = מניה מובילה; Base Breakout = בסיס לפני פריצה; Turnaround Watch = התחלת שיפור; Weak/Ignore = כרגע לא מעניין."),
        "Action State": st.column_config.TextColumn("מצב פעולה", help="PLACE ORDER / WAIT FOR PULLBACK / WAIT FOR BREAKOUT / TURNAROUND WATCH / WAIT / NO TRADE."),
        "Pullback Watch Price": st.column_config.NumberColumn("מחיר מעקב לפולבק", help="אזור כניסה מעניין אם מניה חזקה מדי תתקן לכיוון EMA8. לא פקודת קנייה אוטומטית.", format="%.2f"),
        "Pullback Deep Price": st.column_config.NumberColumn("מחיר פולבק עמוק", help="אזור כניסה עמוק יותר סביב EMA21. מתאים למניות שרצו חזק מדי וצריך לחכות להן.", format="%.2f"),
        "Breakout Watch Price": st.column_config.NumberColumn("מחיר מעקב לפריצה", help="רמה שמעליה מתחיל להיות מעניין לבדוק פריצה. שימושי במיוחד למסלול Base Breakout / Turnaround.", format="%.2f"),
        "Status": st.column_config.TextColumn("סטטוס", help="SIGNAL = איתות ביצוע. WATCH = במעקב. REJECT = לא רלוונטית כרגע."),
        "Decision": st.column_config.TextColumn("החלטה", help="ACTION / WAIT / NO TRADE לפי תנאי המודל."),
        "Reason": st.column_config.TextColumn("סיבה", help="הסיבה המרכזית למה המניה לא נכנסה או למה צריך למכור/להחזיק."),
        "Quantity": st.column_config.NumberColumn("כמות", help="כמות המניות שאתה מחזיק בפועל בתיק הווירטואלי.", format="%.4f"),
        "Avg Entry": st.column_config.NumberColumn("שער כניסה ממוצע", help="שער הקנייה הממוצע שלך בפוזיציה.", format="%.2f"),
        "Market Value": st.column_config.NumberColumn("שווי נוכחי", help="כמות × מחיר נוכחי.", format="$%.2f"),
        "Cost": st.column_config.NumberColumn("עלות", help="כמות × שער כניסה ממוצע.", format="$%.2f"),
        "PnL $": st.column_config.NumberColumn("רווח/הפסד $", help="שווי נוכחי פחות עלות.", format="$%.2f"),
        "PnL %": st.column_config.NumberColumn("רווח/הפסד %", help="אחוז הרווח/הפסד מהכניסה.", format="%.2f%%"),
        "Distance to Trail %": st.column_config.NumberColumn("מרחק מהסטופ %", help="כמה אחוזים המחיר הנוכחי מעל ה-Trailing Stop. נמוך = קרוב ליציאה.", format="%.2f%%"),
        "Initial Stop": st.column_config.NumberColumn("סטופ התחלתי", help="הסטופ המקורי מהיום שנכנסת לעסקה.", format="%.2f"),
        "Last Date": st.column_config.TextColumn("תאריך נתון אחרון", help="תאריך יום המסחר האחרון שהנתונים מתייחסים אליו."),
        "Action": st.column_config.TextColumn("פעולה", help="HOLD = להחזיק. SELL = יציאה לפי המודל."),
        "State": st.column_config.TextColumn("מצב", help="BUY SETUP READY / WAIT FOR BREAKOUT / WAIT FOR PULLBACK / TURNAROUND WATCH / IGNORE."),
        "Buy Trigger": st.column_config.NumberColumn("טריגר קנייה", help="השער שמעליו המודל מציע לשקול כניסה. בדרך כלל פריצה מעל שיא 20/50 יום.", format="%.2f"),
        "Distance to Trigger %": st.column_config.NumberColumn("מרחק לטריגר %", help="כמה המחיר הנוכחי רחוק מטריגר הקנייה.", format="%.2f%%"),
        "Next Action Price": st.column_config.NumberColumn("מחיר פעולה הבא", help="המחיר הבא שרלוונטי לפעולה: טריגר פריצה או מחיר פולבק.", format="%.2f"),
        "Distance to Action %": st.column_config.NumberColumn("מרחק לפעולה %", help="כמה המחיר הנוכחי רחוק ממחיר הפעולה הבא.", format="%.2f%%"),
        "Target 8%": st.column_config.NumberColumn("יעד 8%", help="יעד רווח ראשון לבדיקה — 8% מעל טריגר הקנייה.", format="%.2f"),
        "Target 12%": st.column_config.NumberColumn("יעד 12%", help="יעד רווח שני לבדיקה — 12% מעל טריגר הקנייה.", format="%.2f"),
        "RR 8%": st.column_config.NumberColumn("R/R ל-8%", help="יחס סיכון/סיכוי עד יעד 8%. מעל 1 עדיף.", format="%.2f"),
        "RR 12%": st.column_config.NumberColumn("R/R ל-12%", help="יחס סיכון/סיכוי עד יעד 12%. מעל 1.5 עדיף.", format="%.2f"),
        "Breakout Score": st.column_config.NumberColumn("ציון פריצה", help="ציון פרקטי שמעדיף טריגר קרוב, סיכון סביר, ATR מתאים, בסיס מתכווץ ושיפור כוח יחסי.", format="%.2f"),
        "Action Quality": st.column_config.TextColumn("איכות פעולה", help="READY = פקודה אפשרית; NEAR READY = קרוב אבל חסר דלק/איכות; WATCH = מעקב בלבד."),
        "Quality Notes": st.column_config.TextColumn("הערות איכות", help="מה מונע מהמניה להיות פקודת ביצוע איכותית: RS, ATR, סיכון, R/R, FOMO וכו׳."),
        "Exhaustion Risk": st.column_config.TextColumn("סיכון מימוש/FOMO", help="LOW/MEDIUM/HIGH/MISSED. מזהה מצב שבו המניה חמה מדי או שהטריגר כבר ברח."),
        "What We Need": st.column_config.TextColumn("מה חסר", help="הדבר הבא שצריך לקרות כדי שהמניה תהפוך לפעולה."),
        "Why": st.column_config.TextColumn("למה", help="הסבר קצר למה המניה במצב הנוכחי."),
        "10D Range %": st.column_config.NumberColumn("טווח 10 ימים", help="טווח תנודת המחיר ב-10 ימי המסחר האחרונים. נמוך יותר יכול להעיד על התבססות.", format="%.1f%%"),
        "ATR Pinch": st.column_config.NumberColumn("ATR Pinch", help="ATR5 חלקי ATR20. נמוך מ-0.75 מצביע על התכווצות תנודתיות; נמוך מ-0.65 הוא קפיץ חזק.", format="%.2f"),
        "VAR 15d": st.column_config.NumberColumn("VAR 15d", help="יחס ווליום בימי עלייה מול ימי ירידה ב-15 ימים. מעל 1.25/1.5 מרמז על איסוף יחסי.", format="%.2f"),
        "Inside Day": st.column_config.CheckboxColumn("Inside Day", help="הנר היומי הסגור האחרון היה בתוך הטווח של היום שלפניו."),
        "Tight Day": st.column_config.CheckboxColumn("Tight Day", help="הנר היומי הסגור האחרון היה צר ביחס לתנודתיות הרגילה."),
        "SMA200": st.column_config.NumberColumn("SMA200", help="ממוצע נע פשוט ל-200 ימי מסחר. משמש כסינון מגמה ראשי.", format="%.2f"),
        "Account": st.column_config.TextColumn("סוג תיק", help="אמת או וירטואלי. מאפשר לסכם בנפרד השקעות אמיתיות וסימולציות."),
        "Open PnL $": st.column_config.NumberColumn("רווח פתוח $", help="רווח/הפסד על פוזיציות שעדיין פתוחות.", format="$%.2f"),
        "Open PnL %": st.column_config.NumberColumn("רווח פתוח %", help="אחוז רווח/הפסד על הפוזיציה הפתוחה.", format="%.2f%%"),
        "Realized PnL": st.column_config.NumberColumn("רווח ממומש", help="רווח/הפסד ממכירות שכבר בוצעו לפי יומן הפעולות.", format="$%.2f"),
    }



def empty_ledger():
    return pd.DataFrame(columns=[
        "Date", "Account", "Action", "Ticker", "Quantity", "Price", "Initial Stop", "Note"
    ])


def normalize_ledger(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Date", "Account", "Action", "Ticker", "Quantity", "Price", "Initial Stop", "Note"]
    if df is None or df.empty:
        return empty_ledger()

    out = df.copy()
    for col in required:
        if col not in out.columns:
            out[col] = "" if col in ["Date", "Account", "Action", "Ticker", "Note"] else 0.0

    out = out[required]
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out["Account"] = out["Account"].astype(str).replace({"real": "אמת", "virtual": "וירטואלי", "REAL": "אמת", "VIRTUAL": "וירטואלי"})
    out["Action"] = out["Action"].astype(str).replace({"buy": "BUY", "sell": "SELL", "קניה": "BUY", "מכירה": "SELL"})
    out["Action"] = out["Action"].str.upper().str.strip()
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce").fillna(0.0)
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce").fillna(0.0)
    out["Initial Stop"] = pd.to_numeric(out["Initial Stop"], errors="coerce").fillna(0.0)
    return out


def ledger_to_holdings(ledger: pd.DataFrame):
    ledger = normalize_ledger(ledger)
    errors = []
    state = {}

    for idx, tx in ledger.iterrows():
        account = str(tx["Account"]).strip() or "אמת"
        action = str(tx["Action"]).strip().upper()
        ticker = str(tx["Ticker"]).strip().upper()
        qty = safe_float(tx["Quantity"], 0)
        price = safe_float(tx["Price"], 0)
        stop = safe_float(tx["Initial Stop"], 0)
        date = tx["Date"]

        if not ticker or qty <= 0 or price <= 0 or action not in ["BUY", "SELL"]:
            continue

        key = (account, ticker)
        if key not in state:
            state[key] = {
                "Account": account,
                "Ticker": ticker,
                "Quantity": 0.0,
                "Cost Basis": 0.0,
                "Avg Entry": 0.0,
                "Initial Stop Total": 0.0,
                "First Entry Date": date,
                "Realized PnL": 0.0,
            }

        pos = state[key]

        if action == "BUY":
            if pos["Quantity"] <= 0:
                pos["First Entry Date"] = date

            pos["Cost Basis"] += qty * price
            if stop > 0:
                pos["Initial Stop Total"] += qty * stop
            pos["Quantity"] += qty
            pos["Avg Entry"] = pos["Cost Basis"] / pos["Quantity"] if pos["Quantity"] > 0 else 0

        elif action == "SELL":
            if qty > pos["Quantity"] + 1e-9:
                errors.append({
                    "Row": idx + 1,
                    "Ticker": ticker,
                    "Account": account,
                    "Error": f"מכירה של {qty} גדולה מהכמות הקיימת {pos['Quantity']:.4f}",
                })
                continue

            avg_entry = pos["Avg Entry"] if pos["Avg Entry"] > 0 else price
            realized = qty * (price - avg_entry)
            pos["Realized PnL"] += realized
            pos["Cost Basis"] -= qty * avg_entry

            if pos["Quantity"] > 0 and pos["Initial Stop Total"] > 0:
                pos["Initial Stop Total"] *= max(0.0, (pos["Quantity"] - qty) / pos["Quantity"])

            pos["Quantity"] -= qty
            if pos["Quantity"] <= 1e-9:
                pos["Quantity"] = 0.0
                pos["Cost Basis"] = 0.0
                pos["Avg Entry"] = 0.0
                pos["Initial Stop Total"] = 0.0
            else:
                pos["Avg Entry"] = pos["Cost Basis"] / pos["Quantity"]

    rows = []
    for pos in state.values():
        if pos["Quantity"] > 0:
            avg_stop = (pos["Initial Stop Total"] / pos["Quantity"]) if pos["Initial Stop Total"] > 0 else 0.0
            rows.append({
                "Account": pos["Account"],
                "Ticker": pos["Ticker"],
                "Quantity": pos["Quantity"],
                "Avg Entry": pos["Avg Entry"],
                "Entry Date": pos["First Entry Date"],
                "Initial Stop": avg_stop,
                "Cost": pos["Cost Basis"],
                "Realized PnL": pos["Realized PnL"],
            })

    return pd.DataFrame(rows), pd.DataFrame(errors)


def analyze_unified_portfolio(ledger: pd.DataFrame):
    holdings, errors = ledger_to_holdings(ledger)
    if holdings.empty:
        return holdings, errors

    analyzed = []
    for _, row in holdings.iterrows():
        status = analyze_open_position(
            row["Ticker"],
            safe_float(row["Avg Entry"]),
            row["Entry Date"],
            safe_float(row["Initial Stop"])
        )

        current = safe_float(status.get("Current", np.nan))
        qty = safe_float(row["Quantity"])
        avg_entry = safe_float(row["Avg Entry"])
        cost = safe_float(row["Cost"])
        market_value = qty * current if np.isfinite(current) else np.nan
        open_pnl = market_value - cost if np.isfinite(market_value) else np.nan
        open_pnl_pct = (current / avg_entry - 1) * 100 if np.isfinite(current) and avg_entry else np.nan

        analyzed.append({
            "Account": row["Account"],
            "Ticker": row["Ticker"],
            "Quantity": round(qty, 4),
            "Avg Entry": round(avg_entry, 2),
            "Current": round(current, 2) if np.isfinite(current) else np.nan,
            "Cost": round(cost, 2),
            "Market Value": round(market_value, 2) if np.isfinite(market_value) else np.nan,
            "Open PnL $": round(open_pnl, 2) if np.isfinite(open_pnl) else np.nan,
            "Open PnL %": round(open_pnl_pct, 2) if np.isfinite(open_pnl_pct) else np.nan,
            "Realized PnL": round(safe_float(row["Realized PnL"]), 2),
            "Action": status.get("Action", ""),
            "Reason": status.get("Reason", ""),
            "Trailing Stop": status.get("Trailing Stop", np.nan),
            "Current Protection Stop": status.get("Current Protection Stop", np.nan),
            "Profit Checkpoint": status.get("Profit Checkpoint", np.nan),
            "Distance to Trail %": status.get("Distance to Trail %", np.nan),
            "EMA21": status.get("EMA21", np.nan),
            "Exit Close Level": status.get("Exit Close Level", np.nan),
            "Initial Stop": status.get("Initial Stop", row["Initial Stop"]),
            "Entry Date": row["Entry Date"],
            "Last Date": status.get("Last Date", ""),
        })

    return pd.DataFrame(analyzed), errors


def summarize_portfolio(df: pd.DataFrame, include_virtual: bool):
    if df.empty:
        return {"cost": 0, "value": 0, "open_pnl": 0, "realized": 0, "total_pnl": 0, "pnl_pct": 0, "count": 0}

    scope = df if include_virtual else df[df["Account"] == "אמת"]
    if scope.empty:
        return {"cost": 0, "value": 0, "open_pnl": 0, "realized": 0, "total_pnl": 0, "pnl_pct": 0, "count": 0}

    cost = float(scope["Cost"].sum())
    value = float(scope["Market Value"].sum())
    open_pnl = float(scope["Open PnL $"].sum())
    realized = float(scope["Realized PnL"].sum()) if "Realized PnL" in scope else 0
    total_pnl = open_pnl + realized
    pnl_pct = (total_pnl / cost * 100) if cost else 0

    return {
        "cost": cost,
        "value": value,
        "open_pnl": open_pnl,
        "realized": realized,
        "total_pnl": total_pnl,
        "pnl_pct": pnl_pct,
        "count": len(scope),
    }



def unique_existing_columns(columns, df):
    """Return existing DataFrame columns without duplicates, preserving order."""
    seen = set()
    result = []
    for col in columns:
        if col in df.columns and col not in seen:
            result.append(col)
            seen.add(col)
    return result


def dedupe_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive fix for Streamlit/PyArrow duplicate-column errors."""
    if df is None or df.empty:
        return df
    return df.loc[:, ~pd.Index(df.columns).duplicated()].copy()



def safe_profile_name(profile: str) -> str:
    profile = str(profile).strip().lower()
    allowed = "".join(ch for ch in profile if ch.isalnum() or ch in ["_", "-"])
    return allowed or "user1"


def ledger_file_path(profile: str) -> Path:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    return LEDGER_DIR / f"{safe_profile_name(profile)}_ledger.csv"


def load_profile_ledger(profile: str) -> pd.DataFrame:
    path = ledger_file_path(profile)
    if path.exists():
        try:
            return normalize_ledger(pd.read_csv(path))
        except Exception:
            return empty_ledger()
    return empty_ledger()


def save_profile_ledger(profile: str, ledger: pd.DataFrame):
    path = ledger_file_path(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_ledger(ledger).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def reset_profile_ledger(profile: str):
    path = ledger_file_path(profile)
    if path.exists():
        path.unlink()
    return empty_ledger()



# ==========================================================
# 9. UI
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
    st.markdown("<h1 style='text-align: center;'>🎯 SwingHunter V12.3 — Coiled Breakout Cheat Entry</h1>", unsafe_allow_html=True)
    st.info(
        "V12.1 מוסיפה Coiled Breakout Cheat Entry: רק סטאפ עם דלק אמיתי ל-8%-12% נשאר BUY SETUP READY; פריצות כבדות/חלשות עוברות ל-NEAR READY או Watch. "
        "המערכת מסכמת רווח/הפסד לתיק אמת בלבד וגם לאמת+וירטואלי, וממשיכה לתת HOLD/SELL לפי EMA21 ו-Trailing Stop."
    )

    st.sidebar.header("הגדרות קצרות")

    if "refresh_counter" not in st.session_state:
        st.session_state["refresh_counter"] = 0
    if "last_refresh_time" not in st.session_state:
        st.session_state["last_refresh_time"] = "טרם בוצע רענון ידני"

    if st.sidebar.button("🔄 רענן נתונים עכשיו", use_container_width=True):
        st.session_state["refresh_counter"] += 1
        st.session_state["last_refresh_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.cache_data.clear()
        st.rerun()

    st.sidebar.caption(f"רענון אחרון: {st.session_state['last_refresh_time']}")
    months = st.sidebar.slider("תקופת בדיקה היסטורית (חודשים)", 3, 24, 12)
    starting_bank = st.sidebar.number_input("בנק התחלתי ($)", value=50000, step=5000)
    max_risk_pct = st.sidebar.slider("סיכון מקסימלי לעסקה (%)", 4.0, 15.0, 9.5, 0.5)
    position_pct = st.sidebar.slider("אחוז מהבנק לכל עסקה", 5.0, 25.0, 10.0, 1.0) / 100

    params = StrategyParams(max_risk_pct=max_risk_pct, position_pct=position_pct)

    tab_daily, tab_portfolio, tab_backtest = st.tabs(["🚀 מה עושים היום", "💼 תיק השקעות", "🔬 בדיקת בנק"])

    with tab_daily:
        st.markdown("## 🚀 V12 — תוכנית פעולה לפריצות קרובות")
        st.caption(
            "המטרה כאן היא לא להציג 80 מניות עם חורים, אלא תוכנית פעולה: "
            "מה מוכן לקנייה, מה מחכה לפריצה, מה מחכה לפולבק ומה לא מעניין כרגע."
        )

        if st.button("⚡ הפק תוכנית פעולה להיום", use_container_width=True):
            with st.spinner("סורק מניות ומחשב תוכניות פעולה פרקטיות..."):
                df_action, df_watch, df_ignore = get_today_breakout_action_plan(params)

                total = len(df_action) + len(df_watch) + len(df_ignore)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("פקודות מוכנות", len(df_action))
                k2.metric("Near Ready / מעקב", len(df_watch))
                k3.metric("מוסתר / Ignore", len(df_ignore))
                k4.metric("סה״כ נבדקו", total)

                st.markdown("## ✅ פקודות לביצוע / כמעט לביצוע")
                action_cols = [
                    "Ticker", "State", "Setup Type", "Current", "Buy Trigger", "Distance to Trigger %",
                    "Stop", "Risk %", "Target 8%", "Target 12%", "RR 8%", "RR 12%",
                    "Breakout Score", "Action Quality", "Exhaustion Risk", "Quality Notes", "Why", "EMA21", "SMA200", "RS5", "RS20", "RSI", "ATR%", "ATR Pinch", "VAR 15d", "Inside Day", "Tight Day", "20D Run", "10D Range %"
                ]

                if not df_action.empty:
                    df_action = dedupe_dataframe_columns(df_action)
                    cols = unique_existing_columns(action_cols, df_action)
                    st.dataframe(df_action[cols], use_container_width=True, hide_index=True, column_config=get_column_config())
                else:
                    st.warning("אין כרגע פקודת קנייה מספיק נקייה. זה בסדר — לא חייבים לסחור כל יום.")

                st.markdown("## 👀 Near Ready / מועמדות למעקב עם מחיר פעולה הבא")
                watch_cols = [
                    "Ticker", "State", "Setup Type", "Current", "Next Action Price", "Distance to Action %",
                    "What We Need", "Why", "Breakout Score", "Action Quality", "Exhaustion Risk", "Quality Notes", "Buy Trigger", "Pullback Watch Price",
                    "Pullback Deep Price", "Stop", "Risk %", "Target 8%", "Target 12%",
                    "EMA21", "SMA200", "RS5", "RS20", "RSI", "ATR%", "ATR Pinch", "VAR 15d", "Inside Day", "Tight Day", "20D Run", "Run Zone", "10D Range %"
                ]

                if not df_watch.empty:
                    df_watch = dedupe_dataframe_columns(df_watch)
                    cols = unique_existing_columns(watch_cols, df_watch)
                    st.dataframe(df_watch[cols], use_container_width=True, hide_index=True, column_config=get_column_config())
                else:
                    st.info("אין מועמדות מעקב כרגע.")

                with st.expander("🧹 Ignore / מניות לא רלוונטיות כרגע"):
                    if not df_ignore.empty:
                        ignore_cols = [
                            "Ticker", "State", "Setup Type", "Current", "Why", "What We Need",
                            "EMA21", "SMA200", "RS5", "RS20", "RSI", "ATR%", "ATR Pinch", "VAR 15d", "Inside Day", "Tight Day", "20D Run", "Run Zone"
                        ]
                        df_ignore = dedupe_dataframe_columns(df_ignore)
                        cols = unique_existing_columns(ignore_cols, df_ignore)
                        st.dataframe(df_ignore[cols], use_container_width=True, hide_index=True, column_config=get_column_config())
                    else:
                        st.write("אין מניות ב-Ignore.")

                combined_radar = pd.concat([df_watch, df_ignore], ignore_index=True) if not df_watch.empty or not df_ignore.empty else pd.DataFrame()

                zip_bytes = build_zip_report(
                    pd.DataFrame(),
                    pd.DataFrame(),
                    pd.DataFrame(),
                    {"App Version": APP_VERSION},
                    df_action,
                    combined_radar
                )
                st.download_button(
                    "⬇️ הורד ZIP יומי עם תוכנית פעולה",
                    zip_bytes,
                    file_name=f"swinghunter_{APP_VERSION}_daily_action_plan.zip",
                    mime="application/zip",
                    use_container_width=True
                )

    with tab_portfolio:
        st.markdown("## 💼 תיק השקעות — אמת + וירטואלי")
        st.caption(
            "זה המקום היחיד לניהול פוזיציות. קנית/מכרת בפועל או וירטואלית — מזינים פעולה. "
            "המערכת מחשבת החזקות, שווי נוכחי, רווח/הפסד, ו-HOLD/SELL לפי EMA21/Trailing Stop."
        )

        st.info(
            "חדש ב-V11.5: התיק נשמר אוטומטית לפי פרופיל. אפשר לעבוד עם user1 / user2 / user3, "
            "ולא צריך להעלות CSV בכל פתיחה. כפתור איפוס מוחק רק את הפרופיל הפעיל."
        )

        profile = st.selectbox("בחר תיק / משתמש", ["user1", "user2", "user3"], key="ledger_profile")

        if "active_ledger_profile" not in st.session_state or st.session_state["active_ledger_profile"] != profile:
            st.session_state["active_ledger_profile"] = profile
            st.session_state["ledger"] = load_profile_ledger(profile)

        if "ledger" not in st.session_state:
            st.session_state["ledger"] = load_profile_ledger(profile)

        st.caption(f"שמירה מקומית פעילה לפרופיל: {profile}")

        pcol1, pcol2, pcol3 = st.columns(3)

        with pcol1:
            if st.button("💾 שמור עכשיו", use_container_width=True):
                save_profile_ledger(profile, st.session_state["ledger"])
                st.success("נשמר.")

        with pcol2:
            uploaded_ledger = st.file_uploader("ייבוא CSV לפרופיל הפעיל", type=["csv"], key=f"ledger_upload_{profile}")
            if uploaded_ledger is not None:
                try:
                    st.session_state["ledger"] = normalize_ledger(pd.read_csv(uploaded_ledger))
                    save_profile_ledger(profile, st.session_state["ledger"])
                    st.success("הקובץ נטען ונשמר לפרופיל הפעיל.")
                except Exception:
                    st.error("לא הצלחתי לקרוא את הקובץ. ודא שזה CSV.")

        with pcol3:
            confirm_reset = st.checkbox("אשר איפוס", key=f"confirm_reset_{profile}")
            if st.button("🗑️ איפוס התיק הפעיל", use_container_width=True):
                if confirm_reset:
                    st.session_state["ledger"] = reset_profile_ledger(profile)
                    st.success("התיק אופס.")
                    st.rerun()
                else:
                    st.warning("סמן קודם 'אשר איפוס'.")

        st.markdown("### הוספת פעולה")
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        with c1:
            account = st.selectbox("תיק", ["אמת", "וירטואלי"], key="tx_account")
        with c2:
            action = st.selectbox("פעולה", ["BUY", "SELL"], key="tx_action")
        with c3:
            ticker = st.text_input("Ticker", value="", key="tx_ticker").upper().strip()
        with c4:
            qty = st.number_input("כמות", min_value=0.0, step=0.01, key="tx_qty")
        with c5:
            price = st.number_input("מחיר", min_value=0.0, step=0.01, key="tx_price")
        with c6:
            initial_stop = st.number_input("Initial Stop", min_value=0.0, step=0.01, key="tx_stop")

        note = st.text_input("הערה", value="", key="tx_note")

        if st.button("➕ הוסף פעולה ליומן", use_container_width=True):
            if not ticker or qty <= 0 or price <= 0:
                st.error("צריך Ticker, כמות ומחיר תקינים.")
            else:
                new_row = pd.DataFrame([{
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                    "Account": account,
                    "Action": action,
                    "Ticker": ticker,
                    "Quantity": qty,
                    "Price": price,
                    "Initial Stop": initial_stop,
                    "Note": note,
                }])
                st.session_state["ledger"] = normalize_ledger(pd.concat([st.session_state["ledger"], new_row], ignore_index=True))
                save_profile_ledger(profile, st.session_state["ledger"])
                st.success("הפעולה נוספה ונשמרה.")

        st.markdown("### יומן פעולות")
        ledger_edit = st.data_editor(
            st.session_state["ledger"],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key=f"ledger_editor_{profile}",
            column_config={
                "Date": st.column_config.TextColumn("Date"),
                "Account": st.column_config.SelectboxColumn("Account", options=["אמת", "וירטואלי"]),
                "Action": st.column_config.SelectboxColumn("Action", options=["BUY", "SELL"]),
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, step=0.01),
                "Price": st.column_config.NumberColumn("Price", min_value=0.0, step=0.01),
                "Initial Stop": st.column_config.NumberColumn("Initial Stop", min_value=0.0, step=0.01),
                "Note": st.column_config.TextColumn("Note"),
            }
        )

        edited_ledger = normalize_ledger(ledger_edit)
        if not edited_ledger.equals(st.session_state["ledger"]):
            st.session_state["ledger"] = edited_ledger
            save_profile_ledger(profile, st.session_state["ledger"])
            st.caption("השינויים בטבלה נשמרו אוטומטית.")

        st.download_button(
            "⬇️ הורד גיבוי CSV",
            st.session_state["ledger"].to_csv(index=False).encode("utf-8-sig"),
            file_name=f"swinghunter_{profile}_portfolio_ledger.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("🔎 נתח תיק השקעות עכשיו", use_container_width=True):
            df_portfolio, df_errors = analyze_unified_portfolio(st.session_state["ledger"])

            if not df_errors.empty:
                st.error("יש שגיאות ביומן הפעולות:")
                st.dataframe(df_errors, use_container_width=True, hide_index=True)

            if df_portfolio.empty:
                st.warning("אין החזקות פתוחות לניתוח.")
            else:
                summary_real = summarize_portfolio(df_portfolio, include_virtual=False)
                summary_all = summarize_portfolio(df_portfolio, include_virtual=True)

                st.markdown("### סיכום תיק אמת בלבד")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("עלות", f"${summary_real['cost']:,.0f}")
                r2.metric("שווי נוכחי", f"${summary_real['value']:,.0f}")
                r3.metric("רווח/הפסד כולל", f"${summary_real['total_pnl']:,.0f}", f"{summary_real['pnl_pct']:.2f}%")
                r4.metric("פוזיציות", summary_real["count"])

                st.markdown("### סיכום כולל אמת + וירטואלי")
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("עלות", f"${summary_all['cost']:,.0f}")
                a2.metric("שווי נוכחי", f"${summary_all['value']:,.0f}")
                a3.metric("רווח/הפסד כולל", f"${summary_all['total_pnl']:,.0f}", f"{summary_all['pnl_pct']:.2f}%")
                a4.metric("פוזיציות", summary_all["count"])

                cols = [
                    "Account", "Ticker", "Action", "Reason", "Quantity", "Avg Entry", "Current",
                    "Market Value", "Open PnL $", "Open PnL %", "Realized PnL",
                    "Trailing Stop", "Current Protection Stop", "Profit Checkpoint", "Distance to Trail %", "EMA21", "Exit Close Level",
                    "Initial Stop", "Entry Date", "Last Date"
                ]
                df_portfolio = dedupe_dataframe_columns(df_portfolio)
                cols = unique_existing_columns(cols, df_portfolio)
                st.dataframe(df_portfolio[cols], use_container_width=True, hide_index=True, column_config=get_column_config())

                st.download_button(
                    "⬇️ הורד ניתוח תיק CSV",
                    df_portfolio.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"swinghunter_{APP_VERSION}_{profile}_portfolio_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        st.markdown("#### איך מבצעים מכירה?")
        st.write(
            "מוסיף פעולה חדשה מסוג SELL עם אותה מניה וכמות למכירה. הכמות חייבת להיות קטנה או שווה לכמות הפתוחה. "
            "אם מכרת הכול — הכמות תתאפס ולא תופיע בהחזקות. אם מכרת חלק — הכמות שנותרה תמשיך להיות מנוהלת."
        )

    with tab_backtest:
        st.markdown(f"### 🧪 Banked Backtest — {months} חודשים — בנק ${starting_bank:,.0f} — 10% מהבנק לכל כניסה")

        if st.button("⚙️ הרץ בדיקת בנק", type="primary"):
            with st.spinner("מריץ Backtest עם בנק, כניסה חוזרת אחרי מכירה מלאה, ויציאה מלאה בכל יעד/סטופ..."):
                data = fetch_backtest_data(months)
                df_equity, df_trades, ticker_summary, metrics = run_banked_backtest(
                    data,
                    months,
                    params,
                    starting_bank=float(starting_bank)
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("בנק סופי", f"${metrics['Ending Bank']:,.0f}", f"{metrics['Bank ROI %']:.2f}%")
                c2.metric("Total PnL", f"${metrics['Total PnL']:,.0f}")
                c3.metric("Profit Factor", metrics["Profit Factor"])
                c4.metric("Win Rate", f"{metrics['Win Rate %']:.1f}%")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Trades", metrics["Trades"])
                c6.metric("Avg Win / Loss", f"{metrics['Average Win %']:.2f}% / {metrics['Average Loss %']:.2f}%")
                c7.metric("Max Drawdown", f"{metrics['Max Drawdown %']:.2f}%")
                c8.metric("QQQ Buy&Hold", f"{metrics['QQQ Buy & Hold %']}%", f"${metrics['QQQ Buy & Hold PnL']:,.0f}")

                st.caption(
                    f"מודל בנק: מתחילים עם ${metrics['Starting Bank']:,.0f}. "
                    f"כל עסקה משתמשת ב-{metrics['Position % of Bank']:.0f}% מהבנק בזמן הכניסה. היציאה היא מלאה בשבירת EMA21/Trailing Stop. "
                    f"Turnover: ${metrics['Turnover']:,.0f}. חשיפה ממוצעת: {metrics['Average Exposure %']:.1f}%."
                )

                st.markdown("#### 📈 Bank Equity Curve")
                if not df_equity.empty:
                    st.line_chart(df_equity[["Total Equity"]])

                zip_bytes = build_zip_report(df_equity, df_trades, ticker_summary, metrics)
                st.download_button(
                    "⬇️ הורד ZIP אחד עם כל הבדיקה",
                    zip_bytes,
                    file_name=f"swinghunter_{APP_VERSION}_banked_report.zip",
                    mime="application/zip",
                    use_container_width=True
                )

                if not df_trades.empty:
                    with st.expander("📌 כל העסקאות"):
                        st.dataframe(df_trades, use_container_width=True, hide_index=True, column_config=get_column_config())

                if not ticker_summary.empty:
                    with st.expander("🏷️ PnL לפי טיקר"):
                        st.dataframe(ticker_summary, use_container_width=True, hide_index=True, column_config=get_column_config())
