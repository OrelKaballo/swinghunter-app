import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SwingHunter V10.5 - Hebrew Tooltips Dashboard", layout="wide")
APP_VERSION = "V10.5"

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
    'BA','XOM','CVX','GE'
]

MOMENTUM_TICKERS = {
    'AMD','NVDA','TSLA','DDOG','NET','QCOM','CRWD','PANW','AVGO','AMZN',
    'MSTR','COIN','SMCI','PLTR','ARM','MU','MRVL','TSM','META','GOOGL',
    'HOOD','AFRM','SHOP','NFLX','SNOW','ZS','MDB','SOFI','SQ','PYPL',
    'MARA','RIOT','ASML'
}

NOTIONAL_PER_TRADE = 1000.0
DEFAULT_STARTING_BANK = 50000.0
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
    max_20d_run: float = 60.0
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


@st.cache_data(ttl=3600, show_spinner=False)
def download_single(ticker: str, period: str = "370d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        return flatten_download(df).dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
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
        "Score": 0.0,
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
        "RS5": np.nan,
        "RS20": np.nan,
        "RSI": np.nan,
        "ATR%": np.nan,
        "20D Run": np.nan,
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
        run20 = (last_p / c.iloc[-21] - 1) * 100
        rs5, rs20 = relative_strength_vs(qqq_slice, c)
        regime = market_regime(qqq_slice)

        base.update({
            "Current": round(last_p, 2),
            "Regime": regime,
            "EMA21": round(ema21, 2),
            "Exit Close Level": round(ema21 * 0.995, 2),
            "RS5": round(rs5, 1),
            "RS20": round(rs20, 1),
            "RSI": round(rsi, 1),
            "ATR%": round(atr_pct, 1) if np.isfinite(atr_pct) else np.nan,
            "20D Run": round(run20, 1)
        })

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

        # Hard rejects, but still with full radar info.
        if last_p < sma200:
            base.update(Status="REJECT", Decision="NO TRADE", Reason="מתחת SMA200")
            return base

        if not np.isfinite(atr_pct) or atr_pct < params.min_atr_pct:
            base.update(Status="REJECT", Decision="NO TRADE", Reason=f"ATR% נמוך מדי ({atr_pct:.1f}%)")
            return base

        # Regime filter: normal trades only in BULL_STRONG. Exceptional leaders can pass in weaker regimes.
        if regime != "BULL_STRONG" and not exceptional:
            base.update(Status="WATCH", Decision="WAIT", Reason=f"שוק לא מספיק חזק ({regime})")
            return base

        if (rsi > params.overbought_rsi or run20 > params.max_20d_run) and not exceptional:
            base.update(Status="WATCH", Decision="WAIT", Reason="מתוחה מדי כרגע")
            return base

        if not edge_ok and not exceptional:
            base.update(Status="WATCH", Decision="WAIT", Reason="RS חלש מול QQQ")
            return base

        order_type = None
        entry = 0.0
        setup = ""

        dist_to_res = (res20 / last_p - 1)

        # 1) Breakout near resistance.
        if 0 < dist_to_res < 0.045 and edge_ok:
            entry = round(res20 * 1.002, 2)
            if last_p < entry:
                order_type = "BUY STOP LIMIT"
                setup = "Near Resistance Breakout"

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

        if order_type is None and pullback_quality_ok and (last_p / ema8 - 1) < 0.035:
            entry = min(round(ema8 * 1.003, 2), round(last_p * 0.995, 2))
            if last_p > entry:
                order_type = "BUY LIMIT"
                setup = "RS Pullback"

        # 3) Momentum continuation is WATCH only unless exceptional.
        if order_type is None and exceptional and last_p > ema21 and (prev_hi / last_p - 1) < 0.025:
            entry = round(prev_hi * 1.002, 2)
            if last_p < entry:
                order_type = "BUY STOP LIMIT"
                setup = "Exceptional Momentum Continuation"

        if order_type is None:
            base.update(
                Status="WATCH",
                Decision="WAIT",
                Score=round(score_candidate(rs5, rs20, run20, 1.0, params.max_risk_pct, "WATCH", exceptional, momentum_name), 2),
                Reason="אין נקודת כניסה נקייה כרגע"
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
            "Score": round(score, 2),
            "Setup": setup,
            "Entry": round(entry, 2),
            "Distance to Entry %": round(distance, 2),
            "Stop": stop,
            "Target": "No fixed target",
            "Exit Rule": "Close below EMA21 trail",
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
    }


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
    st.markdown("<h1 style='text-align: center;'>🎯 SwingHunter V10.5 — Hebrew Tooltips Dashboard</h1>", unsafe_allow_html=True)
    st.info(
        "V10.5 מוסיפה הסברים בעברית ב-hover על העמודות, ומציגה EMA21 ורמת יציאה גם במסכי הפקודות/הרדאר. "
        "בתיק הווירטואלי מקבלים שווי עדכני והחלטת HOLD/SELL לפי EMA21 ו-Trailing Stop."
    )

    st.sidebar.header("הגדרות קצרות")
    months = st.sidebar.slider("תקופת בדיקה היסטורית (חודשים)", 3, 24, 12)
    starting_bank = st.sidebar.number_input("בנק התחלתי ($)", value=50000, step=5000)
    max_risk_pct = st.sidebar.slider("סיכון מקסימלי לעסקה (%)", 4.0, 15.0, 9.5, 0.5)
    position_pct = st.sidebar.slider("אחוז מהבנק לכל עסקה", 5.0, 25.0, 10.0, 1.0) / 100

    params = StrategyParams(max_risk_pct=max_risk_pct, position_pct=position_pct)

    tab_daily, tab_portfolio, tab_positions, tab_backtest = st.tabs(["🚀 מה עושים היום", "💼 תיק וירטואלי", "📌 ניהול פוזיציות", "🔬 בדיקת בנק"])

    with tab_daily:
        if st.button("⚡ הפק פקודות/מעקב להיום", use_container_width=True):
            with st.spinner("סורק מניות ומחשב תכניות כניסה/יציאה..."):
                df_orders, df_radar = get_today_actions(params)

                st.markdown("## 🧭 פקודות לביצוע היום")

                if not df_orders.empty:
                    cols = [
                        "Ticker","Action Now","Order","Current","Entry","Distance to Entry %",
                        "Stop","EMA21","Exit Close Level","Exit Rule","Risk %","Setup","Score",
                        "Regime","RS5","RS20","RSI","ATR%","20D Run"
                    ]
                    cols = [c for c in cols if c in df_orders.columns]
                    st.dataframe(df_orders[cols], use_container_width=True, hide_index=True, column_config=get_column_config())
                else:
                    st.warning("אין היום פקודות ביצוע. זה לא אומר שאין מידע — ראה רדאר למטה.")

                st.markdown("## 🔍 רדאר מלא — למה מניות לא נכנסו")
                if not df_radar.empty:
                    cols = [
                        "Ticker","Status","Decision","Score","Reason","Current","Entry","Distance to Entry %",
                        "Stop","Target","Risk %","Setup","Regime","RS5","RS20","RSI","ATR%","20D Run"
                    ]
                    cols = [c for c in cols if c in df_radar.columns]
                    st.dataframe(df_radar[cols].head(80), use_container_width=True, hide_index=True, column_config=get_column_config())

                zip_bytes = build_zip_report(
                    pd.DataFrame(),
                    pd.DataFrame(),
                    pd.DataFrame(),
                    {"App Version": APP_VERSION},
                    df_orders,
                    df_radar
                )
                st.download_button(
                    "⬇️ הורד ZIP יומי עם פקודות ורדאר",
                    zip_bytes,
                    file_name=f"swinghunter_{APP_VERSION}_daily.zip",
                    mime="application/zip",
                    use_container_width=True
                )



    with tab_portfolio:
        st.markdown("## 💼 תיק השקעות וירטואלי")
        st.caption(
            "זה לא מחובר לברוקר. זה יומן מעקב שאתה שומר כ-CSV: מזין פוזיציות, מקבל שווי עדכני והחלטת HOLD/SELL. "
            "אחרי קנייה/מכירה בפועל — מעדכן את הטבלה ושומר CSV חדש."
        )

        uploaded_positions = st.file_uploader("העלה קובץ פוזיציות CSV קודם, אם יש", type=["csv"])

        if uploaded_positions is not None:
            try:
                portfolio_template = pd.read_csv(uploaded_positions)
            except Exception:
                st.error("לא הצלחתי לקרוא את הקובץ. ודא שזה CSV.")
                portfolio_template = build_virtual_portfolio_template()
        else:
            portfolio_template = build_virtual_portfolio_template()

        positions_portfolio = st.data_editor(
            portfolio_template,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, step=0.01),
                "Avg Entry": st.column_config.NumberColumn("Avg Entry", min_value=0.0, step=0.01),
                "Entry Date": st.column_config.TextColumn("Entry Date YYYY-MM-DD"),
                "Initial Stop": st.column_config.NumberColumn("Initial Stop", min_value=0.0, step=0.01),
            }
        )

        st.download_button(
            "⬇️ שמור קובץ פוזיציות לעריכה עתידית",
            positions_portfolio.to_csv(index=False).encode("utf-8-sig"),
            file_name="swinghunter_virtual_portfolio_positions.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("🔎 נתח תיק וירטואלי עכשיו", use_container_width=True):
            df_portfolio = analyze_virtual_portfolio(positions_portfolio)

            if df_portfolio.empty:
                st.warning("אין פוזיציות תקינות לניתוח.")
            else:
                total_cost = df_portfolio["Cost"].sum()
                total_value = df_portfolio["Market Value"].sum()
                total_pnl = df_portfolio["PnL $"].sum()
                total_pnl_pct = (total_value / total_cost - 1) * 100 if total_cost else 0

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("עלות", f"${total_cost:,.0f}")
                k2.metric("שווי נוכחי", f"${total_value:,.0f}")
                k3.metric("רווח/הפסד", f"${total_pnl:,.0f}", f"{total_pnl_pct:.2f}%")
                k4.metric("פוזיציות", len(df_portfolio))

                cols = [
                    "Ticker", "Action", "Reason", "Quantity", "Avg Entry", "Current",
                    "Market Value", "PnL $", "PnL %", "Trailing Stop", "Distance to Trail %",
                    "EMA21", "Exit Close Level", "Initial Stop", "Entry Date", "Last Date"
                ]
                cols = [c for c in cols if c in df_portfolio.columns]
                st.dataframe(df_portfolio[cols], use_container_width=True, hide_index=True, column_config=get_column_config())

                st.download_button(
                    "⬇️ הורד ניתוח תיק CSV",
                    df_portfolio.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"swinghunter_{APP_VERSION}_virtual_portfolio_analysis.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        st.markdown("#### איך מעדכנים פעולה?")
        st.write(
            "קנית? הוסף שורה עם הטיקר, כמות, שער כניסה ותאריך. "
            "מכרת חלק? עדכן את Quantity לכמות שנשארה. מכרת הכול? מחק את השורה או שנה Quantity ל-0. "
            "אחרי כל שינוי שמור CSV חדש כדי להמשיך ממנו בפעם הבאה."
        )

    with tab_positions:
        st.markdown("## 📌 ניהול פוזיציות פתוחות")
        st.caption(
            "כאן מזינים את המניות שכבר קנית בפועל. המערכת מחשבת האם להחזיק או למכור לפי אותה לוגיקת יציאה של V10.2: "
            "Trailing Stop שעולה עם EMA21, ויציאה מלאה כשהמגמה נשברת."
        )

        st.markdown("#### הזן פוזיציות פתוחות")
        st.caption("עמודות חובה: Ticker, Entry, Entry Date. מומלץ להזין גם Initial Stop מהפקודה המקורית.")

        template = build_positions_template()
        positions_input = st.data_editor(
            template,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Entry": st.column_config.NumberColumn("Entry", min_value=0.0, step=0.01),
                "Entry Date": st.column_config.TextColumn("Entry Date YYYY-MM-DD"),
                "Initial Stop": st.column_config.NumberColumn("Initial Stop", min_value=0.0, step=0.01),
            }
        )

        if st.button("🔎 בדוק מה לעשות עם הפוזיציות", use_container_width=True):
            rows = []
            for _, row in positions_input.iterrows():
                ticker = str(row.get("Ticker", "")).strip().upper()
                entry = safe_float(row.get("Entry", np.nan))
                entry_date = row.get("Entry Date", "")
                initial_stop = safe_float(row.get("Initial Stop", np.nan))

                if not ticker or not np.isfinite(entry) or entry <= 0:
                    continue

                rows.append(analyze_open_position(ticker, entry, entry_date, initial_stop))

            if rows:
                df_pos = pd.DataFrame(rows)
                cols = [
                    "Ticker", "Action", "Status", "Reason", "Current", "Entry", "PnL %",
                    "Trailing Stop", "Distance to Trail %", "EMA21", "Exit Close Level",
                    "Initial Stop", "Last Date"
                ]
                cols = [c for c in cols if c in df_pos.columns]
                st.dataframe(df_pos[cols], use_container_width=True, hide_index=True, column_config=get_column_config())

                st.download_button(
                    "⬇️ הורד CSV פוזיציות",
                    df_pos.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"swinghunter_{APP_VERSION}_open_positions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("לא הוזנו פוזיציות תקינות.")

        st.markdown("#### כלל היציאה בפועל")
        st.write(
            "במודל V10.2/V10.3 אין יעד רווח קשיח. בפועל בודקים את הפוזיציה בסוף יום מסחר: "
            "אם הסגירה מתחת EMA21×0.995 — מוכרים. בנוסף, אם המחיר נוגע ב-Trailing Stop — זו יציאה."
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
