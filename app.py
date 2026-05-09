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

st.set_page_config(page_title="SwingHunter V9.2 - Banked Signal Engine", layout="wide")
APP_VERSION = "V9.2"

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
    'AAPL','MSFT','NVDA','TSLA','AMZN','META','GOOGL','NFLX','AMD','AVGO','TSM','QCOM',
    'CRWD','PANW','PLTR','SNOW','DDOG','NET','SMCI','COIN','MSTR','HOOD','SOFI','SQ',
    'PYPL','AFRM','SHOP','BABA','MELI','WMT','TGT','COST','HD','UBER','ABNB','SPOT',
    'DKNG','DIS','NKE','SBUX','MCD','JPM','BAC','GS','MS','V','MA','LLY','NVO','UNH','CAT','BA','MRNA'
]

MOMENTUM_TICKERS = {
    'AMD','NVDA','TSLA','DDOG','NET','QCOM','CRWD','PANW','AVGO','AMZN',
    'MSTR','COIN','SMCI','PLTR','ARM','MU','MRVL','TSM','META','GOOGL',
    'HOOD','AFRM','SHOP','NFLX','SNOW','ZS','MDB','SOFI','SQ','PYPL'
}

NOTIONAL_PER_TRADE = 1000.0
DEFAULT_STARTING_BANK = 50000.0


# ==========================================================
# 3. Parameters
# ==========================================================
@dataclass
class StrategyParams:
    max_risk_pct: float = 9.5
    target_pct: float = 10.0
    min_rr: float = 1.10
    min_atr_pct: float = 1.7
    overbought_rsi: float = 84.0
    max_20d_run: float = 55.0
    stop_atr_breakout: float = 2.5
    stop_atr_pullback: float = 2.1
    rs5_min: float = 1.0
    rs20_min: float = 3.0
    use_momentum_only: bool = True
    max_holding_days: int = 35


# ==========================================================
# 4. Data + Indicators
# ==========================================================
def flatten_download(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels[-1]) == 1:
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
def download_single(ticker: str, period: str = "350d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        df = flatten_download(df)
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_backtest_data(months: int):
    end = datetime.now()
    start = end - timedelta(days=months * 31 + 320)
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
def evaluate_candidate(
    ticker: str,
    c: pd.Series,
    h: pd.Series,
    l: pd.Series,
    qqq_slice: pd.Series,
    params: StrategyParams,
    for_today: bool = False
):
    """
    Returns a concrete trade plan or a reason why not.
    One signal = one theoretical $1,000 trade.
    No cash accounting. No adding to an existing open ticker.
    """
    try:
        if len(c) < 220:
            return None

        last_p = float(c.iloc[-1])
        sma200 = float(c.rolling(200).mean().iloc[-1])
        ema8 = float(c.ewm(span=8, adjust=False).mean().iloc[-1])
        ema21 = float(c.ewm(span=21, adjust=False).mean().iloc[-1])
        res20 = float(h.iloc[-21:-1].max())
        prev_hi = float(h.iloc[-2])

        atr = float(calc_atr_from_series(h, l, c, 14).iloc[-1])
        atr_pct = atr / last_p * 100
        rsi = float(calc_rsi(c, 14).iloc[-1])
        run20 = (last_p / c.iloc[-21] - 1) * 100
        rs5, rs20 = relative_strength_vs(qqq_slice, c)

        regime = market_regime(qqq_slice)

        if params.use_momentum_only and ticker not in MOMENTUM_TICKERS:
            return {"ticker": ticker, "status": "WATCH_ONLY", "reason": "לא מניית מומנטום/צמיחה"}

        if last_p < sma200:
            return {"ticker": ticker, "status": "REJECT", "reason": "מתחת SMA200"}

        if atr_pct < params.min_atr_pct:
            return {"ticker": ticker, "status": "REJECT", "reason": f"ATR% נמוך מדי ({atr_pct:.1f}%)"}

        edge_ok = rs5 > params.rs5_min or rs20 > params.rs20_min

        exceptional_leader = (
            ticker in MOMENTUM_TICKERS
            and rs20 > 25
            and rs5 > 5
            and last_p > ema8
            and last_p > ema21
            and atr_pct >= params.min_atr_pct
        )

        # In weak market, no regular entries. Only exceptional leaders.
        if regime != "BULL_STRONG" and not exceptional_leader:
            return {"ticker": ticker, "status": "REJECT", "reason": f"שוק לא מספיק חזק ({regime})"}

        if (rsi > params.overbought_rsi or run20 > params.max_20d_run) and not exceptional_leader:
            return {"ticker": ticker, "status": "REJECT", "reason": "מתוחה מדי"}

        if not edge_ok and not exceptional_leader:
            return {"ticker": ticker, "status": "REJECT", "reason": "RS חלש מול QQQ"}

        order_type = None
        entry = 0.0
        setup = ""

        dist_to_res = (res20 / last_p - 1)

        # 1) Breakout near resistance
        if 0 < dist_to_res < 0.045 and edge_ok:
            entry = round(res20 * 1.002, 2)
            if last_p < entry:
                order_type = "BUY STOP LIMIT"
                setup = "Near Resistance Breakout"

        # 2) Pullback in a confirmed leader.
        # V9.2: do NOT remove TSLA/COIN/etc. Instead, make pullbacks higher-quality:
        # trend must be orderly, relative strength meaningful, and not a falling knife.
        higher_low_3 = False
        try:
            higher_low_3 = bool(l.iloc[-1] > l.iloc[-2] > l.iloc[-3])
        except Exception:
            higher_low_3 = False

        pullback_quality_ok = (
            last_p > ema8 > ema21
            and rs20 > max(params.rs20_min, 5)
            and rs5 > -2
            and 45 <= rsi <= 74
            and run20 <= 30
            and higher_low_3
        )

        if order_type is None and pullback_quality_ok and (last_p / ema8 - 1) < 0.025:
            base = ema8
            entry = min(round(base * 1.003, 2), round(last_p * 0.995, 2))
            if last_p > entry:
                order_type = "BUY LIMIT"
                setup = "RS Pullback"

        # 3) Momentum continuation
        if order_type is None and last_p > ema21 and (prev_hi / last_p - 1) < 0.025:
            entry = round(prev_hi * 1.002, 2)
            if last_p < entry:
                order_type = "BUY STOP LIMIT"
                setup = "Momentum Continuation"

        if order_type is None or entry <= 0:
            return {"ticker": ticker, "status": "WATCH", "reason": "אין נקודת כניסה נקייה כרגע"}

        if order_type == "BUY LIMIT":
            raw_stop = entry - params.stop_atr_pullback * atr
        else:
            raw_stop = entry - params.stop_atr_breakout * atr

        # Stop is capped by user max-risk percent.
        stop = max(raw_stop, entry * (1 - params.max_risk_pct / 100))
        stop = round(stop, 2)

        risk_pct = (entry - stop) / entry * 100
        target = round(entry * (1 + params.target_pct / 100), 2)
        rr = params.target_pct / risk_pct if risk_pct > 0 else 0

        if risk_pct > params.max_risk_pct:
            return {"ticker": ticker, "status": "REJECT", "reason": f"סיכון גבוה מדי ({risk_pct:.1f}%)"}

        if rr < params.min_rr:
            return {"ticker": ticker, "status": "REJECT", "reason": f"R/R נמוך ({rr:.2f})"}

        distance_to_entry = (entry / last_p - 1) * 100

        if order_type == "BUY LIMIT":
            if distance_to_entry <= -3.0:
                action = "WAIT - רחוקה מעל הלימיט"
            else:
                action = "PLACE LIMIT"
        else:
            if distance_to_entry < 0:
                action = "MISSED / WAIT RESET"
            elif distance_to_entry <= 3.0:
                action = "PLACE STOP LIMIT"
            else:
                action = "WAIT - טריגר רחוק"

        score = (
            rs20 * 1.3
            + rs5 * 0.7
            + min(max(run20, -10), 60) * 0.30
            + rr * 10
            - risk_pct * 1.2
            + (15 if exceptional_leader else 0)
            + (8 if setup == "RS Pullback" else 0)
        )

        return {
            "ticker": ticker,
            "status": "SIGNAL",
            "setup": setup,
            "order_type": order_type,
            "current": round(last_p, 2),
            "entry": round(entry, 2),
            "limit": round(entry * 1.006, 2) if order_type == "BUY STOP LIMIT" else round(entry, 2),
            "stop": stop,
            "target": target,
            "distance_to_entry_pct": round(distance_to_entry, 2),
            "risk_pct": round(risk_pct, 2),
            "target_pct": params.target_pct,
            "rr": round(rr, 2),
            "rs5": round(rs5, 1),
            "rs20": round(rs20, 1),
            "rsi": round(rsi, 1),
            "atr_pct": round(atr_pct, 1),
            "run20": round(run20, 1),
            "regime": regime,
            "action": action,
            "score": round(score, 2),
            "reason": ""
        }

    except Exception as e:
        return {"ticker": ticker, "status": "ERROR", "reason": str(e)[:80]}


def execute_pending_order(order, open_p, high_p, low_p):
    """
    Executes previous-day DAY order on today's OHLC.
    Returns fill price or None.
    """
    if order["order_type"] == "BUY LIMIT":
        if low_p <= order["entry"]:
            return order["entry"]
        return None

    if order["order_type"] == "BUY STOP LIMIT":
        stop_price = order["entry"]
        limit_price = order["limit"]

        # Realistic stop-limit: gap over limit means no fill.
        if open_p > limit_price:
            return None

        if high_p >= stop_price:
            fill = max(open_p, stop_price)
            return fill if fill <= limit_price else None

    return None


# ==========================================================
# 6. Backtest - Clean Signal Test
# ==========================================================
def run_clean_signal_backtest(
    data: pd.DataFrame,
    months: int,
    params: StrategyParams,
    starting_bank: float = DEFAULT_STARTING_BANK,
    diagnostics_top_n: int = 20
):
    """
    Banked signal test:
    - Every new trade uses $1,000 from the bank.
    - If the bank does not have $1,000 available, no new position is opened.
    - When a position closes, $1,000 + PnL returns to the bank.
    - No adding to a ticker that is already open.
    - Sell always closes 100% of the position.
    """
    prices = get_panel(data, "Close")
    highs = get_panel(data, "High")
    lows = get_panel(data, "Low")
    opens = get_panel(data, "Open")

    requested_days = int(months * 21.5)
    start_idx = max(220, len(prices) - requested_days)
    end_idx = len(prices) - 1

    cash_bank = float(starting_bank)
    open_positions = {}  # ticker -> position
    pending_orders = {}  # ticker -> candidate from yesterday

    trade_log = []
    equity_rows = []
    pending_created = 0
    pending_filled = 0
    pending_expired = 0
    pending_rejected_no_cash = 0
    next_trade_id = 1
    turnover = 0.0
    max_open_positions = 0

    def current_open_value(i):
        value = 0.0
        for ticker, pos in open_positions.items():
            try:
                close_p = float(prices[ticker].iloc[i])
                value += NOTIONAL_PER_TRADE * (close_p / pos["entry"])
            except Exception:
                value += NOTIONAL_PER_TRADE
        return value

    for i in range(start_idx, end_idx):
        date = prices.index[i]
        date_str = date.strftime("%Y-%m-%d")

        # ----------------------------------------------
        # 1) Execute pending DAY orders
        # ----------------------------------------------
        processed = set()
        for ticker, order in list(pending_orders.items()):
            if ticker in open_positions:
                processed.add(ticker)
                continue

            if cash_bank < NOTIONAL_PER_TRADE:
                pending_rejected_no_cash += 1
                processed.add(ticker)
                continue

            try:
                open_p = safe_float(opens[ticker].iloc[i])
                high_p = safe_float(highs[ticker].iloc[i])
                low_p = safe_float(lows[ticker].iloc[i])
            except Exception:
                processed.add(ticker)
                continue

            if not np.isfinite(open_p):
                processed.add(ticker)
                continue

            fill_price = execute_pending_order(order, open_p, high_p, low_p)

            if fill_price is not None:
                cash_bank -= NOTIONAL_PER_TRADE
                turnover += NOTIONAL_PER_TRADE

                open_positions[ticker] = {
                    "trade_id": next_trade_id,
                    "ticker": ticker,
                    "entry_date": date_str,
                    "entry_index": i,
                    "entry": fill_price,
                    "stop": order["stop"],
                    "target": order["target"],
                    "setup": order["setup"],
                    "order_type": order["order_type"],
                    "risk_pct": (fill_price - order["stop"]) / fill_price * 100,
                    "target_pct": (order["target"] / fill_price - 1) * 100,
                    "score": order["score"],
                }
                pending_filled += 1
                next_trade_id += 1
                processed.add(ticker)

        pending_expired += len([t for t in pending_orders if t not in processed])
        pending_orders.clear()

        # ----------------------------------------------
        # 2) Manage existing positions - sell full position only
        # ----------------------------------------------
        closed_tickers = []

        for ticker, pos in list(open_positions.items()):
            # Do not exit on the same day as entry due daily OHLC sequencing ambiguity.
            if pos["entry_index"] == i:
                continue

            try:
                high_p = safe_float(highs[ticker].iloc[i])
                low_p = safe_float(lows[ticker].iloc[i])
                close_p = safe_float(prices[ticker].iloc[i])
                close_hist = prices[ticker].iloc[max(0, i - 25):i + 1].dropna()
                ema21 = float(close_hist.ewm(span=21, adjust=False).mean().iloc[-1])
            except Exception:
                continue

            exit_reason = None
            exit_price = None

            # Conservative assumption: if stop and target both touched, stop first.
            if low_p <= pos["stop"]:
                exit_reason = "STOP"
                exit_price = pos["stop"]
            elif high_p >= pos["target"]:
                exit_reason = "TARGET"
                exit_price = pos["target"]
            elif np.isfinite(close_p) and close_p < ema21 * 0.995:
                exit_reason = "EMA21_CLOSE"
                exit_price = close_p
            elif i - pos["entry_index"] >= params.max_holding_days:
                exit_reason = "TIME_EXIT"
                exit_price = close_p

            if exit_reason and exit_price:
                ret_pct = (exit_price / pos["entry"] - 1) * 100
                pnl = NOTIONAL_PER_TRADE * ret_pct / 100
                cash_bank += NOTIONAL_PER_TRADE + pnl

                # QQQ benchmark for same holding window
                try:
                    q_entry = float(prices["QQQ"].loc[pd.to_datetime(pos["entry_date"])])
                    q_exit = float(prices["QQQ"].iloc[i])
                    q_ret_pct = (q_exit / q_entry - 1) * 100
                    q_pnl = NOTIONAL_PER_TRADE * q_ret_pct / 100
                except Exception:
                    q_ret_pct = np.nan
                    q_pnl = np.nan

                trade_log.append({
                    "TradeID": pos["trade_id"],
                    "Ticker": ticker,
                    "EntryDate": pos["entry_date"],
                    "ExitDate": date_str,
                    "HoldingDays": i - pos["entry_index"],
                    "Setup": pos["setup"],
                    "Entry": round(pos["entry"], 2),
                    "Exit": round(exit_price, 2),
                    "ExitReason": exit_reason,
                    "Return %": round(ret_pct, 2),
                    "PnL $1000": round(pnl, 2),
                    "Risk %": round(pos["risk_pct"], 2),
                    "Target %": round(pos["target_pct"], 2),
                    "Score": round(pos["score"], 2),
                    "QQQ Same Window %": round(q_ret_pct, 2) if np.isfinite(q_ret_pct) else np.nan,
                    "QQQ Same Window PnL": round(q_pnl, 2) if np.isfinite(q_pnl) else np.nan
                })

                closed_tickers.append(ticker)

        for ticker in closed_tickers:
            open_positions.pop(ticker, None)

        # ----------------------------------------------
        # 3) Bank equity curve
        # ----------------------------------------------
        open_value = current_open_value(i)
        total_equity = cash_bank + open_value
        open_capital = len(open_positions) * NOTIONAL_PER_TRADE
        max_open_positions = max(max_open_positions, len(open_positions))

        equity_rows.append({
            "Date": date_str,
            "Cash Bank": cash_bank,
            "Open Value": open_value,
            "Total Equity": total_equity,
            "Open Positions": len(open_positions),
            "Open Capital": open_capital,
            "Exposure %": (open_value / total_equity * 100) if total_equity > 0 else 0
        })

        # ----------------------------------------------
        # 4) Generate next-day orders
        # ----------------------------------------------
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

                cand = evaluate_candidate(ticker, c, h, l, qqq_slice, params)
                if cand and cand.get("status") == "SIGNAL":
                    candidates.append(cand)
            except Exception:
                continue

        # Rank candidates, then create as many pending orders as bank can realistically support.
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

        available_slots = int(cash_bank // NOTIONAL_PER_TRADE) - len(pending_orders)
        if available_slots > 0:
            for cand in candidates[:available_slots]:
                pending_orders[cand["ticker"]] = cand
                pending_created += 1

    # Final close of any remaining positions
    final_date = prices.index[end_idx].strftime("%Y-%m-%d")
    for ticker, pos in list(open_positions.items()):
        try:
            final_close = float(prices[ticker].iloc[end_idx])
            ret_pct = (final_close / pos["entry"] - 1) * 100
            pnl = NOTIONAL_PER_TRADE * ret_pct / 100
            cash_bank += NOTIONAL_PER_TRADE + pnl

            try:
                q_entry = float(prices["QQQ"].loc[pd.to_datetime(pos["entry_date"])])
                q_exit = float(prices["QQQ"].iloc[end_idx])
                q_ret_pct = (q_exit / q_entry - 1) * 100
                q_pnl = NOTIONAL_PER_TRADE * q_ret_pct / 100
            except Exception:
                q_ret_pct = np.nan
                q_pnl = np.nan

            trade_log.append({
                "TradeID": pos["trade_id"],
                "Ticker": ticker,
                "EntryDate": pos["entry_date"],
                "ExitDate": final_date,
                "HoldingDays": end_idx - pos["entry_index"],
                "Setup": pos["setup"],
                "Entry": round(pos["entry"], 2),
                "Exit": round(final_close, 2),
                "ExitReason": "FINAL_CLOSE",
                "Return %": round(ret_pct, 2),
                "PnL $1000": round(pnl, 2),
                "Risk %": round(pos["risk_pct"], 2),
                "Target %": round(pos["target_pct"], 2),
                "Score": round(pos["score"], 2),
                "QQQ Same Window %": round(q_ret_pct, 2) if np.isfinite(q_ret_pct) else np.nan,
                "QQQ Same Window PnL": round(q_pnl, 2) if np.isfinite(q_pnl) else np.nan
            })
        except Exception:
            pass

    df_trades = pd.DataFrame(trade_log)
    df_equity = pd.DataFrame(equity_rows).set_index("Date") if equity_rows else pd.DataFrame()

    if not df_equity.empty:
        # After final liquidation, add final point.
        df_equity.loc[final_date, "Cash Bank"] = cash_bank
        df_equity.loc[final_date, "Open Value"] = 0.0
        df_equity.loc[final_date, "Total Equity"] = cash_bank
        df_equity.loc[final_date, "Open Positions"] = 0
        df_equity.loc[final_date, "Open Capital"] = 0
        df_equity.loc[final_date, "Exposure %"] = 0.0

    if not df_trades.empty:
        wins = int((df_trades["PnL $1000"] > 0).sum())
        losses = int((df_trades["PnL $1000"] < 0).sum())
        gross_profit = float(df_trades.loc[df_trades["PnL $1000"] > 0, "PnL $1000"].sum())
        gross_loss = float(-df_trades.loc[df_trades["PnL $1000"] < 0, "PnL $1000"].sum())
        total_pnl = float(df_trades["PnL $1000"].sum())
        avg_return = float(df_trades["Return %"].mean())
        avg_win = float(df_trades.loc[df_trades["PnL $1000"] > 0, "Return %"].mean()) if wins else 0.0
        avg_loss = float(df_trades.loc[df_trades["PnL $1000"] < 0, "Return %"].mean()) if losses else 0.0
        qqq_same_window_pnl = float(df_trades["QQQ Same Window PnL"].sum(skipna=True))
        qqq_same_window_avg = float(df_trades["QQQ Same Window %"].mean(skipna=True))
    else:
        wins = losses = 0
        gross_profit = gross_loss = total_pnl = avg_return = avg_win = avg_loss = 0.0
        qqq_same_window_pnl = qqq_same_window_avg = 0.0

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    win_rate = wins / max(1, wins + losses) * 100
    ending_bank = cash_bank
    roi = (ending_bank / starting_bank - 1) * 100 if starting_bank else 0

    if not df_equity.empty:
        running_max = df_equity["Total Equity"].cummax()
        drawdown_pct = (df_equity["Total Equity"] / running_max - 1) * 100
        max_dd_pct = float(drawdown_pct.min())
        avg_exposure_pct = float(df_equity["Exposure %"].mean())
    else:
        max_dd_pct = 0.0
        avg_exposure_pct = 0.0

    # QQQ buy-and-hold on starting bank for period
    try:
        q = prices["QQQ"].iloc[start_idx:end_idx].dropna()
        qqq_buyhold_pct = (float(q.iloc[-1]) / float(q.iloc[0]) - 1) * 100 if len(q) > 1 else np.nan
        qqq_buyhold_value = starting_bank * (1 + qqq_buyhold_pct / 100)
        qqq_buyhold_pnl = qqq_buyhold_value - starting_bank
    except Exception:
        qqq_buyhold_pct = np.nan
        qqq_buyhold_value = np.nan
        qqq_buyhold_pnl = np.nan

    # Ticker summary
    if not df_trades.empty:
        ticker_summary = (
            df_trades.groupby("Ticker", as_index=False)
            .agg(
                Trades=("TradeID", "count"),
                PnL=("PnL $1000", "sum"),
                AvgReturn=("Return %", "mean"),
                Wins=("PnL $1000", lambda x: int((x > 0).sum())),
                Losses=("PnL $1000", lambda x: int((x < 0).sum())),
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
        "Months": months,
        "Starting Bank": round(starting_bank, 2),
        "Ending Bank": round(ending_bank, 2),
        "Bank ROI %": round(roi, 2),
        "Trade Size": NOTIONAL_PER_TRADE,
        "Max Risk %": params.max_risk_pct,
        "Target %": params.target_pct,
        "Trades": len(df_trades),
        "Wins": wins,
        "Losses": losses,
        "Win Rate %": round(win_rate, 2),
        "Total PnL": round(total_pnl, 2),
        "Average Trade Return %": round(avg_return, 2),
        "Average Win %": round(avg_win, 2),
        "Average Loss %": round(avg_loss, 2),
        "Profit Factor": round(profit_factor, 2) if np.isfinite(profit_factor) else "∞",
        "Max Drawdown %": round(max_dd_pct, 2),
        "Average Exposure %": round(avg_exposure_pct, 2),
        "Max Open Positions": max_open_positions,
        "Turnover": round(turnover, 2),
        "QQQ Buy & Hold %": round(qqq_buyhold_pct, 2) if np.isfinite(qqq_buyhold_pct) else "",
        "QQQ Buy & Hold Ending Value": round(qqq_buyhold_value, 2) if np.isfinite(qqq_buyhold_value) else "",
        "QQQ Buy & Hold PnL": round(qqq_buyhold_pnl, 2) if np.isfinite(qqq_buyhold_pnl) else "",
        "QQQ Same Windows PnL": round(qqq_same_window_pnl, 2),
        "QQQ Same Windows Avg %": round(qqq_same_window_avg, 2),
        "Pending Orders Created": pending_created,
        "Pending Orders Filled": pending_filled,
        "Pending Orders Expired": pending_expired,
        "Pending Rejected No Cash": pending_rejected_no_cash,
    }

    return df_equity, df_trades, ticker_summary, metrics
# ==========================================================
# 7. Live Daily Dashboard
# ==========================================================
def get_today_actions(params: StrategyParams):
    qqq = download_single("QQQ", "350d")
    if qqq.empty:
        return pd.DataFrame(), pd.DataFrame()

    qqq_slice = qqq["Close"].dropna()

    rows = []
    radar = []

    for ticker in WATCHLIST:
        df = download_single(ticker, "350d")
        if df.empty or len(df) < 220:
            continue

        c, h, l = df["Close"], df["High"], df["Low"]
        cand = evaluate_candidate(ticker, c, h, l, qqq_slice, params, for_today=True)

        if not cand:
            continue

        if cand.get("status") == "SIGNAL":
            rows.append({
                "Ticker": cand["ticker"],
                "Action Now": cand["action"],
                "Order": cand["order_type"],
                "Current": cand["current"],
                "Entry": cand["entry"],
                "Distance to Entry %": cand["distance_to_entry_pct"],
                "Stop": cand["stop"],
                "Target": cand["target"],
                "Risk %": cand["risk_pct"],
                "Target %": cand["target_pct"],
                "R/R": cand["rr"],
                "Setup": cand["setup"],
                "Regime": cand["regime"],
                "RS20": cand["rs20"],
                "Score": cand["score"],
            })
        else:
            radar.append({
                "Ticker": cand.get("ticker", ticker),
                "Status": cand.get("status", ""),
                "Reason": cand.get("reason", ""),
            })

    df_orders = pd.DataFrame(rows)
    if not df_orders.empty:
        df_orders = df_orders.sort_values("Score", ascending=False)

    df_radar = pd.DataFrame(radar).head(40)
    return df_orders, df_radar


# ==========================================================
# 8. Export
# ==========================================================
def build_zip_report(df_equity, df_trades, ticker_summary, metrics):
    """
    Robust export that does NOT depend on openpyxl/xlsxwriter.
    Creates one ZIP file containing all CSVs.
    """
    output = BytesIO()
    metrics_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()])

    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("summary.csv", metrics_df.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("signal_equity.csv", df_equity.reset_index().to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("trades.csv", df_trades.to_csv(index=False).encode("utf-8-sig"))
        zf.writestr("ticker_summary.csv", ticker_summary.to_csv(index=False).encode("utf-8-sig"))

    output.seek(0)
    return output.getvalue()


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
    st.markdown("<h1 style='text-align: right;'>🎯 SwingHunter V9.2 — Banked Signal Engine</h1>", unsafe_allow_html=True)
    st.info(
        "V9.2 בודקת בצורה שיותר דומה למה שתיארת: יש בנק, כל כניסה משתמשת ב-$1,000, "
        "אם אין כסף פנוי לא נכנסים, לא מוסיפים למניה שכבר פתוחה, ובמכירה כל הכסף חוזר לבנק."
    )

    st.sidebar.header("הגדרות קצרות")
    months = st.sidebar.slider("תקופת בדיקה היסטורית (חודשים)", 3, 24, 12)
    starting_bank = st.sidebar.number_input("בנק התחלתי ($)", value=50000, step=5000)
    max_risk_pct = st.sidebar.slider("סיכון מקסימלי לעסקה (%)", 4.0, 15.0, 9.5, 0.5)
    target_pct = st.sidebar.slider("יעד רווח לעסקה (%)", 6.0, 18.0, 10.0, 0.5)

    params = StrategyParams(
        max_risk_pct=max_risk_pct,
        target_pct=target_pct
    )

    tab_daily, tab_backtest = st.tabs(["🚀 מה עושים היום", "🔬 בדיקת איתותים נקייה"])

    with tab_daily:
        if st.button("⚡ הפק פקודות/מעקב להיום", use_container_width=True):
            with st.spinner("סורק מניות ומייצר פקודות ברורות..."):
                df_orders, df_radar = get_today_actions(params)

                st.markdown("### 🧭 פקודות לביצוע היום")

                if not df_orders.empty:
                    st.dataframe(df_orders, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ הורד Orders CSV",
                        df_orders.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"swinghunter_{APP_VERSION}_orders_today.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("אין היום פקודות ביצוע. לא בכוח.")

                with st.expander("🔍 רדאר / למה מניות לא נכנסו"):
                    st.dataframe(df_radar, use_container_width=True, hide_index=True)

    with tab_backtest:
        st.markdown(
            f"### 🧪 Banked Signal Backtest — {months} חודשים — בנק ${starting_bank:,.0f} — $1,000 לכל כניסה"
        )

        if st.button("⚙️ הרץ בדיקת איתותים", type="primary"):
            with st.spinner("מריץ Backtest נקי בלי קופה ובלי גודל פוזיציה..."):
                data = fetch_backtest_data(months)
                df_equity, df_trades, ticker_summary, metrics = run_clean_signal_backtest(
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
                    f"כל עסקה משתמשת ב-$1,000 בלבד. כשהיא נמכרת, $1,000 + רווח/הפסד חוזר לבנק. "
                    f"Turnover מצטבר: ${metrics['Turnover']:,.0f}. חשיפה ממוצעת: {metrics['Average Exposure %']:.1f}%."
                )

                st.markdown("#### 📈 Bank Equity Curve")
                st.line_chart(df_equity)

                zip_bytes = build_zip_report(df_equity, df_trades, ticker_summary, metrics)
                st.download_button(
                    "⬇️ הורד קובץ ZIP אחד עם כל הבדיקה",
                    zip_bytes,
                    file_name=f"swinghunter_{APP_VERSION}_clean_signal_report.zip",
                    mime="application/zip",
                    use_container_width=True
                )

                if not df_trades.empty:
                    with st.expander("📌 כל העסקאות"):
                        st.dataframe(df_trades, use_container_width=True, hide_index=True)

                if not ticker_summary.empty:
                    with st.expander("🏷️ PnL לפי טיקר"):
                        st.dataframe(ticker_summary, use_container_width=True, hide_index=True)
