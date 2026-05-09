import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import urllib.request
import xml.etree.ElementTree as ET
import warnings

# ==========================================
# SwingHunter V7 - THE QUANT TERMINAL (FINAL)
# ==========================================

st.set_page_config(page_title="SwingHunter V7 - The Quant Terminal", layout="wide")
warnings.filterwarnings('ignore')

# --- Hardcoded Credentials ---
APP_PASSWORD = "Pk0105Ak2701" 
MY_EMAIL = "orel@peleg-eng.com"

WATCHLIST = [
    'AAPL','MSFT','NVDA','TSLA','AMZN','META','GOOGL','NFLX','AMD','AVGO','TSM','QCOM',
    'CRWD','PANW','PLTR','SNOW','DDOG','NET','SMCI','COIN','MSTR','HOOD','SOFI','SQ',
    'PYPL','AFRM','SHOP','BABA','MELI','WMT','TGT','COST','HD','UBER','ABNB','SPOT',
    'DKNG','DIS','NKE','SBUX','MCD','JPM','BAC','GS','MS','V','MA','LLY','NVO','UNH','CAT','BA','MRNA'
]

HISTORY_FILE = "swinghunter_history.csv"

# ==========================================
# 1. Persistence & Memory
# ==========================================
def save_scan_history(df_scan):
    if df_scan.empty: return
    df_save = df_scan[['מניה', 'החלטה', 'ציון_כולל']].copy()
    df_save["scan_date"] = datetime.now().strftime('%Y-%m-%d')
    try:
        old = pd.read_csv(HISTORY_FILE)
        combined = pd.concat([old, df_save], ignore_index=True).drop_duplicates(subset=['מניה', 'scan_date'], keep='last')
        combined.to_csv(HISTORY_FILE, index=False)
    except:
        df_save.to_csv(HISTORY_FILE, index=False)

def get_setup_persistence(ticker):
    try:
        hist = pd.read_csv(HISTORY_FILE)
        recent = hist[hist["מניה"] == ticker].tail(5)
        if recent.empty: return 0, 0
        watch_days = recent["החלטה"].astype(str).str.contains("למעקב|פעיל|ARMED|Building Pressure").sum()
        score_trend = recent["ציון_כולל"].diff().sum()
        return watch_days, score_trend
    except: return 0, 0

# ==========================================
# 2. Environment Helpers
# ==========================================
@st.cache_data(ttl=3600)
def get_spy_context():
    try:
        spy = yf.download('SPY', period='250d', progress=False)
        spy_close = spy['Close'] if not isinstance(spy.columns, pd.MultiIndex) else spy['Close'].iloc[:, 0]
        sma20 = spy_close.rolling(20).mean().iloc[-1]
        trend = "BULL" if spy_close.iloc[-1] > sma20 else "BEAR"
        return spy_close, trend
    except: return None, "UNKNOWN"

def get_earnings_status(ticker):
    try:
        tkr = yf.Ticker(ticker)
        cal = tkr.get_earnings_dates(limit=5)
        if cal is not None and not cal.empty:
            now = pd.Timestamp.now(tz='UTC')
            future = cal.index[cal.index > now]
            if not future.empty and (future[0] - now).days <= 3:
                return "DANGER", (future[0] - now).days
    except: pass
    return "CLEAR", 0

def get_headlines_sentiment(ticker):
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp: xml_page = resp.read()
        root = ET.fromstring(xml_page)
        items = root.findall('.//item')
        if not items: return "⚪"
        text = " | ".join([item.find('title').text for item in items[:2]]).lower()
        pos = sum(1 for w in ['up', 'surge', 'jump', 'beat', 'growth'] if w in text)
        neg = sum(1 for w in ['down', 'plunge', 'drop', 'miss', 'cut', 'warning'] if w in text)
        return "🟢" if pos > neg else ("🔴" if neg > pos else "⚪")
    except: return "⚪"

# ==========================================
# 3. Core Edge Modules
# ==========================================
def calculate_compression(df):
    try:
        h, l, c, v = df["High"], df["Low"], df["Close"], df["Volume"]
        range_5 = ((h.tail(5).max() - l.tail(5).min()) / c.iloc[-1]) * 100
        avg_range_20 = (((h - l) / c).rolling(20).mean().iloc[-1]) * 100
        v5, v20 = v.tail(5).mean(), v.rolling(20).mean().iloc[-1]
        score = 0
        if range_5 < (avg_range_20 * 0.75): score += 20
        if v5 < (v20 * 0.85): score += 10
        if sum(1 for i in range(-4, 0) if l.iloc[i] >= l.iloc[i-1]) >= 3: score += 15
        return score
    except: return 0

def check_failed_breakdown(df):
    try:
        c, l = df["Close"], df["Low"]
        sup20 = float(l.iloc[-21:-5].min())
        broke = float(l.iloc[-5:-1].min()) < sup20 * 0.99
        reclaimed = float(c.iloc[-1]) > sup20 * 1.005 and float(c.iloc[-1]) > c.rolling(5).mean().iloc[-1]
        if broke and reclaimed: return True, sup20
    except: pass
    return False, 0

def check_post_drift(df):
    try:
        c, o, h, l, v = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
        if v.tail(10).max() > v.rolling(20).mean().iloc[-1] * 2.5:
            idx = v.tail(10).idxmax()
            pos = df.index.get_loc(idx)
            if pos < len(df) - 1:
                ret = (float(c.iloc[pos]) / float(o.iloc[pos]) - 1) * 100
                if ret > -5.0 and float(c.iloc[-1]) > float(l.iloc[pos]):
                    return True, float(h.iloc[pos]), float(l.iloc[pos])
    except: pass
    return False, 0, 0

# ==========================================
# 4. Unified Analysis Engine (Live & Backtest)
# ==========================================
def analyze_edge_unified(ticker, ticker_df, spy_close_series, market_trend, invest_amount, current_pos=[]):
    try:
        if len(ticker_df) < 200: return None
        c, h, l = ticker_df['Close'], ticker_df['High'], ticker_df['Low']
        last_p = float(c.iloc[-1])
        sma200, ema21 = c.rolling(200).mean().iloc[-1], c.ewm(span=21, adjust=False).mean().iloc[-1]
        res20 = float(h.iloc[-21:-1].max())
        dist_res = (res20 / last_p - 1) * 100

        # Modules
        comp_s = calculate_compression(ticker_df)
        rs_20 = ((last_p / c.iloc[-21]) - (spy_close_series.iloc[-1] / spy_close_series.iloc[-21])) * 100
        fail_brk, reclaimed_lvl = check_failed_breakdown(ticker_df)
        drift, ev_h, ev_l = check_post_drift(ticker_df)
        w_days, _ = get_setup_persistence(ticker)

        # Scoring
        score = (15 if last_p > sma200 else 0) + (10 if last_p > ema21 else 0) + (15 if dist_res < 4.0 else 0)
        notes = []
        if comp_s >= 25 and (dist_res < 4.0 or rs_20 > 0): score += comp_s; notes.append("Compression")
        if rs_20 > 5.0: score += 20; notes.append("RS Leader")
        if fail_brk: score += 35; notes.append("Failed Breakdown")
        if drift: score += 25; notes.append("Post-Event Drift")
        if w_days >= 2: score += 15; notes.append("Building Pressure")

        rsi = 100 - (100 / (1 + ((c.diff().where(c.diff() > 0, 0)).rolling(14).mean() / (-c.diff().where(c.diff() < 0, 0)).rolling(14).mean()).iloc[-1]))
        earn_s, earn_d = get_earnings_status(ticker)

        # Status
        icon, decision, reject = "🔴", "Dormant", ""
        if earn_s == "DANGER": icon, decision, reject, score = "⚠️", "DANGER", "דוח קרוב", 0
        elif rsi > 78: icon, decision, reject, score = "🔥", "חם מדי", "RSI גבוה", min(score, 30)
        elif market_trend == "BEAR" and score < 75: icon, decision, reject, score = "🔴", "Dormant", "שוק BEAR", min(score, 40)
        elif score >= 75 and len(notes) > 0: icon, decision = "🟢", "ARMED"
        elif score >= 45: icon, decision = "🟡", "Building Pressure"
        
        order = None
        if decision == "ARMED" and ticker not in current_pos:
            p_type, entry = None, 0
            if "Compression" in notes and dist_res < 3.0: p_type, entry = "BUY STOP LIMIT", round(res20 * 1.002, 2)
            elif fail_brk or (rs_20 > 5.0 and last_p > ema21): p_type, entry = "BUY LIMIT", min(round(ema21 * 1.005, 2), round(last_p * 0.995, 2))
            
            if p_type:
                tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                stop = round(min(ema21, entry - (1.5 * atr)), 2)
                if (entry-stop) > 0 and (entry*0.1)/(entry-stop) >= 1.5:
                    sh = int(invest_amount / entry)
                    if sh > 0:
                        order = {'מניה': ticker, 'פעולה': p_type, 'Edge': " + ".join(notes), 'כניסה': entry, 'כמות': sh, 'סטופ': stop}
        
        return {
            'scanner': {'מניה': ticker, 'החלטה': f"{icon} {decision}", 'ציון_כולל': int(score), 'Edge': " + ".join(notes), 'סיבת פסילה': reject},
            'order': order
        }
    except: return None

# ==========================================
# 5. Backtest Simulation (V7 Engine)
# ==========================================
def run_v7_backtest(data, invest_amount):
    closes, highs, lows, opens = data['Close'], data['High'], data['Low'], data['Open']
    spy_c = data['Close']['SPY']
    cash, pos, pending, logs, equity = 10000.0, {}, {}, [], []
    tot_inv, tot_p, tot_l = 0.0, 0.0, 0.0

    for i in range(200, len(closes) - 1):
        today_str = closes.index[i].strftime('%Y-%m-%d')
        # 1. Fill Pending
        for t, o in list(pending.items()):
            if t in pos or len(pos) >= 5: continue
            try:
                t_o, t_h, t_l = float(opens[t].iloc[i]), float(highs[t].iloc[i]), float(lows[t].iloc[i])
                exec_p, execed = 0, False
                if o['type'] == 'BUY LIMIT' and t_l <= o['price']: exec_p, execed = o['price'], True
                elif o['type'] == 'BUY STOP LIMIT' and t_h >= o['price']: exec_p, execed = max(t_o, o['price']), True
                if execed and cash >= (exec_p * o['shares']):
                    cash -= (exec_p * o['shares']); tot_inv += (exec_p * o['shares'])
                    pos[t] = {'qty': o['shares'], 'ent': exec_p, 'st': o['stop'], 'mode': 'FULL', 'ts': 0.0}
            except: continue
        pending.clear()

        # 2. Position Mgmt (Scale Out)
        for t, p in list(pos.items()):
            try:
                t_h, t_l = float(highs[t].iloc[i]), float(lows[t].iloc[i])
                ema21 = float(closes[t].iloc[:i+1].ewm(span=21, adjust=False).mean().iloc[-1])
                if p['mode'] == 'FULL' and t_h >= (p['ent'] * 1.10):
                    q = p['qty'] // 2
                    if q > 0:
                        rev = (p['ent'] * 1.10 * q); cash += rev; tot_p += (rev - (q * p['ent']))
                        p['qty'] -= q; p['mode'], p['st'] = 'HALF', p['ent']
                if p['mode'] == 'HALF': p['ts'] = max(p['ts'], ema21 * 0.99, p['ent'])
                
                active_stop = p['ts'] if p['mode'] == 'HALF' else p['st']
                if t_l <= active_stop or (p['mode'] == 'HALF' and t_h >= p['ent'] * 1.15):
                    sell_p = p['ent'] * 1.15 if (p['mode'] == 'HALF' and t_h >= p['ent'] * 1.15) else active_stop
                    rev = sell_p * p['qty']; cash += rev
                    pnl = rev - (p['qty'] * p['ent'])
                    if pnl > 0: tot_p += pnl
                    else: tot_l += abs(pnl)
                    logs.append({'Date': today_str, 'Ticker': t, 'PnL': round(pnl, 2)}); del pos[t]
            except: continue
        equity.append(cash + sum(p['qty'] * float(closes[t].iloc[i]) for t, p in pos.items()))

        # 3. New Scan
        spy_slice = spy_c.iloc[:i+1]
        market_trend = "BULL" if float(spy_slice.iloc[-1]) > float(spy_slice.rolling(20).mean().iloc[-1]) else "BEAR"
        if market_trend == "BULL" and len(pos) < 5:
            for t in WATCHLIST:
                if t in pos: continue
                res = analyze_edge_unified(t, data[t].iloc[:i+1], spy_slice, "BULL", invest_amount, list(pos.keys()))
                if res and res['order']:
                    o = res['order']
                    pending[t] = {'type': o['פעולה'], 'price': o['כניסה'], 'stop': o['סטופ'], 'shares': o['כמות']}

    return pd.DataFrame({'Portfolio': equity}, index=closes.index[200:-1]), pd.DataFrame(logs), tot_inv, tot_p, tot_l

# ==========================================
# 6. Streamlit UI
# ==========================================
if "auth" not in st.session_state: st.session_state["auth"] = False
if not st.session_state["auth"]:
    st.title("🔒 Terminal Access")
    if st.text_input("Password:", type="password") == APP_PASSWORD:
        if st.button("Enter"): st.session_state["auth"] = True; st.rerun()
else:
    st.markdown("<h1 style='text-align: right;'>🎯 SwingHunter V7 - The Quant Terminal</h1>", unsafe_allow_html=True)
    st.sidebar.header("Capital Mgmt")
    inv_amount = st.sidebar.number_input("Investment per Trade ($)", value=1000, step=100)
    tab1, tab2 = st.tabs(["🚀 Live Dashboard", "🔬 Backtest Lab"])

    with tab1:
        if st.button("⚡ Execute Daily Scan", use_container_width=True):
            spy_c, trend = get_spy_context()
            results, orders = [], []
            with st.spinner("Running Unified Engine..."):
                for t in WATCHLIST:
                    df = yf.download(t, period='250d', progress=False)
                    res = analyze_edge_unified(t, df, spy_c, trend, inv_amount)
                    if res:
                        results.append(res['scanner'])
                        if res['order']: orders.append(res['order'])
            st.markdown("### 📝 Today's Orders"); st.dataframe(pd.DataFrame(orders)) if orders else st.info("No orders.")
            st.markdown("### 🔍 Market Radar"); df_res = pd.DataFrame(results).sort_values(by='ציון_כולל', ascending=False)
            save_scan_history(df_res); st.dataframe(df_res)

    with tab2:
        st.markdown(f"### 🧪 3-Month Backtest Analysis (Fixed {inv_amount}$)")
        if st.button("⚙️ Start Simulation"):
            with st.spinner("Processing..."):
                data = yf.download(WATCHLIST + ['SPY'], start=(datetime.now()-timedelta(days=340)), progress=False)
                df_e, df_tr, tot_inv, tot_p, tot_l = run_v7_backtest(data, inv_amount)
                net = tot_p - tot_l; roi = (net / 10000.0) * 100
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Invested", f"${tot_inv:,.0f}"); c2.metric("Net Profit", f"${net:,.0f}", f"{roi:.1f}% ROI")
                c3.metric("Profit Factor", f"{round(tot_p/tot_l, 2) if tot_l > 0 else 'N/A'}"); c4.metric("Grade", "9/10 🏆" if roi > 8 else "7/10 🟢")
                st.line_chart(df_e); st.dataframe(df_tr)
