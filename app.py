import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import urllib.request
import xml.etree.ElementTree as ET
import warnings

st.set_page_config(page_title="SwingHunter V6.1 - Edge Engine", layout="wide")
warnings.filterwarnings('ignore')

# ==========================================
# 1. הגדרות וניהול היסטוריה
# ==========================================
try:
    APP_PASSWORD = st.secrets["APP_PASSWORD"]
    MY_EMAIL = st.secrets["MY_EMAIL"]
except:
    APP_PASSWORD = "Pk0105Ak2701" 
    MY_EMAIL = "orel@peleg-eng.com"

WATCHLIST = [
    'AAPL','MSFT','NVDA','TSLA','AMZN','META','GOOGL','NFLX','AMD','AVGO','TSM','QCOM',
    'CRWD','PANW','PLTR','SNOW','DDOG','NET','SMCI','COIN','MSTR','HOOD','SOFI','SQ',
    'PYPL','AFRM','SHOP','BABA','MELI','WMT','TGT','COST','HD','UBER','ABNB','SPOT',
    'DKNG','DIS','NKE','SBUX','MCD','JPM','BAC','GS','MS','V','MA','LLY','NVO','UNH','CAT','BA','MRNA'
]

HISTORY_FILE = "swinghunter_history.csv"

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
        watch_days = recent["החלטה"].astype(str).str.contains("למעקב|פעיל|ARMED").sum()
        score_trend = recent["ציון_כולל"].diff().sum()
        return watch_days, score_trend
    except: return 0, 0

# ==========================================
# 2. פונקציות שוק, דוחות וסנטימנט
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
# 3. מודולי Edge משופרים (V6.1)
# ==========================================
def compression_score(df):
    try:
        high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]
        range_5 = ((high.tail(5).max() - low.tail(5).min()) / close.iloc[-1]) * 100
        avg_range_20 = (((high - low) / close).rolling(20).mean().iloc[-1]) * 100
        vol_5 = vol.tail(5).mean()
        vol_20 = vol.rolling(20).mean().iloc[-1]
        recent_lows = low.tail(5).values
        higher_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] >= recent_lows[i-1])

        score = 0
        if range_5 < (avg_range_20 * 0.75): score += 20
        if vol_5 < (vol_20 * 0.85): score += 10
        if higher_lows >= 3: score += 15
        return score
    except: return 0

def relative_strength_vs_spy(close, spy_close):
    try:
        s_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
        s_20d = (close.iloc[-1] / close.iloc[-21] - 1) * 100
        spy_5d = (spy_close.iloc[-1] / spy_close.iloc[-6] - 1) * 100
        spy_20d = (spy_close.iloc[-1] / spy_close.iloc[-21] - 1) * 100
        return s_5d - spy_5d, s_20d - spy_20d
    except: return 0, 0

def failed_breakdown_recovery(df):
    try:
        close, low = df["Close"], df["Low"]
        sma5 = close.rolling(5).mean().iloc[-1]
        support_20 = float(low.iloc[-21:-5].min())
        broke_support = float(low.iloc[-5:-1].min()) < support_20 * 0.99
        # דורש גם חזרה מעל התמיכה וגם מעל ממוצע 5 כדי לוודא עוצמה
        reclaimed = float(close.iloc[-1]) > support_20 * 1.005 and float(close.iloc[-1]) > sma5
        if broke_support and reclaimed: return True, support_20
    except: pass
    return False, 0

def post_event_drift(df):
    try:
        close, open_p, high, low, vol = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
        recent_vol = vol.tail(10)
        avg_vol = vol.rolling(20).mean().iloc[-1]
        
        if recent_vol.max() > avg_vol * 2.5:
            event_idx = recent_vol.idxmax()
            event_pos = df.index.get_loc(event_idx)
            
            if event_pos < len(df) - 1:
                event_open = float(open_p.iloc[event_pos])
                event_close = float(close.iloc[event_pos])
                event_return = (event_close / event_open - 1) * 100
                
                # פסילה אם יום האירוע היה התרסקות מוחלטת
                if event_return < -5.0: return False, 0, 0
                
                event_high = float(high.iloc[event_pos])
                event_low = float(low.iloc[event_pos])
                
                holds_above_low = float(close.iloc[-1]) > event_low
                higher_lows = low.tail(3).is_monotonic_increasing
                
                if holds_above_low and higher_lows: return True, event_high, event_low
    except: pass
    return False, 0, 0

# ==========================================
# 4. מנוע האנליזה וההחלטות
# ==========================================
def analyze_edge(ticker, spy_close, market_trend, risk_budget):
    try:
        df = yf.download(ticker, period='250d', progress=False)
        if df.empty or len(df) < 200: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        close, highs, lows = df['Close'], df['High'], df['Low']
        last_price = float(close.iloc[-1])
        
        sma200 = close.rolling(200).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        res_20d = float(highs.iloc[-21:-1].max()) 
        dist_to_res = (res_20d / last_price - 1) * 100
        
        tr = pd.concat([highs-lows, (highs-close.shift()).abs(), (lows-close.shift()).abs()], axis=1).max(axis=1)
        atr_val = tr.rolling(14).mean().iloc[-1]
        
        delta = close.diff()
        rsi_val = 100 - (100 / (1 + ((delta.where(delta > 0, 0)).rolling(14).mean() / (-delta.where(delta < 0, 0)).rolling(14).mean()).iloc[-1]))

        # --- מודולי Edge ---
        raw_comp_score = compression_score(df)
        rs_5d, rs_20d = relative_strength_vs_spy(close, spy_close)
        failed_break, reclaimed_level = failed_breakdown_recovery(df)
        drift, event_high, event_low = post_event_drift(df)
        watch_days, score_trend = get_setup_persistence(ticker)

        # ----------------------------------------
        # חישוב ציונים ו-Edge Notes
        # ----------------------------------------
        setup_score = 0
        if last_price > sma200: setup_score += 15
        if last_price > ema21: setup_score += 10
        if dist_to_res < 4.0: setup_score += 15
        
        edge_score = 0
        edge_notes = []
        reject_reason = ""
        
        # התניית Compression - תופס רק אם קרוב להתנגדות או חזק מהשוק
        if raw_comp_score >= 25:
            if dist_to_res < 4.0 or rs_5d > 0 or rs_20d > 0:
                edge_score += raw_comp_score
                edge_notes.append("Compression")
                
        if rs_5d > 2.0 or rs_20d > 5.0:
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

        # ----------------------------------------
        # שומרי סף וסיווג (Guards)
        # ----------------------------------------
        final_rank = setup_score + edge_score
        setup_type = "No Setup"
        instruction = "אין טריגר."
        
        is_overextended = rsi_val > 78
        earn_status, earn_days = get_earnings_status(ticker)

        if earn_status == "DANGER":
            icon, decision = "⚠️", "DANGER"
            instruction = f"דוח כספי בעוד {earn_days} ימים. סכנה בינארית."
            reject_reason = "דוח קרוב"
            final_rank = 0
        elif is_overextended:
            icon, decision = "🔥", "חם מדי"
            instruction = "סכנת רדיפה."
            reject_reason = "RSI גבוה מ-78"
            final_rank = min(final_rank, 30)
        elif market_trend == "BEAR" and edge_score < 40:
            icon, decision = "🔴", "Dormant"
            instruction = "שוק חלש. המניה חסרת אדג' קיצוני."
            reject_reason = "Market BEAR + Edge נמוך"
            final_rank = min(final_rank, 40)
        elif final_rank >= 80 and len(edge_notes) > 0:
            icon, decision = "🟢", "ARMED"
        elif final_rank >= 45:
            icon, decision = "🟡", "Building Pressure"
        else:
            icon, decision = "🔴", "Dormant"
            reject_reason = "ציון נמוך מדי"

        # הגדרת התבנית
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

        # ----------------------------------------
        # ייצור פקודות (רק למצב ARMED שעבר את שומרי הסף)
        # ----------------------------------------
        order_data = None
        if decision == "ARMED":
            p_type, entry, e_disp = None, 0, ""
            
            if setup_type in ["Pre-Breakout Compression", "Post-Event Drift", "Momentum Breakout"]:
                trigger = res_20d if setup_type != "Post-Event Drift" else event_high
                entry = round(trigger * 1.002, 2)
                if last_price < entry:
                    p_type = "BUY STOP LIMIT"
                    e_disp = f"Stop {entry} / Lmt {round(entry*1.008, 2)}"
                else:
                    reject_reason = "כבר פרצה. לא רודפים."
                    icon, decision = "🟡", "Building Pressure" # מוריד סטטוס
            
            elif setup_type in ["Failed Breakdown Recovery", "RS Pullback"]:
                entry = min(round(ema21 * 1.005, 2), round(last_price * 0.995, 2))
                p_type = "BUY LIMIT"
                e_disp = f"{entry}"

            if p_type:
                stop = round(min(ema21, entry - (1.5 * atr_val)), 2)
                if setup_type == "Failed Breakdown Recovery": stop = round(reclaimed_level * 0.985, 2)
                
                risk_ps = entry - stop
                if risk_ps > 0:
                    risk_p = (risk_ps / entry) * 100
                    rr = (entry * 0.10) / risk_ps 
                    
                    if risk_p <= 7.0 and rr >= 1.5:
                        shares = int(risk_budget / risk_ps)
                        if shares > 0:
                            order_data = {
                                'מניה': ticker, 'פעולה': p_type,
                                'Edge': " + ".join(edge_notes),
                                'כניסה': e_disp, 'כמות': shares, 
                                'יעד 10%': round(entry * 1.10, 2), 'יעד 15%': round(entry * 1.15, 2),
                                'סטופ': stop, 'סיכון $': f"${round(shares * risk_ps, 2)}",
                                'R/R': round(rr, 2), 'תוקף': 'DAY ONLY'
                            }
                    else:
                        reject_reason = f"R/R נמוך ({round(rr,1)}) או סיכון גבוה"

        return {
            'scanner': {
                'מניה': ticker, 'החלטה': f"{icon} {decision}", 'ציון_כולל': int(final_rank),
                'תבנית': setup_type, 'Market': market_trend, 'Watch Days': watch_days,
                'Comp. Score': raw_comp_score, 'RS (5D)': f"{round(rs_5d,1)}%", 'RS (20D)': f"{round(rs_20d,1)}%",
                'Edge Notes': " + ".join(edge_notes) if edge_notes else "None",
                'סיבת פסילה': reject_reason
            },
            'order': order_data
        }
    except: return None

# ==========================================
# 5. UI
# ==========================================
if "authenticated" not in st.session_state: st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 Edge Engine Access")
    pwd = st.text_input("סיסמה:", type="password")
    if st.button("כניסה"):
        if pwd == APP_PASSWORD: st.session_state["authenticated"] = True; st.rerun()
else:
    st.markdown("<h1 style='text-align: right;'>🧠 SwingHunter V6.1 - The Fortress Edition</h1>", unsafe_allow_html=True)
    
    st.sidebar.header("ניהול סיכונים")
    risk_sum = st.sidebar.number_input("סיכון מקסימלי לעסקה ($)", value=150, step=50)
    
    if st.button("🚀 הרץ אנליזת פרימיום", use_container_width=True):
        with st.spinner("מחשב אדג', בודק יומן דוחות ומאמת מבנה מחירים..."):
            spy_close, market_trend = get_spy_context()
            if spy_close is not None:
                raw_results = [analyze_edge(t, spy_close, market_trend, risk_sum) for t in WATCHLIST]
                raw_results = [r for r in raw_results if r is not None]
            
                # 1. פקודות היום
                order_list = [r['order'] for r in raw_results if r['order'] is not None]
                st.markdown(f"### 📝 פקודות ביצוע מוגנות (Market: {market_trend})")
                if order_list:
                    df_orders = pd.DataFrame(order_list).sort_values(by="R/R", ascending=False).head(3)
                    st.dataframe(df_orders.style.hide(axis="index"), use_container_width=True)
                else:
                    st.info("אין היום פקודות שעברו את כל שומרי הסף (דוחות, R/R, ומצב שוק).")

                # 2. סורק מעקב
                st.markdown("---")
                st.markdown("### 🔍 רדאר התהוות וניתוח פסילות")
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
            else:
                st.error("שגיאה במשיכת נתוני SPY (שוק כללי).")
