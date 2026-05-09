import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import urllib.request
import xml.etree.ElementTree as ET
import warnings

# 1. הגדרת עמוד
st.set_page_config(page_title="SwingHunter V4.3 - Command Center", layout="wide")
warnings.filterwarnings('ignore')

# ==========================================
# 2. אבטחה והגדרות
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

# ==========================================
# 3. פונקציות עזר
# ==========================================
def get_market_context():
    try:
        spy = yf.download('SPY', period='50d', progress=False)
        last_p = float(spy['Close'].iloc[-1])
        sma20 = float(spy['Close'].rolling(20).mean().iloc[-1])
        return "BULL" if last_p > sma20 else "BEAR"
    except: return "UNKNOWN"

def get_headlines_sentiment(ticker):
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp: xml_page = resp.read()
        root = ET.fromstring(xml_page)
        items = root.findall('.//item')
        if not items: return "⚪ ניטרלי"
        news_text = " | ".join([item.find('title').text for item in items[:2]])
        pos_words = ['up', 'surge', 'jump', 'beat', 'growth', 'upgrade', 'profit']
        neg_words = ['down', 'plunge', 'drop', 'miss', 'cut', 'downgrade', 'loss']
        t_low = news_text.lower()
        pos_c = sum(1 for w in pos_words if w in t_low)
        neg_c = sum(1 for w in neg_words if w in t_low)
        return "🟢 חיובי" if pos_c > neg_c else ("🔴 שלילי" if neg_c > pos_c else "⚪ ניטרלי")
    except: return "⚪ לא ידוע"

def get_earnings_warning(ticker):
    try:
        tkr = yf.Ticker(ticker)
        cal = tkr.get_earnings_dates(limit=3)
        if cal is not None and not cal.empty:
            now = pd.Timestamp.now(tz='UTC')
            future_dates = cal.index[cal.index > now]
            if not future_dates.empty:
                days = (future_dates[0] - now).days
                if 0 <= days <= 3: return True
    except: pass
    return False

# ==========================================
# 4. המנוע המרכזי (The Command Engine V4.3)
# ==========================================
def run_full_analysis(ticker, market_trend, risk_budget):
    try:
        df = yf.download(ticker, period='250d', progress=False)
        if df.empty or len(df) < 200: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        close, highs, lows = df['Close'], df['High'], df['Low']
        last_price = float(close.iloc[-1])
        sma200 = close.rolling(200).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        res_20d = float(highs.iloc[-21:-1].max()) 
        
        tr = pd.concat([highs-lows, (highs-close.shift()).abs(), (lows-close.shift()).abs()], axis=1).max(axis=1)
        atr_val = tr.rolling(14).mean().iloc[-1]
        rel_vol = float(df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1])
        change_20d = float((last_price / close.iloc[-20] - 1) * 100)
        
        delta = close.diff()
        rsi_val = 100 - (100 / (1 + ((delta.where(delta > 0, 0)).rolling(14).mean() / (-delta.where(delta < 0, 0)).rolling(14).mean()).iloc[-1]))

        # לוגיקת סיווג בסיסית לסורק
        score = 0
        decision, icon, setup = "לא לגעת", "🔴", "No Setup"
        instruction = "אין טריגר ברור."
        is_hot = rsi_val > 78 or change_20d > 30
        has_earn = get_earnings_warning(ticker)

        if market_trend == "BULL": score += 10
        if last_price > sma200: score += 15
        else: score -= 20

        if is_hot:
            setup, icon, decision = "Overextended", "🔥", "חם מדי"
            instruction = "סכנת רדיפה."
            score -= 40
        elif last_price > res_20d:
            setup = "Breakout"
            if rel_vol > 1.5:
                score += 30; icon, decision = "🟢", "סטאפ פעיל"
                instruction = "פריצה עם נפח."
            else:
                icon, decision = "🟡", "למעקב"; score += 5
                instruction = "מעל התנגדות ללא נפח."
        elif (res_20d / last_price - 1) < 0.03:
            setup, icon, decision = "Near Resistance", "🟡", "למעקב"
            score += 5; instruction = f"להמתין לפריצה מעל {round(res_20d, 2)}."
        elif last_price > ema21 * 0.99 and last_price < ema21 * 1.03:
            setup, icon, decision = "Pullback", "🟡", "למעקב"
            score += 15; instruction = "תיקון לממוצע 21."

        if rel_vol > 2.0: score += 15
        if has_earn: score -= 15

        # לוגיקת פקודות ביצוע חסינה (Order Generation)
        order_data = None
        p_type = None
        
        if score >= 40 and not is_hot and not has_earn and market_trend == "BULL" and last_price > sma200:
            if setup in ["Breakout", "Near Resistance"]:
                entry = round(res_20d * 1.002, 2)
                # תיקון קריטי: Buy Stop חייב להיות מעל המחיר הנוכחי
                if last_price < entry:
                    l_max = round(res_20d * 1.008, 2)
                    e_disp = f"Stop {entry} / Lmt {l_max}"
                    p_type = "BUY STOP LIMIT"
                else:
                    instruction = "כבר פרצה — לא לרדוף. המתן לפולבק." # מבטל את הפקודה ומעדכן את הסורק
            
            elif setup == "Pullback":
                # תיקון קריטי: Buy Limit חייב להיות מתחת או שווה למחיר הנוכחי
                entry = min(round(ema21 * 1.005, 2), round(last_price * 0.995, 2))
                e_disp = f"{entry}"
                p_type = "BUY LIMIT"

            # אם נוצרה פקודה חוקית, מחשבים סיכון
            if p_type:
                stop = round(min(ema21, entry - (1.5 * atr_val)), 2)
                risk_ps = entry - stop
                
                if risk_ps > 0:
                    risk_p = (risk_ps / entry) * 100
                    rr = (entry * 0.10) / risk_ps # R/R לפי יעד של 10%
                    
                    # סינון קפדני: R/R מעל 1.5, וסיכון לא גדול מ-7%
                    if risk_p <= 7.0 and rr >= 1.5:
                        shares = int(risk_budget / risk_ps)
                        if shares > 0:
                            actual_risk_dollars = round(shares * risk_ps, 2)
                            
                            order_data = {
                                'מניה': ticker,
                                'פעולה': p_type,
                                'תוקף': 'DAY ONLY',
                                'כניסה': e_disp,
                                'כמות': shares,
                                'סיכון $': f"${actual_risk_dollars}",
                                'יעד 10%': round(entry * 1.10, 2),
                                'יעד 15%': round(entry * 1.15, 2),
                                'סטופ': stop,
                                'סיכון %': f"{round(risk_p,1)}%",
                                'R/R': round(rr, 2)
                            }

        return {
            'scanner': {
                'החלטה': f"{icon} {decision}", 'מניה': ticker, 'ציון': int(max(0, score)),
                'סוג סטאפ': setup, 'הוראה': instruction, 'מחיר': round(last_price, 2),
                'RSI': int(rsi_val), 'סנטימנט': get_headlines_sentiment(ticker) if score > 20 else "-"
            },
            'order': order_data
        }
    except: return None

# ==========================================
# 5. UI וממשק משולב
# ==========================================
if "authenticated" not in st.session_state: st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 SwingHunter Access")
    pwd = st.text_input("סיסמה:", type="password")
    if st.button("כניסה"):
        if pwd == APP_PASSWORD: st.session_state["authenticated"] = True; st.rerun()
else:
    st.markdown("<h1 style='text-align: right;'>🎮 מרכז בקרה - SwingHunter V4.3</h1>", unsafe_allow_html=True)
    
    st.sidebar.header("ניהול סיכונים")
    risk_sum = st.sidebar.number_input("סיכון לעסקה ($)", value=150, step=50)
    
    if st.button("🚀 הרץ ניתוח משולב", use_container_width=True):
        with st.spinner("מנתח שוק ומייצר פקודות עבודה מוגנות..."):
            market_trend = get_market_context()
            raw_results = [run_full_analysis(t, market_trend, risk_sum) for t in WATCHLIST]
            raw_results = [r for r in raw_results if r is not None]
        
        # 1. טבלת פקודות
        order_list = [r['order'] for r in raw_results if r['order'] is not None]
        st.markdown("### 📝 פקודות לביצוע היום (Bracket Orders)")
        if order_list:
            df_orders = pd.DataFrame(order_list).sort_values(by="R/R", ascending=False).head(3)
            # הסתרת אינדקס הטבלה למראה נקי יותר
            st.dataframe(df_orders.style.hide(axis="index"), use_container_width=True)
        else:
            st.info("אין פקודות ביצוע שתואמות את כל תנאי הסף להיום. עדיף להמתין.")

        # 2. טבלת סורק 
        st.markdown("---")
        st.markdown("### 🔍 סורק שוק מלא (Watchlist & Analysis)")
        scanner_list = [r['scanner'] for r in raw_results]
        df_scan = pd.DataFrame(scanner_list).sort_values(by="ציון", ascending=False)
        
        def highlight_rows(row):
            val = str(row['החלטה'])
            if "פעיל" in val: return ['background-color: rgba(46, 204, 113, 0.2)'] * len(row)
            if "למעקב" in val: return ['background-color: rgba(241, 196, 15, 0.1)'] * len(row)
            return [''] * len(row)

        st.dataframe(df_scan.style.apply(highlight_rows, axis=1), use_container_width=True)
        st.write(f"📈 **מצב שוק:** {market_trend}")
