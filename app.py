import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import urllib.request
import xml.etree.ElementTree as ET
import warnings

# 1. הגדרת עמוד (חייב להיות ראשון)
st.set_page_config(page_title="SwingHunter Pro V3.5", layout="wide")
warnings.filterwarnings('ignore')

# ==========================================
# 2. אבטחה והגדרות
# ==========================================
try:
    APP_PASSWORD = st.secrets["APP_PASSWORD"]
    MY_EMAIL = st.secrets["MY_EMAIL"]
except:
    APP_PASSWORD = "YOUR_WEBSITE_PASSWORD" 
    MY_EMAIL = "your_email@gmail.com"

WATCHLIST = [
    'AAPL','MSFT','NVDA','TSLA','AMZN','META','GOOGL','NFLX','AMD','AVGO','TSM','QCOM',
    'CRWD','PANW','PLTR','SNOW','DDOG','NET','SMCI','COIN','MSTR','HOOD','SOFI','SQ',
    'PYPL','AFRM','SHOP','BABA','MELI','WMT','TGT','COST','HD','UBER','ABNB','SPOT',
    'DKNG','DIS','NKE','SBUX','MCD','JPM','BAC','GS','MS','V','MA','LLY','NVO','UNH','CAT','BA'
]

# ==========================================
# 3. פונקציות עזר (שוק, חדשות, דוחות)
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
                if 0 <= days <= 7: return True, days
    except: pass
    return False, -1

# ==========================================
# 4. מנוע הניתוח המרכזי
# ==========================================
def analyze_ticker(ticker, market_trend):
    try:
        df = yf.download(ticker, period='250d', progress=False)
        if df.empty or len(df) < 200: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        close, highs, lows = df['Close'], df['High'], df['Low']
        last_price = float(close.iloc[-1])
        
        # אינדיקטורים
        sma200 = close.rolling(200).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        res_20d = float(highs.iloc[-21:-1].max()) 
        
        # ATR תקין
        tr = pd.concat([highs-lows, (highs-close.shift()).abs(), (lows-close.shift()).abs()], axis=1).max(axis=1)
        atr_val = tr.rolling(14).mean().iloc[-1]
        
        rel_vol = float(df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1])
        change_20d = float((last_price / close.iloc[-20] - 1) * 100)
        
        # RSI
        delta = close.diff()
        rsi_val = 100 - (100 / (1 + ((delta.where(delta > 0, 0)).rolling(14).mean() / (-delta.where(delta < 0, 0)).rolling(14).mean()).iloc[-1]))

        # ניקוד (מתחיל מ-0)
        score = 0
        reasons = []
        decision, icon, setup = "לא לגעת", "🔴", "No Setup"
        instruction = "אין טריגר ברור."

        if market_trend == "BULL": score += 10
        if last_price > sma200: score += 15
        else: score -= 20; reasons.append("מתחת ל-SMA200")

        # איתור סטאפ
        is_hot = rsi_val > 78 or change_20d > 30
        if is_hot:
            setup, icon, decision = "Overextended", "🔥", "חם מדי"
            instruction = "מתוחה מדי. סכנת רדיפה."
            score -= 40
        else:
            if last_price > res_20d:
                setup = "Breakout"
                if rel_vol > 1.5:
                    score += 30; icon, decision = "🟢", "סטאפ פעיל"
                    instruction = "פריצה עם נפח חזק."
                else:
                    icon, decision = "🟡", "למעקב"
                    instruction = f"מעל התנגדות ללא נפח ({round(rel_vol,1)})."
            elif (res_20d / last_price - 1) < 0.03:
                setup, icon, decision = "Near Resistance", "🟡", "למעקב"
                instruction = f"להמתין לפריצה מעל {round(res_20d, 2)}."
            elif last_price > ema21 * 0.99 and last_price < ema21 * 1.03:
                setup, icon, decision = "Pullback", "🟡", "למעקב"
                score += 15; instruction = "תיקון לממוצע 21. מחפש היפוך."

        if rel_vol > 2.0: score += 20
        has_earn, earn_days = get_earnings_warning(ticker)
        if has_earn: score -= 15; instruction += f" | דוח עוד {earn_days} ימים!"

        # יעדים ו-R/R
        target_10 = last_price * 1.10
        stop_tight = ema21
        stop_wide = last_price - (atr_val * 1.5)
        
        rr_tight = (target_10 - last_price) / (last_price - stop_tight) if last_price > stop_tight else 0
        
        return {
            'החלטה': f"{icon} {decision}",
            'מניה': ticker,
            'ציון': int(max(0, score)),
            'סוג סטאפ': setup,
            'הוראה': instruction,
            'מחיר': round(last_price, 2),
            'יעד 10%': round(target_10, 2),
            'סטופ': round(stop_tight, 2),
            'R/R (יעד 10%)': round(rr_tight, 2),
            'סנטימנט': get_headlines_sentiment(ticker) if score > 35 else "-"
        }
    except: return None

# ==========================================
# 5. ממשק משתמש ותיקון באג העיצוב
# ==========================================
if "authenticated" not in st.session_state: st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 SwingHunter Access")
    pwd = st.text_input("סיסמה:", type="password")
    if st.button("כניסה"):
        if pwd == APP_PASSWORD: st.session_state["authenticated"] = True; st.rerun()
        else: st.error("שגויה")
else:
    st.title("🎯 SwingHunter Pro V3.5")
    if st.button("🚀 הרץ סריקה", use_container_width=True):
        with st.spinner("סורק מניות ומחשב R/R..."):
            market_trend = get_market_context()
            results = [analyze_ticker(t, market_trend) for t in WATCHLIST]
            results = [r for r in results if r is not None]
        
        if results:
            df_res = pd.DataFrame(results).sort_values(by="ציון", ascending=False)
            st.write(f"📈 **מצב שוק:** {market_trend}")
            
            # פונקציית צביעה חסינה (מטפלת ב-NaN ובמספרים)
            def highlight_active(row):
                # הופך למחרוזת כדי למנוע את ה-TypeError
                val = str(row['החלטה'])
                if "פעיל" in val:
                    return ['background-color: rgba(46, 204, 113, 0.2)'] * len(row)
                return [''] * len(row)

            st.dataframe(df_res.style.apply(highlight_active, axis=1), use_container_width=True)
