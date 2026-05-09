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

# 1. הגדרה ראשונית
st.set_page_config(page_title="SwingHunter Pro V3.4", layout="wide")
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
# 4. המנוע המרכזי - V3.4 (Precision Engine)
# ==========================================
def analyze_ticker(ticker, market_trend):
    try:
        df = yf.download(ticker, period='250d', progress=False)
        if df.empty or len(df) < 200: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        close = df['Close']
        highs = df['High']
        lows = df['Low']
        last_price = float(close.iloc[-1])
        
        # אינדיקטורים
        sma200 = close.rolling(200).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        res_20d = float(highs.iloc[-21:-1].max()) 
        
        # חישוב ATR תקין (תיקון הבאג)
        tr = pd.concat([
            highs - lows,
            (highs - close.shift()).abs(),
            (lows - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr_val = tr.rolling(14).mean().iloc[-1]
        
        # RSI ונתוני ווליום
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi_val = 100 - (100 / (1 + (gain/loss).iloc[-1]))
        rel_vol = float(df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1])
        change_20d = float((last_price / close.iloc[-20] - 1) * 100)

        # --- לוגיקת ניקוד וסיווג ---
        score = 0
        reasons = []
        decision_text = "לא לגעת"
        decision_icon = "🔴"
        setup_type = "No Setup"
        instruction = "אין טריגר ברור כרגע."

        # בדיקת דוחות (סיכון בינארי)
        has_earn, earn_days = get_earnings_warning(ticker)

        # 1. Market & Trend
        if market_trend == "BULL": score += 10
        if last_price > sma200: score += 15
        else: score -= 20; reasons.append("מתחת ל-SMA200")

        # 2. Setup Identification
        is_overextended = rsi_val > 78 or change_20d > 30

        if is_overextended:
            setup_type = "Overextended"
            decision_icon = "🔥"
            decision_text = "חם מדי"
            instruction = f"מניה מתוחה מדי ({round(change_20d)}%). להתרחק."
            score -= 40
        else:
            # פריצה
            if last_price > res_20d:
                setup_type = "Breakout"
                if rel_vol > 1.5:
                    score += 30
                    decision_icon = "🟢"
                    decision_text = "סטאפ פעיל"
                    instruction = "פריצה עם נפח. דרך פתוחה ליעדים."
                else:
                    score += 5
                    decision_icon = "🟡"
                    decision_text = "למעקב"
                    instruction = "מעל התנגדות אך ללא נפח (RelVol: {:.1f}).".format(rel_vol)
            # קרוב להתנגדות
            elif (res_20d / last_price - 1) < 0.03:
                setup_type = "Near Resistance"
                decision_icon = "🟡"
                decision_text = "למעקב"
                instruction = f"להמתין לפריצה מעל {round(res_20d, 2)}."
            # תיקון לממוצע 21
            elif last_price > ema21 * 0.99 and last_price < ema21 * 1.03:
                setup_type = "Pullback to EMA21"
                score += 15
                decision_icon = "🟡"
                decision_text = "למעקב"
                instruction = "תיקון לממוצע 21. מחפש היפוך מומנטום."

        # 3. Volume & Earnings Penalties
        if rel_vol > 2.0: score += 20
        elif rel_vol > 1.5: score += 10
        
        if has_earn:
            score -= 15
            reasons.append(f"דוח בעוד {earn_days} ימים")
            instruction += " | ❗ זהירות: דוח קרוב."

        # 4. ניהול סיכונים (יעד 10% וסטופים)
        target_10 = last_price * 1.10
        stop_tight = ema21 
        stop_wide = last_price - (atr_val * 1.5)
        
        # חישוב R/R (Risk/Reward)
        risk_tight = last_price - stop_tight
        risk_wide = last_price - stop_wide
        reward = target_10 - last_price
        
        rr_tight = reward / risk_tight if risk_tight > 0 else 0
        rr_wide = reward / risk_wide if risk_wide > 0 else 0
        
        if rr_tight > 1.5: score += 10

        status = "✅ עובר" if score >= 60 else "❌ נפסל"
        sentiment = get_headlines_sentiment(ticker) if score > 35 else "⚪-"

        return {
            'החלטה': f"{decision_icon} {decision_text}",
            'מניה': ticker,
            'ציון': int(max(0, score)),
            'סוג סטאפ': setup_type,
            'הוראה': instruction,
            'מחיר': round(last_price, 2),
            'יעד 10%': round(target_10, 2),
            'סטופ הדוק': round(stop_tight, 2),
            'סטופ רחב': round(stop_wide, 2),
            'R/R (הדוק)': round(rr_tight, 2),
            'R/R (רחב)': round(rr_wide, 2),
            'סיבת פסילה': ", ".join(reasons) if reasons else "תקין",
            'סנטימנט': sentiment
        }
    except: return None

# ==========================================
# 5. UI וממשק
# ==========================================
if "authenticated" not in st.session_state: st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 SwingHunter Access")
    pwd = st.text_input("סיסמה:", type="password")
    if st.button("כניסה"):
        if pwd == APP_PASSWORD: st.session_state["authenticated"] = True; st.rerun()
        else: st.error("שגויה")
else:
    st.title("🎯 SwingHunter Pro V3.4")
    st.sidebar.header("הגדרות")
    email_pw = st.sidebar.text_input("Gmail App Password:", type="password")
    
    if st.button("🚀 הרץ סריקה (Precision Engine)", use_container_width=True):
        with st.spinner("מנתח מרווחי ATR ומודד R/R..."):
            market_trend = get_market_context()
            results = []
            bar = st.progress(0)
            for i, t in enumerate(WATCHLIST):
                res = analyze_ticker(t, market_trend)
                if res: results.append(res)
                bar.progress((i+1)/len(WATCHLIST))
        
        if results:
            df_res = pd.DataFrame(results).sort_values(by="ציון", ascending=False)
            st.write(f"📈 **מצב שוק:** {market_trend}")
            
            # עיצוב טבלה
            def color_dec(val):
                if "פעיל" in val: return 'background-color: #2ecc7133; color: #2ecc71; font-weight: bold'
                if "למעקב" in val: return 'color: #f1c40f; font-weight: bold'
                if "חם מדי" in val: return 'color: #e74c3c; font-weight: bold'
                return ''

            st.dataframe(df_res.style.applymap(color_dec, subset=['החלטה']), use_container_width=True)
