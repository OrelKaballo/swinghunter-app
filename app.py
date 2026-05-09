import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. הגדרות משתמש (User Settings) - ערוך כאן!
# ==========================================
APP_PASSWORD = "YOUR_WEBSITE_PASSWORD" # שנה לסיסמה שתשמש אותך לכניסה לאתר
MY_EMAIL = "your_email@gmail.com"      # המייל שלך (שולח ומקבל)

# רשימת מניות חזקות וסחירות (אפשר להוסיף/להוריד)
WATCHLIST = [
    'AAPL','MSFT','NVDA','TSLA','AMZN','META','GOOGL','AMD','AVGO','PLTR',
    'CRWD','PANW','SMCI','COIN','MSTR','HOOD','SOFI','RIVN','UBER','SHOP',
    'SQ','NFLX','DDOG','SNOW','NET','ROKU','AFRM','PYPL','MRVL','INTC',
    'QCOM','TSM','BABA','CRM','NOW','UBER','ABNB','SPOT','DKNG','MARA'
]

# ==========================================
# 2. פונקציות טכניות (The Brain / Engine)
# ==========================================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def get_market_context():
    """בודק את מצב השוק לפי ה-S&P 500"""
    try:
        spy = yf.download('SPY', period='50d', progress=False)
        last_close = spy['Close'].iloc[-1].item() if isinstance(spy['Close'].iloc[-1], pd.Series) else float(spy['Close'].iloc[-1])
        sma20 = spy['Close'].rolling(20).mean().iloc[-1].item() if isinstance(spy['Close'].rolling(20).mean().iloc[-1], pd.Series) else float(spy['Close'].rolling(20).mean().iloc[-1])
        trend = "BULLISH" if last_close > sma20 else "BEARISH"
        return trend
    except:
        return "UNKNOWN"

def analyze_ticker(ticker, market_trend):
    """מנתח מניה בודדת ומחזיר ציון ויעדים דינמיים"""
    try:
        df = yf.download(ticker, period='180d', progress=False)
        if df.empty or len(df) < 60: return None
        
        # השטחת כותרות אם yfinance מחזיר MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df['Close']
        high = df['High']
        low = df['Low']
        vol = df['Volume']

        # חישוב מתנדים
        df['SMA20'] = close.rolling(20).mean()
        df['EMA8'] = close.ewm(span=8, adjust=False).mean()
        df['EMA21'] = close.ewm(span=21, adjust=False).mean()
        df['RSI14'] = rsi(close, 14)
        df['ATR14'] = atr(high, low, close, 14)
        df['Vol20'] = vol.rolling(20).mean()
        df['RelVol'] = vol / df['Vol20']

        last = df.iloc[-1]
        price = float(last['Close'])
        avg_vol = float(last['Vol20'])
        atr_val = float(last['ATR14'])

        # סינון נפח ומחיר מינימלי
        if price < 5 or avg_vol < 1_000_000 or pd.isna(atr_val): return None

        # -- ניהול סיכונים דינמי (ATR) --
        # סטופ לוס: 1.5 פעמים ATR מתחת למחיר
        stop = price - (atr_val * 1.5)
        # יעדים מבוססי תנודתיות
        target1 = price + (atr_val * 2.0)
        target2 = price + (atr_val * 3.5)
        
        risk = price - stop
        reward = target1 - price
        rr = reward / risk if risk > 0 else 0

        # -- לוגיקת סטאפים משופרת --
        setup = []
        # פריצה עם ווליום עולה בנר ירוק
        if price > df['High'].iloc[-2] and last['RelVol'] > 1.2 and price > last['Open']:
            setup.append('Momentum Breakout')
        # מגמה חזקה לטווח קצר
        if price > last['EMA8'] and last['EMA8'] > last['EMA21']:
            setup.append('Strong Trend (8/21)')
        # תיקון תנודתי (Pullback) מעל תמיכה
        if last['RSI14'] < 40 and price > last['SMA20']:
            setup.append('SMA20 Pullback')

        if not setup: return None # מסנן מניות ללא סטאפ
        setup_txt = ' + '.join(setup)

        # -- שיטת ניקוד חדשה וקשוחה --
        score = 50 # ציון התחלתי

        # בונוסים
        if 'Breakout' in setup_txt: score += 15
        if last['RelVol'] > 1.5: score += 15
        if rr >= 2.0: score += 20
        elif rr >= 1.5: score += 10

        # קנסות (ניהול סיכונים)
        if market_trend == "BEARISH": score -= 20 # השוק נגדנו
        if rr < 1.3: score -= 30 # פוסל טריידים מסוכנים מדי
        if last['RSI14'] > 75: score -= 15 # קניית יתר
        if price > last['SMA20'] + (atr_val * 2): score -= 20 # מתוח מדי

        score = max(0, min(100, score)) # תוחם בין 0 ל-100

        if score < 65: return None # מחזיר רק מניות חזקות

        return {
            'Ticker': ticker,
            'Score': int(score),
            'Setup': setup_txt,
            'Price': round(price, 2),
            'Target 1 (ATR)': round(target1, 2),
            'Stop Loss': round(stop, 2),
            'Risk/Reward': round(rr, 2),
            'RSI': round(float(last['RSI14']), 1),
            'RelVol': round(float(last['RelVol']), 2)
        }
    except Exception as e:
        return None

# ==========================================
# 3. פונקציית שליחת המייל
# ==========================================
def send_email_report(df, email_pw):
    msg = MIMEMultipart()
    msg['From'] = MY_EMAIL
    msg['To'] = MY_EMAIL
    msg['Subject'] = f"📈 סורק SwingHunter - {datetime.now().strftime('%d/%m/%Y')}"

    html_table = df.to_html(index=False, justify='center', classes='table table-striped')
    body = f"""
    <html dir="rtl">
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ padding: 8px; text-align: center; border-bottom: 1px solid #ddd; }}
          th {{ background-color: #4CAF50; color: white; }}
        </style>
      </head>
      <body>
        <h2>תוצאות סריקת SwingHunter Pro</h2>
        <p>להלן המניות המובילות שעברו את סינוני ה-Risk/Reward וה-ATR:</p>
        {html_table}
      </body>
    </html>
    """
    msg.attach(MIMEText(body, 'html'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(MY_EMAIL, email_pw)
        server.sendmail(MY_EMAIL, MY_EMAIL, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"שגיאה בשליחת המייל: {e}")
        return False

# ==========================================
# 4. ממשק המשתמש (Streamlit UI)
# ==========================================
def check_password():
    """מוודא שרק אתה יכול להיכנס לאפליקציה"""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    
    if not st.session_state["authenticated"]:
        st.title("🔒 SwingHunter Access")
        pwd_input = st.text_input("הזן סיסמה כדי להמשיך:", type="password")
        if st.button("כניסה"):
            if pwd_input == APP_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("סיסמה שגויה")
        return False
    return True

if check_password():
    st.set_page_config(page_title="SwingHunter Pro", layout="wide")
    st.title("🎯 SwingHunter Pro Dashboard")
    st.markdown("מנוע סריקה חכם מבוסס תנודתיות (ATR) וניהול סיכונים מתקדם.")

    # תפריט צד
    st.sidebar.header("הגדרות סריקה ומייל")
    st.sidebar.write(f"**מייל מוגדר:** {MY_EMAIL}")
    email_app_pw = st.sidebar.text_input("סיסמת אפליקציה של Gmail (לצורך שליחה):", type="password")
    
    if st.button("🚀 התחל סריקת שוק", use_container_width=True):
        with st.spinner("בודק מצב שוק (SPY)..."):
            market_trend = get_market_context()
        
        if market_trend == "BEARISH":
            st.error(f"🚨 אזהרת שוק: ה-S&P 500 במגמת ירידה (מתחת ל-SMA20). המערכת תחמיר עם ניקוד המניות.")
        else:
            st.success(f"✅ מצב שוק: חיובי (BULLISH). הסורק מחפש מומנטום.")

        # הרצת הסריקה עם סרגל התקדמות
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, ticker in enumerate(WATCHLIST):
            status_text.text(f"סורק את: {ticker} ({i+1}/{len(WATCHLIST)})...")
            res = analyze_ticker(ticker, market_trend)
            if res:
                results.append(res)
            progress_bar.progress((i + 1) / len(WATCHLIST))

        status_text.text("הסריקה הושלמה!")
        
        # הצגת תוצאות
        if results:
            df_res = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
            st.dataframe(df_res, use_container_width=True)

            # שליחת המייל אוטומטית אם הוזנה סיסמת אפליקציה
            if email_app_pw:
                with st.spinner("שולח דו\"ח למייל..."):
                    if send_email_report(df_res, email_app_pw):
                        st.balloons()
                        st.success("📩 הדו\"ח נשלח בהצלחה למייל שלך!")
            else:
                st.warning("⚠️ לא הוזנה 'סיסמת אפליקציה' של Gmail בתפריט הצד. התוצאות לא נשלחו למייל.")
        else:
            st.info("לא נמצאו מניות שעומדות בקריטריונים הקשוחים של הסורק כרגע.")
