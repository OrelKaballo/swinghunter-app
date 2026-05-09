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

warnings.filterwarnings('ignore')

# ==========================================
# 1. הגדרות (שנה לנתונים שלך)
# ==========================================
APP_PASSWORD = "Pk0105Ak2701" # סיסמת כניסה לאתר
MY_EMAIL = "orel@peleg-eng.com"      # המייל שלך להתראות

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

# ==========================================
# 2. מנועי מידע וניתוח חדשות (Google News)
# ==========================================
def get_external_news_and_sentiment(ticker):
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            xml_page = response.read()
        
        root = ET.fromstring(xml_page)
        items = root.findall('.//item')
        if not items:
            return "אין חדשות לאחרונה", "⚪ ניטרלי"
        
        titles = [item.find('title').text for item in items[:2]]
        news_text = " | ".join(titles)
        
        pos_words = ['up', 'surge', 'jump', 'beat', 'growth', 'buy', 'upgrade', 'high', 'profit', 'win', 'soar']
        neg_words = ['down', 'plunge', 'drop', 'miss', 'cut', 'sell', 'downgrade', 'low', 'loss', 'lawsuit', 'crash']
        
        text_lower = news_text.lower()
        pos_count = sum(1 for word in pos_words if f" {word} " in f" {text_lower} ")
        neg_count = sum(1 for word in neg_words if f" {word} " in f" {text_lower} ")
        
        if pos_count > neg_count: sentiment = "🟢 חיובי"
        elif neg_count > pos_count: sentiment = "🔴 שלילי"
        else: sentiment = "⚪ ניטרלי"
            
        return news_text, sentiment
    except Exception as e:
        return "שגיאה בשליפת חדשות מ-Google", "⚪ לא ידוע"

def check_upcoming_earnings(ticker):
    try:
        tkr = yf.Ticker(ticker)
        cal = tkr.get_earnings_dates(limit=3)
        if cal is not None and not cal.empty:
            now = pd.Timestamp.now(tz='UTC')
            future_dates = cal.index[cal.index > now]
            if not future_dates.empty:
                next_date = future_dates[0]
                days = (next_date - now).days
                if 0 <= days <= 7:
                    return True, days
    except:
        pass
    return False, -1

# ==========================================
# 3. מנוע ניתוח טכני
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
    try:
        spy = yf.download('SPY', period='50d', progress=False)
        last_close = float(spy['Close'].iloc[-1])
        sma20 = float(spy['Close'].rolling(20).mean().iloc[-1])
        return "חיובי (מעל ממוצע 20)" if last_close > sma20 else "שלילי (מתחת לממוצע 20)"
    except:
        return "לא ידוע"

def analyze_ticker(ticker, market_trend):
    try:
        df = yf.download(ticker, period='250d', progress=False)
        if df.empty or len(df) < 200: return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df['Close']
        high = df['High']
        low = df['Low']
        vol = df['Volume']

        df['SMA20'] = close.rolling(20).mean()
        df['SMA200'] = close.rolling(200).mean()
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

        if price < 5 or avg_vol < 1_000_000 or pd.isna(atr_val): return None

        stop = price - (atr_val * 1.5)
        target1 = price + (atr_val * 2.0)
        
        risk = price - stop
        reward = target1 - price
        rr = reward / risk if risk > 0 else 0

        setup = []
        if price > df['High'].iloc[-2] and last['RelVol'] > 1.2 and price > last['Open']:
            setup.append('פריצת מומנטום')
        if price > last['EMA8'] and last['EMA8'] > last['EMA21']:
            setup.append('מגמה קצרת-טווח חזקה')
        if last['RSI14'] < 40 and price > last['SMA20']:
            setup.append('תיקון טכני על תמיכה')

        setup_txt = ' + '.join(setup) if setup else "אין טריגר כניסה"

        score = 50 
        remarks = []
        display_ticker = ticker

        if not setup:
            score = 0
            remarks.append("נפסל: אין תבנית")
        else:
            if 'פריצה' in setup_txt: score += 15
            if last['RelVol'] > 1.5: score += 15
            if rr >= 2.0: score += 20
            elif rr >= 1.5: score += 10

            if "שלילי" in market_trend: 
                score -= 20
                remarks.append("אזהרה: שוק חלש")
            if rr < 1.3: 
                score -= 30
                remarks.append("נפסל: סיכוי-סיכון נמוך")
            if last['RSI14'] > 75: 
                score -= 15
                remarks.append("אזהרה: קניית יתר (RSI>75)")
            if price > last['SMA20'] + (atr_val * 2): 
                score -= 20
                remarks.append("אזהרה: מחיר מתוח מדי")
                
            if price < float(last['SMA200']):
                score -= 20
                remarks.append("נפסל: מגמה ארוכת-טווח שלילית (מתחת ל-200)")

            if score >= 50:
                has_earnings, days = check_upcoming_earnings(ticker)
                if has_earnings:
                    display_ticker = f"❗ {ticker}"
                    score -= 10 
                    remarks.append(f"סכנה: דוח בעוד {days} ימים")

        score = max(0, min(100, score)) 
        status = "✅ עובר" if score >= 65 else "❌ נפסל"

        news_text, sentiment = get_external_news_and_sentiment(ticker) if score >= 65 else ("-", "-")

        return {
            'סטטוס': status,
            'מניה': display_ticker,
            'ציון': int(score),
            'סנטימנט בחדשות': sentiment,
            'סיבת פסילה / אזהרות': ", ".join(remarks) if remarks else "תקין",
            'תבנית טכנית': setup_txt,
            'מחיר נוכחי': round(price, 2),
            'יעד ראשון': round(target1, 2),
            'סטופ-לוס': round(stop, 2),
            'יחס R/R': round(rr, 2),
            'כותרות מהרשת': news_text
        }
    except Exception as e:
        return None

# ==========================================
# 4. פונקציית שליחת המייל שחזרה הביתה
# ==========================================
def send_email_report(df, email_pw):
    df_passed = df[df['סטטוס'] == '✅ עובר']
    
    msg = MIMEMultipart()
    msg['From'] = MY_EMAIL
    msg['To'] = MY_EMAIL
    msg['Subject'] = f"📈 דו\"ח סריקת מניות פרו - {datetime.now().strftime('%d/%m/%Y')}"

    html_table = df_passed.to_html(index=False, justify='center', classes='table table-striped')
    body = f"""
    <html dir="rtl">
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; }}
          table {{ border-collapse: collapse; width: 100%; direction: rtl; }}
          th, td {{ padding: 8px; text-align: center; border-bottom: 1px solid #ddd; }}
          th {{ background-color: #2ecc71; color: white; }}
        </style>
      </head>
      <body>
        <h2>סיכום סריקה יומית - מניות נבחרות</h2>
        <p>להלן המניות שעמדו בכל הקריטריונים המחמירים של המערכת (ציון 65 ומעלה):</p>
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
# 5. ממשק משתמש מלא (UI)
# ==========================================
def check_password():
    if "authenticated" not in st.session_state:
