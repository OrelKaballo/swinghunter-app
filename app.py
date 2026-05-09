import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import urllib.request
import xml.etree.ElementTree as ET
import warnings

st.set_page_config(page_title="SwingHunter V6.4 - Quant Terminal", layout="wide")
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
        watch_days = recent["החלטה"].astype(str).str.contains("למעקב|פעיל|ARMED|Building Pressure").sum()
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
# 3. מודולי Edge
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
                if (event_close / event_open - 1) * 100 < -5.0: return False, 0, 0
                
                event_high = float(high.iloc[event_pos])
                event_low = float(low.iloc[event_pos])
                if float(close.iloc[-1]) > event_low and low.tail(3).is_monotonic_increasing:
                    return True, event_high, event_low
    except: pass
    return False, 0, 0

# ==========================================
# 4. מנוע האנליזה היומי (Live Engine)
# ==========================================
def analyze_edge(ticker, spy_close, market_trend, investment_budget):
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

        raw_comp_score = compression_score(df)
        rs_5d, rs_20d = relative_strength_vs_spy(close, spy_close)
        failed_break, reclaimed_level = failed_breakdown_recovery(df)
        drift, event_high, event_low = post_event_drift(df)
        watch_days, score_trend = get_setup_persistence(ticker)

        setup_score = 0
        if last_price > sma200: setup_score += 15
        if last_price > ema21: setup_score += 10
        if dist_to_res < 4.0: setup_score += 15
        
        edge_score = 0
        edge_notes = []
        reject_reason = ""
        
        if raw_comp_score >= 25 and (dist_to_res < 4.0 or rs_5d > 0 or rs_20d > 0):
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

        final_rank = setup_score + edge_score
        setup_type = "No Setup"
        is_overextended = rsi_val > 78
        earn_status, earn_days = get_earnings_status(ticker)

        if earn_status == "DANGER":
            icon, decision, reject_reason, final_rank = "⚠️", "DANGER", "דוח קרוב", 0
        elif is_overextended:
            icon, decision, reject_reason, final_rank = "🔥", "חם מדי", "RSI גבוה מ-78", min(final_rank, 30)
        elif market_trend == "BEAR" and edge_score < 40:
            icon, decision, reject_reason, final_rank = "🔴", "Dormant", "Market BEAR + Edge נמוך", min(final_rank, 40)
        elif final_rank >= 80 and len(edge_notes) > 0:
            icon, decision = "🟢", "ARMED"
        elif final_rank >= 45:
            icon, decision = "🟡", "Building Pressure"
        else:
            icon, decision, reject_reason = "🔴", "Dormant", "ציון נמוך מדי"

        if "Compression" in edge_notes and dist_to_res < 3.0: setup_type = "Pre-Breakout Compression"
        elif failed_break: setup_type = "Failed Breakdown Recovery"
        elif drift: setup_type = "Post-Event Drift"
        elif "RS Leader" in edge_notes and last_price > ema21: setup_type = "RS Pullback"
        elif last_price > res_20d: setup_type = "Momentum Breakout"

        order_data = None
        if decision == "ARMED":
            p_type, entry, e_disp = None, 0, ""
            
            if setup_type in ["Pre-Breakout Compression", "Post-Event Drift", "Momentum Breakout"]:
                trigger = res_20d if setup_type != "Post-Event Drift" else event_high
                entry = round(trigger * 1.002, 2)
                if last_price < entry:
                    p_type, e_disp = "BUY STOP LIMIT", f"Stop {entry} / Lmt {round(entry*1.008, 2)}"
                else:
                    reject_reason, icon, decision = "כבר פרצה. לא רודפים.", "🟡", "Building Pressure"
            
            elif setup_type in ["Failed Breakdown Recovery", "RS Pullback"]:
                entry = min(round(ema21 * 1.005, 2), round(last_price * 0.995, 2))
                p_type, e_disp = "BUY LIMIT", f"{entry}"

            if p_type:
                stop = round(min(ema21, entry - (1.5 * atr_val)), 2)
                if setup_type == "Failed Breakdown Recovery": stop = round(reclaimed_level * 0.985, 2)
                
                risk_ps = entry - stop
                if risk_ps > 0 and (entry * 0.10) / risk_ps >= 1.5:
                    # שינוי מהותי: כמות המניות מחושבת לפי תקציב השקעה קבוע של 1000$ (ולא לפי סיכון)
                    shares = int(investment_budget / entry)
                    if shares > 0:
                        order_data = {
                            'מניה': ticker, 'פעולה': p_type, 'Edge': " + ".join(edge_notes),
                            'כניסה': e_disp, 'כמות': shares, 
                            'השקעה $': f"${round(shares * entry, 2)}",
                            'יעד 10%': round(entry * 1.10, 2), 'יעד 15%': round(entry * 1.15, 2),
                            'סטופ': stop, 'R/R': round((entry * 0.10) / risk_ps, 2), 'תוקף': 'DAY ONLY'
                        }
                else: reject_reason = "R/R נמוך מ-1.5"

        return {
            'scanner': {
                'מניה': ticker, 'החלטה': f"{icon} {decision}", 'ציון_כולל': int(final_rank),
                'תבנית': setup_type, 'Market': market_trend, 'Watch Days': watch_days,
                'Comp. Score': raw_comp_score, 'RS (5D)': f"{round(rs_5d,1)}%", 'RS (20D)': f"{round(rs_20d,1)}%",
                'Edge Notes': " + ".join(edge_notes) if edge_notes else "None", 'סיבת פסילה': reject_reason
            },
            'order': order_data
        }
    except: return None

# ==========================================
# 5. מנוע הסימולציה (Backtester Engine)
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_backtest_data(months):
    end = datetime.now()
    start = end - timedelta(days=months * 30 + 250)
    return yf.download(WATCHLIST + ['SPY'], start=start, end=end, progress=False)

def run_backtest_simulation(data, investment_per_trade, starting_capital=10000.0):
    prices, highs, lows, opens = data['Close'], data['High'], data['Low'], data['Open']
    
    cash = starting_capital
    positions = {}
    pending_orders = {}
    equity_curve, dates, trade_log = [], [], []
    
    wins, losses = 0, 0
    total_invested_dollars = 0.0 # סופר כל דולר שהושקע בעסקאות
    total_gross_profit = 0.0     # סופר רק רווחים מתנועות חיוביות
    total_gross_loss = 0.0       # סופר רק הפסדים

    start_idx = 200 
    for i in range(start_idx, len(prices) - 1):
        today_str = prices.index[i].strftime('%Y-%m-%d')
        
        # 1. הפעלת פקודות מאתמול (רק אם אנחנו לא כבר מחזיקים את המניה)
        executed_tickers = []
        for ticker, order in pending_orders.items():
            if ticker in positions: continue # כלל ברזל: לא קונים מניה שכבר מחזיקים בה
            
            try:
                t_open, t_high, t_low = float(opens[ticker].iloc[i]), float(highs[ticker].iloc[i]), float(lows[ticker].iloc[i])
            except: continue
            if pd.isna(t_open): continue
            
            executed, exec_price = False, 0
            if order['type'] == 'BUY LIMIT' and t_low <= order['price']:
                exec_price, executed = order['price'], True
            elif order['type'] == 'BUY STOP LIMIT' and t_high >= order['price']:
                exec_price, executed = max(t_open, order['price']), True
                
            if executed:
                cost = exec_price * order['shares']
                if cash >= cost: # יש מספיק מזומן בחשבון
                    cash -= cost
                    total_invested_dollars += cost # רישום השקעה
                    positions[ticker] = {
                        'shares': order['shares'], 'entry': exec_price,
                        'target': order['target'], 'stop': order['stop']
                    }
                    executed_tickers.append(ticker)
        
        pending_orders.clear() # פקודות שלא נתפסו נמחקות
        
        # 2. ניהול פוזיציות וסגירתן ברווח/הפסד
        closed_tickers = []
        for ticker, pos in positions.items():
            try: t_high, t_low = float(highs[ticker].iloc[i]), float(lows[ticker].iloc[i])
            except: continue
            
            sell_price = 0
            if t_low <= pos['stop']:
                sell_price = pos['stop']
                losses += 1
            elif t_high >= pos['target']:
                sell_price = pos['target']
                wins += 1
                
            if sell_price > 0:
                revenue = sell_price * pos['shares']
                cash += revenue
                pnl = revenue - (pos['entry'] * pos['shares'])
                
                # חלוקה לרווח והפסד
                if pnl > 0: total_gross_profit += pnl
                else: total_gross_loss += abs(pnl)
                
                trade_log.append({'Date': today_str, 'Ticker': ticker, 'Invested': round(pos['entry']*pos['shares'],2), 'PnL': round(pnl, 2)})
                closed_tickers.append(ticker)
                
        for t in closed_tickers: del positions[t]

        # 3. שערוך תיק יומי
        portfolio_value = cash
        for ticker, pos in positions.items():
            t_close = prices[ticker].iloc[i]
            if not pd.isna(t_close): portfolio_value += pos['shares'] * float(t_close)
                
        equity_curve.append(portfolio_value)
        dates.append(today_str)
        
        # 4. סריקה למחר (רק למניות שלא מוחזקות כרגע בתיק!)
        try:
            spy_slice = prices['SPY'].iloc[i-20:i+1]
            market_bull = float(spy_slice.iloc[-1]) > float(spy_slice.rolling(20).mean().iloc[-1])
        except: continue
        
        if not market_bull: continue # בשוק יורד לא קונים חדשות
        
        for ticker in WATCHLIST:
            if ticker in positions: continue # כבר "בפנים" - לא מייצר סטאפ חדש
            
            try:
                c_hist, h_hist, l_hist = prices[ticker].iloc[i-200:i+1], highs[ticker].iloc[i-21:i+1], lows[ticker].iloc[i-21:i+1]
                if len(c_hist) < 200 or pd.isna(c_hist.iloc[-1]): continue
                
                last_p = float(c_hist.iloc[-1])
                sma200, ema21, res_20d = float(c_hist.mean()), float(c_hist.ewm(span=21, adjust=False).mean().iloc[-1]), float(h_hist.iloc[:-1].max())
                
                if last_p < sma200: continue
                
                tr = pd.concat([h_hist-l_hist, (h_hist-c_hist.shift()).abs(), (l_hist-c_hist.shift()).abs()], axis=1).max(axis=1)
                atr_val = float(tr.rolling(14).mean().iloc[-1])
                
                order_type, entry_price = None, 0
                dist_to_res = (res_20d / last_p - 1)
                
                if 0 < dist_to_res < 0.03: 
                    entry_price = round(res_20d * 1.002, 2)
                    if last_p < entry_price: order_type = 'BUY STOP LIMIT'
                elif last_p > ema21 and (last_p / ema21 - 1) < 0.03:
                    entry_price = round(ema21 * 1.005, 2)
                    if last_p > entry_price: order_type = 'BUY LIMIT'
                        
                if order_type:
                    stop_price = round(min(ema21, entry_price - (1.5 * atr_val)), 2)
                    if entry_price - stop_price > 0 and (entry_price * 0.10) / (entry_price - stop_price) >= 1.5:
                        # השקעה קבועה של 1000$ (או מה שהוגדר)
                        shares = int(investment_per_trade / entry_price)
                        if shares > 0 and (shares * entry_price) <= cash:
                            pending_orders[ticker] = {
                                'type': order_type, 'price': entry_price,
                                'target': round(entry_price * 1.10, 2), 'stop': stop_price, 'shares': shares
                            }
            except: continue

    df_equity = pd.DataFrame({'Date': dates, 'Portfolio Value': equity_curve}).set_index('Date')
    df_trades = pd.DataFrame(trade_log)
    return df_equity, df_trades, wins, losses, total_invested_dollars, total_gross_profit, total_gross_loss

# ==========================================
# 6. UI Dashboard
# ==========================================
if "authenticated" not in st.session_state: st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 Terminal Access")
    pwd = st.text_input("סיסמה:", type="password")
    if st.button("כניסה"):
        if pwd == APP_PASSWORD: st.session_state["authenticated"] = True; st.rerun()
else:
    st.markdown("<h1 style='text-align: right;'>🎯 SwingHunter V6.4 - Bottom Line Edition</h1>", unsafe_allow_html=True)
    
    st.sidebar.header("ניהול כספי")
    investment_amount = st.sidebar.number_input("סכום קבוע להשקעה בכל עסקה ($)", value=1000, step=100)
    
    tab_daily, tab_backtest = st.tabs(["🚀 מסך עבודה יומי", "🔬 מעבדת סימולציות"])
    
    # --- לשונית עבודה יומית ---
    with tab_daily:
        if st.button("⚡ הפק תוכנית עבודה להיום", use_container_width=True):
            with st.spinner("מנתח את השוק ומייצר פקודות..."):
                spy_close, market_trend = get_spy_context()
                if spy_close is not None:
                    raw_results = [analyze_edge(t, spy_close, market_trend, investment_amount) for t in WATCHLIST]
                    raw_results = [r for r in raw_results if r is not None]
                
                    order_column_config = {
                        "מניה": st.column_config.TextColumn("מניה"), "פעולה": st.column_config.TextColumn("פעולה"),
                        "Edge": st.column_config.TextColumn("Edge"), "כניסה": st.column_config.TextColumn("כניסה"),
                        "כמות": st.column_config.NumberColumn("כמות", help=f"מספר מניות לקנייה. מחושב כדי להגיע לכ-{investment_amount}$ השקעה"),
                        "השקעה $": st.column_config.TextColumn("השקעה $", help="סכום הכסף המדויק שינעל בפקודה הזו (הכמות כפול שער הכניסה)"),
                        "יעד 10%": st.column_config.NumberColumn("יעד 10%"), "יעד 15%": st.column_config.NumberColumn("יעד 15%"),
                        "סטופ": st.column_config.NumberColumn("סטופ"), "R/R": st.column_config.NumberColumn("R/R"),
                        "תוקף": st.column_config.TextColumn("תוקף")
                    }

                    order_list = [r['order'] for r in raw_results if r['order'] is not None]
                    st.markdown(f"### 📝 פקודות (סכום להשקעה מבוקש בעסקה: {investment_amount}$)")
                    if order_list:
                        df_orders = pd.DataFrame(order_list).sort_values(by="R/R", ascending=False).head(3)
                        st.dataframe(df_orders.style.hide(axis="index"), column_config=order_column_config, use_container_width=True)
                    else: st.info("אין היום פקודות שעברו את כל שומרי הסף.")

                    scan_column_config = {"החלטה": st.column_config.TextColumn("החלטה"), "סיבת פסילה": st.column_config.TextColumn("סיבת פסילה")}
                    st.markdown("---")
                    st.markdown("### 🔍 רדאר שוק")
                    scanner_list = [r['scanner'] for r in raw_results]
                    df_scan = pd.DataFrame(scanner_list).sort_values(by="ציון_כולל", ascending=False)
                    save_scan_history(df_scan)
                    
                    def color_logic(row):
                        val = str(row['החלטה'])
                        if "ARMED" in val: return ['background-color: rgba(46, 204, 113, 0.2)'] * len(row)
                        if "Building Pressure" in val: return ['background-color: rgba(241, 196, 15, 0.1)'] * len(row)
                        if "DANGER" in val: return ['background-color: rgba(231, 76, 60, 0.1)'] * len(row)
                        return [''] * len(row)
                    st.dataframe(df_scan.style.apply(color_logic, axis=1), column_config=scan_column_config, use_container_width=True)
                else: st.error("שגיאה במשיכת נתוני השוק.")

    # --- לשונית מעבדה וסטטיסטיקה ---
    with tab_backtest:
        st.markdown(f"### 🧪 בדיקת אלגוריתם - 3 חודשים (השקעה של {investment_amount}$ בעסקה)")
        if st.button("⚙️ הרץ בדיקה היסטורית (לוקח כדקה)", type="primary"):
            with st.spinner("מריץ אלפי סימולציות. זה עשוי לקחת קצת זמן..."):
                data = fetch_backtest_data(months=3)
                df_eq, df_trades, wins, losses, tot_invested, tot_g_profit, tot_g_loss = run_backtest_simulation(data, investment_amount, 10000.0)
                
                total_trades = wins + losses
                net_pnl = tot_g_profit - tot_g_loss
                final_val = df_eq['Portfolio Value'].iloc[-1] if not df_eq.empty else 10000.0
                roi = ((final_val / 10000.0) - 1) * 100
                
                # מתן ציון למערכת
                grade_str, desc_str = "", ""
                if roi > 15: grade_str, desc_str = "10/10 🏆", "מכונת מזומנים. המערכת קוראת את השוק בצורה מבריקה."
                elif roi > 5: grade_str, desc_str = "8/10 🟢", "אלגוריתם חזק ויציב. מייצר אלפא יפה על הכסף."
                elif roi > 0: grade_str, desc_str = "6/10 🟡", "רווחי אבל מתקשה לייצר תנופה אגרסיבית (שומר על ההון)."
                elif roi > -5: grade_str, desc_str = "4/10 🟠", "הפסד קל. השוק כנראה חותך את הסטופים."
                else: grade_str, desc_str = "2/10 🔴", "ביצועים חלשים. נדרש כיול מחדש ללוגיקת הכניסות."

                st.markdown("#### 💰 שורה תחתונה (Bottom Line)")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("סך הכל השקעה מצטברת", f"${tot_invested:,.0f}")
                col2.metric("הרווחנו בעסקאות טובות", f"+${tot_g_profit:,.0f}")
                col3.metric("הפסדנו בעסקאות רעות", f"-${tot_g_loss:,.0f}")
                col4.metric("רווח נקי בכיס (Net PnL)", f"${net_pnl:,.0f}", delta=f"{roi:.1f}% תשואה על התיק")
                
                st.info(f"**ציון המערכת לתקופה זו: {grade_str}** | {desc_str}")
                
                st.markdown("#### 📈 התפתחות שווי התיק (מ-10,000$)")
                st.line_chart(df_eq)
                
                if not df_trades.empty:
                    with st.expander("📝 יומן עסקאות מפורט (לראות כל דולר שנכנס ויצא)"):
                        st.dataframe(df_trades, use_container_width=True)
