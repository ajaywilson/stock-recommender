import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_market_calendars as mcal
import warnings
import json

warnings.filterwarnings("ignore")

# =======================
# CONFIG
# =======================
TOP_N = 10
TOTAL_CAPITAL = 3000
PORTFOLIO_FILE = "portfolio.json"

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# =======================
# Portfolio helpers
# =======================
def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return {}
    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)

def save_portfolio(p):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2)

portfolio = load_portfolio()

# =======================
# Market day check
# =======================
def is_trading_day():
    nse = mcal.get_calendar("NSE")
    today = datetime.now().date()
    schedule = nse.schedule(start_date=today, end_date=today)
    return not schedule.empty

if not is_trading_day():
    print("Market closed today. Exiting.")
    exit()

# =======================
# Telegram helpers
# =======================
def send_text(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

# =======================
# Load symbols
# =======================
url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
symbols = pd.read_csv(url)["Symbol"].dropna().unique().tolist()
symbols = [s.strip() + ".NS" for s in symbols if isinstance(s, str)]

nifty = yf.download("^NSEI", period="6mo", progress=False)["Close"]

results = []
stock_data = {}

# =======================
# Scan stocks
# =======================
for sym in symbols:
    try:
        df = yf.download(sym, period="6mo", progress=False)
        if df is None or df.empty or len(df) < 70:
            continue

        df = df.dropna()
        if len(df) < 70:
            continue

        stock = sym.replace(".NS", "")
        stock_data[stock] = df.copy()

        close = df["Close"]
        ret_1m = float(close.iloc[-1] / close.iloc[-21] - 1)
        ret_3m = float(close.iloc[-1] / close.iloc[-63] - 1)

        rs_stock = float(close.iloc[-1] / close.iloc[-63])
        rs_index = float(nifty.iloc[-1] / nifty.iloc[-63])
        rel_strength = float(rs_stock / rs_index)

        volatility = float(close.pct_change().std())

        score = (
            ret_1m * 0.4 +
            ret_3m * 0.3 +
            rel_strength * 0.2 -
            volatility * 0.1
        )

        exp_return = (0.6 * ret_1m + 0.4 * ret_3m) * 100
        results.append((stock, float(close.iloc[-1]), score, exp_return, volatility))

    except Exception:
        continue

df = pd.DataFrame(results, columns=["Stock", "Price", "Score", "ExpReturn", "Volatility"])
df = df.sort_values(by="Score", ascending=False)
top = df.head(TOP_N).copy()

# =======================
# Exit logic
# =======================
def exit_signal(df):
    close = df["Close"]
    ema10 = close.ewm(span=10).mean()
    recent_high = df["High"].tail(10).max()
    today_close = float(close.iloc[-1])
    today_open = float(df["Open"].iloc[-1])

    below_ema = today_close < float(ema10.iloc[-1])
    trailing_stop = today_close < 0.94 * float(recent_high)

    if below_ema:
        return "🚨 Exit: Below 10EMA"
    elif trailing_stop:
        return "🚨 Exit: Trailing stop hit"
    else:
        return "✅ Hold"

# =======================
# Evaluate existing holdings (SELL logic)
# =======================
sell_list = []
for stock in list(portfolio.keys()):
    if stock not in stock_data:
        continue

    df_recent = stock_data[stock].tail(30)
    signal = exit_signal(df_recent)

    if "🚨 Exit" in signal:
        sell_list.append(stock)

# Remove sold stocks from portfolio
for s in sell_list:
    portfolio.pop(s, None)

# =======================
# BUY allocation (same as your logic)
# =======================
signals = {row.Stock: "Neutral" for row in top.itertuples()}

weights = {
    "Neutral": 1.0
}

alloc = top.copy()
alloc["weight"] = 1.0
alloc["adj_score"] = alloc["Score"]
alloc["shares"] = 0
alloc["invested"] = 0.0

remaining = TOTAL_CAPITAL

while True:
    bought = False
    for i, row in alloc.iterrows():
        if remaining >= row["Price"]:
            alloc.loc[i, "shares"] += 1
            alloc.loc[i, "invested"] += row["Price"]
            remaining -= row["Price"]
            bought = True
    if not bought:
        break

alloc = alloc[alloc["shares"] > 0]

# Add buys to portfolio
for row in alloc.itertuples():
    portfolio[row.Stock] = {
        "shares": int(row.shares),
        "buy_price": float(row.Price),
        "date": datetime.now().strftime("%Y-%m-%d")
    }

save_portfolio(portfolio)

# =======================
# Telegram message
# =======================
today = datetime.now().strftime("%Y-%m-%d")
msg = f"📊 Daily Stock Bot ({today})\n\n"

if sell_list:
    msg += "📤 SELL TOMORROW\n"
    for s in sell_list:
        msg += f"{s}\n"
else:
    msg += "📤 SELL TOMORROW: None\n"

msg += "\n📥 BUY TOMORROW\n"
for row in alloc.itertuples():
    msg += f"{row.Stock} – Buy {row.shares} shares (~₹{int(row.invested)})\n"

msg += "\n📦 CURRENT PORTFOLIO\n"
for s, info in portfolio.items():
    msg += f"{s}: {info['shares']} shares @ ₹{info['buy_price']}\n"

send_text(msg)
print("Done.")
