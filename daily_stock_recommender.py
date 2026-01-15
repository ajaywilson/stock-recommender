import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_market_calendars as mcal
import warnings

warnings.filterwarnings("ignore")

# =======================
# CONFIG
# =======================
TOP_N = 10
TOTAL_CAPITAL = 3000

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

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

def send_photo(path, caption=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(path, "rb") as img:
        requests.post(url, files={"photo": img},
                      data={"chat_id": CHAT_ID, "caption": caption})

# =======================
# Load Nifty 500
# =======================
url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
symbols = pd.read_csv(url)["Symbol"].tolist()
symbols = [s + ".NS" for s in symbols]

print(f"Scanning {len(symbols)} stocks...")

nifty = yf.download("^NSEI", period="6mo", progress=False, threads=False)["Close"]

results = []
stock_data = {}

# =======================
# Scan stocks
# =======================
for sym in symbols:
    try:
        df = yf.download(sym, period="6mo", progress=False, threads=False)

        if df.empty or len(df) < 70:
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

    except:
        continue

df = pd.DataFrame(results, columns=["Stock","Price","Score","ExpReturn","Volatility"])
df = df.sort_values(by="Score", ascending=False)
top = df.head(TOP_N).copy()

# =======================
# Labels
# =======================
def strength_label(score):
    if score >= 0.45: return "🟢 Strong"
    elif score >= 0.30: return "🟡 Moderate"
    else: return "🔴 Weak"

def risk_label(vol):
    if vol < 0.015: return "🟢 Low Risk"
    elif vol < 0.025: return "🟡 Medium Risk"
    else: return "🔴 High Risk"

# =======================
# Price Action
# =======================
signals = {}

for stock in top["Stock"]:
    df2 = stock_data[stock].tail(25)

    high = df2["High"]
    low = df2["Low"]
    close = df2["Close"]
    open_ = df2["Open"]

    today_close = float(close.iloc[-1])
    today_open  = float(open_.iloc[-1])
    today_high  = float(high.iloc[-1])
    today_low   = float(low.iloc[-1])

    recent_high = float(high.iloc[:-1].max())
    full_high   = float(high.max())

    is_breakout = today_close > recent_high
    near_resistance = today_close > 0.97 * full_high

    bullish = today_close > today_open
    body_ratio = abs(today_close - today_open) / (today_high - today_low + 1e-6)
    strong_candle = bullish and body_ratio > 0.6

    if is_breakout:
        signals[stock] = "📈 Breakout (Go for it)"
    elif near_resistance:
        signals[stock] = "⚠️ Near Resistance (Wait)"
    elif strong_candle:
        signals[stock] = "🟢 Strong Candle (Consider entry)"
    else:
        signals[stock] = "Neutral"

# =======================
# Telegram message
# =======================
today = datetime.now().strftime("%Y-%m-%d")
msg = f"📊 Daily Stock Picks ({today})\n\n"

for i, row in enumerate(top.itertuples(), 1):
    msg += (
        f"{i}. {row.Stock} ₹{round(row.Price,2)} | "
        f"Score {round(row.Score,3)} | {strength_label(row.Score)} | "
        f"Risk: {risk_label(row.Volatility)} | "
        f"Exp 1M: {round(row.ExpReturn,2)}%\n"
        f"   ➤ Price Action: {signals[row.Stock]}\n"
    )

# =======================
# Allocation respecting price-action
# =======================
msg += f"\n💰 Investment Plan (₹{TOTAL_CAPITAL})\n\n"

weights = {
    "📈 Breakout (Go for it)": 1.3,
    "🟢 Strong Candle (Consider entry)": 1.1,
    "Neutral": 1.0,
    "⚠️ Near Resistance (Wait)": 0.3
}

alloc = top.copy()
alloc["weight"] = alloc["Stock"].apply(lambda s: weights.get(signals[s], 0))
alloc["adj_score"] = alloc["Score"] * alloc["weight"]

alloc = alloc[alloc["weight"] > 0]  # remove bad signals
alloc = alloc.sort_values(by="adj_score", ascending=False)

alloc["shares"] = 0
alloc["invested"] = 0.0
remaining = TOTAL_CAPITAL
MAX_PER_STOCK = 0.30 * TOTAL_CAPITAL

# Buy greedily by adjusted priority
while True:
    bought = False
    for i, row in alloc.iterrows():
        if remaining >= row["Price"] and alloc.loc[i, "invested"] + row["Price"] <= MAX_PER_STOCK:
            alloc.loc[i, "shares"] += 1
            alloc.loc[i, "invested"] += row["Price"]
            remaining -= row["Price"]
            bought = True
    if not bought:
        break

alloc = alloc[alloc["shares"] > 0]

for row in alloc.itertuples():
    msg += f"{row.Stock} – Buy {row.shares} shares (₹{int(row.invested)})\n"

msg += f"\nRemaining cash: ₹{int(remaining)}"

send_text(msg)
print("Done.")
