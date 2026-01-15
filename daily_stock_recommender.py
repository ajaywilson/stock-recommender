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

# =======================
# Scan stocks
# =======================
for sym in symbols:
    try:
        df = yf.download(sym, period="6mo", progress=False, threads=False)

        if df.empty or len(df) < 70:
            continue

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

        results.append((sym.replace(".NS",""), float(close.iloc[-1]), score, exp_return))

    except:
        continue

df = pd.DataFrame(results, columns=["Stock","Price","Score","ExpReturn"])
df = df.dropna()

if df.empty:
    send_text("⚠️ No valid data today.")
    exit()

df = df.sort_values(by="Score", ascending=False)
top = df.head(TOP_N).copy()

# =======================
# Message with strength labels
# =======================
today = datetime.now().strftime("%Y-%m-%d")
msg = f"📊 Daily Stock Picks ({today})\n\n"

def label(score):
    if score >= 0.45:
        return "🟢 Strong"
    elif score >= 0.30:
        return "🟡 Moderate"
    else:
        return "🔴 Weak"

for i, row in enumerate(top.itertuples(), 1):
    msg += (
        f"{i}. {row.Stock} ₹{round(row.Price,2)} | "
        f"Score {round(row.Score,3)} | "
        f"{label(row.Score)} | "
        f"Exp 1M: {round(row.ExpReturn,2)}%\n"
    )

# =======================
# Improved allocation logic (aggressive capital usage)
# =======================
msg += f"\n💰 Investment Plan (₹{TOTAL_CAPITAL})\n\n"

MAX_PER_STOCK = 0.30 * TOTAL_CAPITAL
PRICE_LIMIT = TOTAL_CAPITAL  # allow any price <= total capital

alloc = top[top["Price"] <= PRICE_LIMIT].copy()

if alloc.empty:
    msg += "No affordable stocks for current capital."
    send_text(msg)
    exit()

# Initialize 1 share of as many top stocks as possible
alloc["shares"] = 0
alloc["invested"] = 0.0

remaining = TOTAL_CAPITAL

# First pass: give 1 share to top-ranked affordable stocks
for i, row in alloc.iterrows():
    if remaining >= row["Price"]:
        alloc.loc[i, "shares"] = 1
        alloc.loc[i, "invested"] = row["Price"]
        remaining -= row["Price"]

# Second pass: greedy buying by score until money exhausted
alloc = alloc.sort_values(by="Score", ascending=False)

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

# Print allocation
for row in alloc.itertuples():
    msg += f"{row.Stock} – Buy {row.shares} shares (₹{int(row.invested)})\n"

msg += f"\nRemaining cash: ₹{int(remaining)}"

# =======================
# Portfolio expectation
# =======================
future_val = 0
for row in alloc.itertuples():
    stock_exp = top[top["Stock"] == row.Stock]["ExpReturn"].values[0]
    future_val += row.invested * (1 + stock_exp / 100)

future_val += remaining
gain = future_val - TOTAL_CAPITAL
pct = gain / TOTAL_CAPITAL * 100

msg += (
    f"\n\n📈 Expected portfolio value after 1 month: ₹{int(future_val)}"
    f"\n(Expected gain: +₹{int(gain)} / +{round(pct,2)}%)"
)

send_text(msg)

# =======================
# Charts for top 3
# =======================
for stock in top.head(3)["Stock"]:
    try:
        df = yf.download(stock + ".NS", period="1mo", progress=False, threads=False)
        if df.empty:
            continue

        plt.figure()
        plt.plot(df["Close"])
        plt.title(stock)
        plt.xlabel("Date")
        plt.ylabel("Stock Price (₹)")
        plt.xticks(rotation=90)
        plt.grid()
        plt.tight_layout()

        path = f"/tmp/{stock}.png"
        plt.savefig(path)
        plt.close()

        send_photo(path, caption=f"{stock} – 1 Month Chart")
    except:
        continue

print("Done.")
