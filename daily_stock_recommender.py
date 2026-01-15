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
# Scan all stocks (no filtering)
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

        results.append((
            sym.replace(".NS", ""),
            float(close.iloc[-1]),
            float(score)
        ))

    except:
        continue

# =======================
# Build dataframe safely
# =======================
df = pd.DataFrame(results, columns=["Stock", "Price", "Score"])
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
df = df.dropna()

if df.empty:
    send_text("⚠️ No valid stock data today.")
    exit()

df = df.sort_values(by="Score", ascending=False)
top = df.head(TOP_N).copy()

# =======================
# Telegram Picks Message
# =======================
today = datetime.now().strftime("%Y-%m-%d")
msg = f"📊 Daily Stock Picks ({today})\n\n"

for i, row in enumerate(top.itertuples(), 1):
    if row.Score > 0.3:
        label = "🟢"
    elif row.Score > 0.15:
        label = "🟡"
    else:
        label = "🔴"

    msg += f"{i}. {row.Stock} ₹{round(row.Price,2)} | Score: {round(row.Score,3)} {label}\n"

# =======================
# Improved Allocation Logic
# =======================
msg += f"\n💰 Investment Plan (₹{TOTAL_CAPITAL})\n\n"

MAX_PER_STOCK = 0.30 * TOTAL_CAPITAL
MIN_STOCKS = 4
PRICE_LIMIT = 0.60 * TOTAL_CAPITAL

alloc_df = top[top["Price"] <= PRICE_LIMIT].copy()

if alloc_df.empty:
    msg += "All stocks too expensive for current capital."
    send_text(msg)
    exit()

alloc_df["Score"] = alloc_df["Score"].clip(lower=0.01)
total_score = alloc_df["Score"].sum()

alloc_df["weight"] = alloc_df["Score"] / total_score
alloc_df["ideal_amount"] = alloc_df["weight"] * TOTAL_CAPITAL
alloc_df["ideal_amount"] = alloc_df["ideal_amount"].clip(upper=MAX_PER_STOCK)

alloc_df["shares"] = (alloc_df["ideal_amount"] / alloc_df["Price"]).astype(int)
alloc_df["invested"] = alloc_df["shares"] * alloc_df["Price"]

alloc_df = alloc_df[alloc_df["shares"] > 0].copy()

# Ensure minimum diversification
if len(alloc_df) < MIN_STOCKS:
    alloc_df = top[top["Price"] <= PRICE_LIMIT].head(MIN_STOCKS).copy()
    alloc_df["shares"] = 1
    alloc_df["invested"] = alloc_df["Price"]

remaining = TOTAL_CAPITAL - alloc_df["invested"].sum()

# Greedy redistribution
alloc_df = alloc_df.sort_values(by="Score", ascending=False)

while True:
    bought = False
    for i, row in alloc_df.iterrows():
        if remaining >= row["Price"] and alloc_df.loc[i, "invested"] + row["Price"] <= MAX_PER_STOCK:
            alloc_df.loc[i, "shares"] += 1
            alloc_df.loc[i, "invested"] += row["Price"]
            remaining -= row["Price"]
            bought = True
    if not bought:
        break

# Add allocation to message
for row in alloc_df.itertuples():
    msg += f"{row.Stock} – Buy {row.shares} shares (₹{int(row.invested)})\n"

msg += f"\nRemaining cash: ₹{int(remaining)}"

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

        img_path = f"/tmp/{stock}.png"
        plt.savefig(img_path)
        plt.close()

        send_photo(img_path, caption=f"{stock} – 1 Month Chart")
    except:
        continue

print("Done.")
