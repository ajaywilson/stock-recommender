import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_market_calendars as mcal

# =======================
# CONFIG
# =======================
TOP_N = 10
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# =======================
# Market day check (NSE)
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
# RSI
# =======================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean()
    return 100 - (100 / (1 + rs))

# =======================
# Load Nifty 500
# =======================
url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
symbols = pd.read_csv(url)["Symbol"].tolist()
symbols = [s + ".NS" for s in symbols]

print(f"Scanning {len(symbols)} stocks...")

# Download Nifty index for relative strength
nifty = yf.download("^NSEI", period="6mo", progress=False)["Close"]

results = []

# =======================
# Scan all stocks
# =======================
for sym in symbols:
    try:
        df = yf.download(sym, period="6mo", progress=False)

        if df.empty or len(df) < 70:
            continue

        close = df["Close"]
        volume = df["Volume"]

        ret_1m = close.iloc[-1] / close.iloc[-21] - 1
        ret_3m = close.iloc[-1] / close.iloc[-63] - 1

        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]

        rsi_val = rsi(close).iloc[-1]

        vol_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]

        # Relative strength vs Nifty
        rs_stock = close.iloc[-1] / close.iloc[-63]
        rs_index = nifty.iloc[-1] / nifty.iloc[-63]
        rel_strength = rs_stock / rs_index

        # Filters (quality gates)
        if not (close.iloc[-1] > ma20 > ma50):
            continue
        if not (45 < rsi_val < 65):
            continue
        if vol_ratio < 1.1:
            continue

        volatility = close.pct_change().std()

        score = (
            ret_1m * 0.4 +
            ret_3m * 0.3 +
            rel_strength * 0.2 -
            volatility * 0.1
        )

        results.append((sym.replace(".NS", ""), close.iloc[-1], score))

    except:
        continue

# =======================
# Rank stocks
# =======================
df = pd.DataFrame(results, columns=["Stock", "Price", "Score"])
df = df.sort_values(by="Score", ascending=False)

top = df.head(TOP_N)

# =======================
# Send Telegram text
# =======================
today = datetime.now().strftime("%Y-%m-%d")
msg = f"📊 Daily Stock Picks ({today})\n\n"

for i, row in enumerate(top.itertuples(), 1):
    strength = "🟢 Strong" if row.Score > 0.25 else "🟡 Moderate"
    msg += f"{i}. {row.Stock} ₹{round(row.Price,2)} | Score: {round(row.Score,3)} | {strength}\n"

send_text(msg)

# =======================
# Send charts for top 3
# =======================
for stock in top.head(3)["Stock"]:
    df = yf.download(stock + ".NS", period="1mo", progress=False)
    plt.figure()
    plt.plot(df["Close"])
    plt.title(stock)
    plt.grid()
    plt.tight_layout()
    img_path = f"/tmp/{stock}.png"
    plt.savefig(img_path)
    plt.close()
    send_photo(img_path, caption=f"{stock} – 1 Month Chart")

print("Done.")
