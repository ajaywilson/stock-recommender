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
# Scan all stocks (NO FILTERING)
# =======================
for sym in symbols:
    try:
        df = yf.download(sym, period="6mo", progress=False, threads=False)

        if df.empty or len(df) < 70:
            continue

        close = df["Close"]
        volume = df["Volume"]

        # Factors
        ret_1m = float(close.iloc[-1] / close.iloc[-21] - 1)
        ret_3m = float(close.iloc[-1] / close.iloc[-63] - 1)

        rsi_val = float(rsi(close).iloc[-1])

        vol_ratio = float(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1])

        # Relative strength vs Nifty
        rs_stock = float(close.iloc[-1] / close.iloc[-63])
        rs_index = float(nifty.iloc[-1] / nifty.iloc[-63])
        rel_strength = float(rs_stock / rs_index)

        volatility = float(close.pct_change().std())

        # Score (always numeric now)
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

    except Exception as e:
        continue  # silently skip bad symbols

# =======================
# Build dataframe safely
# =======================
df = pd.DataFrame(results, columns=["Stock", "Price", "Score"])

# Force numeric types (extra safety)
df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Drop broken rows
df = df.dropna(subset=["Score", "Price"])

if df.empty:
    send_text("⚠️ No valid stock data today.")
    print("No usable data.")
    exit()

# Sort safely
df = df.sort_values(by="Score", ascending=False)

top = df.head(TOP_N)

# =======================
# Send Telegram text
# =======================
today = datetime.now().strftime("%Y-%m-%d")
msg = f"📊 Daily Stock Picks ({today})\n\n"

for i, row in enumerate(top.itertuples(), 1):
    if row.Score > 0.3:
        label = "🟢 Strong"
    elif row.Score > 0.15:
        label = "🟡 Moderate"
    else:
        label = "🔴 Weak"

    msg += f"{i}. {row.Stock} ₹{round(row.Price,2)} | Score: {round(row.Score,3)} | {label}\n"

send_text(msg)

# =======================
# Send charts for top 3
# =======================
for stock in top.head(3)["Stock"]:
    try:
        df = yf.download(stock + ".NS", period="1mo", progress=False, threads=False)
        if df.empty:
            continue

        plt.figure()
        plt.plot(df["Close"])

        plt.title(stock)
        plt.ylabel("Stock Price (₹)")
        plt.xlabel("Date")

        plt.xticks(rotation=90)

        plt.grid()
        plt.tight_layout()

        plt.savefig(img_path)
        plt.close()
        send_photo(img_path, caption=f"{stock} – 1 Month Chart")
    except:
        continue

print("Done.")

