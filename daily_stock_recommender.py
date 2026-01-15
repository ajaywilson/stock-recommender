import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import warnings

warnings.filterwarnings("ignore")

# =======================
# CONFIG
# =======================
TOP_N = 10
TOTAL_CAPITAL = 3000
LOG_FILE = "performance_log.csv"

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
# CALIBRATION (uses history)
# =======================
calibration_factor = 1.0

if os.path.exists(LOG_FILE):
    hist = pd.read_csv(LOG_FILE)
    past = hist.dropna(subset=["RealizedReturn","ExpReturn"])
    if len(past) > 10:
        calibration_factor = past["RealizedReturn"].mean() / past["ExpReturn"].mean()

top["AdjExpReturn"] = top["ExpReturn"] * calibration_factor

# =======================
# Build message
# =======================
today = datetime.now().strftime("%Y-%m-%d")
msg = f"📊 Daily Stock Picks ({today})\n\n"

for i, row in enumerate(top.itertuples(), 1):
    msg += (
        f"{i}. {row.Stock} ₹{round(row.Price,2)} | "
        f"Score {round(row.Score,3)} | "
        f"Exp 1M: {round(row.AdjExpReturn,2)}%\n"
    )

# =======================
# Allocation logic
# =======================
msg += f"\n💰 Investment Plan (₹{TOTAL_CAPITAL})\n\n"

MAX_PER = 0.30 * TOTAL_CAPITAL
PRICE_LIMIT = 0.60 * TOTAL_CAPITAL

alloc = top[top["Price"] <= PRICE_LIMIT].copy()

alloc["Score"] = alloc["Score"].clip(lower=0.01)
alloc["weight"] = alloc["Score"] / alloc["Score"].sum()
alloc["ideal"] = (alloc["weight"] * TOTAL_CAPITAL).clip(upper=MAX_PER)

alloc["shares"] = (alloc["ideal"] / alloc["Price"]).astype(int)
alloc["invested"] = alloc["shares"] * alloc["Price"]
alloc = alloc[alloc["shares"] > 0]

remaining = TOTAL_CAPITAL - alloc["invested"].sum()

for row in alloc.itertuples():
    msg += f"{row.Stock} – Buy {row.shares} shares (₹{int(row.invested)})\n"

msg += f"\nRemaining cash: ₹{int(remaining)}"

# =======================
# Portfolio expectation
# =======================
future_val = 0
for row in alloc.itertuples():
    future_val += row.invested * (1 + row.AdjExpReturn / 100)

future_val += remaining
gain = future_val - TOTAL_CAPITAL
pct = gain / TOTAL_CAPITAL * 100

msg += f"\n\n📈 Expected portfolio value after 1 month: ₹{int(future_val)}"
msg += f"\n(Expected gain: +₹{int(gain)} / +{round(pct,2)}%)"

send_text(msg)

# =======================
# Save tracking log
# =======================
log_rows = []
for row in top.itertuples():
    log_rows.append({
        "Date": today,
        "Stock": row.Stock,
        "Price": row.Price,
        "ExpReturn": row.AdjExpReturn,
        "RealizedReturn": np.nan
    })

log_df = pd.DataFrame(log_rows)

if os.path.exists(LOG_FILE):
    old = pd.read_csv(LOG_FILE)
    log_df = pd.concat([old, log_df], ignore_index=True)

log_df.to_csv(LOG_FILE, index=False)

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
