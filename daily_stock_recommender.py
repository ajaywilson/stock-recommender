import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import json
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
# JSON always next to script
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio.json")

print(f"📁 Portfolio file located at: {PORTFOLIO_FILE}")

# =======================
# Portfolio helpers
# =======================
def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return {"_meta": {"cash": TOTAL_CAPITAL}}
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            data = json.load(f)
            if "_meta" not in data:
                data["_meta"] = {"cash": TOTAL_CAPITAL}
            return data
    except:
        return {"_meta": {"cash": TOTAL_CAPITAL}}

def save_portfolio(p):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(p, f, indent=2)

portfolio = load_portfolio()
cash_available = float(portfolio["_meta"]["cash"])

# =======================
# Market day check
# =======================
def is_trading_day():
    nse = mcal.get_calendar("NSE")
    today = datetime.now().date()
    schedule = nse.schedule(start_date=today, end_date=today)
    return not schedule.empty

# Uncomment in production
# if not is_trading_day():
#     print("Market closed today. Exiting.")
#     exit()

# =======================
# Telegram helpers
# =======================
def send_text(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

# =======================
# Load Nifty 500
# =======================
url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
symbols = pd.read_csv(url)["Symbol"].dropna().unique().tolist()
symbols = [s.strip() + ".NS" for s in symbols if isinstance(s, str)]

print(f"Scanning {len(symbols)} stocks...")

nifty = yf.download("^NSEI", period="6mo", progress=False, threads=False)["Close"]

results = []
stock_data = {}

# =======================
# Scan stocks (UNCHANGED)
# =======================
for sym in symbols:
    try:
        df = yf.download(sym, period="6mo", progress=False, threads=False)

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
# Labels (UNCHANGED)
# =======================
def strength_label(score):
    if score >= 0.45: return "🟢 Strong"
    elif score >= 0.30: return "🟡 Moderate"
    else: return "🔴 Weak"

def risk_label(vol):
    if vol < 0.015: return "🟢 Low Risk"
    elif vol < 0.025: return "🟡 Medium Risk"
    else: return "🔴 High Risk"

def holding_period(vol):
    if vol < 0.015:
        return "3–6 weeks"
    elif vol < 0.025:
        return "1–3 weeks"
    else:
        return "3–7 days"

# =======================
# Exit logic (UNCHANGED)
# =======================
def exit_signal(df):
    close = df["Close"]
    open_ = df["Open"]
    high = df["High"]
    low = df["Low"]

    ema10 = close.ewm(span=10).mean()
    recent_high = high.tail(10).max()

    today_close = float(close.iloc[-1])
    today_open = float(open_.iloc[-1])

    below_ema = today_close < float(ema10.iloc[-1])
    trailing_stop = today_close < 0.94 * float(recent_high)

    body_ratio = abs(today_close - today_open) / (float(high.iloc[-1]) - float(low.iloc[-1]) + 1e-6)
    bearish_candle = today_close < today_open and body_ratio > 0.6

    momentum_break = (today_close / float(close.iloc[-6]) - 1) < -0.02

    if below_ema:
        return "🚨 Exit: Below 10EMA"
    elif trailing_stop:
        return "🚨 Exit: Trailing stop hit"
    elif bearish_candle:
        return "🚨 Exit: Bearish candle"
    elif momentum_break:
        return "🚨 Exit: Momentum breakdown"
    else:
        return "✅ Hold"

# =======================
# Price Action (UNCHANGED)
# =======================
signals = {}

for stock in top["Stock"]:
    if stock not in stock_data:
        continue

    df2 = stock_data[stock].tail(25)

    if df2.empty or len(df2) < 5:
        continue

    high = df2["High"]
    close = df2["Close"]
    open_ = df2["Open"]

    today_close = float(close.iloc[-1])
    today_open = float(open_.iloc[-1])

    recent_high = float(high.iloc[:-1].max())
    full_high = float(high.max())

    is_breakout = today_close > recent_high
    near_resistance = today_close > 0.97 * full_high

    bullish = today_close > today_open

    high_last = float(df2["High"].iloc[-1])
    low_last = float(df2["Low"].iloc[-1])

    body_ratio = abs(today_close - today_open) / (high_last - low_last + 1e-6)
    strong_candle = bool(bullish and body_ratio > 0.6)

    if is_breakout:
        signals[stock] = "📈 Breakout (Go for it)"
    elif near_resistance:
        signals[stock] = "⚠️ Near Resistance (Wait)"
    elif strong_candle:
        signals[stock] = "🟢 Strong Candle (Consider entry)"
    else:
        signals[stock] = "Neutral"

# =======================
# SELL from actual holdings (NEW)
# =======================
sell_list = []
for stock in list(portfolio.keys()):
    if stock == "_meta" or stock not in stock_data:
        continue

    df_recent = stock_data[stock].tail(30)
    exit_msg = exit_signal(df_recent)

    if "🚨 Exit" in exit_msg:
        shares = portfolio[stock]["shares"]
        price = float(stock_data[stock]["Close"].iloc[-1])
        cash_available += shares * price
        sell_list.append((stock, exit_msg))
        portfolio.pop(stock, None)

# =======================
# Telegram message (ORIGINAL OUTPUT KEPT)
# =======================
today = datetime.now().strftime("%Y-%m-%d")
msg = f"📊 Daily Stock Picks ({today})\n\n"

for i, row in enumerate(top.itertuples(), 1):
    if row.Stock not in stock_data:
        continue

    df_recent = stock_data[row.Stock].tail(30)
    exit_msg = exit_signal(df_recent)
    hold_time = holding_period(row.Volatility)

    msg += (
        f"{i}. {row.Stock} ₹{round(row.Price,2)} | "
        f"Score {round(row.Score,3)} | {strength_label(row.Score)} | "
        f"Risk: {risk_label(row.Volatility)}\n"
        f"   ➤ Price Action: {signals.get(row.Stock, 'Neutral')}\n"
        f"   ➤ Holding Period: {hold_time}\n"
        f"   ➤ Exit Signal: {exit_msg}\n\n"
    )

# =======================
# Allocation (UNCHANGED logic, but uses available cash)
# =======================
msg += f"\n💰 Investment Plan (Available: ₹{int(cash_available)})\n\n"

weights = {
    "📈 Breakout (Go for it)": 1.3,
    "🟢 Strong Candle (Consider entry)": 1.1,
    "Neutral": 1.0,
    "⚠️ Near Resistance (Wait)": 0.3
}

alloc = top.copy()
alloc["weight"] = alloc["Stock"].apply(lambda s: weights.get(signals.get(s, "Neutral"), 0))
alloc["adj_score"] = alloc["Score"] * alloc["weight"]

alloc = alloc[alloc["weight"] > 0]
alloc = alloc.sort_values(by="adj_score", ascending=False)

alloc["shares"] = 0
alloc["invested"] = 0.0
remaining = cash_available
MAX_PER_STOCK = 0.30 * TOTAL_CAPITAL

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
    portfolio[row.Stock] = {
        "shares": int(row.shares),
        "price": float(row.Price),
        "date": today
    }

msg += f"\nRemaining cash: ₹{int(remaining)}"

# =======================
# Buy/Sell summary (NEW but added after original output)
# =======================
msg += "\n\n📌 TRADE SUMMARY FOR NEXT SESSION\n"

if sell_list:
    msg += "\n📤 SELL TOMORROW\n"
    for stock, reason in sell_list:
        msg += f"{stock} – {reason}\n"
else:
    msg += "\n📤 SELL TOMORROW\nNone ✅\n"

if not alloc.empty:
    msg += "\n📥 BUY TOMORROW\n"
    for row in alloc.itertuples():
        msg += f"{row.Stock} – Buy {row.shares} shares (~₹{int(row.invested)})\n"
else:
    msg += "\n📥 BUY TOMORROW\nNo fresh buys today\n"

# =======================
# Save portfolio + send Telegram
# =======================
portfolio["_meta"]["cash"] = round(remaining, 2)
save_portfolio(portfolio)

send_text(msg)
print("Done. Telegram sent. Portfolio updated.")
