import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
TOP_N = 10
LOOKBACK = 60          # indicator lookback
HOLD_DAYS = 21         # 1 month hold
TRAIN_DAYS = 252       # ~12 months training window
TEST_DAYS = 63         # ~3 months test window

UNIVERSE_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

# -------------------------
# RSI
# -------------------------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(period).mean() / loss.rolling(period).mean()
    return 100 - (100 / (1 + rs))

# -------------------------
# Load universe
# -------------------------
symbols = pd.read_csv(UNIVERSE_URL)["Symbol"].tolist()
symbols = [s + ".NS" for s in symbols]

print(f"Universe: {len(symbols)} stocks")

# -------------------------
# Download 3 years of data
# -------------------------
print("Downloading price data...")
data = yf.download(symbols, period="3y", group_by="ticker", progress=False)
nifty = yf.download("^NSEI", period="3y", progress=False)["Close"]

dates = nifty.index

# -------------------------
# Factor scoring function
# -------------------------
def score_stock(df, nifty_slice):
    close = df["Close"]
    volume = df["Volume"]

    if len(close) < 63:
        return None

    ret_1m = close.iloc[-1] / close.iloc[-21] - 1
    ret_3m = close.iloc[-1] / close.iloc[-63] - 1

    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    rsi_val = rsi(close).iloc[-1]

    vol_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]

    rs_stock = close.iloc[-1] / close.iloc[-63]
    rs_index = nifty_slice.iloc[-1] / nifty_slice.iloc[-63]
    rel_strength = rs_stock / rs_index

    # Filters (quality gate)
    if not (close.iloc[-1] > ma20 > ma50):
        return None
    if not (45 < rsi_val < 65):
        return None
    if vol_ratio < 1.1:
        return None

    volatility = close.pct_change().std()

    score = (
        ret_1m * 0.4 +
        ret_3m * 0.3 +
        rel_strength * 0.2 -
        volatility * 0.1
    )

    return score

# -------------------------
# Walk-forward loop
# -------------------------
all_oos_returns = []

print("Running walk-forward validation...")

for start in range(TRAIN_DAYS, len(dates) - TEST_DAYS - HOLD_DAYS, TEST_DAYS):
    train_end = dates[start]
    test_start = dates[start + 1]
    test_end = dates[start + TEST_DAYS]

    print(f"\nTrain end: {train_end.date()} | Test window: {test_start.date()} → {test_end.date()}")

    test_returns = []

    for i in range(start + LOOKBACK, start + TEST_DAYS - HOLD_DAYS):
        today = dates[i]
        future = dates[i + HOLD_DAYS]

        scores = []

        nifty_slice = nifty.loc[:today]

        for sym in symbols:
            try:
                df = data[sym].loc[:today].dropna()
                if len(df) < LOOKBACK:
                    continue

                s = score_stock(df, nifty_slice)
                if s is not None:
                    scores.append((sym, s))
            except:
                continue

        top = sorted(scores, key=lambda x: x[1], reverse=True)[:TOP_N]

        rets = []
        for sym, _ in top:
            try:
                p0 = data[sym].loc[today]["Close"]
                p1 = data[sym].loc[future]["Close"]
                rets.append((p1 / p0) - 1)
            except:
                continue

        if rets:
            test_returns.append(np.mean(rets))

    if test_returns:
        avg = np.mean(test_returns)
        print(f"  OOS avg return: {avg*100:.2f}%")
        all_oos_returns.extend(test_returns)

# -------------------------
# Final evaluation
# -------------------------
all_oos_returns = np.array(all_oos_returns)

print("\n========== WALK-FORWARD RESULTS ==========")
print(f"Total test periods: {len(all_oos_returns)}")
print(f"Win rate: {(all_oos_returns > 0).mean()*100:.2f}%")
print(f"Average return per period: {all_oos_returns.mean()*100:.2f}%")
print(f"Best period: {all_oos_returns.max()*100:.2f}%")
print(f"Worst period: {all_oos_returns.min()*100:.2f}%")

# Equity curve
equity = (1 + all_oos_returns).cumprod()

plt.figure(figsize=(10,6))
plt.plot(equity, label="Walk-forward Equity")
plt.title("Out-of-sample Walk-forward Equity Curve")
plt.xlabel("Test trades")
plt.ylabel("Growth of 1 unit")
plt.grid()
plt.legend()
plt.show()
