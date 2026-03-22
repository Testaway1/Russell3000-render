# Supertrend + MA Screener — Russell 3000

A web app version of the supertrend_trial_russell3000.py screener.
Runs locally — no CORS issues, uses yfinance directly.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

## Usage

- Leave the ticker box **blank** to screen the full Russell 3000 (fetched live from iShares IWV)
- Or enter **custom tickers** (e.g. `AAPL, MSFT, NVDA`) for a quick test
- Click **Run Screen** — progress updates live as batches complete
- Results are sortable by any column
- Click **CSV** to download the current results

## Signal criteria

**LONG** — all must be true:
1. Supertrend uptrend (direction = −1)
2. MA10 > MA20
3. MA10 crossed above MA20 within last 10 bars
4. Low ≤ MA20 and close > MA20 (touched and bounced)
5. Close > Open (bullish candle)

**SHORT** — all must be true:
1. Supertrend downtrend (direction = +1)
2. MA10 < MA20
3. MA10 crossed below MA20 within last 10 bars
4. High ≥ MA20 and close < MA20 (touched and rejected)
5. Close < Open (bearish candle)

## Deploy online (optional)

To run from anywhere, deploy to a free tier on **Railway** or **Render**:

```bash
# Railway
npm install -g @railway/cli
railway login
railway init
railway up
```

Or **Render**: connect your GitHub repo, set build command to
`pip install -r requirements.txt` and start command to `python app.py`.
