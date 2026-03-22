"""
============================================================
  Supertrend + MA Confluence Screener — Web App
  Run:  python app.py
  Open: http://localhost:5000
============================================================
"""

import warnings
import threading
import uuid
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO

from flask import Flask, jsonify, render_template_string, request, Response
import json

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
ATR_PERIOD        = 10
FACTOR            = 3.0
MA_FAST           = 10
MA_SLOW           = 20
MA_CROSS_LOOKBACK = 10
HISTORY_PERIOD    = "1y"
INTERVAL          = "1d"
BATCH_SIZE        = 50

IWV_URL = (
    "https://www.ishares.com/us/products/239707/ishares-russell-3000-etf"
    "/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
)

# In-memory job store  { job_id: { status, results, errors, progress, total } }
jobs = {}

# ── TICKER FETCH ──────────────────────────────────────────────────────────────
def fetch_iwv_tickers():
    try:
        r = requests.get(IWV_URL, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text), skiprows=9, dtype=str)
        if "Ticker" not in df.columns:
            raise ValueError("Ticker column not found")
        tickers = (
            df["Ticker"].dropna().astype(str).str.strip().str.upper()
        )
        tickers = tickers[tickers.str.match(r'^[A-Z]{1,5}$')].unique().tolist()
        if len(tickers) < 400:
            raise ValueError(f"Only {len(tickers)} tickers returned")
        return tickers, "live"
    except Exception as e:
        return [], str(e)

# ── INDICATORS ────────────────────────────────────────────────────────────────
def calc_supertrend(df):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + FACTOR * atr
    basic_lower = hl2 - FACTOR * atr
    n = len(df)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    direction   = np.ones(n)
    st_line     = np.zeros(n)
    for i in range(1, n):
        fu = (basic_upper.iloc[i]
              if basic_upper.iloc[i] < final_upper[i-1]
              or close.iloc[i-1] > final_upper[i-1]
              else final_upper[i-1])
        fl = (basic_lower.iloc[i]
              if basic_lower.iloc[i] > final_lower[i-1]
              or close.iloc[i-1] < final_lower[i-1]
              else final_lower[i-1])
        final_upper[i] = fu
        final_lower[i] = fl
        if st_line[i-1] == final_upper[i-1]:
            direction[i] = 1 if close.iloc[i] <= final_upper[i] else -1
        else:
            direction[i] = -1 if close.iloc[i] >= final_lower[i] else 1
        st_line[i] = final_lower[i] if direction[i] == -1 else final_upper[i]
    return pd.Series(direction, index=df.index)

def calc_indicators(df):
    df = df.copy()
    df["ma_fast"]   = df["Close"].rolling(MA_FAST).mean()
    df["ma_slow"]   = df["Close"].rolling(MA_SLOW).mean()
    df["direction"] = calc_supertrend(df).values
    return df

def ma_cross_within(df):
    fast, slow = df["ma_fast"].values, df["ma_slow"].values
    last = len(df) - 1
    bull = bear = False
    for i in range(last, 0, -1):
        if (last - i) > MA_CROSS_LOOKBACK:
            break
        if any(pd.isna(v) for v in [fast[i], slow[i], fast[i-1], slow[i-1]]):
            continue
        if fast[i-1] <= slow[i-1] and fast[i] > slow[i]:
            return True, False
        if fast[i-1] >= slow[i-1] and fast[i] < slow[i]:
            return False, True
    return bull, bear

def check_signals(df):
    if len(df) < MA_SLOW + MA_CROSS_LOOKBACK + 2:
        return None
    row = df.iloc[-1]
    if pd.isna(row["ma_fast"]) or pd.isna(row["ma_slow"]):
        return None
    direction = row["direction"]
    mf, ms    = row["ma_fast"], row["ma_slow"]
    bull, bear = ma_cross_within(df)
    if (direction == -1 and mf > ms and bull
            and row["Low"] <= ms and row["Close"] > ms and row["Close"] > row["Open"]):
        return "LONG"
    if (direction == 1 and mf < ms and bear
            and row["High"] >= ms and row["Close"] < ms and row["Close"] < row["Open"]):
        return "SHORT"
    return None

# ── BATCH DOWNLOAD ────────────────────────────────────────────────────────────
def download_batch(tickers):
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers, period=HISTORY_PERIOD, interval=INTERVAL,
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )
    except Exception:
        return {}
    if raw is None or raw.empty:
        return {}

    result = {}
    min_bars = MA_SLOW + ATR_PERIOD + 5

    if len(tickers) == 1:
        df = raw.dropna(how="all")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) >= min_bars:
            result[tickers[0]] = df
        return result

    cols = raw.columns
    if not isinstance(cols, pd.MultiIndex):
        return result

    level0  = set(cols.get_level_values(0).unique())
    fields  = {"Close", "High", "Low", "Open", "Volume"}
    fl      = 0 if fields & level0 else 1
    tl      = 1 - fl

    for tkr in tickers:
        try:
            tc = cols[cols.get_level_values(tl) == tkr]
            if tc.empty:
                continue
            sub = raw[tc].copy()
            sub.columns = sub.columns.get_level_values(fl)
            sub = sub.dropna(how="all")
            ohlc = [c for c in ["Open","High","Low","Close"] if c in sub.columns]
            sub = sub.dropna(subset=ohlc, how="all")
            if len(sub) >= min_bars:
                result[tkr] = sub
        except Exception:
            continue
    return result

# ── BACKGROUND SCAN JOB ───────────────────────────────────────────────────────
def run_scan(job_id, tickers):
    job = jobs[job_id]
    job["total"]    = len(tickers)
    job["done"]     = 0
    job["results"]  = []
    job["errors"]   = 0
    job["status"]   = "running"
    job["current"]  = ""

    batches = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

    for batch in batches:
        if job.get("stop"):
            break
        job["current"] = ", ".join(batch[:5]) + ("..." if len(batch) > 5 else "")
        try:
            price_data = download_batch(batch)
        except Exception:
            job["errors"] += len(batch)
            job["done"]   += len(batch)
            continue

        for tkr in batch:
            if tkr not in price_data:
                job["errors"] += 1
                job["done"]   += 1
                continue
            try:
                df  = calc_indicators(price_data[tkr])
                sig = check_signals(df)
            except Exception:
                job["errors"] += 1
                job["done"]   += 1
                continue

            if sig:
                last = df.iloc[-1]
                job["results"].append({
                    "ticker": tkr,
                    "signal": sig,
                    "close":  round(float(last["Close"]), 2),
                    "ma10":   round(float(last["ma_fast"]), 2),
                    "ma20":   round(float(last["ma_slow"]), 2),
                    "date":   df.index[-1].strftime("%Y-%m-%d"),
                })
            job["done"] += 1

    job["status"] = "done"

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/tickers")
def api_tickers():
    tickers, status = fetch_iwv_tickers()
    return jsonify({"tickers": tickers, "count": len(tickers), "status": status})

@app.route("/api/scan", methods=["POST"])
def api_scan():
    data    = request.json or {}
    custom  = data.get("tickers", "")
    if custom:
        tickers = [t.strip().upper() for t in custom.replace(",", " ").split() if t.strip()]
    else:
        tickers, _ = fetch_iwv_tickers()
        if not tickers:
            return jsonify({"error": "Could not fetch Russell 3000 tickers"}), 500

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "starting", "results": [], "errors": 0, "done": 0, "total": 0}
    t = threading.Thread(target=run_scan, args=(job_id, tickers), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})

@app.route("/api/job/<job_id>")
def api_job(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status":   job["status"],
        "done":     job["done"],
        "total":    job["total"],
        "errors":   job["errors"],
        "current":  job.get("current", ""),
        "results":  job["results"],
    })

@app.route("/api/job/<job_id>/stop", methods=["POST"])
def api_stop(job_id):
    job = jobs.get(job_id)
    if job:
        job["stop"] = True
    return jsonify({"ok": True})

# ── HTML FRONTEND ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Supertrend + MA Screener — Russell 3000</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f4f4f2;color:#1a1a18;min-height:100vh}
.shell{max-width:1040px;margin:0 auto;padding:36px 20px 60px}
.hd-eye{font-size:10px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:#888;margin-bottom:3px}
.hd-title{font-size:26px;font-weight:600;letter-spacing:-.02em;margin-bottom:3px}
.hd-sub{font-size:12px;color:#888;margin-bottom:24px}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end;margin-bottom:20px}
.field{display:flex;flex-direction:column;gap:4px;flex:1;min-width:220px}
.lbl{font-size:11px;font-weight:500;color:#666}
input[type=text]{height:38px;padding:0 12px;border-radius:6px;border:1px solid #d4d4d0;background:#fff;font-size:13px;color:#1a1a18;width:100%}
input[type=text]:focus{outline:none;border-color:#555}
input:disabled{background:#f0f0ee;color:#aaa}
.btn{height:38px;padding:0 22px;border-radius:6px;font-size:13px;font-weight:500;border:1px solid #d4d4d0;background:#fff;color:#1a1a18;white-space:nowrap;transition:background .12s}
.btn:hover{background:#eee}
.btn.primary{background:#1a1a18;color:#fff;border-color:#1a1a18}
.btn.primary:hover{background:#333}
.btn.danger{background:#fff0f0;border-color:#f5a0a0;color:#c00}
.btn:disabled{opacity:.45;cursor:not-allowed}
.prog-wrap{margin-bottom:20px}
.prog-meta{display:flex;justify-content:space-between;font-size:11px;color:#888;margin-bottom:5px}
.prog-bg{height:4px;background:#e0e0dc;border-radius:2px;overflow:hidden}
.prog-fill{height:100%;background:#1a1a18;border-radius:2px;transition:width .4s}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}
.card{background:#fff;border:1px solid #e4e4e0;border-radius:8px;padding:14px 16px}
.card-lbl{font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#888;margin-bottom:4px}
.card-val{font-size:26px;font-weight:600;letter-spacing:-.02em}
.cn{color:#1a1a18}.cl{color:#0a7a50}.cs{color:#b83020}.ce{color:#c07000}
.tabs{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap}
.tab{padding:5px 16px;font-size:12px;font-weight:500;border-radius:20px;border:1px solid #d4d4d0;background:#fff;color:#555;cursor:pointer}
.tab:hover{border-color:#aaa}
.tab.on{background:#1a1a18;color:#fff;border-color:#1a1a18}
.tab.on-l{background:#0a7a50;color:#fff;border-color:#0a7a50}
.tab.on-s{background:#b83020;color:#fff;border-color:#b83020}
.tbl-wrap{border:1px solid #e4e4e0;border-radius:8px;overflow:hidden;background:#fff;margin-bottom:20px}
table{width:100%;border-collapse:collapse;font-size:13px}
thead tr{background:#f8f8f6}
th{padding:9px 14px;font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:#888;border-bottom:1px solid #e4e4e0;white-space:nowrap;cursor:pointer;user-select:none}
th:hover{color:#333}
td{padding:9px 14px;border-bottom:1px solid #f0f0ee}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:#fafaf8}
.num{text-align:right;font-family:'SF Mono',Consolas,monospace;font-size:12px}
.badge{display:inline-block;font-size:10px;font-weight:700;letter-spacing:.06em;padding:2px 8px;border-radius:4px}
.bl{background:#e0f5ec;color:#0a7a50}
.bs{background:#fde8e4;color:#b83020}
.empty{text-align:center;padding:48px;color:#888;font-size:14px}
.legend{padding:14px 16px;border:1px solid #e4e4e0;border-radius:8px;background:#fff;font-size:12px;color:#666;line-height:2}
.legend-title{font-weight:600;color:#1a1a18;margin-bottom:2px;font-size:13px}
.ll{color:#0a7a50;font-weight:600}.ls{color:#b83020;font-weight:600}
.dl-btn{padding:6px 14px;font-size:12px;border-radius:6px;border:1px solid #d4d4d0;background:#fff;cursor:pointer;float:right;margin-top:-2px}
.dl-btn:hover{background:#f0f0ee}
@media(max-width:600px){.cards{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="shell">
  <div class="hd-eye">Supertrend + MA Confluence</div>
  <div class="hd-title">Signal Screener</div>
  <div class="hd-sub">Daily bars &nbsp;·&nbsp; Supertrend ATR(10) Factor 3.0 &nbsp;·&nbsp; MA10/MA20 &nbsp;·&nbsp; Russell 3000</div>

  <div class="controls">
    <div class="field">
      <div class="lbl">Custom tickers — leave blank to screen full Russell 3000</div>
      <input type="text" id="customInput" placeholder="e.g. AAPL, MSFT, NVDA, TSLA">
    </div>
    <button class="btn primary" id="runBtn" onclick="startScan()">&#9654;&nbsp; Run Screen</button>
    <button class="btn danger"  id="stopBtn" onclick="stopScan()" style="display:none">&#9632;&nbsp; Stop</button>
  </div>

  <div class="prog-wrap" id="progWrap" style="display:none">
    <div class="prog-meta">
      <span id="progLabel">Scanning...</span>
      <span id="progCount">0 / 0</span>
    </div>
    <div class="prog-bg"><div class="prog-fill" id="progFill" style="width:0%"></div></div>
  </div>

  <div class="cards" id="cards" style="display:none">
    <div class="card"><div class="card-lbl">Signals</div><div class="card-val cn" id="cTotal">0</div></div>
    <div class="card"><div class="card-lbl">Long</div><div class="card-val cl" id="cLong">0</div></div>
    <div class="card"><div class="card-lbl">Short</div><div class="card-val cs" id="cShort">0</div></div>
    <div class="card"><div class="card-lbl">Errors</div><div class="card-val ce" id="cErr">0</div></div>
  </div>

  <div id="tabsWrap" style="display:none">
    <div class="tabs">
      <button class="tab on"   id="tabAll"   onclick="setFilter('ALL')">All</button>
      <button class="tab"      id="tabLong"  onclick="setFilter('LONG')">Long</button>
      <button class="tab"      id="tabShort" onclick="setFilter('SHORT')">Short</button>
      <button class="dl-btn"   onclick="downloadCSV()">&#8595; CSV</button>
    </div>
  </div>

  <div id="tblWrap" style="display:none" class="tbl-wrap">
    <table>
      <thead>
        <tr>
          <th onclick="sortBy('ticker')">Ticker <span id="s-ticker">↕</span></th>
          <th>Signal</th>
          <th onclick="sortBy('close')" style="text-align:right">Close ($) <span id="s-close">↕</span></th>
          <th onclick="sortBy('ma10')"  style="text-align:right">MA10 ($) <span id="s-ma10">↕</span></th>
          <th onclick="sortBy('ma20')"  style="text-align:right">MA20 ($) <span id="s-ma20">↕</span></th>
          <th onclick="sortBy('date')">Date <span id="s-date">↕</span></th>
        </tr>
      </thead>
      <tbody id="tblBody"></tbody>
    </table>
  </div>

  <div id="emptyMsg" class="empty" style="display:none">No signals found matching all criteria.</div>

  <div class="legend">
    <div class="legend-title">Signal criteria — all conditions must be true simultaneously</div>
    <span class="ll">LONG</span> — Supertrend uptrend (dir=−1) &nbsp;·&nbsp; MA10 &gt; MA20 &nbsp;·&nbsp;
    Bullish cross ≤10 bars ago &nbsp;·&nbsp; Low ≤ MA20, close &gt; MA20 &nbsp;·&nbsp; Close &gt; Open<br>
    <span class="ls">SHORT</span> — Supertrend downtrend (dir=+1) &nbsp;·&nbsp; MA10 &lt; MA20 &nbsp;·&nbsp;
    Bearish cross ≤10 bars ago &nbsp;·&nbsp; High ≥ MA20, close &lt; MA20 &nbsp;·&nbsp; Close &lt; Open
  </div>
</div>

<script>
let currentJobId = null;
let pollTimer    = null;
let allResults   = [];
let filter       = 'ALL';
let sortKey      = 'ticker';
let sortAsc      = true;

async function startScan() {
  const custom = document.getElementById('customInput').value.trim();
  const res    = await fetch('/api/scan', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ tickers: custom })
  });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }

  currentJobId = data.job_id;
  allResults   = [];
  filter       = 'ALL';

  document.getElementById('runBtn').style.display  = 'none';
  document.getElementById('stopBtn').style.display = '';
  document.getElementById('customInput').disabled  = true;
  document.getElementById('progWrap').style.display = '';
  document.getElementById('cards').style.display    = 'grid';
  document.getElementById('tabsWrap').style.display = 'none';
  document.getElementById('tblWrap').style.display  = 'none';
  document.getElementById('emptyMsg').style.display = 'none';

  pollTimer = setInterval(pollJob, 1500);
}

async function pollJob() {
  if (!currentJobId) return;
  const res  = await fetch(`/api/job/${currentJobId}`);
  const data = await res.json();

  const pct = data.total ? Math.round(data.done / data.total * 100) : 0;
  document.getElementById('progFill').style.width = pct + '%';
  document.getElementById('progCount').textContent = `${data.done} / ${data.total} (${pct}%)`;
  document.getElementById('progLabel').textContent =
    data.status === 'done' ? 'Complete' : `Scanning: ${data.current}`;

  allResults = data.results;
  updateCards(data);
  renderTable();

  if (data.status === 'done') {
    clearInterval(pollTimer);
    document.getElementById('runBtn').style.display  = '';
    document.getElementById('stopBtn').style.display = 'none';
    document.getElementById('customInput').disabled  = false;
    if (data.results.length > 0) {
      document.getElementById('tabsWrap').style.display = '';
      document.getElementById('tblWrap').style.display  = '';
    } else {
      document.getElementById('emptyMsg').style.display = '';
    }
  }
}

async function stopScan() {
  if (!currentJobId) return;
  await fetch(`/api/job/${currentJobId}/stop`, { method: 'POST' });
}

function updateCards(data) {
  const longs  = data.results.filter(r => r.signal === 'LONG').length;
  const shorts = data.results.filter(r => r.signal === 'SHORT').length;
  document.getElementById('cTotal').textContent = data.results.length;
  document.getElementById('cLong').textContent  = longs;
  document.getElementById('cShort').textContent = shorts;
  document.getElementById('cErr').textContent   = data.errors;
}

function setFilter(f) {
  filter = f;
  ['ALL','LONG','SHORT'].forEach(x => {
    const btn = document.getElementById('tab' + x.charAt(0) + x.slice(1).toLowerCase());
    btn.className = 'tab' + (f === x ? (x==='ALL'?' on':x==='LONG'?' on-l':' on-s') : '');
  });
  // fix tab IDs
  document.getElementById('tabAll').className   = 'tab' + (f==='ALL'  ?' on':'');
  document.getElementById('tabLong').className  = 'tab' + (f==='LONG' ?' on-l':'');
  document.getElementById('tabShort').className = 'tab' + (f==='SHORT'?' on-s':'');
  renderTable();
}

function sortBy(key) {
  if (sortKey === key) sortAsc = !sortAsc;
  else { sortKey = key; sortAsc = true; }
  ['ticker','close','ma10','ma20','date'].forEach(k => {
    document.getElementById('s-' + k).textContent =
      sortKey === k ? (sortAsc ? '↑' : '↓') : '↕';
  });
  renderTable();
}

function renderTable() {
  const shown = filter === 'ALL' ? allResults : allResults.filter(r => r.signal === filter);
  const sorted = [...shown].sort((a, b) => {
    const av = isNaN(parseFloat(a[sortKey])) ? a[sortKey] : parseFloat(a[sortKey]);
    const bv = isNaN(parseFloat(b[sortKey])) ? b[sortKey] : parseFloat(b[sortKey]);
    return sortAsc ? (av < bv ? -1 : av > bv ? 1 : 0) : (av > bv ? -1 : av < bv ? 1 : 0);
  });
  document.getElementById('tblBody').innerHTML = sorted.map(r => `
    <tr>
      <td><b>${r.ticker}</b></td>
      <td><span class="badge ${r.signal==='LONG'?'bl':'bs'}">${r.signal}</span></td>
      <td class="num">${r.close}</td>
      <td class="num">${r.ma10}</td>
      <td class="num">${r.ma20}</td>
      <td style="color:#888;font-size:12px">${r.date}</td>
    </tr>`).join('');
  if (shown.length > 0) {
    document.getElementById('tblWrap').style.display  = '';
    document.getElementById('emptyMsg').style.display = 'none';
  }
}

function downloadCSV() {
  const shown = filter === 'ALL' ? allResults : allResults.filter(r => r.signal === filter);
  const header = 'Ticker,Signal,Close,MA10,MA20,Date';
  const rows = shown.map(r => `${r.ticker},${r.signal},${r.close},${r.ma10},${r.ma20},${r.date}`);
  const csv  = [header, ...rows].join('\\n');
  const a    = document.createElement('a');
  a.href     = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
  a.download = `signals_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("\\n" + "="*55)
    print("  Supertrend + MA Screener — Russell 3000")
    print("  Open: http://localhost:5000")
    print("="*55 + "\\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
