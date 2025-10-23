# app.py
# MARUTI stock forecasting dashboard
# Uses your files: MARUTI.csv, maruti_lr_model.pkl, maruti_lstm_model.h5, maruti_scaler.pkl
# Home page = two buttons (BSE / NIFTY). BSE shows LR vs LSTM vs Quantum (PennyLane).

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel

import os
from pathlib import Path
import datetime as dt
import io, base64

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# PennyLane quantum pieces
import pennylane as qml
from pennylane import numpy as pnp

# -----------------------------
# App & paths
# -----------------------------
app = FastAPI(title="Stock Forecasting Dashboard – MARUTI", version="2.0")
BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Your artifact paths (can override with env vars if needed)
DATA_CSV_PATH   = os.getenv("DATA_CSV_PATH",   str(BASE_DIR / "MARUTI.csv"))
LR_MODEL_PATH   = os.getenv("LR_MODEL_PATH",   str(BASE_DIR / "maruti_lr_model.pkl"))
LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", str(BASE_DIR / "maruti_lstm_model.h5"))
SCALER_PATH     = os.getenv("SCALER_PATH",     str(BASE_DIR / "maruti_scaler.pkl"))

# LSTM window length (set to your model’s expected timesteps)
LSTM_WINDOW = int(os.getenv("LSTM_WINDOW", "60"))

# Quantum window/qubits (mirrors your TF-IDF → AngleEmbedding flow; small for demo)
Q_WINDOW = int(os.getenv("Q_WINDOW", "8"))   # window size used for quantum pattern features
N_QUBITS = Q_WINDOW                          # 1 feature per qubit (AngleEmbedding)

# -----------------------------
# Globals loaded once
# -----------------------------
lr_model = None
lstm_model = None
scaler = None

df = None  # MARUTI dataframe with Date, Close, optional Volume

# quantum index built from history (windows -> quantum features), like your TF-IDF matrix
q_dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(q_dev)
def quantum_encoder(x):
    # x is length N_QUBITS (already normalized/angle-scaled)
    qml.templates.AngleEmbedding(x, wires=range(N_QUBITS))
    qml.templates.BasicEntanglerLayers(weights=pnp.ones((1, N_QUBITS)), wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

quantum_feature_matrix = None   # shape: [num_windows, N_QUBITS]
quantum_windows_next = None     # next-day close following each window

# -----------------------------
# Utilities
# -----------------------------
def assert_exists(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at: {path}")

def load_maruti_csv() -> pd.DataFrame:
    assert_exists(DATA_CSV_PATH, "MARUTI.csv")
    raw = pd.read_csv(DATA_CSV_PATH)

    # normalize columns
    cols = {c: c.strip().lower() for c in raw.columns}
    raw = raw.rename(columns=cols)

    date_col = next((c for c in ["date","timestamp","datetime"] if c in raw.columns), None)
    close_col = next((c for c in ["close","adj close","closing price","close_price"] if c in raw.columns), None)
    if date_col is None or close_col is None:
        raise ValueError("CSV must contain a date column (date/timestamp) and a close column (close/adj close/closing price).")

    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    out = pd.DataFrame({
        "Date": raw[date_col],
        "Close": raw[close_col].astype(float)
    })

    # optional volume
    for cand in ["volume","vol","total trade quantity","shares"]:
        if cand in raw.columns:
            out["Volume"] = pd.to_numeric(raw[cand], errors="coerce")
            break

    return out

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def make_price_chart(dates, closes, title: str) -> str:
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(dates, closes)
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("Close")
    fig.autofmt_xdate()
    return fig_to_b64(fig)

def make_lr_lstm_quant_bar(last_close, lr_pred, lstm_pred, q_pred, title: str) -> str:
    fig = plt.figure()
    ax = fig.gca()
    labels = ["Last Close", "LR (pkl)", "LSTM (h5)", "Quantum kNN"]
    vals = [
        float(last_close),
        float(lr_pred) if lr_pred == lr_pred else 0.0,
        float(lstm_pred) if lstm_pred == lstm_pred else 0.0,
        float(q_pred) if q_pred == q_pred else 0.0
    ]
    ax.bar(labels, vals)
    ax.set_title(title); ax.set_ylabel("Price")
    return fig_to_b64(fig)

def make_forward_overlay(dates, closes, next_dates, lr_values, lstm_values, q_values, title: str) -> str:
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(dates, closes)
    if next_dates:
        if lr_values:   ax.scatter(next_dates[:1], lr_values[:1], s=40)  # only day-1 for LR
        if lstm_values: ax.scatter(next_dates, lstm_values, s=40)
        if q_values:    ax.scatter(next_dates[:1], q_values[:1], s=40)   # day-1 for quantum baseline
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("Close / Predicted")
    fig.autofmt_xdate()
    return fig_to_b64(fig)

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = (delta.clip(lower=0)).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def make_rsi_chart(dates, rsi_vals, title: str) -> str:
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(dates, rsi_vals)
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("RSI")
    fig.autofmt_xdate()
    return fig_to_b64(fig)

def make_volume_chart(dates, volume, title: str) -> str:
    fig = plt.figure()
    ax = fig.gca()
    ax.bar(dates, volume)
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("Volume")
    fig.autofmt_xdate()
    return fig_to_b64(fig)

def make_sparkline(series: pd.Series) -> str:
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(series.tail(60))
    ax.set_xticks([]); ax.set_yticks([])
    return fig_to_b64(fig)

# -----------------------------
# Load models
# -----------------------------
def load_artifacts():
    global lr_model, lstm_model, scaler
    if lr_model is None:
        assert_exists(LR_MODEL_PATH, "LR model (.pkl)")
        lr_model = joblib.load(LR_MODEL_PATH)
    if lstm_model is None:
        assert_exists(LSTM_MODEL_PATH, "LSTM model (.h5)")
        from tensorflow.keras.models import load_model
        lstm_model = load_model(LSTM_MODEL_PATH)
    if scaler is None:
        assert_exists(SCALER_PATH, "Scaler (.pkl)")
        scaler = joblib.load(SCALER_PATH)

# -----------------------------
# LR & LSTM predictions
# -----------------------------
def predict_next_close_lr(closes: np.ndarray) -> float:
    if lr_model is None:
        return float("nan")
    last = closes[-1]
    ret5 = (closes[-1] / closes[-6]) - 1.0 if len(closes) >= 6 else 0.0
    sma5 = float(np.mean(closes[-5:])) if len(closes) >= 5 else float(last)
    X = np.array([[last, ret5, sma5]])
    try:
        return float(lr_model.predict(X)[0])
    except Exception:
        return float(last)

def predict_next_close_lstm(closes: np.ndarray) -> float:
    if lstm_model is None or scaler is None:
        return float("nan")
    try:
        c = closes.reshape(-1, 1)
        scaled = scaler.transform(c).flatten()
        if len(scaled) < LSTM_WINDOW:
            pad = np.full((LSTM_WINDOW - len(scaled),), scaled[0])
            seq = np.concatenate([pad, scaled]).reshape(1, LSTM_WINDOW, 1)
        else:
            seq = scaled[-LSTM_WINDOW:].reshape(1, LSTM_WINDOW, 1)
        pred_scaled = lstm_model.predict(seq, verbose=0)[0][0]
        inv = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1))[0][0]
        return float(inv)
    except Exception:
        return float(closes[-1])

def predict_n_days_lstm(closes: np.ndarray, n_days: int) -> list[float]:
    preds, history = [], closes.astype(float).tolist()
    for _ in range(n_days):
        nxt = predict_next_close_lstm(np.array(history))
        preds.append(nxt)
        history.append(nxt)
    return preds

# -----------------------------
# Quantum pattern forecasting (adapts your TF-IDF + encoder + cosine_similarity)
# We build a library of windows -> quantum features; for a query window (latest),
# we find top-k most similar (cosine in quantum feature space) and average the
# following-day closes as the forecast.
# -----------------------------
def build_quantum_index(series: pd.Series):
    global quantum_feature_matrix, quantum_windows_next
    values = series.values.astype(float)
    if len(values) <= Q_WINDOW:
        quantum_feature_matrix = None
        quantum_windows_next = None
        return

    # create sliding windows
    windows = []
    next_vals = []
    for i in range(len(values) - Q_WINDOW):
        w = values[i:i+Q_WINDOW]
        nrm = w / (np.linalg.norm(w) + 1e-12)           # like normalize(..., norm="l2")
        angles = nrm                                     # already in [-1,1], AngleEmbedding accepts real angles
        qfeat = np.array(quantum_encoder(pnp.array(angles)), dtype=np.float64)
        windows.append(qfeat)
        next_vals.append(values[i+Q_WINDOW])             # the next-day close after the window

    quantum_feature_matrix = np.vstack(windows)          # [num_windows, N_QUBITS]
    quantum_windows_next = np.array(next_vals)           # [num_windows]

def quantum_forecast_next(series: pd.Series, top_k: int = 5) -> float:
    if quantum_feature_matrix is None or quantum_windows_next is None:
        return float("nan")
    vals = series.values.astype(float)
    if len(vals) < Q_WINDOW:
        return float("nan")

    query = vals[-Q_WINDOW:]
    qn = query / (np.linalg.norm(query) + 1e-12)
    qfeat = np.array(quantum_encoder(pnp.array(qn)), dtype=np.float64).reshape(1, -1)

    sims = cosine_similarity(qfeat, quantum_feature_matrix).flatten()
    top_idx = sims.argsort()[::-1][:max(1, top_k)]
    # average the next-day closes of top matches
    return float(np.mean(quantum_windows_next[top_idx]))

# -----------------------------
# Load data + build quantum index once at startup
# -----------------------------
df = load_maruti_csv()
build_quantum_index(df["Close"])

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home(request: Request):
    # Minimal hero page with two buttons (per your screenshot)
    meta = {
        "year": dt.datetime.now().year,
        "title": "Welcome to Stock Forecasting Dashboard"
    }
    return templates.TemplateResponse("index.html", {"request": request, "meta": meta})

@app.get("/bse")
def bse(request: Request, days: int = 7):
    try:
        load_artifacts()
        closes = df["Close"].values.astype(float)
        last_close = float(closes[-1])

        # predictions
        lr_next = predict_next_close_lr(closes)
        lstm_preds = predict_n_days_lstm(closes, days)
        lstm_day1 = float(lstm_preds[0]) if lstm_preds else float("nan")
        q_next = quantum_forecast_next(df["Close"], top_k=5)

        # charts
        price_img = make_price_chart(df["Date"], df["Close"], "MARUTI – Close Price")
        comp_bar = make_lr_lstm_quant_bar(last_close, lr_next, lstm_day1, q_next, "LR vs LSTM vs Quantum")
        next_dates = [df["Date"].iloc[-1] + pd.Timedelta(days=i+1) for i in range(days)]
        lr_values = [lr_next] + [np.nan]*(days-1)
        q_values  = [q_next]  + [np.nan]*(days-1)
        forward_overlay = make_forward_overlay(
            df["Date"].tail(120), df["Close"].tail(120).values,
            next_dates, lr_values, lstm_preds, q_values,
            f"Last 120 + Next {days} day(s)"
        )

        return templates.TemplateResponse(
            "bse.html",
            {
                "request": request,
                "last_close": f"{last_close:.2f}",
                "lr_next": f"{lr_next:.2f}" if lr_next == lr_next else "NA",
                "lstm_day1": f"{lstm_day1:.2f}" if lstm_day1 == lstm_day1 else "NA",
                "q_next": f"{q_next:.2f}" if q_next == q_next else "NA",
                "price_img": price_img,
                "comp_bar": comp_bar,
                "forward_overlay": forward_overlay,
                "days": days,
            },
        )
    except Exception as e:
        return templates.TemplateResponse("bse.html", {"request": request, "error": str(e), "days": days})

@app.get("/nifty")
def nifty(request: Request):
    try:
        s20 = sma(df["Close"], 20)
        s50 = sma(df["Close"], 50)
        rsi14 = rsi(df["Close"], 14)

        overlay_img = make_price_chart(df["Date"], df["Close"], "MARUTI – Price")
        # Add SMA overlay
        fig = plt.figure(); ax = fig.gca()
        ax.plot(df["Date"], df["Close"], label="Close")
        ax.plot(df["Date"], s20, label="SMA 20")
        ax.plot(df["Date"], s50, label="SMA 50")
        ax.set_title("MARUTI – Price + SMA(20/50)"); ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
        fig.autofmt_xdate()
        overlay_img = fig_to_b64(fig)

        rsi_img = make_rsi_chart(df["Date"], rsi14, "MARUTI – RSI(14)")
        volume_img = None
        if "Volume" in df.columns:
            volume_img = make_volume_chart(df["Date"], df["Volume"], "MARUTI – Volume")

        latest = {
            "date": str(df["Date"].iloc[-1].date()),
            "close": f"{float(df['Close'].iloc[-1]):.2f}",
            "sma20": f"{float(s20.iloc[-1]):.2f}" if not np.isnan(s20.iloc[-1]) else None,
            "sma50": f"{float(s50.iloc[-1]):.2f}" if not np.isnan(s50.iloc[-1]) else None,
            "rsi14": f"{float(rsi14.iloc[-1]):.2f}" if not np.isnan(rsi14.iloc[-1]) else None,
        }

        return templates.TemplateResponse(
            "nifty.html",
            {"request": request, "latest": latest, "overlay_img": overlay_img, "rsi_img": rsi_img, "volume_img": volume_img},
        )
    except Exception as e:
        return templates.TemplateResponse("nifty.html", {"request": request, "error": str(e)})

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
