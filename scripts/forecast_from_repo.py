import argparse
import datetime as dt
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_SM = True
except Exception:
    HAS_SM = False


def prom_query_range(base_url: str, query: str, start: float, end: float, step: int) -> Dict:
    url = f"{base_url.rstrip('/')}/api/v1/query_range"
    resp = requests.get(url, params={"query": query, "start": start, "end": end, "step": f"{step}s"}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus error: {data}")
    return data


def series_to_pd(values: List[List]) -> pd.Series:
    if not values:
        return pd.Series(dtype=float)
    df = pd.DataFrame(values, columns=["ts", "value"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("ts").sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df["value"]


def run_hw(series: pd.Series, seasonal: int, z: float) -> Dict[str, float]:
    if series.empty:
        return {"yhat": float("nan"), "yhat_lower": float("nan"), "yhat_upper": float("nan")}
    if HAS_SM and series.shape[0] >= max(10, seasonal + 2):
        m = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=seasonal, initialization_method="estimated")
        fit = m.fit(optimized=True)
        yhat = float(fit.forecast(1).iloc[0])
        resid = series - fit.fittedvalues.reindex(series.index)
        resid = resid.replace([np.inf, -np.inf], np.nan).dropna()
        sd = float(resid.std(ddof=1)) if resid.shape[0] > 1 else 0.0
        return {"yhat": yhat, "yhat_lower": yhat - z * sd, "yhat_upper": yhat + z * sd}
    # Fallback EWMA
    ewma = series.ewm(span=max(3, seasonal // 4)).mean()
    yhat = float(ewma.iloc[-1])
    diff = series - series.shift(1)
    diff = diff.replace([np.inf, -np.inf], np.nan).dropna()
    sd = float(diff.std(ddof=1)) if diff.shape[0] > 1 else 0.0
    return {"yhat": yhat, "yhat_lower": yhat - z * sd, "yhat_upper": yhat + z * sd}


def main() -> int:
    ap = argparse.ArgumentParser(description="Test forecast using repo context; output JSON for quick evaluation")
    ap.add_argument("--prom", default="http://192.168.10.18:9090")
    ap.add_argument("--query", required=True)
    ap.add_argument("--hours", type=int, default=48)
    ap.add_argument("--step", type=int, default=300)
    ap.add_argument("--seasonal", type=int, default=288)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--output", default="-")
    args = ap.parse_args()

    end = dt.datetime.utcnow().timestamp()
    start = end - args.hours * 3600
    data = prom_query_range(args.prom, args.query, start, end, args.step)
    results = (data.get("data") or {}).get("result") or []
    results = results[: args.limit]

    out: List[Dict] = []
    for r in results:
        labels: Dict[str, str] = r.get("metric", {})
        ser = series_to_pd(r.get("values", []))
        fc = run_hw(ser, args.seasonal, args.z)
        out.append({"labels": labels, "forecast": fc})

    payload = {
        "query": args.query,
        "series_count": len(out),
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "results": out,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output == "-":
        print(text)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


