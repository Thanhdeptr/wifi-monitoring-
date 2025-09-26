import argparse
import datetime as dt
import json
import math
import os
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


def holt_winters_forecast(series: pd.Series, seasonal_periods: int, z: float) -> Dict[str, float]:
    if series.empty:
        return {"yhat": math.nan, "yhat_lower": math.nan, "yhat_upper": math.nan}
    if HAS_SM and series.shape[0] >= max(10, seasonal_periods + 2):
        try:
            model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=seasonal_periods, initialization_method="estimated")
            fit = model.fit(optimized=True)
            yhat = float(fit.forecast(1).iloc[0])
            resid = series - fit.fittedvalues.reindex(series.index)
            resid = resid.replace([np.inf, -np.inf], np.nan).dropna()
            sd = float(resid.std(ddof=1)) if resid.shape[0] > 1 else 0.0
            return {"yhat": yhat, "yhat_lower": yhat - z * sd, "yhat_upper": yhat + z * sd}
        except Exception:
            pass
    # Fallback: EWMA
    ewma = series.ewm(span=max(3, seasonal_periods // 4)).mean()
    yhat = float(ewma.iloc[-1])
    diff = series - series.shift(1)
    diff = diff.replace([np.inf, -np.inf], np.nan).dropna()
    sd = float(diff.std(ddof=1)) if diff.shape[0] > 1 else 0.0
    return {"yhat": yhat, "yhat_lower": yhat - z * sd, "yhat_upper": yhat + z * sd}


def main() -> int:
    ap = argparse.ArgumentParser(description="Forecast selected Prometheus time series and output JSON")
    ap.add_argument("--prom", default=os.getenv("PROM_URL", "http://127.0.0.1:9090"), help="Prometheus base URL")
    ap.add_argument("--query", required=True, help="PromQL query for query_range (vector of series)")
    ap.add_argument("--hours", type=int, default=48, help="History lookback in hours")
    ap.add_argument("--step", type=int, default=300, help="Step (seconds) for query_range")
    ap.add_argument("--seasonal", type=int, default=288, help="Seasonal periods (e.g., 288 for 5m step ~ 24h)")
    ap.add_argument("--z", type=float, default=2.0, help="Z-score for bounds")
    ap.add_argument("--limit", type=int, default=20, help="Limit number of series to forecast")
    ap.add_argument("--output", default="-", help="Output file path or - for stdout")
    args = ap.parse_args()

    end = dt.datetime.utcnow().timestamp()
    start = end - args.hours * 3600
    data = prom_query_range(args.prom, args.query, start, end, args.step)
    series_list = (data.get("data") or {}).get("result") or []
    series_list = series_list[: args.limit]

    out: List[Dict] = []
    for s in series_list:
        labels: Dict[str, str] = s.get("metric", {})
        ser = series_to_pd(s.get("values", []))
        fc = holt_winters_forecast(ser, args.seasonal, args.z)
        out.append({"labels": labels, "forecast": fc})

    payload = {
        "query": args.query,
        "prom": args.prom,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "series_count": len(out),
        "results": out,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output == "-":
        print(text)
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


