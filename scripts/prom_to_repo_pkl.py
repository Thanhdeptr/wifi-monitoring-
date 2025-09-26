import argparse
import os
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import requests


def prom_query_range(base_url: str, query: str, start: int, end: int, step: int) -> Dict:
    url = f"{base_url.rstrip('/')}/api/v1/query_range"
    resp = requests.get(
        url,
        params={"query": query, "start": start, "end": end, "step": f"{step}s"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus error: {json.dumps(data)[:300]}")
    return data


def values_to_df(values: List[List]) -> pd.DataFrame:
    if not values:
        return pd.DataFrame(columns=["timestamps", "values"])
    df = pd.DataFrame(values, columns=["ts", "value"])  # [[epoch, str(val)], ...]
    df["timestamps"] = pd.to_datetime(df["ts"], unit="s")
    df["values"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.drop(columns=["ts", "value"])  # keep only expected columns
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values(by=["timestamps"])  # ensure monotonic time
    return df


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch Prometheus time series and save as pickle compatible with external/forecast-prometheus fourier/prophet scripts"
        )
    )
    ap.add_argument("--prom", default=os.getenv("PROM_URL", "http://127.0.0.1:9090"), help="Prometheus base URL")
    ap.add_argument("--query", required=True, help="PromQL returning vector of series")
    ap.add_argument("--hours", type=int, default=48, help="Lookback window in hours")
    ap.add_argument("--step", type=int, default=300, help="Step in seconds for query_range")
    ap.add_argument("--metric", default="custom", help="Metric name used to name output pickle file")
    ap.add_argument(
        "--outdir",
        default="external/forecast-prometheus/pkl_data",
        help="Output directory where <metric>_dataframes.pkl will be written",
    )
    ap.add_argument("--limit", type=int, default=20, help="Limit number of series")
    args = ap.parse_args()

    end = int(pd.Timestamp.utcnow().timestamp())
    start = end - args.hours * 3600
    data = prom_query_range(args.prom, args.query, start, end, args.step)
    results: List[Dict] = (data.get("data") or {}).get("result") or []
    results = results[: args.limit]

    # Build dict[str(labels) -> DataFrame(timestamps, values)]
    dfs: Dict[str, pd.DataFrame] = {}
    for r in results:
        labels = r.get("metric", {})
        key = str(labels)
        df = values_to_df(r.get("values", []))
        if not df.empty:
            dfs[key] = df

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    out_path = os.path.join(args.outdir, f"{args.metric}_dataframes.pkl")
    pd.to_pickle(dfs, out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


