import os
import sys
import json
import datetime as dt
from typing import Dict, List, Tuple, Optional, Any

import requests


PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://192.168.10.18:9090")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# LLM/Ollama configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.10.32:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
LLM_ENABLE = os.getenv("LLM_ENABLE", "1") not in ("0", "false", "False")
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "300"))


def _read_slack_webhook_from_secret() -> str:
    """Read Slack webhook URL from secrets/slack_webhook relative to this file."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        secret_path = os.path.join(base_dir, "..", "secrets", "slack_webhook")
        with open(secret_path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:
        return ""


# Fallback: if env var is empty, read from secrets file
if not SLACK_WEBHOOK_URL:
    SLACK_WEBHOOK_URL = _read_slack_webhook_from_secret()


def prom_query(expr: str) -> List[Dict]:
    """Run an instant query against Prometheus and return result vector."""
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": expr},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Prometheus error for {expr}: {resp.status_code} {resp.text}")
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus error for {expr}: {data}")
    return data["data"]["result"]


def prom_query_range(expr: str, start: dt.datetime, end: dt.datetime, step: str = "5m") -> List[Dict]:
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={
            "query": expr,
            "start": int(start.timestamp()),
            "end": int(end.timestamp()),
            "step": step,
        },
        timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"Prometheus range error for {expr}: {resp.status_code} {resp.text}")
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus range error for {expr}: {data}")
    return data["data"]["result"]


def prom_query_range_seconds(expr: str, start_epoch: int, end_epoch: int, step_seconds: int = 300) -> List[Dict]:
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={
            "query": expr,
            "start": start_epoch,
            "end": end_epoch,
            "step": f"{step_seconds}s",
        },
        timeout=90,
    )
    if not resp.ok:
        raise RuntimeError(f"Prometheus range error for {expr}: {resp.status_code} {resp.text}")
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus range error for {expr}: {data}")
    return data["data"]["result"]


# ------------------------------
# Utilities for feature engineering
# ------------------------------

def _percentile(nums: List[float], p: float) -> float:
    if not nums:
        return 0.0
    if p <= 0:
        return min(nums)
    if p >= 100:
        return max(nums)
    arr = sorted(nums)
    n = len(arr)
    pos = (n - 1) * (p / 100.0)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return arr[lo] * (1 - frac) + arr[hi] * frac


def _median(nums: List[float]) -> float:
    if not nums:
        return 0.0
    return _percentile(nums, 50)


def _mad(nums: List[float]) -> float:
    if not nums:
        return 0.0
    med = _median(nums)
    abs_dev = [abs(x - med) for x in nums]
    return _median(abs_dev)


def _coef_var(nums: List[float]) -> float:
    if not nums:
        return 0.0
    mean = sum(nums) / len(nums)
    if mean == 0:
        return 0.0
    # population std
    var = sum((x - mean) ** 2 for x in nums) / len(nums)
    return (var ** 0.5) / mean


def _slope_per_hour(timestamps: List[float], values: List[float]) -> float:
    # Simple linear regression slope normalized per hour
    if len(values) < 2:
        return 0.0
    # Use indices if timestamps are not strictly increasing
    if not timestamps or len(timestamps) != len(values):
        x = list(range(len(values)))
        # assume 5m step
        seconds_per_step = 300.0
    else:
        x = timestamps
        # estimate typical step
        diffs = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:]) if t2 > t1]
        seconds_per_step = (sum(diffs) / len(diffs)) if diffs else 300.0
    n = len(values)
    mean_x = sum(x) / n
    mean_y = sum(values) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, values))
    varx = sum((xi - mean_x) ** 2 for xi in x)
    if varx == 0:
        return 0.0
    slope_per_second = cov / varx
    return slope_per_second * 3600.0


def _time_over_threshold_minutes(values: List[float], thresholds: Tuple[float, float], step_seconds: float) -> Tuple[float, float, int]:
    warn, crit = thresholds
    over_warn = sum(1 for v in values if v >= warn)
    over_crit = sum(1 for v in values if v >= crit)
    spikes = over_crit
    minutes_per_step = step_seconds / 60.0
    return over_warn * minutes_per_step, over_crit * minutes_per_step, spikes


def _anomaly_rate(nums: List[float]) -> float:
    if not nums:
        return 0.0
    med = _median(nums)
    mad = _mad(nums)
    if mad == 0:
        return 0.0
    upper = med + 3 * mad
    lower = med - 3 * mad
    anomalies = sum(1 for x in nums if x > upper or x < lower)
    return anomalies / len(nums)


def _safe_float_series(series: List[Dict]) -> Tuple[List[float], List[float]]:
    # Returns timestamps (epoch seconds) and numeric values
    if not series:
        return [], []
    values = [(float(ts), float(v)) for ts, v in series[0]["values"] if v not in ("NaN", "Inf", "+Inf", "-Inf")]
    if not values:
        return [], []
    ts = [t for t, _ in values]
    vs = [v for _, v in values]
    return ts, vs


def _estimate_step_seconds(timestamps: List[float]) -> float:
    if len(timestamps) < 2:
        return 300.0
    diffs = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:]) if t2 > t1]
    return (sum(diffs) / len(diffs)) if diffs else 300.0


def _load_llm_static_context() -> Dict[str, Any]:
    # Try load configs/llm_static.json; fall back to defaults
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(base_dir, "..", "configs", "llm_static.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
                if isinstance(obj, dict):
                    return obj
                # If file content is not a JSON object, ignore and use defaults
    except Exception:
        pass
    # Defaults
    return {
        "business_hours": "08:00-18:00 Asia/Ho_Chi_Minh",
        "severity": {"ok": 0, "warning": 1, "critical": 2},
        "thresholds": {
            "cpu_percent": {"warning": 70.0, "critical": 85.0},
            "ram_percent": {"warning": 75.0, "critical": 90.0},
            "ping_ms": {"warning": 60.0, "critical": 120.0},
            # If WAN capacities unknown, evaluate relatively
            "wan_download_bps": {"warning": 20_000_000.0, "critical": 10_000_000.0},
            "wan_upload_bps": {"warning": 10_000_000.0, "critical": 5_000_000.0},
        },
        "notes": "Có thể override qua configs/llm_static.json"
    }


def _daily_bins_utc(days: int = 7) -> List[Tuple[int, int, str]]:
    """Return list of (start_epoch, end_epoch, label_date) for each day in UTC, recent first including today."""
    end = dt.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=1)
    bins: List[Tuple[int, int, str]] = []
    for i in range(days):
        day_end = end - dt.timedelta(days=i)
        day_start = day_end - dt.timedelta(days=1)
        bins.append((int(day_start.timestamp()), int(day_end.timestamp()), day_start.strftime("%Y-%m-%d")))
    return list(reversed(bins))


def _series_values(series: List[Dict]) -> List[float]:
    if not series:
        return []
    values = [(float(ts), float(v)) for ts, v in series[0].get("values", []) if v not in ("NaN", "Inf", "+Inf", "-Inf")]
    return [v for _, v in values]


def _agg_nums(nums: List[float]) -> Dict[str, float]:
    if not nums:
        return {"avg": 0.0, "p95": 0.0, "max": 0.0, "cv": 0.0}
    avg = sum(nums) / len(nums)
    return {
        "avg": avg,
        "p95": _percentile(nums, 95),
        "max": max(nums),
        "cv": _coef_var(nums),
    }


def collect_7d_summary() -> Tuple[List[str], Dict[str, Any]]:
    """Return human lines and a compact dict for last 7 days comparisons."""
    bins = _daily_bins_utc(7)
    lines: List[str] = ["*7d So sánh xu hướng*"]
    summary: Dict[str, Any] = {"days": []}

    for start_epoch, end_epoch, label in bins:
        # CPU and RAM by gw aggregated to overall (take worst p95 across gw per day)
        cpu_series = prom_query_range_seconds("avg by (gw) (hrProcessorLoad)", start_epoch, end_epoch, 300)
        ram_expr = (
            '100 * avg by (gw) (hrStorageUsed{hrStorageIndex="1"} / hrStorageSize{hrStorageIndex="1"})'
        )
        ram_series = prom_query_range_seconds(ram_expr, start_epoch, end_epoch, 300)

        cpu_p95s = []
        for s in cpu_series:
            nums = _series_values([s])
            if nums:
                cpu_p95s.append(_percentile(nums, 95))
        ram_p95s = []
        for s in ram_series:
            nums = _series_values([s])
            if nums:
                ram_p95s.append(_percentile(nums, 95))

        cpu_p95_worst = max(cpu_p95s) if cpu_p95s else 0.0
        ram_p95_worst = max(ram_p95s) if ram_p95s else 0.0

        # Ping overall (max p95 among gateways)
        ping_p95s: List[float] = []
        for gw in ["GW1", "GW2", "GW4", "GW5"]:
            ping_series = prom_query_range_seconds(f"speedtest_ping_latency_milliseconds{{gateway=\"{gw}\"}}", start_epoch, end_epoch, 300)
            nums = _series_values(ping_series)
            if nums:
                ping_p95s.append(_percentile(nums, 95))
        ping_p95_worst = max(ping_p95s) if ping_p95s else 0.0

        # Errors per day total
        err_expr = "sum by (gw) (increase(ifInErrors[24h]) + increase(ifOutErrors[24h]))"
        # Approx by querying at day end instant
        try:
            err_total = int(sum(float(r["value"][1]) for r in prom_query(f"sum by (gw) (increase(ifInErrors[{(end_epoch-start_epoch)//3600}h]))")))
        except Exception:
            err_total = 0

        summary_day = {
            "date": label,
            "cpu_p95_worst": cpu_p95_worst,
            "ram_p95_worst": ram_p95_worst,
            "ping_p95_worst": ping_p95_worst,
            "errors_total": err_total,
        }
        summary["days"].append(summary_day)
        lines.append(
            f"- {label}: CPU p95 worst {cpu_p95_worst:.1f}% | RAM p95 worst {ram_p95_worst:.1f}% | Ping p95 worst {ping_p95_worst:.1f} ms | Errors {err_total}"
        )

    return lines, summary


def _ollama_chat(messages: List[Dict[str, str]], timeout: Optional[int] = None) -> Optional[str]:
    try:
        resp = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "keep_alive": "5m",
                "options": {"temperature": 0.2, "num_ctx": 8192},
            },
            timeout=timeout or OLLAMA_TIMEOUT_SEC,
        )
        if not resp.ok:
            return f"__ERR__ HTTP {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        # new format: {message:{content: "..."}}
        msg = (data.get("message") or {}).get("content")
        if msg:
            return msg
        # fallback older generate format
        legacy = data.get("response")
        return legacy if legacy is not None else ""
    except Exception as exc:  # noqa: BLE001
        return f"__ERR__ EXC {type(exc).__name__}: {str(exc)[:200]}"


def _top_n_items(d: Dict[str, Dict[str, Any]], keys: List[str], n: int, reverse: bool = True) -> Dict[str, Dict[str, Any]]:
    if not d:
        return {}
    def score(v: Dict[str, Any]) -> float:
        vals = []
        for k in keys:
            x = v.get(k, 0.0)
            try:
                vals.append(float(x))
            except Exception:
                vals.append(0.0)
        return sum(vals)
    items = sorted(((k, v) for k, v in d.items() if v), key=lambda kv: score(kv[1]), reverse=reverse)
    return {k: v for k, v in items[:n]}


def _reduce_for_llm(payload: Dict[str, Any], top_n: int = 3) -> Dict[str, Any]:
    th = payload.get("thresholds", {})
    cpu = payload.get("cpu_by_gw", {})
    ram = payload.get("ram_by_gw", {})
    ping = payload.get("ping_by_gw", {})
    spd = payload.get("speedtest_by_line", {})
    err = payload.get("errors_by_gw", {})
    hist7 = payload.get("history_7d", {})
    cmp7 = payload.get("compare_7d", {})
    reduced = {
        "window": payload.get("window", {}),
        "thresholds": th,
        "cpu_by_gw_top": _top_n_items(cpu, ["time_over_critical_min", "p95", "cv"], top_n, True),
        "ram_by_gw_top": _top_n_items(ram, ["time_over_critical_min", "p95", "cv"], top_n, True),
        "ping_by_gw_top": _top_n_items(ping, ["time_over_critical_min", "p95"], top_n, True),
        "speedtest_lines_top": _top_n_items(
            {k: {**v, "under_sum": (v.get("time_dl_under_warn_min", 0.0) + v.get("time_ul_under_warn_min", 0.0))} for k, v in spd.items()},
            ["under_sum"],
            top_n,
            True,
        ),
        "errors_total": int(sum((float(v) for v in err.values()), 0.0)),
        "errors_top": dict(sorted(err.items(), key=lambda kv: kv[1], reverse=True)[:top_n]),
        "static": payload.get("static", {}),
        "history_7d": hist7,
        "compare_7d": cmp7,
    }
    return reduced


def collect_llm_assessment() -> List[str]:
    """Compute robust features and ask LLM for a daily health assessment."""
    if not LLM_ENABLE:
        return []

    end = dt.datetime.utcnow()
    start = end - dt.timedelta(hours=24)

    static_ctx = _load_llm_static_context()
    th = static_ctx.get("thresholds", {})

    # CPU by gw
    cpu_series = prom_query_range("avg by (gw) (hrProcessorLoad)", start, end, step="5m")
    # RAM by gw
    mem_expr_idx1 = '100 * avg by (gw) (hrStorageUsed{hrStorageIndex="1"} / hrStorageSize{hrStorageIndex="1"})'
    mem_series = prom_query_range(mem_expr_idx1, start, end, step="5m")
    if not mem_series:
        mem_expr_descr = (
            '100 * avg by (gw) (hrStorageUsed{hrStorageType="1.3.6.1.2.1.25.2.1.2"} / hrStorageSize{hrStorageType="1.3.6.1.2.1.25.2.1.2"})'
        )
        mem_series = prom_query_range(mem_expr_descr, start, end, step="5m")

    # Ping by gateway
    gateways = ["GW1", "GW2", "GW4", "GW5"]

    # Speedtest by line
    wan_lines = [f"WAN{i}" for i in range(1, 8 + 1)]

    def map_by_label(series: List[Dict], label: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for s in series:
            key = s.get("metric", {}).get(label, "")
            ts, vs = _safe_float_series([s])
            if not vs:
                out[key] = {}
                continue
            step_seconds = _estimate_step_seconds(ts)
            warn = th.get("cpu_percent", {}).get("warning", 70.0) if label == "gw" else 0.0
            crit = th.get("cpu_percent", {}).get("critical", 85.0) if label == "gw" else 0.0
            if "hrStorage" in json.dumps(s.get("metric", {})):
                warn = th.get("ram_percent", {}).get("warning", 75.0)
                crit = th.get("ram_percent", {}).get("critical", 90.0)
            owarn_m, ocrit_m, spikes = _time_over_threshold_minutes(vs, (warn, crit), step_seconds)
            out[key] = {
                "avg": (sum(vs) / len(vs)),
                "p95": _percentile(vs, 95),
                "max": max(vs),
                "cv": _coef_var(vs),
                "slope_per_hour": _slope_per_hour(ts, vs),
                "anomaly_rate": _anomaly_rate(vs),
                "time_over_warning_min": owarn_m,
                "time_over_critical_min": ocrit_m,
                "spikes": spikes,
                "peak_time": dt.datetime.utcfromtimestamp(ts[vs.index(max(vs))]).strftime("%H:%M") if ts else "-",
            }
        return out

    cpu_by_gw = map_by_label(cpu_series, "gw")
    # For RAM, rebuild series by gw key since we lose context inside helper
    ram_by_gw: Dict[str, Dict[str, Any]] = {}
    for s in mem_series:
        key = s.get("metric", {}).get("gw", "")
        ts, vs = _safe_float_series([s])
        if not vs:
            ram_by_gw[key] = {}
            continue
        step_seconds = _estimate_step_seconds(ts)
        warn = th.get("ram_percent", {}).get("warning", 75.0)
        crit = th.get("ram_percent", {}).get("critical", 90.0)
        owarn_m, ocrit_m, spikes = _time_over_threshold_minutes(vs, (warn, crit), step_seconds)
        ram_by_gw[key] = {
            "avg": (sum(vs) / len(vs)),
            "p95": _percentile(vs, 95),
            "max": max(vs),
            "cv": _coef_var(vs),
            "slope_per_hour": _slope_per_hour(ts, vs),
            "anomaly_rate": _anomaly_rate(vs),
            "time_over_warning_min": owarn_m,
            "time_over_critical_min": ocrit_m,
            "spikes": spikes,
            "peak_time": dt.datetime.utcfromtimestamp(ts[vs.index(max(vs))]).strftime("%H:%M") if ts else "-",
        }

    ping_by_gw: Dict[str, Dict[str, Any]] = {}
    for gw in gateways:
        series = prom_query_range(f"speedtest_ping_latency_milliseconds{{gateway=\"{gw}\"}}", start, end, step="5m")
        ts, vs = _safe_float_series(series)
        if not vs:
            ping_by_gw[gw] = {}
            continue
        step_seconds = _estimate_step_seconds(ts)
        warn = th.get("ping_ms", {}).get("warning", 60.0)
        crit = th.get("ping_ms", {}).get("critical", 120.0)
        owarn_m, ocrit_m, spikes = _time_over_threshold_minutes(vs, (warn, crit), step_seconds)
        ping_by_gw[gw] = {
            "avg": (sum(vs) / len(vs)),
            "p95": _percentile(vs, 95),
            "max": max(vs),
            "time_over_warning_min": owarn_m,
            "time_over_critical_min": ocrit_m,
            "spikes": spikes,
            "peak_time": dt.datetime.utcfromtimestamp(ts[vs.index(max(vs))]).strftime("%H:%M") if ts else "-",
        }

    speedtest_by_line: Dict[str, Dict[str, Any]] = {}
    for line in wan_lines:
        dl_series = prom_query_range(f"speedtest_download_bits_per_second{{line=\"{line}\"}}", start, end, step="5m")
        ul_series = prom_query_range(f"speedtest_upload_bits_per_second{{line=\"{line}\"}}", start, end, step="5m")
        ts_dl, vs_dl = _safe_float_series(dl_series)
        ts_ul, vs_ul = _safe_float_series(ul_series)
        warn_dl = th.get("wan_download_bps", {}).get("warning", 20_000_000.0)
        crit_dl = th.get("wan_download_bps", {}).get("critical", 10_000_000.0)
        warn_ul = th.get("wan_upload_bps", {}).get("warning", 10_000_000.0)
        crit_ul = th.get("wan_upload_bps", {}).get("critical", 5_000_000.0)
        step_seconds = _estimate_step_seconds(ts_dl or ts_ul)
        owarn_dl_m, ocrit_dl_m, _ = _time_over_threshold_minutes(vs_dl, (warn_dl, crit_dl), step_seconds) if vs_dl else (0.0, 0.0, 0)
        # For bandwidth, "under-performance" is low values, so also compute time below warning
        under_warn_dl_m = sum(1 for v in vs_dl if v <= warn_dl) * (step_seconds / 60.0) if vs_dl else 0.0
        under_warn_ul_m = sum(1 for v in vs_ul if v <= warn_ul) * (step_seconds / 60.0) if vs_ul else 0.0
        speedtest_by_line[line] = {
            "dl_avg": (sum(vs_dl) / len(vs_dl)) if vs_dl else 0.0,
            "dl_p10": _percentile(vs_dl, 10) if vs_dl else 0.0,
            "dl_min": min(vs_dl) if vs_dl else 0.0,
            "ul_avg": (sum(vs_ul) / len(vs_ul)) if vs_ul else 0.0,
            "ul_p10": _percentile(vs_ul, 10) if vs_ul else 0.0,
            "ul_min": min(vs_ul) if vs_ul else 0.0,
            "time_dl_under_warn_min": under_warn_dl_m,
            "time_ul_under_warn_min": under_warn_ul_m,
        }

    # Interface errors (24h increase)
    expr_err = "sum by (gw) (increase(ifInErrors[24h]) + increase(ifOutErrors[24h]))"
    err = {r["metric"].get("gw", ""): float(r["value"][1]) for r in prom_query(expr_err)}

    # Ước lượng step_seconds an toàn kể cả khi không có series CPU
    try:
        if cpu_series:
            ts_any, _vs_any = _safe_float_series([cpu_series[0]])
        else:
            ts_any = []
        step_seconds_any = _estimate_step_seconds(ts_any)
    except Exception:
        step_seconds_any = 300.0

    payload = {
        "window": {
            "start_utc": start.strftime("%Y-%m-%dT%H:%MZ"),
            "end_utc": end.strftime("%Y-%m-%dT%H:%MZ"),
            "step_seconds": step_seconds_any or 300.0,
        },
        "thresholds": th,
        "cpu_by_gw": cpu_by_gw,
        "ram_by_gw": ram_by_gw,
        "ping_by_gw": ping_by_gw,
        "speedtest_by_line": speedtest_by_line,
        "errors_by_gw": err,
        "static": {k: v for k, v in static_ctx.items() if k != "thresholds"},
    }

    # Add 7d history summary to payload (compact) and comparative metrics vs today
    try:
        _lines7d, summary7d = collect_7d_summary()
        payload["history_7d"] = summary7d
        # Comparative today vs last 7 days median and p95
        days = summary7d.get("days", [])
        if days:
            # today's label is last element's date if bins reversed at build time
            today_label = days[-1]["date"]
            hist = days[:-1] if len(days) > 1 else []
            def agg(lst, key):
                vals = [float(d.get(key, 0.0)) for d in lst if d.get(key) is not None]
                return {
                    "median": _percentile(vals, 50) if vals else 0.0,
                    "p95": _percentile(vals, 95) if vals else 0.0,
                    "avg": (sum(vals)/len(vals)) if vals else 0.0,
                }
            today = days[-1]
            payload["compare_7d"] = {
                "today_date": today_label,
                "cpu_p95_worst": {"today": today.get("cpu_p95_worst", 0.0), "hist": agg(hist, "cpu_p95_worst")},
                "ram_p95_worst": {"today": today.get("ram_p95_worst", 0.0), "hist": agg(hist, "ram_p95_worst")},
                "ping_p95_worst": {"today": today.get("ping_p95_worst", 0.0), "hist": agg(hist, "ping_p95_worst")},
                "errors_total": {"today": today.get("errors_total", 0), "hist": agg(hist, "errors_total")},
            }
        else:
            payload["compare_7d"] = {}
    except Exception:
        payload["history_7d"] = {"days": []}
        payload["compare_7d"] = {}

    system_prompt = (
        "Bạn là kỹ sư vận hành mạng. Hãy đánh giá sức khỏe hạ tầng trong 24h qua dựa trên các feature thống kê, "
        "ưu tiên tính ổn định theo thời gian (time-over-threshold, burstiness, slope, anomaly) thay vì chỉ lấy trung bình. "
        "Trả lời tiếng Việt, súc tích. Phân loại mức độ: Tốt / Cảnh báo / Sự cố."
    )
    reduced_payload = _reduce_for_llm(payload, top_n=3)

    user_prompt = (
        "Context tĩnh (ngưỡng/giờ làm việc/ghi chú):\n" + json.dumps({k: v for k, v in static_ctx.items()}, ensure_ascii=False) +
        "\n\nFeature 24h đã xử lý (top-N) + Lịch sử 7 ngày và so sánh hôm nay vs 7 ngày:\n" + json.dumps(reduced_payload, ensure_ascii=False) +
        "\n\nYêu cầu: \n"
        "1) Tình trạng tổng quát.\n"
        "2) 3-6 gạch đầu dòng nêu lý do chính (nêu gw/line, chỉ số, thời điểm).\n"
        "3) Hành động khuyến nghị ngắn gọn.\n"
        "4) So sánh 7 ngày (định tính): mô tả khác biệt hôm nay so với 6 ngày trước (CPU/RAM/Ping/Errors) nhưng KHÔNG in số liệu 7 ngày; dùng các cụm như 'tăng nhẹ', 'cao hơn đáng kể', 'ổn định hơn', 'xấu hơn nhiều'.\n"
        "5) Nêu 2-4 điểm cần chú ý hoặc theo dõi (ngắn gọn).\n"
        "6) Một câu kết luận (<= 120 ký tự), KHÔNG chứa số liệu 7 ngày."
    )

    llm_text = _ollama_chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    title = "*LLM Đánh giá tình trạng (24h)*"
    if not llm_text:
        # Fallback heuristic
        worst_cpu = max((v.get("p95", 0.0) for v in cpu_by_gw.values() if v), default=0.0)
        worst_ram = max((v.get("p95", 0.0) for v in ram_by_gw.values() if v), default=0.0)
        worst_ping = max((v.get("p95", 0.0) for v in ping_by_gw.values() if v), default=0.0)
        err_total = int(sum(err.values()))
        summary = (
            f"- CPU p95 tệ nhất: {worst_cpu:.1f}% | RAM p95: {worst_ram:.1f}% | Ping p95: {worst_ping:.1f} ms | Errors: {err_total}\n"
            f"- Không gọi được LLM tại {OLLAMA_URL}. Dùng tóm tắt heuristic."
        )
        return [title, summary]

    return [title, llm_text]


def format_value(v: float, unit: str = "") -> str:
    if unit == "bps":
        # humanize bits per second (approx)
        for suffix in ["bps", "Kbps", "Mbps", "Gbps", "Tbps"]:
            if v < 1000:
                return f"{v:.2f} {suffix}"
            v /= 1000
        return f"{v:.2f} Pbps"
    if unit == "%":
        return f"{v:.1f}%"
    return f"{v:.2f}"


def collect_cpu_ram_24h_by_gw() -> List[str]:
    # Thống kê theo 24h và kèm thời điểm đỉnh
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(hours=24)

    # CPU: trung bình theo gw tại mỗi bước, rồi tính avg/max/time-max trên 24h
    cpu_series = prom_query_range("avg by (gw) (hrProcessorLoad)", start, end, step="5m")

    def series_stats_by_label(series: List[Dict], label: str) -> Dict[str, Tuple[float, float, float]]:
        out: Dict[str, Tuple[float, float, float]] = {}
        for s in series:
            key = s["metric"].get(label, "")
            values = [(float(ts), float(v)) for ts, v in s["values"] if v not in ("NaN", "Inf", "+Inf", "-Inf")]
            if not values:
                out[key] = (0.0, 0.0, 0.0)
                continue
            nums = [v for _, v in values]
            avg = sum(nums) / len(nums)
            vmax_ts, vmax = max(values, key=lambda x: x[1])[0], max(nums)
            out[key] = (avg, vmax, vmax_ts)
        return out

    cpu_stats = series_stats_by_label(cpu_series, "gw")

    # RAM % theo gw: ưu tiên cách đơn giản theo hrStorageIndex=1 (nếu thiết bị mapping như bạn nêu),
    # nếu không có dữ liệu thì fallback sang lọc theo mô tả "Physical memory".
    mem_expr_idx1 = (
        '100 * avg by (gw) (hrStorageUsed{hrStorageIndex="1"} / hrStorageSize{hrStorageIndex="1"})'
    )
    mem_series = prom_query_range(mem_expr_idx1, start, end, step="5m")
    if not mem_series:
        mem_expr_descr = (
            '100 * avg by (gw) (hrStorageUsed{hrStorageType="1.3.6.1.2.1.25.2.1.2"} / hrStorageSize{hrStorageType="1.3.6.1.2.1.25.2.1.2"})'
        )
        mem_series = prom_query_range(mem_expr_descr, start, end, step="5m")
    mem_stats = series_stats_by_label(mem_series, "gw")

    tstr = lambda ts: dt.datetime.utcfromtimestamp(ts).strftime("%H:%M") if ts else "-"
    lines = ["*CPU & RAM by Gateway (24h)*"]
    for gw in sorted(set(list(cpu_stats.keys()) + list(mem_stats.keys()))):
        c_avg, c_max, c_t = cpu_stats.get(gw, (0.0, 0.0, 0.0))
        m_avg, m_max, m_t = mem_stats.get(gw, (0.0, 0.0, 0.0))
        lines.append(
            f"- {gw}: CPU avg {format_value(c_avg, '%')}, max {format_value(c_max, '%')} @ {tstr(c_t)} | "
            f"RAM avg {format_value(m_avg, '%')}, max {format_value(m_max, '%')} @ {tstr(m_t)}"
        )
    return lines


def collect_mem_24h() -> List[str]:
    base = (
        "100 * ("
        "sum by (device,gw)(hrStorageUsed{hrStorageType=\"1.3.6.1.2.1.25.2.1.2\"} * hrStorageAllocationUnits{hrStorageType=\"1.3.6.1.2.1.25.2.1.2\"}) / "
        "sum by (device,gw)(hrStorageSize{hrStorageType=\"1.3.6.1.2.1.25.2.1.2\"} * hrStorageAllocationUnits{hrStorageType=\"1.3.6.1.2.1.25.2.1.2\"})"
        ")"
    )
    expr_avg = f"avg_over_time(({base})[24h])"
    expr_max = f"max_over_time(({base})[24h])"
    avg_res = {tuple(r["metric"].get(k, "") for k in ("device", "gw")): float(r["value"][1]) for r in prom_query(expr_avg)}
    max_res = {tuple(r["metric"].get(k, "") for k in ("device", "gw")): float(r["value"][1]) for r in prom_query(expr_max)}
    lines = ["*Memory (24h)*"]
    for key in sorted(avg_res):
        device, gw = key
        lines.append(
            f"- {device} [{gw}]: avg {format_value(avg_res[key], '%')}, max {format_value(max_res.get(key, 0.0), '%')}"
        )
    return lines


# Removed WAN 95th percentile section as per user request


def collect_speedtest_by_line() -> List[str]:
    # Average/min and timestamp per WAN line over last 24h
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(hours=24)
    lines = ["*Speedtest by Line (24h)*"]

    wan_lines = [f"WAN{i}" for i in range(1, 9)]
    for line in wan_lines:
        dl_series = prom_query_range(
            f"speedtest_download_bits_per_second{{line=\"{line}\"}}",
            start,
            end,
        )
        ul_series = prom_query_range(
            f"speedtest_upload_bits_per_second{{line=\"{line}\"}}",
            start,
            end,
        )
        ping_series = prom_query_range(
            f"speedtest_ping_latency_milliseconds{{line=\"{line}\"}}",
            start,
            end,
        )

        def stats(series: List[Dict]) -> Tuple[float, float, float]:
            if not series:
                return 0.0, 0.0, 0.0
            values = [(float(ts), float(v)) for ts, v in series[0]["values"] if v not in ("NaN", "Inf", "+Inf", "-Inf")]
            if not values:
                return 0.0, 0.0, 0.0
            nums = [v for _, v in values]
            avg = sum(nums) / len(nums)
            vmin = min(nums)
            tmin = min(values, key=lambda x: x[1])[0]
            return avg, vmin, tmin

        dl_avg, dl_min, dl_t = stats(dl_series)
        ul_avg, ul_min, ul_t = stats(ul_series)
        ping_avg, _, _ = stats(ping_series)

        tmin_str = lambda ts: dt.datetime.utcfromtimestamp(ts).strftime("%H:%M") if ts else "-"
        lines.append(
            f"- {line}: DL avg {format_value(dl_avg, 'bps')}, min {format_value(dl_min, 'bps')} @ {tmin_str(dl_t)} | "
            f"UL avg {format_value(ul_avg, 'bps')}, min {format_value(ul_min, 'bps')} @ {tmin_str(ul_t)} | "
            f"Ping avg {ping_avg:.1f} ms"
        )

    return lines


def collect_ping_by_gw() -> List[str]:
    # Avg and max ping over 24h by gateway (with time of max)
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(hours=24)
    gateways = ["GW1", "GW2", "GW4", "GW5"]

    def stats(series: List[Dict]) -> Tuple[float, float, float]:
        if not series:
            return 0.0, 0.0, 0.0
        values = [(float(ts), float(v)) for ts, v in series[0]["values"] if v not in ("NaN", "Inf", "+Inf", "-Inf")]
        if not values:
            return 0.0, 0.0, 0.0
        nums = [v for _, v in values]
        avg = sum(nums) / len(nums)
        tmax, vmax = max(values, key=lambda x: x[1])
        return avg, vmax, tmax

    tstr = lambda ts: dt.datetime.utcfromtimestamp(ts).strftime("%H:%M") if ts else "-"
    lines = ["*Ping by Gateway (24h)*"]
    for gw in gateways:
        series = prom_query_range(
            f"speedtest_ping_latency_milliseconds{{gateway=\"{gw}\"}}",
            start,
            end,
        )
        avg, vmax, tmax = stats(series)
        lines.append(f"- {gw}: avg {avg:.1f} ms, max {vmax:.1f} ms @ {tstr(tmax)}")
    return lines


def collect_errors_by_gw() -> List[str]:
    # Errors today by gw (sum of in/out)
    expr_err = (
        "sum by (gw) (increase(ifInErrors[24h]) + increase(ifOutErrors[24h]))"
    )
    err = {r["metric"].get("gw", ""): float(r["value"][1]) for r in prom_query(expr_err)}
    lines = ["*Interface Errors by Gateway (24h)*"]
    for gw in sorted(err.keys()):
        lines.append(f"- {gw}: errors {int(err.get(gw, 0))}")
    return lines


def build_report() -> str:
    today = dt.datetime.now().strftime("%Y-%m-%d")
    parts: List[str] = []
    parts.append(f"*Daily Network Report* — {today}")
    # LLM assessment first (if enabled)
    try:
        parts.extend(collect_llm_assessment())
    except Exception as _exc:  # noqa: BLE001
        # Always show a visible section even if failed, with detailed error
        err_type = type(_exc).__name__
        parts.extend([
            "*LLM Đánh giá tình trạng (24h)*",
            f"- Lỗi khi tạo đánh giá LLM: {err_type}: {repr(_exc)}",
        ])
    parts.extend(collect_cpu_ram_24h_by_gw())
    parts.extend(collect_speedtest_by_line())
    parts.extend(collect_errors_by_gw())
    return "\n".join(parts)


def _chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_report_blocks() -> List[Dict]:
    today = dt.datetime.now().strftime("%Y-%m-%d")
    blocks: List[Dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Daily Network Report — {today}", "emoji": True},
        }
    ]

    def add_section_from_lines(lines: List[str]) -> None:
        if not lines:
            return
        title, *rows = lines
        rows = [r for r in rows if r.strip()]
        body = "\n".join(rows)
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"{title}\n{body}"}})

    # LLM Assessment
    blocks.append({"type": "divider"})
    try:
        add_section_from_lines(collect_llm_assessment())
    except Exception as _exc:  # noqa: BLE001
        # Add error section instead of hiding, with details
        err_type = type(_exc).__name__
        add_section_from_lines([
            "*LLM Đánh giá tình trạng (24h)*",
            f"- Lỗi khi tạo đánh giá LLM: {err_type}: {repr(_exc)}",
        ])

    # CPU & RAM
    blocks.append({"type": "divider"})
    add_section_from_lines(collect_cpu_ram_24h_by_gw())

    # Speedtest by Line
    blocks.append({"type": "divider"})
    add_section_from_lines(collect_speedtest_by_line())

    # Interface Errors
    blocks.append({"type": "divider"})
    add_section_from_lines(collect_errors_by_gw())

    return blocks


def build_report_attachments() -> List[Dict]:
    # Attachment có viền màu cam Grafana để nhấn mạnh link
    return [
        {
            "color": "#F46800",
            "text": "<http://192.168.10.18:3001/dashboards?tag=network|Open Grafana dashboard>",
        }
    ]


def post_to_slack(text: str, blocks: Optional[List[Dict]] = None, attachments: Optional[List[Dict]] = None) -> None:
    # If webhook is not set, just print to terminal (dry-run mode)
    if not SLACK_WEBHOOK_URL:
        print(text)
        return
    payload: Dict = {"text": text}
    if blocks:
        payload["blocks"] = blocks
    if attachments:
        payload["attachments"] = attachments
    resp = requests.post(
        SLACK_WEBHOOK_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()


def main() -> int:
    try:
        start_ts = dt.datetime.now()
        print(f"[{start_ts:%Y-%m-%d %H:%M:%S}] Cron start", flush=True)

        report = build_report()
        blocks = build_report_blocks()
        attachments = build_report_attachments()
        post_to_slack(report, blocks, attachments)

        end_ts = dt.datetime.now()
        duration_s = (end_ts - start_ts).total_seconds()
        print(f"[{end_ts:%Y-%m-%d %H:%M:%S}] Report sent to Slack. duration={duration_s:.1f}s", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        now_ts = dt.datetime.now()
        print(f"[{now_ts:%Y-%m-%d %H:%M:%S}] Failed: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


