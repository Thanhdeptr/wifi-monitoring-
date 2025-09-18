import os
import sys
import json
import datetime as dt
from typing import Dict, List, Tuple

import requests


PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://192.168.10.18:9090")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


def prom_query(expr: str) -> List[Dict]:
    """Run an instant query against Prometheus and return result vector."""
    resp = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": expr},
        timeout=30,
    )
    resp.raise_for_status()
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
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"Prometheus range error for {expr}: {data}")
    return data["data"]["result"]


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
        "100 * avg by (gw) ("
        "hrStorageUsed{hrStorageIndex=\\\"1\\\"} / on (instance,hrStorageIndex) hrStorageSize{hrStorageIndex=\\\"1\\\"}"
        ")"
    )
    mem_series = prom_query_range(mem_expr_idx1, start, end, step="5m")
    if not mem_series:
        mem_expr_descr = (
            "100 * avg by (gw) ("
            "(hrStorageUsed and on (hrStorageIndex,instance) hrStorageDescr{hrStorageDescr=\\\"Physical memory\\\"}) / on (instance,hrStorageIndex) "
            "(hrStorageSize and on (hrStorageIndex,instance) hrStorageDescr{hrStorageDescr=\\\"Physical memory\\\"})"
            ")"
        )
        mem_series = prom_query_range(mem_expr_descr, start, end, step="5m")
    mem_stats = series_stats_by_label(mem_series, "gw")

    tstr = lambda ts: dt.datetime.utcfromtimestamp(ts).strftime("%H:%M") if ts else "-"
    lines = ["*CPU & RAM theo Gateway (24h)*"]
    for gw in sorted(set(list(cpu_stats.keys()) + list(mem_stats.keys()))):
        c_avg, c_max, c_t = cpu_stats.get(gw, (0.0, 0.0, 0.0))
        m_avg, m_max, m_t = mem_stats.get(gw, (0.0, 0.0, 0.0))
        lines.append(
            f"• {gw}: CPU avg {format_value(c_avg, '%')}, max {format_value(c_max, '%')} @ {tstr(c_t)} | "
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
            f"• {device} [{gw}]: avg {format_value(avg_res[key], '%')}, max {format_value(max_res.get(key, 0.0), '%')}"
        )
    return lines


# Removed WAN 95th percentile section as per user request


def collect_speedtest_by_line() -> List[str]:
    # Average/min and timestamp per WAN line over last 24h
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(hours=24)
    lines = ["*Speedtest theo Line (24h)*"]

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
            f"• {line}: DL avg {format_value(dl_avg, 'bps')}, min {format_value(dl_min, 'bps')} @ {tmin_str(dl_t)} | "
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
    lines = ["*Ping theo Gateway (24h)*"]
    for gw in gateways:
        series = prom_query_range(
            f"speedtest_ping_latency_milliseconds{{gateway=\"{gw}\"}}",
            start,
            end,
        )
        avg, vmax, tmax = stats(series)
        lines.append(f"• {gw}: avg {avg:.1f} ms, max {vmax:.1f} ms @ {tstr(tmax)}")
    return lines


def collect_errors_by_gw() -> List[str]:
    # Errors today by gw (sum of in/out)
    expr_err = (
        "sum by (gw) (increase(ifInErrors[24h]) + increase(ifOutErrors[24h]))"
    )
    err = {r["metric"].get("gw", ""): float(r["value"][1]) for r in prom_query(expr_err)}
    lines = ["*Lỗi giao diện theo Gateway (24h)*"]
    for gw in sorted(err.keys()):
        lines.append(f"• {gw}: errors {int(err.get(gw, 0))}")
    return lines


def build_report() -> str:
    today = dt.datetime.now().strftime("%Y-%m-%d")
    sections: List[str] = [f"*Daily Network Report* — {today}"]
    sections += collect_cpu_ram_24h_by_gw()
    # 95th percentile WAN traffic removed per user request
    sections += [""] + collect_speedtest_by_line()
    sections += [""] + collect_errors_by_gw()
    return "\n".join(sections)


def post_to_slack(text: str) -> None:
    # If webhook is not set, just print to terminal (dry-run mode)
    if not SLACK_WEBHOOK_URL:
        print(text)
        return
    payload = {"text": text}
    resp = requests.post(
        SLACK_WEBHOOK_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()


def main() -> int:
    try:
        report = build_report()
        post_to_slack(report)
        print("Report sent to Slack.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


