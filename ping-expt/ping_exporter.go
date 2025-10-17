package main

import (
	"context"
	"log"
	"net/http"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// === PHẦN 1: LOGIC PING (Lấy từ code gốc của bạn) ===

type PingResult struct {
	Success      bool
	PacketLoss   float64
	AvgLatencyMs float64
	MaxLatencyMs float64
	MinLatencyMs float64
}

func runPing(ctx context.Context, target, bindAddress string) PingResult {
	args := []string{"-c", "10", "-W", "2", target}
	if bindAddress != "" {
		args = append([]string{"-I", bindAddress}, args...)
	}

	cmd := exec.CommandContext(ctx, "ping", args...)
	log.Printf("Đang chạy lệnh ping: %s", strings.Join(cmd.Args, " "))

	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Lệnh ping thất bại. Lỗi: %v. Output: %s", err, string(output))
		return PingResult{Success: false}
	}

	return parsePingOutput(string(output))
}

func parsePingOutput(out string) PingResult {
	var loss float64 = 100.0
	var min, avg, max float64

	lossRegex := regexp.MustCompile(`(\d+(\.\d+)?)% packet loss`)
	if match := lossRegex.FindStringSubmatch(out); len(match) > 1 {
		loss, _ = strconv.ParseFloat(match[1], 64)
	}

	rttRegex := regexp.MustCompile(`rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms`)
	if match := rttRegex.FindStringSubmatch(out); len(match) == 5 {
		min, _ = strconv.ParseFloat(match[1], 64)
		avg, _ = strconv.ParseFloat(match[2], 64)
		max, _ = strconv.ParseFloat(match[3], 64)
	} else {
		return PingResult{Success: false, PacketLoss: loss}
	}

	log.Printf("Phân tích ping thành công: loss=%.2f%%, avg=%.2fms", loss, avg)
	return PingResult{
		Success:      true,
		PacketLoss:   loss,
		AvgLatencyMs: avg,
		MaxLatencyMs: max,
		MinLatencyMs: min,
	}
}

// === PHẦN 2: LOGIC COLLECTOR CHO PING ===

type PingCollector struct {
	target      string
	bindAddress string

	// Định nghĩa các metrics
	pingUp                *prometheus.Desc
	pingPacketLossPercent *prometheus.Desc
	pingLatencyAvgMs      *prometheus.Desc
}

func NewPingCollector(target, bindAddress string) *PingCollector {
	labels := prometheus.Labels{"target": target, "bind_address": bindAddress}
	return &PingCollector{
		target:      target,
		bindAddress: bindAddress,
		pingUp: prometheus.NewDesc(
			"ping_up",
			"Ping probe successful (1 for success, 0 for failure).",
			nil, labels,
		),
		pingPacketLossPercent: prometheus.NewDesc(
			"ping_packet_loss_percent",
			"Ping packet loss in percent.",
			nil, labels,
		),
		pingLatencyAvgMs: prometheus.NewDesc(
			"ping_latency_average_ms",
			"Ping average latency in milliseconds.",
			nil, labels,
		),
	}
}

func (c *PingCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.pingUp
	ch <- c.pingPacketLossPercent
	ch <- c.pingLatencyAvgMs
}

func (c *PingCollector) Collect(ch chan<- prometheus.Metric) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second) // Timeout 10s cho ping
	defer cancel()

	result := runPing(ctx, c.target, c.bindAddress)

	if result.Success {
		ch <- prometheus.MustNewConstMetric(c.pingUp, prometheus.GaugeValue, 1)
		ch <- prometheus.MustNewConstMetric(c.pingPacketLossPercent, prometheus.GaugeValue, result.PacketLoss)
		ch <- prometheus.MustNewConstMetric(c.pingLatencyAvgMs, prometheus.GaugeValue, result.AvgLatencyMs)
	} else {
		ch <- prometheus.MustNewConstMetric(c.pingUp, prometheus.GaugeValue, 0)
		ch <- prometheus.MustNewConstMetric(c.pingPacketLossPercent, prometheus.GaugeValue, 100)
		ch <- prometheus.MustNewConstMetric(c.pingLatencyAvgMs, prometheus.GaugeValue, 0)
	}
}

// === PHẦN 3: HTTP SERVER ĐỂ CHẠY EXPORTER ===

func probeHandler(w http.ResponseWriter, r *http.Request) {
	target := r.URL.Query().Get("target")
	bindAddress := r.URL.Query().Get("bind_address")

	if target == "" {
		http.Error(w, "'target' parameter is required", http.StatusBadRequest)
		return
	}

	registry := prometheus.NewRegistry()
	collector := NewPingCollector(target, bindAddress)
	registry.MustRegister(collector)

	h := promhttp.HandlerFor(registry, promhttp.HandlerOpts{})
	h.ServeHTTP(w, r)
}

func main() {
	http.HandleFunc("/probe", probeHandler)
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`<html><body><h1>Ping Exporter Test</h1><p><a href="/probe?target=8.8.8.8">Test Probe</a></p></body></html>`))
	})

	log.Println("Ping Exporter Test đang chạy trên cổng :9999")
	if err := http.ListenAndServe(":9999", nil); err != nil {
		log.Fatalf("Lỗi không thể khởi động server: %v", err)
	}
}
