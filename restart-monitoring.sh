#!/bin/bash

echo "=== Restarting WiFi Monitoring Stack ==="

# Stop all services
echo "Stopping services..."
docker-compose down

# Wait a moment
sleep 5

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Check service status
echo "=== Service Status ==="
docker-compose ps

echo ""
echo "=== Access Information ==="
echo "Prometheus: http://192.168.10.18:9090"
echo "Grafana: http://192.168.10.18:3001 (admin/admin)"
echo "SNMP Exporter: http://192.168.10.18:9116"
echo ""
echo "=== Import Grafana Dashboard ==="
echo "1. Open Grafana at http://192.168.10.18:3001"
echo "2. Login with admin/admin"
echo "3. Go to '+' -> Import"
echo "4. Upload the grafana-dashboard.json file"
echo "5. Configure Prometheus datasource as: http://prometheus:9090"
echo ""
echo "=== Test SNMP Connection ==="
echo "Testing SNMP connection to router..."
curl -s "http://192.168.10.18:9116/snmp?target=192.168.10.4&module=if_mib&auth=public_v2c" | head -20
