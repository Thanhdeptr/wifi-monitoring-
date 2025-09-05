#!/bin/bash

echo "==pull=="
git pull

echo "=== Restarting WiFi Monitoring Stack ==="

docker-compose down
sleep 2
docker-compose up -d

sleep 5

echo "=== Service Status ==="
docker-compose ps

