#!/bin/bash

echo "🧪 Testing iperf3 server connectivity..."

# Test 1: Basic connectivity
echo "1. Testing basic connectivity to iperf3 server..."
ping -c 3 192.168.10.175

# Test 2: Port connectivity  
echo "2. Testing port 5201 connectivity..."
nc -zv 192.168.10.175 5201

# Test 3: Manual iperf3 test
echo "3. Testing manual iperf3 command..."
iperf3 -J -t 5 -c 192.168.10.175 -p 5201

# Test 4: Test with bind address
echo "4. Testing with bind address..."
iperf3 -J -t 5 -c 192.168.10.175 -p 5201 -B 192.168.10.160

# Test 5: Check iperf3 server status
echo "5. Checking if iperf3 server is running..."
netstat -tlnp | grep 5201

echo "✅ Tests completed!"
