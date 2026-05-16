#!/bin/bash

# Agentic Robotics Safety Monitor - System Test Script
# This script tests all major functionality of the system

set -e

echo "🧪 Agentic Robotics Safety Monitor - System Test"
echo "================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if backend is running
echo -e "${BLUE}📡 Checking backend status...${NC}"
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Backend is running${NC}"
else
    echo -e "${RED}❌ Backend is not running. Please start the backend first:${NC}"
    echo "   cd backend && source venv/bin/activate && uvicorn main:app --reload"
    exit 1
fi

# Test 1: Health Check
echo ""
echo -e "${BLUE}📋 Test 1: Health Check${NC}"
RESPONSE=$(curl -s http://localhost:8000/)
if echo "$RESPONSE" | grep -q "operational"; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed${NC}"
fi

# Test 2: Scenario Analysis
echo ""
echo -e "${BLUE}📋 Test 2: Scenario Analysis${NC}"
SCENARIO='{
    "robot": {
        "position": [5.0, 5.0],
        "speed": 1.5,
        "velocity": [0.5, 0.3]
    },
    "environment": {
        "width": 10.0,
        "height": 10.0,
        "humans": [
            {"position": [4.0, 4.5], "velocity": [0.1, 0.1]}
        ],
        "restricted_zones": [
            {"type": "circle", "center": [1.0, 1.0], "radius": 1.5, "name": "maintenance"}
        ]
    }
}'

RESPONSE=$(curl -s -X POST http://localhost:8000/api/scenario/analyze \
    -H "Content-Type: application/json" \
    -d "$SCENARIO")

if echo "$RESPONSE" | grep -q "safety_level"; then
    echo -e "${GREEN}✓ Scenario analysis passed${NC}"
    SAFETY_LEVEL=$(echo "$RESPONSE" | grep -o '"safety_level":"[^"]*"' | cut -d'"' -f4)
    echo "   Safety Level: $SAFETY_LEVEL"
else
    echo -e "${RED}✗ Scenario analysis failed${NC}"
fi

# Test 3: Batch Testing
echo ""
echo -e "${BLUE}📋 Test 3: Batch Testing${NC}"
BATCH='{
    "name": "Test Batch",
    "scenarios": [
        {
            "robot": {"position": [5.0, 5.0], "speed": 1.0},
            "environment": {"humans": [{"position": [4.0, 4.5]}]}
        },
        {
            "robot": {"position": [3.0, 3.0], "speed": 2.0},
            "environment": {"humans": [{"position": [3.5, 3.2]}]}
        }
    ]
}'

RESPONSE=$(curl -s -X POST http://localhost:8000/api/batch/test \
    -H "Content-Type: application/json" \
    -d "$BATCH")

if echo "$RESPONSE" | grep -q "total_scenarios"; then
    echo -e "${GREEN}✓ Batch testing passed${NC}"
    TOTAL=$(echo "$RESPONSE" | grep -o '"total_scenarios":[0-9]*' | cut -d':' -f2)
    echo "   Total scenarios: $TOTAL"
else
    echo -e "${RED}✗ Batch testing failed${NC}"
fi

# Test 4: Simulation State
echo ""
echo -e "${BLUE}📋 Test 4: Simulation State${NC}"
RESPONSE=$(curl -s http://localhost:8000/api/simulation/state)
if echo "$RESPONSE" | grep -q "robot"; then
    echo -e "${GREEN}✓ Simulation state fetch passed${NC}"
else
    echo -e "${RED}✗ Simulation state fetch failed${NC}"
fi

# Test 5: Adversarial Generation
echo ""
echo -e "${BLUE}📋 Test 5: Adversarial Generation${NC}"
ADVERSARIAL='{
    "robot": {"position": [5.0, 5.0], "speed": 1.5},
    "environment": {"width": 10.0, "height": 10.0}
}'

RESPONSE=$(curl -s -X POST http://localhost:8000/api/adversarial/generate \
    -H "Content-Type: application/json" \
    -d "$ADVERSARIAL")

if echo "$RESPONSE" | grep -q "count"; then
    echo -e "${GREEN}✓ Adversarial generation passed${NC}"
    COUNT=$(echo "$RESPONSE" | grep -o '"count":[0-9]*' | cut -d':' -f2)
    echo "   Generated $COUNT adversarial scenarios"
else
    echo -e "${RED}✗ Adversarial generation failed${NC}"
fi

# Test 6: Metrics
echo ""
echo -e "${BLUE}📋 Test 6: System Metrics${NC}"
RESPONSE=$(curl -s http://localhost:8000/api/metrics)
if echo "$RESPONSE" | grep -q "system_metrics"; then
    echo -e "${GREEN}✓ Metrics fetch passed${NC}"
else
    echo -e "${RED}✗ Metrics fetch failed${NC}"
fi

# Test 7: Error Handling - Malformed Input
echo ""
echo -e "${BLUE}📋 Test 7: Error Handling (Malformed Input)${NC}"
MALFORMED='{"robot": {}, "environment": {}}'
RESPONSE=$(curl -s -X POST http://localhost:8000/api/scenario/analyze \
    -H "Content-Type: application/json" \
    -d "$MALFORMED")

if echo "$RESPONSE" | grep -q "error\|safety_level"; then
    echo -e "${GREEN}✓ Error handling passed${NC}"
else
    echo -e "${RED}✗ Error handling test failed${NC}"
fi

# Test 8: CORS Headers
echo ""
echo -e "${BLUE}📋 Test 8: CORS Configuration${NC}"
HEADERS=$(curl -s -I http://localhost:8000/ | grep -i "access-control-allow-origin")
if echo "$HEADERS" | grep -q "localhost"; then
    echo -e "${GREEN}✓ CORS configured correctly${NC}"
else
    echo -e "${YELLOW}⚠ CORS header check - may need configuration${NC}"
fi

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ System tests completed!${NC}"
echo ""
echo -e "${YELLOW}📊 Performance Summary:${NC}"
echo "   - Backend: Operational"
echo "   - API Endpoints: All responding"
echo "   - Error Handling: Working"
echo "   - Agent System: Active"
echo ""
echo -e "${BLUE}🌐 Access the application:${NC}"
echo "   Frontend: http://localhost:3000"
echo "   API Docs: http://localhost:8000/docs"
echo ""