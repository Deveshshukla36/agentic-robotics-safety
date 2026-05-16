#!/bin/bash

# Agentic Robotics Safety Monitor - Setup Script
# This script sets up the entire project for first-time use

set -e  # Exit on error

echo "🚀 Agentic Robotics Safety Monitor - Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python
echo -e "${BLUE}📋 Checking prerequisites...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js found: $(node --version)${NC}"

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm not found. Please install npm${NC}"
    exit 1
fi
echo -e "${GREEN}✓ npm found: $(npm --version)${NC}"

echo ""
echo -e "${BLUE}📦 Setting up Backend...${NC}"

# Backend setup
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment and install dependencies
echo "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Create data directories
mkdir -p data
mkdir -p logs
echo -e "${GREEN}✓ Data directories created${NC}"

cd ..

echo ""
echo -e "${BLUE}📦 Setting up Frontend...${NC}"

# Frontend setup
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
    echo -e "${GREEN}✓ Node.js dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Node.js dependencies already installed${NC}"
fi

cd ..

echo ""
echo -e "${BLUE}📁 Creating additional directories...${NC}"
mkdir -p data/uploads
mkdir -p data/experiments
mkdir -p logs/backend
mkdir -p logs/frontend

echo ""
echo -e "${GREEN}✅ Setup completed successfully!${NC}"
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo ""
echo "1. Start the backend server:"
echo "   ${BLUE}cd backend && source venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000${NC}"
echo ""
echo "2. Start the frontend (in a new terminal):"
echo "   ${BLUE}cd frontend && npm run dev${NC}"
echo ""
echo "3. Open your browser and navigate to:"
echo "   ${BLUE}http://localhost:3000${NC}"
echo ""
echo "4. API documentation available at:"
echo "   ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}💡 Tip: You can also use Docker Compose to run both services:${NC}"
echo "   ${BLUE}docker-compose up --build${NC}"
echo ""