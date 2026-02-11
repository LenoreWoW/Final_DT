#!/bin/bash

# Quantum Trail - Main Startup Script
# This script initializes and starts the complete quantum computing platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-development}
SKIP_DEPS=${2:-false}

echo -e "${BLUE}🚀 Starting Quantum Trail Platform...${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    local max_attempts=30
    local attempt=1
    
    echo -e "${YELLOW}Waiting for $service to be ready...${NC}"
    while ! nc -z $host $port >/dev/null 2>&1; do
        if [ $attempt -ge $max_attempts ]; then
            echo -e "${RED}❌ $service failed to start after $max_attempts attempts${NC}"
            return 1
        fi
        echo -e "${YELLOW}Attempt $attempt/$max_attempts: $service not ready, waiting...${NC}"
        sleep 2
        ((attempt++))
    done
    echo -e "${GREEN}✅ $service is ready${NC}"
}

# Check prerequisites
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is required but not installed${NC}"
    exit 1
fi

if ! command_exists docker; then
    echo -e "${RED}❌ Docker is required but not installed${NC}"
    exit 1
fi

if ! command_exists docker-compose; then
    echo -e "${RED}❌ Docker Compose is required but not installed${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found, copying from .env.example${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env from template${NC}"
        echo -e "${YELLOW}⚠️  Please review and update .env with your configurations${NC}"
    else
        echo -e "${RED}❌ .env.example not found${NC}"
        exit 1
    fi
fi

# Load environment variables
source .env

# Install Python dependencies
if [ "$SKIP_DEPS" != "true" ]; then
    echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
    python3 -m pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
fi

# Start infrastructure services with Docker Compose
echo -e "${BLUE}🐳 Starting infrastructure services...${NC}"
docker-compose up -d postgres redis nginx prometheus grafana

# Wait for database to be ready
wait_for_service localhost 5432 "PostgreSQL"

# Initialize database
echo -e "${BLUE}🗄️  Initializing database...${NC}"
python3 scripts/init_db.py
echo -e "${GREEN}✅ Database initialized${NC}"

# Start Celery workers
echo -e "${BLUE}👷 Starting Celery workers...${NC}"
if [ "$ENVIRONMENT" = "production" ]; then
    celery -A dt_project.celery_app worker --loglevel=info --detach --pidfile=/var/run/celery/worker.pid
    celery -A dt_project.celery_app beat --loglevel=info --detach --pidfile=/var/run/celery/beat.pid
else
    celery -A dt_project.celery_app worker --loglevel=debug --detach --pidfile=./celery_worker.pid
    celery -A dt_project.celery_app beat --loglevel=debug --detach --pidfile=./celery_beat.pid
fi
echo -e "${GREEN}✅ Celery workers started${NC}"

# Start the main application
echo -e "${BLUE}🌟 Starting Quantum Trail web application...${NC}"
if [ "$ENVIRONMENT" = "production" ]; then
    gunicorn --bind 0.0.0.0:8000 --workers 4 --timeout 120 dt_project.web_interface.app:app
else
    export FLASK_ENV=development
    export FLASK_DEBUG=1
    export FLASK_RUN_PORT=8000
    python3 -m dt_project.web_interface.app &
    APP_PID=$!
fi

# Wait for application to start
wait_for_service localhost 8000 "Quantum Trail Application"

echo -e "${GREEN}🎉 Quantum Trail Platform started successfully!${NC}"
echo -e "${BLUE}📱 Application URLs:${NC}"
echo -e "   🌐 Main Application: http://localhost:8000"
echo -e "   📊 Grafana Dashboard: http://localhost:3000"
echo -e "   📈 Prometheus Metrics: http://localhost:9090"
echo -e "   🔍 Admin Interface: http://localhost:8000/admin"

if [ "$ENVIRONMENT" = "development" ]; then
    echo -e "${BLUE}🛠️  Development Mode Active${NC}"
    echo -e "   📝 API Documentation: http://localhost:8000/docs"
    echo -e "   🧪 GraphQL Playground: http://localhost:8000/graphql"
    echo -e "${YELLOW}💡 Run 'python3 scripts/demo.py' for a quick demonstration${NC}"
    
    # Keep the script running in development mode
    echo -e "${BLUE}Press Ctrl+C to stop the application${NC}"
    trap 'echo -e "${YELLOW}Stopping services...${NC}"; kill $APP_PID 2>/dev/null; docker-compose down; exit 0' INT
    wait $APP_PID
fi