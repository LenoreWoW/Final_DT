#!/bin/bash

# Quantum Trail - Stop Script
# Gracefully stops all services of the quantum computing platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🛑 Stopping Quantum Trail Platform...${NC}"

# Function to stop process by PID file
stop_process_by_pid() {
    local pid_file=$1
    local service_name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        echo -e "${YELLOW}Stopping $service_name (PID: $pid)...${NC}"
        
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid"
            
            # Wait for graceful shutdown
            local count=0
            while kill -0 "$pid" 2>/dev/null && [ $count -lt 30 ]; do
                sleep 1
                ((count++))
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${YELLOW}Force stopping $service_name...${NC}"
                kill -KILL "$pid" 2>/dev/null || true
            fi
            
            echo -e "${GREEN}✅ $service_name stopped${NC}"
        else
            echo -e "${YELLOW}⚠️  $service_name was not running${NC}"
        fi
        
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}⚠️  $service_name PID file not found${NC}"
    fi
}

# Stop Flask application (if running in background)
echo -e "${BLUE}🌐 Stopping web application...${NC}"
pkill -f "python.*app.py" || echo -e "${YELLOW}⚠️  Web application was not running${NC}"

# Stop Celery workers
echo -e "${BLUE}👷 Stopping Celery workers...${NC}"
stop_process_by_pid "./celery_worker.pid" "Celery Worker"
stop_process_by_pid "./celery_beat.pid" "Celery Beat"
stop_process_by_pid "/var/run/celery/worker.pid" "Celery Worker (production)"
stop_process_by_pid "/var/run/celery/beat.pid" "Celery Beat (production)"

# Stop any remaining Celery processes
pkill -f "celery.*worker" || echo -e "${YELLOW}⚠️  No additional Celery workers found${NC}"
pkill -f "celery.*beat" || echo -e "${YELLOW}⚠️  No additional Celery beat processes found${NC}"

# Stop Docker services
echo -e "${BLUE}🐳 Stopping Docker services...${NC}"
if command -v docker-compose >/dev/null 2>&1; then
    if [ -f "docker-compose.yml" ]; then
        docker-compose down
        echo -e "${GREEN}✅ Docker services stopped${NC}"
    else
        echo -e "${YELLOW}⚠️  docker-compose.yml not found${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Docker Compose not available${NC}"
fi

# Clean up any remaining processes
echo -e "${BLUE}🧹 Cleaning up remaining processes...${NC}"

# Stop any Python processes related to the project
pkill -f "python.*dt_project" || echo -e "${YELLOW}⚠️  No additional Python processes found${NC}"

# Stop any gunicorn processes
pkill -f "gunicorn.*dt_project" || echo -e "${YELLOW}⚠️  No Gunicorn processes found${NC}"

# Remove temporary files
echo -e "${BLUE}🗑️  Removing temporary files...${NC}"
rm -f ./celery_worker.pid ./celery_beat.pid
rm -f /tmp/quantum_trail_*.tmp 2>/dev/null || true

# Check if any services are still running
echo -e "${BLUE}🔍 Checking for remaining processes...${NC}"
remaining_processes=$(pgrep -f "quantum|celery|dt_project" || true)

if [ -n "$remaining_processes" ]; then
    echo -e "${YELLOW}⚠️  Some processes may still be running:${NC}"
    ps -p $remaining_processes -o pid,cmd || true
    echo -e "${YELLOW}   Run 'pkill -f quantum' to force stop all related processes${NC}"
else
    echo -e "${GREEN}✅ All processes stopped cleanly${NC}"
fi

echo -e "${GREEN}🏁 Quantum Trail Platform stopped successfully!${NC}"

# Show final status
if command -v docker >/dev/null 2>&1; then
    running_containers=$(docker ps --filter "name=quantum" -q 2>/dev/null || true)
    if [ -n "$running_containers" ]; then
        echo -e "${YELLOW}⚠️  Some Docker containers may still be running${NC}"
        echo -e "${YELLOW}   Run 'docker ps' to check container status${NC}"
    fi
fi

echo -e "${BLUE}💡 To restart the platform, run: ./start.sh${NC}"