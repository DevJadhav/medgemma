#!/bin/bash
# Start Ray Cluster Script
# Starts Ray head and worker nodes for distributed training
#
# Usage:
#   ./scripts/start_ray_cluster.sh                    # Start local cluster
#   ./scripts/start_ray_cluster.sh --workers 4       # With 4 workers
#   ./scripts/start_ray_cluster.sh --gpu              # With GPU support

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
NUM_WORKERS=1
USE_GPU=false
USE_DOCKER=true
RAY_PORT=6379
DASHBOARD_PORT=8265

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        --local)
            USE_DOCKER=false
            shift
            ;;
        --port)
            RAY_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --workers N    Number of worker nodes (default: 1)"
            echo "  --gpu          Enable GPU support"
            echo "  --local        Run locally instead of Docker"
            echo "  --port PORT    Ray port (default: 6379)"
            echo "  -h, --help     Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

if $USE_DOCKER; then
    log_info "Starting Ray cluster using Docker..."
    
    # Build training image if needed
    if ! docker images | grep -q "medai-training"; then
        log_info "Building training image..."
        docker build -f docker/training.Dockerfile -t medai-training:latest --target ray-head .
    fi
    
    # Start Ray head
    log_info "Starting Ray head node..."
    if $USE_GPU; then
        docker-compose --profile training --profile gpu up -d ray-head
    else
        docker-compose --profile training up -d ray-head
    fi
    
    # Wait for head to be ready
    log_info "Waiting for Ray head to be ready..."
    sleep 10
    
    # Start workers
    if [[ $NUM_WORKERS -gt 0 ]]; then
        log_info "Starting $NUM_WORKERS Ray worker(s)..."
        if $USE_GPU; then
            docker-compose --profile training --profile gpu up -d --scale ray-worker=$NUM_WORKERS ray-worker
        else
            docker-compose --profile training up -d --scale ray-worker=$NUM_WORKERS ray-worker
        fi
    fi
    
    # Check status
    sleep 5
    log_info "Checking Ray cluster status..."
    docker-compose exec ray-head ray status || log_warning "Could not get Ray status"
    
else
    log_info "Starting Ray cluster locally..."
    
    # Check if Ray is installed
    if ! command -v ray &> /dev/null; then
        log_error "Ray is not installed. Install with: pip install 'ray[default]'"
        exit 1
    fi
    
    # Start head node
    log_info "Starting Ray head node on port $RAY_PORT..."
    ray start --head --port=$RAY_PORT --dashboard-host=0.0.0.0 --dashboard-port=$DASHBOARD_PORT
    
    # Get head address
    HEAD_ADDRESS="localhost:$RAY_PORT"
    
    # Start workers
    if [[ $NUM_WORKERS -gt 0 ]]; then
        log_info "Starting $NUM_WORKERS Ray worker(s)..."
        for i in $(seq 1 $NUM_WORKERS); do
            ray start --address=$HEAD_ADDRESS
        done
    fi
    
    # Check status
    sleep 3
    ray status
fi

log_success "Ray cluster started!"
echo ""
echo "Access points:"
echo "  - Ray Dashboard: http://localhost:$DASHBOARD_PORT"
echo "  - Ray Client: ray://localhost:10001"
echo ""
echo "To stop the cluster:"
if $USE_DOCKER; then
    echo "  docker-compose --profile training down"
else
    echo "  ray stop"
fi
