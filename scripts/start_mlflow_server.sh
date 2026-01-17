#!/bin/bash
# Start MLflow Tracking Server
# Starts MLflow with PostgreSQL backend and MinIO artifact storage
#
# Usage:
#   ./scripts/start_mlflow_server.sh              # Start with Docker
#   ./scripts/start_mlflow_server.sh --local      # Start locally

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
USE_DOCKER=true
MLFLOW_PORT=5000

# Load environment
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            USE_DOCKER=false
            shift
            ;;
        --port)
            MLFLOW_PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --local        Run MLflow locally instead of Docker"
            echo "  --port PORT    MLflow port (default: 5000)"
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

# Database and storage configuration
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-medai}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-changeme}"
POSTGRES_DB="mlflow"

MINIO_ENDPOINT="${MINIO_ENDPOINT:-localhost:9000}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"

BACKEND_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
ARTIFACT_ROOT="s3://mlflow-artifacts"

if $USE_DOCKER; then
    log_info "Starting MLflow with Docker..."
    
    # Ensure PostgreSQL and MinIO are running
    log_info "Ensuring PostgreSQL and MinIO are running..."
    docker-compose up -d postgres minio
    sleep 5
    
    # Create MLflow database if it doesn't exist
    log_info "Creating MLflow database..."
    docker-compose exec -T postgres psql -U "${POSTGRES_USER}" -tc "SELECT 1 FROM pg_database WHERE datname = 'mlflow'" | grep -q 1 || \
        docker-compose exec -T postgres psql -U "${POSTGRES_USER}" -c "CREATE DATABASE mlflow;"
    
    # Create MinIO bucket if it doesn't exist
    log_info "Creating MLflow artifacts bucket..."
    docker-compose exec -T minio mc alias set local http://localhost:9000 "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" 2>/dev/null || true
    docker-compose exec -T minio mc mb local/mlflow-artifacts 2>/dev/null || log_info "Bucket may already exist"
    
    # Start MLflow service
    log_info "Starting MLflow service..."
    docker-compose --profile training up -d mlflow
    
    # Wait for MLflow to be ready
    log_info "Waiting for MLflow to be ready..."
    for i in {1..30}; do
        if curl -s "http://localhost:${MLFLOW_PORT}/health" > /dev/null 2>&1; then
            log_success "MLflow is ready!"
            break
        fi
        sleep 2
    done
    
else
    log_info "Starting MLflow locally..."
    
    # Check if MLflow is installed
    if ! command -v mlflow &> /dev/null; then
        log_error "MLflow is not installed. Install with: pip install mlflow"
        exit 1
    fi
    
    # Set AWS credentials for MinIO
    export AWS_ACCESS_KEY_ID="${MINIO_ACCESS_KEY}"
    export AWS_SECRET_ACCESS_KEY="${MINIO_SECRET_KEY}"
    export MLFLOW_S3_ENDPOINT_URL="http://${MINIO_ENDPOINT}"
    
    # Verify PostgreSQL connection
    log_info "Verifying PostgreSQL connection..."
    if ! pg_isready -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" > /dev/null 2>&1; then
        log_error "Cannot connect to PostgreSQL. Ensure it's running."
        exit 1
    fi
    
    # Create database if needed
    PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -tc "SELECT 1 FROM pg_database WHERE datname = 'mlflow'" | grep -q 1 || \
        PGPASSWORD="${POSTGRES_PASSWORD}" psql -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -c "CREATE DATABASE mlflow;"
    
    log_info "Starting MLflow server..."
    mlflow server \
        --host 0.0.0.0 \
        --port "${MLFLOW_PORT}" \
        --backend-store-uri "${BACKEND_URI}" \
        --default-artifact-root "${ARTIFACT_ROOT}" \
        --serve-artifacts
fi

log_success "MLflow tracking server started!"
echo ""
echo "Access points:"
echo "  - MLflow UI: http://localhost:${MLFLOW_PORT}"
echo "  - Tracking URI: http://localhost:${MLFLOW_PORT}"
echo ""
echo "To set tracking URI in Python:"
echo "  import mlflow"
echo "  mlflow.set_tracking_uri('http://localhost:${MLFLOW_PORT}')"
echo ""
echo "To stop the server:"
if $USE_DOCKER; then
    echo "  docker-compose --profile training down mlflow"
else
    echo "  Press Ctrl+C or kill the process"
fi
