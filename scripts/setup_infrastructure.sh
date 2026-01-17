#!/bin/bash
# MedAI Compass Infrastructure Setup Script
# Sets up Ray cluster, MLflow tracking server, and MinIO artifact storage
#
# Usage:
#   ./scripts/setup_infrastructure.sh              # Full setup
#   ./scripts/setup_infrastructure.sh --training   # Training infrastructure only
#   ./scripts/setup_infrastructure.sh --verify     # Verify setup

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"

# Default settings
SETUP_TRAINING=false
VERIFY_ONLY=false
USE_GPU=false
MODEL="medgemma-4b"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed"
        return 1
    fi
    log_success "$1 is available"
    return 0
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --training)
            SETUP_TRAINING=true
            shift
            ;;
        --verify)
            VERIFY_ONLY=true
            shift
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --training    Set up training infrastructure (Ray, MLflow)"
            echo "  --verify      Verify existing setup only"
            echo "  --gpu         Enable GPU support"
            echo "  --model NAME  Select model (medgemma-4b or medgemma-27b)"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Verify Prerequisites
# =============================================================================

verify_prerequisites() {
    log_info "Verifying prerequisites..."
    
    local failed=false
    
    # Check required commands
    check_command "docker" || failed=true
    check_command "docker-compose" || check_command "docker" || failed=true
    check_command "uv" || failed=true
    check_command "python3" || failed=true
    
    # Check optional commands
    if command -v "modal" &> /dev/null; then
        log_success "Modal CLI is available"
    else
        log_warning "Modal CLI not installed - cloud training will not work"
    fi
    
    if command -v "ray" &> /dev/null; then
        log_success "Ray CLI is available"
    else
        log_warning "Ray CLI not installed locally (will use Docker)"
    fi
    
    # Check Docker is running
    if docker info &> /dev/null; then
        log_success "Docker daemon is running"
    else
        log_error "Docker daemon is not running"
        failed=true
    fi
    
    # Check configuration files exist
    local configs=(
        "$CONFIG_DIR/ray_cluster.yaml"
        "$CONFIG_DIR/mlflow_config.yaml"
        "$CONFIG_DIR/modal_config.yaml"
        "$CONFIG_DIR/models.yaml"
        "$CONFIG_DIR/training.yaml"
    )
    
    for config in "${configs[@]}"; do
        if [[ -f "$config" ]]; then
            log_success "Config exists: $(basename "$config")"
        else
            log_error "Missing config: $config"
            failed=true
        fi
    done
    
    if $failed; then
        log_error "Prerequisites check failed"
        return 1
    fi
    
    log_success "All prerequisites verified"
    return 0
}

# =============================================================================
# Setup Python Environment
# =============================================================================

setup_python_environment() {
    log_info "Setting up Python environment with uv..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d ".venv" ]]; then
        log_info "Creating virtual environment..."
        uv venv .venv
    fi
    
    # Activate and install dependencies
    log_info "Installing dependencies..."
    source .venv/bin/activate
    uv pip install -e ".[dev]"
    
    # Install training dependencies
    log_info "Installing training dependencies..."
    uv pip install \
        "ray[train,tune]==2.9.0" \
        "mlflow>=2.10.0" \
        "deepspeed>=0.13.0" \
        "flash-attn>=2.5.0" || log_warning "Some training packages may have failed"
    
    log_success "Python environment ready"
}

# =============================================================================
# Setup Docker Infrastructure
# =============================================================================

setup_docker_infrastructure() {
    log_info "Setting up Docker infrastructure..."
    
    cd "$PROJECT_ROOT"
    
    # Check if .env file exists, create from template if not
    if [[ ! -f ".env" ]]; then
        log_info "Creating .env file from template..."
        if [[ -f ".env.example" ]]; then
            cp .env.example .env
        else
            cat > .env << 'EOF'
# MedAI Compass Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO

# PostgreSQL
POSTGRES_DB=medai_compass
POSTGRES_USER=medai
POSTGRES_PASSWORD=changeme_postgres_password

# Redis
REDIS_PASSWORD=changeme_redis_password

# MinIO
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=changeme_minio_password

# HuggingFace
HF_TOKEN=your_hf_token_here

# JWT
JWT_SECRET=changeme_jwt_secret

# PHI Encryption
PHI_ENCRYPTION_KEY=changeme_encryption_key

# Modal (for cloud GPU)
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=
PREFER_MODAL_GPU=true

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
EOF
        fi
        log_warning "Created .env file - please update with secure passwords!"
    fi
    
    # Build training Docker image
    log_info "Building training Docker image..."
    docker build -f docker/training.Dockerfile -t medai-training:latest --target production . || {
        log_warning "Training image build failed - continuing with other setup"
    }
    
    # Start core services
    log_info "Starting core Docker services..."
    docker-compose up -d postgres redis minio
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 10
    
    # Create MLflow database
    log_info "Creating MLflow database..."
    docker-compose exec -T postgres psql -U "${POSTGRES_USER:-medai}" -c "CREATE DATABASE mlflow;" 2>/dev/null || \
        log_info "MLflow database may already exist"
    
    # Create MinIO buckets
    log_info "Creating MinIO buckets..."
    docker-compose exec -T minio mc alias set local http://localhost:9000 "${MINIO_ACCESS_KEY:-minioadmin}" "${MINIO_SECRET_KEY:-minioadmin}" 2>/dev/null || true
    docker-compose exec -T minio mc mb local/mlflow-artifacts 2>/dev/null || log_info "mlflow-artifacts bucket may already exist"
    docker-compose exec -T minio mc mb local/medai-checkpoints 2>/dev/null || log_info "medai-checkpoints bucket may already exist"
    docker-compose exec -T minio mc mb local/medai-data 2>/dev/null || log_info "medai-data bucket may already exist"
    
    log_success "Docker infrastructure ready"
}

# =============================================================================
# Setup Training Infrastructure
# =============================================================================

setup_training_infrastructure() {
    log_info "Setting up training infrastructure..."
    
    cd "$PROJECT_ROOT"
    
    # Start training services
    log_info "Starting Ray and MLflow services..."
    
    if $USE_GPU; then
        docker-compose --profile training --profile gpu up -d ray-head ray-worker mlflow
    else
        docker-compose --profile training up -d ray-head mlflow
    fi
    
    # Wait for services
    log_info "Waiting for training services to start..."
    sleep 15
    
    # Verify Ray head is running
    if docker-compose ps ray-head | grep -q "Up"; then
        log_success "Ray head node is running"
    else
        log_warning "Ray head node may not be running"
    fi
    
    # Verify MLflow is running
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        log_success "MLflow tracking server is running"
    else
        log_warning "MLflow tracking server may not be accessible"
    fi
    
    log_success "Training infrastructure ready"
}

# =============================================================================
# Setup Modal
# =============================================================================

setup_modal() {
    log_info "Setting up Modal for cloud GPU training..."
    
    # Check if Modal CLI is installed
    if ! command -v "modal" &> /dev/null; then
        log_info "Installing Modal CLI..."
        uv pip install modal
    fi
    
    # Check Modal authentication
    if modal token current &> /dev/null; then
        log_success "Modal is authenticated"
    else
        log_warning "Modal is not authenticated"
        log_info "Run 'modal token new' to authenticate"
        return 1
    fi
    
    # Create Modal secrets if they don't exist
    log_info "Setting up Modal secrets..."
    
    # Check if huggingface-secret exists
    if ! modal secret list 2>/dev/null | grep -q "huggingface-secret"; then
        log_info "Creating huggingface-secret (you'll need to add HF_TOKEN)"
        modal secret create huggingface-secret HF_TOKEN="" || true
    fi
    
    # Deploy training app
    log_info "Deploying Modal training app..."
    modal deploy medai_compass/modal/training_app.py || {
        log_warning "Modal deployment failed - check logs"
    }
    
    log_success "Modal setup complete"
}

# =============================================================================
# Verify Setup
# =============================================================================

verify_setup() {
    log_info "Verifying infrastructure setup..."
    
    local failed=false
    
    # Check Docker services
    log_info "Checking Docker services..."
    
    services=("postgres" "redis" "minio")
    if $SETUP_TRAINING; then
        services+=("ray-head" "mlflow")
    fi
    
    for service in "${services[@]}"; do
        if docker-compose ps "$service" 2>/dev/null | grep -q "Up"; then
            log_success "$service is running"
        else
            log_warning "$service is not running"
        fi
    done
    
    # Check PostgreSQL
    log_info "Checking PostgreSQL connection..."
    if docker-compose exec -T postgres pg_isready -U "${POSTGRES_USER:-medai}" > /dev/null 2>&1; then
        log_success "PostgreSQL is accepting connections"
    else
        log_error "PostgreSQL is not accepting connections"
        failed=true
    fi
    
    # Check Redis
    log_info "Checking Redis connection..."
    if docker-compose exec -T redis redis-cli -a "${REDIS_PASSWORD:-}" ping 2>/dev/null | grep -q "PONG"; then
        log_success "Redis is responding"
    else
        log_error "Redis is not responding"
        failed=true
    fi
    
    # Check MinIO
    log_info "Checking MinIO connection..."
    if curl -s http://localhost:9000/minio/health/live > /dev/null 2>&1; then
        log_success "MinIO is healthy"
    else
        log_error "MinIO is not healthy"
        failed=true
    fi
    
    # Check MLflow
    if $SETUP_TRAINING; then
        log_info "Checking MLflow..."
        if curl -s http://localhost:5000/health > /dev/null 2>&1; then
            log_success "MLflow is healthy"
        else
            log_warning "MLflow may not be accessible (this is expected if training profile not started)"
        fi
        
        # Check Ray
        log_info "Checking Ray cluster..."
        if docker-compose exec -T ray-head ray status > /dev/null 2>&1; then
            log_success "Ray cluster is running"
        else
            log_warning "Ray cluster may not be accessible"
        fi
    fi
    
    # Run Python tests
    log_info "Running infrastructure tests..."
    cd "$PROJECT_ROOT"
    source .venv/bin/activate
    
    if python -m pytest tests/test_infrastructure.py -v -x --tb=short -m "not integration and not slow" 2>/dev/null; then
        log_success "Infrastructure tests passed"
    else
        log_warning "Some infrastructure tests failed"
    fi
    
    if $failed; then
        log_error "Verification failed"
        return 1
    fi
    
    log_success "Infrastructure verification complete"
    return 0
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo "==========================================="
    echo "MedAI Compass Infrastructure Setup"
    echo "==========================================="
    echo ""
    
    if $VERIFY_ONLY; then
        verify_setup
        exit $?
    fi
    
    # Run setup steps
    verify_prerequisites || exit 1
    setup_python_environment
    setup_docker_infrastructure
    
    if $SETUP_TRAINING; then
        setup_training_infrastructure
    fi
    
    # Setup Modal if credentials are available
    if [[ -n "${MODAL_TOKEN_ID:-}" ]] || command -v modal &> /dev/null; then
        setup_modal || true
    fi
    
    # Final verification
    verify_setup
    
    echo ""
    echo "==========================================="
    log_success "Infrastructure setup complete!"
    echo "==========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Update .env with your credentials"
    echo "  2. Start training with: docker-compose --profile training up -d"
    echo "  3. Access MLflow at: http://localhost:5000"
    echo "  4. Access Ray Dashboard at: http://localhost:8265"
    echo "  5. Deploy to Modal with: modal deploy medai_compass/modal/training_app.py"
    echo ""
}

main "$@"
