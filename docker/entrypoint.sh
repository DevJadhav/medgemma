#!/bin/bash
# Entrypoint script for MedAI Compass Docker container
# Performs startup validation before launching the main process

set -e

echo "=============================================="
echo "  MedAI Compass - Container Startup"
echo "=============================================="

# Check required environment variables
REQUIRED_VARS=(
    "POSTGRES_PASSWORD"
    "REDIS_PASSWORD"
    "JWT_SECRET"
    "PHI_ENCRYPTION_KEY"
)

MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: Missing required environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "To generate secrets, run:"
    echo "  python scripts/generate_secrets.py"
    echo ""
    exit 1
fi

echo "✓ Required environment variables set"

# Verify HuggingFace token (warning only)
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "⚠ WARNING: HF_TOKEN not set - MedGemma model access may fail"
    echo "  Get a token at: https://huggingface.co/settings/tokens"
fi

# Check Modal configuration (info only - not required)
if [ -z "$MODAL_TOKEN_ID" ] || [ -z "$MODAL_TOKEN_SECRET" ]; then
    echo "ℹ INFO: Modal tokens not set - using local GPU/CPU"
    echo "  For cloud GPU access, run: modal token new"
else
    echo "✓ Modal cloud GPU configured"
fi

# Wait for PostgreSQL
echo ""
echo "Waiting for PostgreSQL..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if pg_isready -h "${POSTGRES_HOST:-postgres}" -p "${POSTGRES_PORT:-5432}" -U "${POSTGRES_USER:-medai}" > /dev/null 2>&1; then
        echo "✓ PostgreSQL is ready"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Waiting for PostgreSQL... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: PostgreSQL not available after $MAX_RETRIES attempts"
    exit 1
fi

# Wait for Redis
echo ""
echo "Waiting for Redis..."
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if redis-cli -h "${REDIS_HOST:-redis}" -p "${REDIS_PORT:-6379}" -a "${REDIS_PASSWORD}" ping > /dev/null 2>&1; then
        echo "✓ Redis is ready"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Waiting for Redis... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "WARNING: Redis not available - some features may be limited"
fi

# Run connection verification (non-blocking)
echo ""
echo "Verifying connections..."
python scripts/verify_connections.py --quick --skip-db 2>/dev/null || true

echo ""
echo "=============================================="
echo "  Starting application..."
echo "=============================================="
echo ""

# Execute the main command
exec "$@"
