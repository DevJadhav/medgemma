# Docker Deployment Guide

Deploy MedAI Compass using Docker for production or development.

## Prerequisites

- Docker Desktop 4.0+
- Docker Compose 2.0+
- 8GB+ RAM recommended
- GPU (optional, for inference)

## Quick Start

```bash
# Copy environment file
cp .env.example .env

# Edit credentials
nano .env

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

## Services Overview

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | MedAI Compass API |
| `postgres` | 5432 | PostgreSQL database |
| `redis` | 6379 | Redis cache |
| `minio` | 9000/9001 | Object storage (DICOM) |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Dashboards |
| `vllm` | 8080 | vLLM inference (GPU) |
| `triton` | 8001-8003 | Triton inference (GPU) |

## Configuration

### Environment Variables

```bash
# Core
ENVIRONMENT=production
LOG_LEVEL=INFO
API_PORT=8000

# Database
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_USER=medai
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=medai_compass

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Storage
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=your_minio_secret

# HuggingFace
HF_TOKEN=hf_your_token

# Security
JWT_SECRET=your_jwt_secret
PHI_ENCRYPTION_KEY=your_fernet_key
```

## Profiles

### CPU Only (Default)

```bash
docker-compose up -d
```

### With GPU (vLLM/Triton)

```bash
docker-compose --profile gpu up -d
```

## Health Checks

```bash
# API health
curl http://localhost:8000/health

# All services
docker-compose ps

# Logs
docker-compose logs -f api
docker-compose logs -f postgres
```

## Scaling

### Horizontal Scaling

```bash
# Scale API replicas
docker-compose up -d --scale api=3
```

### Resource Limits

Edit `docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
```

## Volumes

| Volume | Purpose |
|--------|---------|
| `postgres-data` | Database persistence |
| `redis-data` | Cache persistence |
| `minio-data` | Object storage |
| `model-cache` | HuggingFace models |
| `prometheus-data` | Metrics history |
| `grafana-data` | Dashboards |

### Backup

```bash
# Database backup
docker-compose exec postgres pg_dump -U medai medai_compass > backup.sql

# Restore
docker-compose exec -T postgres psql -U medai medai_compass < backup.sql
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs api

# Rebuild
docker-compose build --no-cache api
docker-compose up -d
```

### Database Connection Issues

```bash
# Check postgres is healthy
docker-compose exec postgres pg_isready

# Reset database
docker-compose down -v
docker-compose up -d
```

### Out of Memory

```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory

# Or reduce service memory
docker-compose down
# Edit docker-compose.yml memory limits
docker-compose up -d
```
