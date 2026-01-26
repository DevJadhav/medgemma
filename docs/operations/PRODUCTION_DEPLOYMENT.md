# Production Deployment Guide

This guide covers end-to-end production deployment of MedAI Compass, from infrastructure setup to monitoring and maintenance.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Configuration](#configuration)
4. [Deployment Options](#deployment-options)
5. [Health Checks & Monitoring](#health-checks--monitoring)
6. [Scaling](#scaling)
7. [Backup & Recovery](#backup--recovery)
8. [Maintenance](#maintenance)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended | Production |
|-----------|---------|-------------|------------|
| CPU | 8 cores | 16 cores | 32+ cores |
| RAM | 32 GB | 64 GB | 128+ GB |
| Storage | 100 GB SSD | 500 GB NVMe | 1+ TB NVMe |
| GPU | None (Modal) | A100 40GB | H100 80GB × 4 |
| Network | 1 Gbps | 10 Gbps | 25+ Gbps |

### Software Requirements

```bash
# Required
Python >= 3.10
Docker >= 24.0
Docker Compose >= 2.20
Git >= 2.40

# Optional (for local GPU)
CUDA >= 12.0
cuDNN >= 8.9
NVIDIA Driver >= 535

# Optional (for Apple Silicon)
macOS >= 13.0
Xcode Command Line Tools
```

### Accounts & Tokens

| Service | Required | Purpose |
|---------|----------|---------|
| HuggingFace | Yes | MedGemma model access |
| Modal | Optional | Cloud H100 GPU |
| Splunk/ELK | Optional | Log aggregation |
| AWS/GCP/Azure | Optional | Cloud deployment |

---

## Infrastructure Setup

### Step 1: Clone and Install

```bash
# Clone repository
git clone https://github.com/DevJadhav/medgemma.git
cd medgemma

# Install uv (if not installed)
pip install uv

# Install dependencies
uv sync

# Verify installation
uv run python -c "import medai_compass; print('OK')"
```

### Step 2: Generate Secrets

```bash
# Generate encryption keys and secrets
python scripts/generate_secrets.py

# This creates/updates .env with:
# - PHI_ENCRYPTION_KEY (Fernet key for PHI encryption)
# - JWT_SECRET (Random 64-char hex string)
# - POSTGRES_PASSWORD (Random 32-char string)
# - REDIS_PASSWORD (Random 32-char string)
```

### Step 3: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your values
nano .env
```

**Required Environment Variables:**

```bash
# Model Access (REQUIRED)
HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Database (auto-generated, but verify)
POSTGRES_PASSWORD=your_secure_password_here
REDIS_PASSWORD=your_redis_password_here

# Security (auto-generated)
PHI_ENCRYPTION_KEY=your_fernet_key_here
JWT_SECRET=your_jwt_secret_here

# Model Configuration
MEDGEMMA_MODEL_NAME=medgemma-27b  # or medgemma-4b

# Environment
MEDAI_ENVIRONMENT=production  # development, staging, production
```

**Optional Variables:**

```bash
# Modal Cloud GPU (if using)
MODAL_TOKEN_ID=ak-xxxxxxxxxx
MODAL_TOKEN_SECRET=as-xxxxxxxxxx

# SIEM Integration
SPLUNK_HEC_URL=https://your-splunk:8088/services/collector
SPLUNK_HEC_TOKEN=your-hec-token
ELASTICSEARCH_HOSTS=http://elasticsearch:9200

# Monitoring
GRAFANA_ADMIN_PASSWORD=your_grafana_password
```

### Step 4: Modal Setup (Cloud GPU)

If using Modal for H100 GPU access:

```bash
# Install Modal CLI
uv pip install modal

# Authenticate
uv run modal token new

# Create secrets
uv run modal secret create huggingface-secret \
    HF_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Create volumes for model caching
uv run modal volume create medgemma-model-cache
uv run modal volume create medgemma-checkpoints

# Deploy inference service
uv run modal deploy medai_compass/modal/app.py

# Verify deployment
uv run modal app list
```

### Step 5: Database Initialization

```bash
# Start database services only
docker compose up -d postgres redis

# Wait for PostgreSQL to be ready
docker compose exec postgres pg_isready -U medai

# Run database migrations (if any)
# docker compose exec api alembic upgrade head

# Verify database
docker compose exec postgres psql -U medai -d medai_compass -c "SELECT 1;"
```

---

## Configuration

### Hydra Configuration

For training and inference, MedAI Compass uses Hydra configuration:

```bash
# View default configuration
cat config/hydra/config.yaml

# Override via command line
python -m medai_compass.train model=medgemma_27b training=qlora

# Use pre-defined experiment profile
python -m medai_compass.train +experiment=production
```

### Application Configuration

Main configuration file: `config/medai_config.yaml`

```yaml
# Environment-specific settings
environment: production

# Confidence thresholds
confidence:
  high_confidence: 0.90
  medium_confidence: 0.80
  low_confidence: 0.70

# Escalation settings
escalation:
  auto_escalate_phi_breach: true
  critical_findings:
    - pneumothorax
    - stroke
    - myocardial_infarction
    - aortic_dissection

# Inference settings
inference:
  timeout_seconds: 90
  max_tokens: 4096
  retry_attempts: 3

# Security settings
security:
  encryption_algorithm: AES-256-GCM
  key_rotation_days: 30
  session_timeout_minutes: 60

# HIPAA compliance
retention:
  audit_log_years: 6
  phi_data_years: 6
```

---

## Deployment Options

### Option 1: Docker Compose (Recommended for Most Deployments)

```bash
# Start all services
docker compose up -d

# Verify all services are running
docker compose ps

# Expected output:
# NAME                    STATUS              PORTS
# medai-api-1            running             0.0.0.0:8000->8000/tcp
# medai-frontend-1       running             0.0.0.0:3001->3000/tcp
# medai-postgres-1       running             0.0.0.0:5432->5432/tcp
# medai-redis-1          running             0.0.0.0:6379->6379/tcp
# medai-prometheus-1     running             0.0.0.0:9090->9090/tcp
# medai-grafana-1        running             0.0.0.0:3000->3000/tcp
# medai-minio-1          running             0.0.0.0:9001->9001/tcp
```

### Option 2: Docker Compose with GPU

```bash
# Requires NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Start with GPU profile
docker compose --profile gpu up -d

# Verify GPU is available
docker compose exec api python -c "import torch; print(torch.cuda.is_available())"
```

### Option 3: Docker Compose with Modal (Cloud GPU)

```bash
# Start with Modal profile (no local GPU needed)
docker compose --profile modal up -d

# Verify Modal connection
curl http://localhost:8000/api/v1/inference/status
```

### Option 4: Docker Compose with Logging (ELK Stack)

```bash
# Start with logging profile
docker compose --profile logging up -d

# Access Kibana
open http://localhost:5601
```

### Option 5: Kubernetes (Production Scale)

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n medai-compass
kubectl get svc -n medai-compass
```

---

## Health Checks & Monitoring

### Health Check Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `GET /health` | Overall health | `{"status": "healthy", ...}` |
| `GET /health/ready` | Kubernetes readiness | `{"ready": true}` |
| `GET /health/live` | Kubernetes liveness | `{"live": true}` |
| `GET /metrics` | Prometheus metrics | Prometheus text format |

```bash
# Check health
curl http://localhost:8000/health | jq

# Check readiness
curl http://localhost:8000/health/ready

# Check liveness
curl http://localhost:8000/health/live

# Get metrics
curl http://localhost:8000/metrics
```

### Prometheus Metrics

Key metrics to monitor:

```promql
# Request rate
rate(medai_request_total[5m])

# Error rate
rate(medai_request_total{status="error"}[5m]) / rate(medai_request_total[5m])

# P99 latency
histogram_quantile(0.99, rate(medai_request_duration_seconds_bucket[5m]))

# Escalation rate
rate(medai_escalation_total[1h])

# PHI detection rate
rate(medai_phi_detection_total[1h])

# GPU utilization (if local GPU)
nvidia_gpu_utilization_percent
```

### Grafana Dashboards

Access Grafana at http://localhost:3000 (default: admin/admin)

Pre-configured dashboards:
1. **MedAI Overview**: Request rates, latency, errors
2. **Agent Performance**: Per-agent metrics
3. **Escalations**: Escalation tracking
4. **Infrastructure**: CPU, memory, disk, network

### Alerting Rules

Configure in `docker/prometheus/alerts.yaml`:

```yaml
groups:
  - name: medai_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(medai_request_total{status="error"}[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"

      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(medai_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency exceeds 2 seconds"

      - alert: EscalationBacklog
        expr: medai_pending_escalations > 10
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Too many pending escalations"
```

---

## Scaling

### Horizontal Scaling (API Servers)

```bash
# Scale API replicas
docker compose up -d --scale api=3

# Or with Docker Swarm
docker service scale medai_api=3

# Or with Kubernetes
kubectl scale deployment medai-api -n medai-compass --replicas=3
```

### Vertical Scaling (GPU Workers)

```bash
# Add more GPU workers (Kubernetes)
kubectl apply -f k8s/gpu-worker-deployment.yaml

# Or scale Modal deployment
# Edit medai_compass/modal/app.py:
#   @stub.function(gpu="H100", concurrency_limit=10)
```

### Load Balancing

```nginx
# nginx.conf
upstream medai_api {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://medai_api;
        proxy_set_header X-Request-ID $request_id;
    }
}
```

### Auto-Scaling Configuration

```yaml
# Ray Serve autoscaling
autoscaling_config:
  min_replicas: 1
  max_replicas: 10
  target_num_ongoing_requests_per_replica: 5
  upscale_delay_s: 30
  downscale_delay_s: 300
```

---

## Backup & Recovery

### Database Backup

```bash
# PostgreSQL backup
docker compose exec postgres pg_dump -U medai medai_compass > backup_$(date +%Y%m%d).sql

# Automated daily backup (cron)
0 2 * * * docker compose -f /path/to/docker-compose.yml exec -T postgres pg_dump -U medai medai_compass | gzip > /backups/medai_$(date +\%Y\%m\%d).sql.gz

# Restore from backup
docker compose exec -T postgres psql -U medai medai_compass < backup_20260126.sql
```

### MinIO Backup

```bash
# Backup MinIO data
docker compose exec minio mc mirror /data /backup

# Or use mc CLI
mc mirror minio/medai-data s3/backup-bucket/medai-data
```

### Redis Backup

```bash
# Trigger RDB snapshot
docker compose exec redis redis-cli BGSAVE

# Copy snapshot
docker cp medai-redis-1:/data/dump.rdb ./backups/redis_$(date +%Y%m%d).rdb
```

### Disaster Recovery

1. **RTO (Recovery Time Objective)**: 1 hour
2. **RPO (Recovery Point Objective)**: 1 hour

Recovery procedure:
```bash
# 1. Restore PostgreSQL
docker compose up -d postgres
cat backup.sql | docker compose exec -T postgres psql -U medai medai_compass

# 2. Restore Redis (optional, can rebuild)
docker compose up -d redis
docker cp redis_backup.rdb medai-redis-1:/data/dump.rdb
docker compose restart redis

# 3. Restore MinIO
docker compose up -d minio
mc mirror s3/backup-bucket/medai-data minio/medai-data

# 4. Start application
docker compose up -d
```

---

## Maintenance

### Log Rotation

```bash
# Docker log rotation (daemon.json)
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "5"
  }
}
```

### Certificate Renewal

```bash
# Let's Encrypt renewal (if using)
certbot renew --quiet
docker compose restart nginx
```

### Dependency Updates

```bash
# Update Python dependencies
uv sync --upgrade

# Update Docker images
docker compose pull
docker compose up -d

# Test after updates
uv run pytest tests/ -v
```

### Database Maintenance

```bash
# Vacuum PostgreSQL
docker compose exec postgres vacuumdb -U medai -d medai_compass --analyze

# Reindex
docker compose exec postgres reindexdb -U medai -d medai_compass

# Check table sizes
docker compose exec postgres psql -U medai -d medai_compass -c "
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC;
"
```

### Audit Log Management

```bash
# Archive old audit logs (6+ years can be archived)
docker compose exec postgres psql -U medai -d medai_compass -c "
INSERT INTO audit_logs_archive
SELECT * FROM audit_logs
WHERE timestamp < NOW() - INTERVAL '6 years';

DELETE FROM audit_logs
WHERE timestamp < NOW() - INTERVAL '6 years';
"
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| API not starting | Missing env vars | Check `.env` file |
| Model loading fails | No HF token | Set `HUGGING_FACE_HUB_TOKEN` |
| GPU not detected | NVIDIA drivers | Install NVIDIA Container Toolkit |
| High latency | Insufficient GPU | Scale up or use Modal |
| Database errors | Connection pool | Increase `max_connections` |

### Debug Commands

```bash
# View logs
docker compose logs -f api

# Check API health
curl -v http://localhost:8000/health

# Test inference
curl -X POST http://localhost:8000/api/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test", "max_tokens": 10}'

# Check GPU status
docker compose exec api python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"

# Check PostgreSQL connections
docker compose exec postgres psql -U medai -c "SELECT count(*) FROM pg_stat_activity;"
```

---

## Production Checklist

Before going live, verify:

- [ ] All secrets are set and secure
- [ ] Database backups are configured
- [ ] Monitoring and alerting are active
- [ ] Health checks pass
- [ ] Load testing completed
- [ ] Security audit completed
- [ ] HIPAA compliance verified
- [ ] Disaster recovery tested
- [ ] Documentation updated
- [ ] Runbook created

---

## Support

For issues:
1. Check logs: `docker compose logs -f`
2. Check health: `curl http://localhost:8000/health`
3. Review documentation
4. Open GitHub issue: https://github.com/DevJadhav/medgemma/issues
