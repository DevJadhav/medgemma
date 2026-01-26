# ADR-002: State Persistence Strategy

## Status
Accepted

## Date
2025-11-20

## Context

MedAI Compass runs as a horizontally-scaled service with multiple API server instances. We need a state persistence strategy that:

1. Enables workflow resumption after failures
2. Supports multi-instance deployments
3. Provides audit trail for HIPAA compliance
4. Handles session management efficiently

### Key Challenges

- LangGraph workflows may be interrupted and need to resume
- Multiple API instances must share workflow state
- HIPAA requires 6-year audit log retention
- Low-latency session lookups are critical for UX

## Decision

We will use a **three-tier state persistence architecture**:

### Tier 1: Redis (Hot State)
- **Purpose**: Session storage, caching, rate limiting
- **TTL**: Minutes to hours
- **Data**: Session tokens, cached responses, rate limit counters

```python
# Session management
redis.setex(f"session:{session_id}", 3600, session_data)
```

### Tier 2: PostgreSQL (Warm State)
- **Purpose**: Workflow checkpoints, audit logs, user data
- **TTL**: 6 years (HIPAA) for audit logs, indefinite for checkpoints
- **Data**: LangGraph checkpoints, escalations, user profiles

```python
# LangGraph checkpoint
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(conn_string)
```

### Tier 3: MinIO (Cold State)
- **Purpose**: DICOM images, model artifacts, archived reports
- **TTL**: Configurable per object type
- **Data**: Binary files, large objects

```python
# DICOM storage
minio_client.put_object("dicom", object_name, file_data)
```

## Consequences

### Positive
- Appropriate storage tier for each data type
- PostgreSQL checkpointing enables multi-instance deployments
- Redis provides sub-millisecond session lookups
- MinIO handles large file storage efficiently
- Clear HIPAA compliance path with PostgreSQL audit logs

### Negative
- Three storage systems to operate and maintain
- Increased operational complexity
- Potential for inconsistency during failures
- Need for data synchronization logic

### Mitigation
- Use Docker Compose for local development with all services
- Implement health checks for all storage backends
- Design for eventual consistency where acceptable
- Use transactions for critical operations

## Implementation Details

### PostgreSQL Schema

```sql
-- LangGraph checkpoints
CREATE TABLE checkpoints (
    thread_id VARCHAR(255) PRIMARY KEY,
    checkpoint JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit logs (HIPAA compliant)
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL,
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    hash VARCHAR(64),  -- SHA-256 for tamper detection
    previous_hash VARCHAR(64),  -- Chain for integrity
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for efficient queries
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
```

### Redis Key Patterns

```
session:{session_id}          → Session data (TTL: 1 hour)
cache:response:{hash}         → Cached response (TTL: 5 min)
ratelimit:user:{user_id}      → Rate limit counter (TTL: 1 min)
ratelimit:ip:{ip_address}     → IP rate limit (TTL: 1 min)
```

## Alternatives Considered

### 1. Single PostgreSQL for Everything
- **Rejected**: PostgreSQL not optimal for high-throughput session lookups
- Redis provides 10-100x better latency for hot data

### 2. MongoDB for Checkpoints
- **Rejected**: LangGraph has native PostgreSQL support
- PostgreSQL provides better ACID guarantees for audit logs

### 3. AWS S3 Instead of MinIO
- **Rejected**: Want to support on-premises deployments
- MinIO is S3-compatible and can be replaced with S3 if needed

## References

- [LangGraph Persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [HIPAA Data Retention Requirements](https://www.hhs.gov/hipaa/)
- [Redis Data Structures](https://redis.io/docs/data-types/)
