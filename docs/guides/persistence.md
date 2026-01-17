# Persistence & Multi-Instance Deployment

This guide covers conversation persistence for multi-instance deployments.

## Overview

MedAI Compass supports persistent conversation storage for:
- Multi-instance deployments (load-balanced)
- Session continuity across server restarts
- Patient conversation history retrieval
- Audit trail compliance

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                             │
└─────────────────────────────────────────────────────────────┘
            │                │                │
    ┌───────┴───────┐ ┌──────┴──────┐ ┌──────┴──────┐
    │   Instance 1  │ │  Instance 2 │ │  Instance 3 │
    └───────────────┘ └─────────────┘ └─────────────┘
            │                │                │
            └────────────────┼────────────────┘
                             │
    ┌────────────────────────┴────────────────────────┐
    │                                                  │
    ▼                                                  ▼
┌─────────┐                                    ┌────────────┐
│  Redis  │  <- Cache (fast reads)             │ PostgreSQL │ <- Durable storage
│ Cluster │     TTL: 24h                       │  (Primary) │    Retention: 6 years
└─────────┘                                    └────────────┘
```

## Conversation Store

### Configuration

```python
from medai_compass.utils.persistence import ConversationStore

# Full configuration
store = ConversationStore(
    redis_url="redis://localhost:6379",
    postgres_dsn="postgresql://user:pass@localhost/medai",
    cache_ttl=86400,  # 24 hours
    enable_encryption=True
)

# Initialize connections
await store.initialize()
```

### Environment Variables

```bash
# Redis (cache layer)
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your_redis_password

# PostgreSQL (persistent storage)
DATABASE_URL=postgresql://user:password@localhost:5432/medai
POSTGRES_PASSWORD=your_postgres_password

# Encryption
PHI_ENCRYPTION_KEY=your_fernet_key  # For encrypting PHI at rest
```

### Saving Messages

```python
from medai_compass.utils.persistence import ConversationMessage
from datetime import datetime, timezone

# Create a message
message = ConversationMessage(
    session_id="session-abc123",
    role="user",  # or "assistant"
    content="What does this X-ray show?",
    timestamp=datetime.now(timezone.utc),
    metadata={
        "patient_id": "P001",
        "encounter_id": "ENC001"
    }
)

# Save to store
await store.save_message(message)
```

### Retrieving History

```python
# Get recent conversation history
history = await store.get_conversation_history(
    session_id="session-abc123",
    limit=20  # Last 20 messages
)

for msg in history:
    print(f"[{msg.role}]: {msg.content}")

# Get all conversations for a patient
patient_sessions = await store.get_patient_conversations(
    patient_id="P001"
)

for session in patient_sessions:
    print(f"Session: {session['session_id']}, Started: {session['created_at']}")
```

### Deleting Conversations

```python
# Delete a specific conversation (HIPAA right to delete)
await store.delete_conversation(session_id="session-abc123")
```

## LangGraph Checkpointing

For diagnostic workflows, checkpoints enable:
- Resuming interrupted analyses
- State persistence across instances
- Debugging workflow execution

### Configuration

```python
from medai_compass.agents.diagnostic.graph import create_diagnostic_graph

# Create graph with PostgreSQL checkpointing
graph = create_diagnostic_graph(
    postgres_dsn="postgresql://user:pass@localhost/medai"
)

# Run with thread ID (for checkpointing)
result = await graph.ainvoke(
    {"input": analysis_request},
    config={"configurable": {"thread_id": "analysis-001"}}
)
```

### Resuming Workflows

```python
from medai_compass.agents.diagnostic.graph import resume_diagnostic_workflow

# Resume from last checkpoint
result = await resume_diagnostic_workflow(
    thread_id="analysis-001",
    postgres_dsn="postgresql://user:pass@localhost/medai"
)

print(f"Resumed from state: {result['state']}")
```

### Database Schema

The required tables are created automatically, or via init script:

```sql
-- LangGraph checkpoints (in docker/postgres/init.sql)
CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Conversation state
CREATE TABLE IF NOT EXISTS conversation_state (
    session_id TEXT PRIMARY KEY,
    patient_id TEXT,
    messages JSONB NOT NULL DEFAULT '[]',
    context JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX idx_conversation_patient ON conversation_state(patient_id);
```

## Redis Caching

Redis provides fast reads with automatic expiration:

### Cache Strategy

| Data | TTL | Purpose |
|------|-----|---------|
| Recent messages | 24 hours | Fast conversation retrieval |
| Session state | 1 hour | Active session context |
| Model outputs | 5 minutes | Repeated query optimization |

### Cache Keys

```
conversation:{session_id}      # Message history
session:{session_id}:state     # Current session state
patient:{patient_id}:sessions  # Patient's session list
```

### Manual Cache Operations

```python
# Clear cache for a session
await store.clear_cache(session_id="session-abc123")

# Force refresh from PostgreSQL
history = await store.get_conversation_history(
    session_id="session-abc123",
    skip_cache=True
)
```

## Communication Agent Integration

The ConversationManager automatically uses persistence:

```python
from medai_compass.agents.communication import ConversationManager, PatientMessage

manager = ConversationManager()
await manager.initialize()

# Process message (automatically persisted)
response = await manager.process_message(PatientMessage(
    message_id="msg-001",
    patient_id="P001",
    content="What are my test results?"
))

# History is automatically loaded/saved
print(f"Conversation length: {len(manager.conversation_history)}")
```

### Session Management

```python
# Start new session
session_id = await manager.start_session(patient_id="P001")

# Load existing session
await manager.load_session(session_id="session-abc123")

# End session (preserves history)
await manager.end_session()
```

## Fallback Behavior

The persistence layer gracefully degrades:

```
PostgreSQL + Redis → Full persistence with caching
PostgreSQL only   → Persistence without cache
Redis only        → Cache with no durability
Neither           → In-memory (single instance only)
```

```python
# Check backend status
status = await store.get_status()
print(f"Redis: {status['redis']}")      # 'connected' or 'unavailable'
print(f"PostgreSQL: {status['postgres']}")  # 'connected' or 'unavailable'
print(f"Mode: {status['mode']}")        # 'full', 'postgres', 'redis', 'memory'
```

## HIPAA Compliance

### Encryption at Rest

```python
# Enable PHI encryption (requires PHI_ENCRYPTION_KEY)
store = ConversationStore(enable_encryption=True)

# Messages are encrypted before storage
# Automatic decryption on retrieval
```

### Audit Trail

```python
# All operations are logged
# Logs include:
# - Operation type (save, retrieve, delete)
# - Session/patient IDs (hashed)
# - Timestamp
# - User/system identifier
# - Success/failure status
```

### Data Retention

```python
# Configure retention period
store = ConversationStore(
    retention_days=365 * 6  # 6 years for HIPAA
)

# Automatic cleanup job (run via cron)
await store.cleanup_expired()
```

## Multi-Instance Deployment

### Docker Compose

```yaml
services:
  api-1:
    image: medai-compass:latest
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@db:5432/medai
    
  api-2:
    image: medai-compass:latest
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@db:5432/medai

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    
  db:
    image: postgres:15-alpine
    volumes:
      - ./docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medai-compass
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: medai-secrets
              key: redis-url
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: medai-secrets
              key: database-url
```

## Monitoring

### Prometheus Metrics

```python
# Exposed metrics
medai_persistence_operations_total{operation="save|get|delete", backend="redis|postgres"}
medai_persistence_latency_seconds{operation="save|get|delete"}
medai_persistence_cache_hits_total
medai_persistence_cache_misses_total
medai_conversations_active
```

### Health Checks

```bash
# Check persistence health
curl http://localhost:8000/health | jq .persistence

# Response:
{
  "persistence": {
    "redis": "healthy",
    "postgres": "healthy",
    "mode": "full"
  }
}
```

## Troubleshooting

### "Redis connection refused"
- Check Redis is running: `docker ps | grep redis`
- Verify REDIS_URL format
- Check firewall/network settings

### "PostgreSQL connection failed"
- Verify DATABASE_URL format
- Check PostgreSQL logs: `docker logs postgres`
- Ensure init.sql ran successfully

### "Session not found"
- Check session ID spelling
- Verify TTL hasn't expired
- Check if running in memory-only mode

### "Slow conversation loading"
- Enable Redis caching
- Add indexes to PostgreSQL
- Consider pagination for long histories
