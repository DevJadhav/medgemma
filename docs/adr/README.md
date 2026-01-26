# Architectural Decision Records (ADR)

This directory contains the Architectural Decision Records for MedAI Compass. ADRs document significant architectural decisions made during the development of the platform.

## Index

| ID | Title | Status | Date |
|----|-------|--------|------|
| [ADR-001](./ADR-001-multi-agent-framework.md) | Multi-Agent Framework Selection | Accepted | 2025-11 |
| [ADR-002](./ADR-002-state-persistence.md) | State Persistence Strategy | Accepted | 2025-11 |
| [ADR-003](./ADR-003-inference-backend.md) | GPU Inference Backend Selection | Accepted | 2025-12 |
| [ADR-004](./ADR-004-training-strategy.md) | Distributed Training Strategy | Accepted | 2025-12 |
| [ADR-005](./ADR-005-guardrails-architecture.md) | Guardrails Architecture | Accepted | 2025-12 |
| [ADR-006](./ADR-006-configuration-management.md) | Configuration Management with Hydra | Accepted | 2026-01 |

## ADR Template

```markdown
# ADR-XXX: Title

## Status
[Proposed | Accepted | Deprecated | Superseded by ADR-YYY]

## Context
[What is the issue that we're seeing that is motivating this decision?]

## Decision
[What is the change that we're proposing and/or doing?]

## Consequences
[What becomes easier or more difficult as a result of this change?]

## Alternatives Considered
[What other options were considered and why were they rejected?]
```

## How to Add a New ADR

1. Copy the template above
2. Create a new file: `ADR-XXX-short-title.md`
3. Fill in all sections
4. Update the index in this README
5. Submit for review
