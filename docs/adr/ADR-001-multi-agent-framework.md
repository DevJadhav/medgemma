# ADR-001: Multi-Agent Framework Selection

## Status
Accepted

## Date
2025-11-15

## Context

MedAI Compass requires a multi-agent system to handle three distinct domains:
1. **Diagnostic**: Medical image analysis with complex, multi-step workflows
2. **Workflow**: Clinical operations like scheduling, documentation, prior authorization
3. **Communication**: Patient engagement with triage, education, and follow-up

We needed to select agent frameworks that best fit each domain's requirements while enabling integration through a master orchestrator.

### Requirements

| Requirement | Diagnostic | Workflow | Communication |
|-------------|-----------|----------|---------------|
| State Management | Critical | Medium | Low |
| Checkpointing | Required | Nice-to-have | Not needed |
| Role Delegation | N/A | Critical | Moderate |
| Conversational | No | No | Yes |
| Human-in-Loop | Required | Optional | Required |

## Decision

We will use three different agent frameworks, each optimized for its domain:

### LangGraph for Diagnostic Agent
- **Rationale**: Medical image analysis requires explicit state machines with checkpointing
- **Key Features Used**:
  - StateGraph for workflow definition
  - Conditional edges for confidence-based routing
  - PostgreSQL checkpointer for state persistence
  - Async execution support

### CrewAI for Workflow Agent
- **Rationale**: Clinical workflows naturally map to roles with task delegation
- **Key Features Used**:
  - Role-based agents (scheduler, documenter, prior auth)
  - Task assignment and execution
  - Crew coordination patterns
  - Goal and backstory definitions

### AutoGen for Communication Agent
- **Rationale**: Patient communication requires conversational agents with oversight
- **Key Features Used**:
  - Group chat for multi-agent coordination
  - Oversight proxy for human-in-loop
  - Async messaging patterns
  - Reply functions for custom logic

## Consequences

### Positive
- Each agent domain uses a framework optimized for its requirements
- Clear separation of concerns between domains
- Independent scaling and deployment of each agent
- Flexibility to swap frameworks if better options emerge

### Negative
- Three different frameworks to maintain and upgrade
- Learning curve for developers across all three
- Integration complexity at the orchestrator level
- Potential version conflicts between frameworks

### Mitigation
- Master orchestrator abstracts framework differences
- Unified interface for all agents (process_request → response)
- Comprehensive documentation for each framework
- Integration tests validate cross-framework communication

## Alternatives Considered

### 1. Single Framework (LangChain only)
- **Rejected**: LangChain's agent types don't provide optimal patterns for all three domains
- Role-based agents less natural than CrewAI
- Conversational agents less mature than AutoGen

### 2. Custom Agent Framework
- **Rejected**: High development cost and maintenance burden
- Existing frameworks have active communities and updates
- Would delay project delivery significantly

### 3. LangGraph for All Domains
- **Rejected**: Overkill for simpler workflow and communication tasks
- CrewAI and AutoGen provide more natural abstractions
- Would increase boilerplate code

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
