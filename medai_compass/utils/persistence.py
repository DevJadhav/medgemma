"""Conversation Persistence Layer for Multi-Instance Support.

Provides persistent storage for conversation state using:
- Redis for fast session cache and pub/sub
- PostgreSQL for durable conversation history

This enables horizontal scaling with multiple API instances.
"""

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ConversationStore:
    """
    Persistent conversation storage with Redis cache and PostgreSQL backend.
    
    Supports:
    - Fast session lookup via Redis
    - Durable storage in PostgreSQL
    - Automatic cache invalidation
    - Multi-instance synchronization
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        postgres_url: Optional[str] = None,
        session_ttl_hours: int = 24
    ):
        """
        Initialize conversation store.
        
        Args:
            redis_url: Redis connection URL
            postgres_url: PostgreSQL connection URL
            session_ttl_hours: Session TTL in hours
        """
        self.redis_url = redis_url or self._build_redis_url()
        self.postgres_url = postgres_url or self._build_postgres_url()
        self.session_ttl = timedelta(hours=session_ttl_hours)
        
        self._redis_client = None
        self._pg_pool = None
    
    def _build_redis_url(self) -> Optional[str]:
        """Build Redis URL from environment variables."""
        host = os.environ.get("REDIS_HOST", "localhost")
        port = os.environ.get("REDIS_PORT", "6379")
        password = os.environ.get("REDIS_PASSWORD")
        
        if password:
            return f"redis://:{password}@{host}:{port}/0"
        return f"redis://{host}:{port}/0"
    
    def _build_postgres_url(self) -> Optional[str]:
        """Build PostgreSQL URL from environment variables."""
        host = os.environ.get("POSTGRES_HOST")
        password = os.environ.get("POSTGRES_PASSWORD")
        
        if not host or not password:
            return None
        
        user = os.environ.get("POSTGRES_USER", "medai")
        db = os.environ.get("POSTGRES_DB", "medai_compass")
        port = os.environ.get("POSTGRES_PORT", "5432")
        
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    
    async def _get_redis(self):
        """Get Redis client, creating if needed."""
        if self._redis_client is None:
            try:
                import redis.asyncio as redis
                self._redis_client = redis.from_url(
                    self.redis_url,
                    decode_responses=True
                )
                await self._redis_client.ping()
                logger.info("Connected to Redis for conversation cache")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self._redis_client = None
        return self._redis_client
    
    async def _get_pg_pool(self):
        """Get PostgreSQL connection pool, creating if needed."""
        if self._pg_pool is None and self.postgres_url:
            try:
                import asyncpg
                self._pg_pool = await asyncpg.create_pool(
                    self.postgres_url,
                    min_size=2,
                    max_size=10
                )
                logger.info("Connected to PostgreSQL for conversation persistence")
            except Exception as e:
                logger.warning(f"PostgreSQL not available: {e}")
                self._pg_pool = None
        return self._pg_pool
    
    async def save_conversation(
        self,
        session_id: str,
        patient_id: str,
        state: dict,
        messages: list[dict],
        context: Optional[dict] = None
    ) -> bool:
        """
        Save conversation state.
        
        Args:
            session_id: Session identifier
            patient_id: Patient identifier
            state: Current conversation state
            messages: List of messages
            context: Additional context
            
        Returns:
            True if saved successfully
        """
        data = {
            "session_id": session_id,
            "patient_id": patient_id,
            "state": state,
            "messages": messages,
            "context": context or {},
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Save to Redis cache
        redis = await self._get_redis()
        if redis:
            try:
                cache_key = f"conversation:{session_id}"
                await redis.setex(
                    cache_key,
                    int(self.session_ttl.total_seconds()),
                    json.dumps(data)
                )
            except Exception as e:
                logger.warning(f"Failed to cache conversation: {e}")
        
        # Save to PostgreSQL
        pool = await self._get_pg_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO conversation_state 
                            (session_id, patient_id, state, messages, context, updated_at, expires_at)
                        VALUES ($1, $2, $3, $4, $5, NOW(), NOW() + $6::interval)
                        ON CONFLICT (session_id) DO UPDATE SET
                            state = EXCLUDED.state,
                            messages = EXCLUDED.messages,
                            context = EXCLUDED.context,
                            updated_at = NOW(),
                            expires_at = NOW() + $6::interval
                    """,
                        session_id,
                        patient_id,
                        json.dumps(state),
                        json.dumps(messages),
                        json.dumps(context or {}),
                        f"{self.session_ttl.total_seconds()} seconds"
                    )
                return True
            except Exception as e:
                logger.error(f"Failed to save conversation to PostgreSQL: {e}")
                return False
        
        return redis is not None
    
    async def get_conversation(
        self,
        session_id: str
    ) -> Optional[dict]:
        """
        Get conversation state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Conversation data or None
        """
        # Try Redis cache first
        redis = await self._get_redis()
        if redis:
            try:
                cache_key = f"conversation:{session_id}"
                cached = await redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis cache miss: {e}")
        
        # Fall back to PostgreSQL
        pool = await self._get_pg_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT session_id, patient_id, state, messages, context, updated_at
                        FROM conversation_state
                        WHERE session_id = $1 
                        AND (expires_at IS NULL OR expires_at > NOW())
                    """, session_id)
                    
                    if row:
                        data = {
                            "session_id": row["session_id"],
                            "patient_id": row["patient_id"],
                            "state": json.loads(row["state"]) if row["state"] else {},
                            "messages": json.loads(row["messages"]) if row["messages"] else [],
                            "context": json.loads(row["context"]) if row["context"] else {},
                            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None
                        }
                        
                        # Populate Redis cache
                        if redis:
                            try:
                                cache_key = f"conversation:{session_id}"
                                await redis.setex(
                                    cache_key,
                                    int(self.session_ttl.total_seconds()),
                                    json.dumps(data)
                                )
                            except Exception:
                                pass
                        
                        return data
            except Exception as e:
                logger.error(f"Failed to get conversation from PostgreSQL: {e}")
        
        return None
    
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        agent_name: Optional[str] = None,
        triage_result: Optional[dict] = None,
        requires_review: bool = False
    ) -> bool:
        """
        Add a message to conversation.
        
        Args:
            session_id: Session identifier
            role: Message role (patient, agent, clinician)
            content: Message content
            agent_name: Name of responding agent
            triage_result: Optional triage assessment
            requires_review: Whether message needs clinician review
            
        Returns:
            True if added successfully
        """
        message = {
            "role": role,
            "content": content,
            "agent_name": agent_name,
            "triage_result": triage_result,
            "requires_review": requires_review,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Get existing conversation
        conversation = await self.get_conversation(session_id)
        if conversation is None:
            logger.warning(f"Conversation not found: {session_id}")
            return False
        
        # Add message
        messages = conversation.get("messages", [])
        messages.append(message)
        
        # Save updated conversation
        return await self.save_conversation(
            session_id=session_id,
            patient_id=conversation["patient_id"],
            state=conversation.get("state", {}),
            messages=messages,
            context=conversation.get("context")
        )
    
    async def delete_conversation(self, session_id: str) -> bool:
        """
        Delete conversation.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted
        """
        # Delete from Redis
        redis = await self._get_redis()
        if redis:
            try:
                cache_key = f"conversation:{session_id}"
                await redis.delete(cache_key)
            except Exception as e:
                logger.warning(f"Failed to delete from Redis: {e}")
        
        # Delete from PostgreSQL
        pool = await self._get_pg_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "DELETE FROM conversation_state WHERE session_id = $1",
                        session_id
                    )
                return True
            except Exception as e:
                logger.error(f"Failed to delete conversation: {e}")
                return False
        
        return redis is not None
    
    async def get_patient_conversations(
        self,
        patient_id: str,
        limit: int = 10
    ) -> list[dict]:
        """
        Get recent conversations for a patient.
        
        Args:
            patient_id: Patient identifier
            limit: Maximum conversations to return
            
        Returns:
            List of conversation summaries
        """
        pool = await self._get_pg_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT session_id, state, updated_at
                        FROM conversation_state
                        WHERE patient_id = $1
                        ORDER BY updated_at DESC
                        LIMIT $2
                    """, patient_id, limit)
                    
                    return [
                        {
                            "session_id": row["session_id"],
                            "state": json.loads(row["state"]) if row["state"] else {},
                            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None
                        }
                        for row in rows
                    ]
            except Exception as e:
                logger.error(f"Failed to get patient conversations: {e}")
        
        return []
    
    async def close(self):
        """Close all connections."""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
        
        if self._pg_pool:
            await self._pg_pool.close()
            self._pg_pool = None


# Global singleton instance
_conversation_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Get or create global conversation store instance."""
    global _conversation_store
    
    if _conversation_store is None:
        _conversation_store = ConversationStore()
    
    return _conversation_store
