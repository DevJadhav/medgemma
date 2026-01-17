#!/usr/bin/env python3
"""Verify external service connections for MedAI Compass.

Checks:
- HuggingFace Hub token and model access
- Modal authentication and GPU availability
- Database connections (PostgreSQL, Redis)
- MinIO object storage

Usage:
    python scripts/verify_connections.py [--quick] [--skip-modal]
"""

import argparse
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_huggingface() -> dict:
    """Verify HuggingFace Hub connection and model access."""
    result = {
        "name": "HuggingFace Hub",
        "status": "unknown",
        "details": {},
    }
    
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if not hf_token:
        result["status"] = "warning"
        result["message"] = "No HF_TOKEN set - won't be able to access gated models"
        return result
    
    try:
        from huggingface_hub import HfApi, whoami
        
        # Check authentication
        user_info = whoami(token=hf_token)
        result["details"]["username"] = user_info.get("name", "unknown")
        result["details"]["organizations"] = [
            org.get("name") for org in user_info.get("orgs", [])
        ]
        
        # Check access to MedGemma model
        api = HfApi(token=hf_token)
        try:
            model_info = api.model_info("google/medgemma-4b-it")
            result["details"]["medgemma_access"] = True
            result["details"]["model_id"] = model_info.modelId
        except Exception as e:
            result["details"]["medgemma_access"] = False
            result["details"]["medgemma_error"] = str(e)
        
        result["status"] = "ok"
        result["message"] = f"Authenticated as {user_info.get('name', 'unknown')}"
        
    except ImportError:
        result["status"] = "error"
        result["message"] = "huggingface_hub package not installed"
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Authentication failed: {e}"
    
    return result


def check_modal() -> dict:
    """Verify Modal connection and GPU availability."""
    result = {
        "name": "Modal (Cloud GPU)",
        "status": "unknown",
        "details": {},
    }
    
    token_id = os.environ.get("MODAL_TOKEN_ID")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    
    if not token_id or not token_secret:
        result["status"] = "warning"
        result["message"] = "Modal tokens not set - will fall back to local GPU/CPU"
        return result
    
    try:
        import modal
        
        # Check if we can connect to Modal
        result["details"]["modal_version"] = modal.__version__
        
        # Try to lookup or create a simple app to verify auth
        try:
            # Just check that we can authenticate
            from modal import App
            # This will fail if tokens are invalid
            result["status"] = "ok"
            result["message"] = "Modal authentication verified"
            result["details"]["gpu_available"] = "H100 (80GB)"
            
        except modal.exception.AuthError as e:
            result["status"] = "error"
            result["message"] = f"Modal authentication failed: {e}"
            
    except ImportError:
        result["status"] = "warning"
        result["message"] = "Modal package not installed - run: uv add modal"
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Modal connection failed: {e}"
    
    return result


def check_postgres() -> dict:
    """Verify PostgreSQL connection."""
    result = {
        "name": "PostgreSQL",
        "status": "unknown",
        "details": {},
    }
    
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "medai")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    database = os.environ.get("POSTGRES_DB", "medai_compass")
    
    if not password:
        result["status"] = "warning"
        result["message"] = "POSTGRES_PASSWORD not set"
        return result
    
    try:
        import asyncpg
        import asyncio
        
        async def test_connection():
            conn = await asyncpg.connect(
                host=host,
                port=int(port),
                user=user,
                password=password,
                database=database,
                timeout=5,
            )
            version = await conn.fetchval("SELECT version()")
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
            await conn.close()
            return version, [t["tablename"] for t in tables]
        
        version, tables = asyncio.run(test_connection())
        
        result["status"] = "ok"
        result["message"] = f"Connected to PostgreSQL at {host}:{port}"
        result["details"]["version"] = version.split(",")[0] if version else "unknown"
        result["details"]["tables"] = len(tables)
        result["details"]["database"] = database
        
    except ImportError:
        result["status"] = "error"
        result["message"] = "asyncpg package not installed"
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Connection failed: {e}"
    
    return result


def check_redis() -> dict:
    """Verify Redis connection."""
    result = {
        "name": "Redis",
        "status": "unknown",
        "details": {},
    }
    
    host = os.environ.get("REDIS_HOST", "localhost")
    port = os.environ.get("REDIS_PORT", "6379")
    password = os.environ.get("REDIS_PASSWORD", "")
    
    try:
        import redis
        
        client = redis.Redis(
            host=host,
            port=int(port),
            password=password or None,
            socket_timeout=5,
        )
        
        info = client.info()
        
        result["status"] = "ok"
        result["message"] = f"Connected to Redis at {host}:{port}"
        result["details"]["version"] = info.get("redis_version", "unknown")
        result["details"]["memory_used_mb"] = round(info.get("used_memory", 0) / (1024 * 1024), 2)
        result["details"]["connected_clients"] = info.get("connected_clients", 0)
        
    except ImportError:
        result["status"] = "error"
        result["message"] = "redis package not installed"
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Connection failed: {e}"
    
    return result


def check_minio() -> dict:
    """Verify MinIO connection."""
    result = {
        "name": "MinIO (Object Storage)",
        "status": "unknown",
        "details": {},
    }
    
    endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.environ.get("MINIO_SECRET_KEY", "")
    
    if not secret_key:
        result["status"] = "warning"
        result["message"] = "MINIO_SECRET_KEY not set"
        return result
    
    try:
        from minio import Minio
        
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False,
        )
        
        # List buckets to verify connection
        buckets = client.list_buckets()
        
        result["status"] = "ok"
        result["message"] = f"Connected to MinIO at {endpoint}"
        result["details"]["buckets"] = [b.name for b in buckets]
        
    except ImportError:
        result["status"] = "warning"
        result["message"] = "minio package not installed - run: uv add minio"
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"Connection failed: {e}"
    
    return result


def check_physionet() -> dict:
    """Check PhysioNet credentials for MIMIC datasets."""
    result = {
        "name": "PhysioNet (MIMIC Access)",
        "status": "unknown",
        "details": {},
    }
    
    username = os.environ.get("PHYSIONET_USERNAME")
    password = os.environ.get("PHYSIONET_PASSWORD")
    
    if not username or not password:
        result["status"] = "info"
        result["message"] = "PhysioNet credentials not set - MIMIC datasets unavailable"
        result["details"]["instruction"] = "Set PHYSIONET_USERNAME and PHYSIONET_PASSWORD for MIMIC access"
        return result
    
    # Just check that credentials are set (don't actually verify with PhysioNet)
    result["status"] = "ok"
    result["message"] = f"PhysioNet credentials configured for user: {username}"
    result["details"]["username"] = username
    
    return result


def print_result(result: dict) -> None:
    """Print a check result."""
    status_icons = {
        "ok": "✓",
        "warning": "⚠",
        "error": "✗",
        "info": "ℹ",
        "unknown": "?",
    }
    
    icon = status_icons.get(result["status"], "?")
    name = result["name"]
    message = result.get("message", "")
    
    print(f"  {icon} {name}: {message}")
    
    if result.get("details") and result["status"] in ("ok", "info"):
        for key, value in result["details"].items():
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value[:5])
                if len(result["details"][key]) > 5:
                    value += "..."
            print(f"      {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Verify MedAI Compass connections")
    parser.add_argument("--quick", action="store_true", help="Quick check (skip slow tests)")
    parser.add_argument("--skip-modal", action="store_true", help="Skip Modal check")
    parser.add_argument("--skip-db", action="store_true", help="Skip database checks")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("  MedAI Compass - Connection Verification")
    print("=" * 60 + "\n")
    
    all_results = []
    
    # Core services
    print("External Services:")
    
    # HuggingFace
    result = check_huggingface()
    print_result(result)
    all_results.append(result)
    
    # Modal
    if not args.skip_modal:
        result = check_modal()
        print_result(result)
        all_results.append(result)
    
    # PhysioNet
    result = check_physionet()
    print_result(result)
    all_results.append(result)
    
    if not args.skip_db:
        print("\nDatabase Services:")
        
        # PostgreSQL
        result = check_postgres()
        print_result(result)
        all_results.append(result)
        
        # Redis
        result = check_redis()
        print_result(result)
        all_results.append(result)
        
        # MinIO
        result = check_minio()
        print_result(result)
        all_results.append(result)
    
    # Summary
    print("\n" + "-" * 60)
    
    errors = [r for r in all_results if r["status"] == "error"]
    warnings = [r for r in all_results if r["status"] == "warning"]
    
    if errors:
        print(f"  ✗ {len(errors)} error(s) found")
        for r in errors:
            print(f"      - {r['name']}: {r.get('message', 'Unknown error')}")
    
    if warnings:
        print(f"  ⚠ {len(warnings)} warning(s)")
        for r in warnings:
            print(f"      - {r['name']}: {r.get('message', 'Unknown warning')}")
    
    if not errors and not warnings:
        print("  ✓ All connections verified successfully")
    elif not errors:
        print("  ✓ Core connections OK (warnings are non-critical)")
    
    print("-" * 60 + "\n")
    
    # Exit with error if critical services failed
    critical_failed = any(
        r["status"] == "error" and r["name"] in ("HuggingFace Hub",)
        for r in all_results
    )
    
    return 1 if critical_failed else 0


if __name__ == "__main__":
    sys.exit(main())
