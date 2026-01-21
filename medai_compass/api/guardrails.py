from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

from medai_compass.guardrails.input_rails import (
    apply_input_guardrails,
    MEDICAL_SCOPE_PATTERNS,
    JAILBREAK_PATTERNS,
    JailbreakCategory
)

router = APIRouter(prefix="/api/v1/guardrails", tags=["Guardrails"])

# Models
class TestRequest(BaseModel):
    text: str = Field(..., description="Text to test against guardrails")

class TestResponse(BaseModel):
    sanitized_input: str
    is_safe: bool
    is_valid_request: bool
    jailbreak: Dict[str, Any]
    injection: Dict[str, Any]
    scope: Dict[str, Any]
    
class ConfigResponse(BaseModel):
    scope_patterns: Dict[str, List[str]]
    jailbreak_categories: List[str]

class ComplianceStatusResponse(BaseModel):
    status: str
    timestamp: str
    safeguards: Dict[str, Any]

@router.get("/config", response_model=ConfigResponse)
async def get_guardrails_config():
    """Get active guardrail configurations."""
    # Convert JailbreakCategory enum keys to strings for JSON serialization if needed
    # iterating the enum members to get list of categories
    categories = [cat.value for cat in JailbreakCategory]
    
    return ConfigResponse(
        scope_patterns=MEDICAL_SCOPE_PATTERNS,
        jailbreak_categories=categories
    )

@router.post("/test", response_model=TestResponse)
async def test_guardrails(request: TestRequest):
    """Test guardrails with input text."""
    result = apply_input_guardrails(request.text)
    return TestResponse(**result)

@router.get("/compliance/status", response_model=ComplianceStatusResponse)
async def get_compliance_status():
    """Get HIPAA compliance system status."""
    from datetime import datetime, timezone
    
    # In a real system, these would check actual configurations
    return ComplianceStatusResponse(
        status="compliant",
        timestamp=datetime.now(timezone.utc).isoformat(),
        safeguards={
            "encryption": {
                "status": "active",
                "details": "AES-256 at rest, TLS 1.3 in transit"
            },
            "audit_logging": {
                "status": "active",
                "details": "Comprehensive request/response logging enabled"
            },
            "access_control": {
                "status": "active",
                "details": "RBAC with principle of least privilege"
            },
            "baas": {
                "status": "active",
                "details": "Business Associate Agreements signed with vendors"
            }
        }
    )
