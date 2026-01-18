"""
API contract validation for MedAI Compass.

Provides:
- OpenAPI schema validation
- Request/response contract validation
- Backward compatibility checks
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContractViolation:
    """API contract violation."""
    
    endpoint: str
    method: str
    violation_type: str  # schema, backward_compat, required_field
    description: str
    expected: Any
    actual: Any


@dataclass
class APIContract:
    """API contract definition."""
    
    endpoint: str
    method: str
    request_schema: Optional[Dict] = None
    response_schema: Optional[Dict] = None
    required_headers: List[str] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    
    @classmethod
    def from_openapi(cls, spec: Dict, path: str, method: str) -> "APIContract":
        """
        Create contract from OpenAPI spec.
        
        Args:
            spec: OpenAPI specification
            path: API path
            method: HTTP method
            
        Returns:
            APIContract
        """
        path_spec = spec.get("paths", {}).get(path, {})
        method_spec = path_spec.get(method.lower(), {})
        
        # Extract request schema
        request_body = method_spec.get("requestBody", {})
        request_schema = None
        if request_body:
            content = request_body.get("content", {})
            json_content = content.get("application/json", {})
            request_schema = json_content.get("schema")
        
        # Extract response schema
        responses = method_spec.get("responses", {})
        success_response = responses.get("200", {}) or responses.get("201", {})
        response_content = success_response.get("content", {})
        json_response = response_content.get("application/json", {})
        response_schema = json_response.get("schema")
        
        return cls(
            endpoint=path,
            method=method.upper(),
            request_schema=request_schema,
            response_schema=response_schema,
        )


@dataclass
class APIContractReport:
    """API contract validation report."""
    
    valid: bool
    violations: List[ContractViolation]
    contracts_checked: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "contracts_checked": self.contracts_checked,
            "violations": [
                {
                    "endpoint": v.endpoint,
                    "method": v.method,
                    "violation_type": v.violation_type,
                    "description": v.description,
                }
                for v in self.violations
            ],
            "timestamp": self.timestamp,
        }
    
    def get_summary(self) -> str:
        """Get summary."""
        lines = ["API Contract Validation Report", "=" * 50]
        lines.append(f"Contracts Checked: {self.contracts_checked}")
        lines.append(f"Status: {'VALID' if self.valid else 'INVALID'}")
        
        if self.violations:
            lines.append(f"\nViolations ({len(self.violations)}):")
            for v in self.violations:
                lines.append(f"  - {v.method} {v.endpoint}: {v.description}")
        
        return "\n".join(lines)


class ContractValidator:
    """
    API contract validator.
    """
    
    def __init__(self, openapi_spec: Optional[Dict] = None):
        """
        Initialize contract validator.
        
        Args:
            openapi_spec: OpenAPI specification
        """
        self.spec = openapi_spec
        self.violations: List[ContractViolation] = []
    
    def validate_request(
        self,
        contract: APIContract,
        request: Dict,
    ) -> bool:
        """
        Validate request against contract.
        
        Args:
            contract: API contract
            request: Request data
            
        Returns:
            True if valid
        """
        if not contract.request_schema:
            return True
        
        # Check required fields
        for field in contract.required_fields:
            if field not in request:
                self.violations.append(ContractViolation(
                    endpoint=contract.endpoint,
                    method=contract.method,
                    violation_type="required_field",
                    description=f"Missing required field: {field}",
                    expected=field,
                    actual=None,
                ))
                return False
        
        return True
    
    def validate_response(
        self,
        contract: APIContract,
        response: Dict,
        status_code: int,
    ) -> bool:
        """
        Validate response against contract.
        
        Args:
            contract: API contract
            response: Response data
            status_code: HTTP status code
            
        Returns:
            True if valid
        """
        if not contract.response_schema:
            return True
        
        # Basic type checking
        expected_type = contract.response_schema.get("type")
        if expected_type == "object" and not isinstance(response, dict):
            self.violations.append(ContractViolation(
                endpoint=contract.endpoint,
                method=contract.method,
                violation_type="schema",
                description=f"Expected object, got {type(response).__name__}",
                expected="object",
                actual=type(response).__name__,
            ))
            return False
        
        return True
    
    def validate_backward_compatibility(
        self,
        old_spec: Dict,
        new_spec: Dict,
    ) -> APIContractReport:
        """
        Check backward compatibility between specs.
        
        Args:
            old_spec: Previous OpenAPI spec
            new_spec: New OpenAPI spec
            
        Returns:
            APIContractReport
        """
        self.violations = []
        contracts_checked = 0
        
        old_paths = old_spec.get("paths", {})
        new_paths = new_spec.get("paths", {})
        
        for path, methods in old_paths.items():
            for method, _ in methods.items():
                if method.startswith("x-"):
                    continue
                
                contracts_checked += 1
                
                # Check if endpoint still exists
                if path not in new_paths:
                    self.violations.append(ContractViolation(
                        endpoint=path,
                        method=method.upper(),
                        violation_type="backward_compat",
                        description=f"Endpoint removed in new version",
                        expected=path,
                        actual=None,
                    ))
                elif method not in new_paths.get(path, {}):
                    self.violations.append(ContractViolation(
                        endpoint=path,
                        method=method.upper(),
                        violation_type="backward_compat",
                        description=f"Method removed in new version",
                        expected=method,
                        actual=None,
                    ))
        
        return APIContractReport(
            valid=len(self.violations) == 0,
            violations=self.violations,
            contracts_checked=contracts_checked,
        )
    
    def validate_all(self, app: Any = None) -> APIContractReport:
        """
        Validate all API contracts.
        
        Args:
            app: FastAPI application
            
        Returns:
            APIContractReport
        """
        self.violations = []
        contracts_checked = 0
        
        if app is None:
            # Return empty report
            return APIContractReport(
                valid=True,
                violations=[],
                contracts_checked=0,
            )
        
        # Extract OpenAPI spec from app
        if hasattr(app, "openapi"):
            spec = app.openapi()
            
            for path, methods in spec.get("paths", {}).items():
                for method, details in methods.items():
                    if method.startswith("x-"):
                        continue
                    
                    contracts_checked += 1
                    contract = APIContract.from_openapi(spec, path, method)
                    
                    # Validate contract has required fields
                    if not details.get("responses"):
                        self.violations.append(ContractViolation(
                            endpoint=path,
                            method=method.upper(),
                            violation_type="schema",
                            description="Missing responses definition",
                            expected="responses",
                            actual=None,
                        ))
        
        return APIContractReport(
            valid=len(self.violations) == 0,
            violations=self.violations,
            contracts_checked=contracts_checked,
        )
