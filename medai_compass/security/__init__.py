"""MedAI Compass Security Module.

Provides HIPAA-compliant security features:
- PHI encryption (AES-256)
- JWT token management with MFA support
- Role-based access control
- Audit logging with 6-year retention
- Security auditing (OWASP Top 10)
- HIPAA compliance checking
- API contract validation
- Penetration testing integration
"""

from medai_compass.security.encryption import (
    PHIEncryptor,
    hash_for_audit,
    derive_key_from_password,
)
from medai_compass.security.auth import (
    TokenManager,
    RoleBasedAccessControl,
)
from medai_compass.security.audit import (
    AuditLogger,
    AuditEntry,
    # Phase 9 additions
    AuditConfig,
    AuditResult,
    AuditFinding,
    AuditSeverity,
    SecurityAudit,
    OWASPAudit,
    APISecurityAudit,
)
from medai_compass.security.hipaa import (
    HIPAACompliance,
    HIPAAViolation,
    HIPAAReport,
    HIPAACategory,
    ComplianceStatus,
)
from medai_compass.security.contracts import (
    APIContract,
    ContractValidator,
    APIContractReport,
    ContractViolation,
)
from medai_compass.security.penetration import (
    PenetrationTestConfig,
    PenetrationTestRunner,
    PenetrationTestResult,
    PenetrationTestFinding,
    PenetrationTestSeverity,
)


class HIPAASecurityManager:
    """
    Unified HIPAA security manager.

    Combines encryption, authentication, authorization,
    and audit logging for comprehensive security.
    """

    def __init__(self, encryption_key: bytes):
        """
        Initialize HIPAA security manager.

        Args:
            encryption_key: 32-byte encryption key for PHI
        """
        self.encryptor = PHIEncryptor(encryption_key)
        self.rbac = RoleBasedAccessControl()
        self.audit = AuditLogger()

    def encrypt_phi(self, data: str) -> str:
        """
        Encrypt PHI data.

        Args:
            data: PHI to encrypt

        Returns:
            Encrypted data
        """
        return self.encryptor.encrypt(data)

    def decrypt_phi(self, data: str) -> str:
        """
        Decrypt PHI data.

        Args:
            data: Encrypted PHI

        Returns:
            Decrypted data
        """
        return self.encryptor.decrypt(data)

    def validate_access(self, roles: list[str], required_permission: str) -> bool:
        """
        Validate access based on roles and required permission.

        Args:
            roles: User's roles
            required_permission: Permission required for access

        Returns:
            True if access granted, False otherwise
        """
        return self.rbac.has_permission(roles, required_permission)

    def log_access(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        phi_accessed: bool = True
    ) -> str:
        """
        Log an access event.

        Args:
            user_id: User performing action
            action: Action taken
            resource_type: Type of resource
            resource_id: Resource identifier
            phi_accessed: Whether PHI was accessed

        Returns:
            Event ID
        """
        return self.audit.log_access(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            phi_accessed=phi_accessed,
            outcome="success"
        )


__all__ = [
    # Encryption
    "PHIEncryptor",
    "hash_for_audit",
    "derive_key_from_password",
    # Auth
    "TokenManager",
    "RoleBasedAccessControl",
    # Audit
    "AuditLogger",
    "AuditEntry",
    # Security Audit (Phase 9)
    "AuditConfig",
    "AuditResult",
    "AuditFinding",
    "AuditSeverity",
    "SecurityAudit",
    "OWASPAudit",
    "APISecurityAudit",
    # HIPAA Compliance (Phase 9)
    "HIPAACompliance",
    "HIPAAViolation",
    "HIPAAReport",
    "HIPAACategory",
    "ComplianceStatus",
    # API Contracts (Phase 9)
    "APIContract",
    "ContractValidator",
    "APIContractReport",
    "ContractViolation",
    # Penetration Testing (Phase 9)
    "PenetrationTestConfig",
    "PenetrationTestRunner",
    "PenetrationTestResult",
    "PenetrationTestFinding",
    "PenetrationTestSeverity",
    # Manager
    "HIPAASecurityManager",
]
