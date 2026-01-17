"""
Authentication and authorization module for HIPAA compliance.

Provides JWT token management with MFA verification support
and session timeout enforcement.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt


class TokenManager:
    """
    JWT token manager with HIPAA compliance features.

    Supports:
    - Short session timeouts (15 minutes default per HIPAA)
    - MFA verification status tracking
    - Role-based claims
    """

    def __init__(
        self,
        secret_key: str,
        session_timeout_minutes: int = 15,
        algorithm: str = "HS256"
    ):
        """
        Initialize token manager.

        Args:
            secret_key: Secret key for JWT signing
            session_timeout_minutes: Session timeout in minutes (HIPAA: 15)
            algorithm: JWT signing algorithm
        """
        self.secret_key = secret_key
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.algorithm = algorithm

    def generate_token(
        self,
        user_id: str,
        roles: list[str],
        mfa_verified: bool = False,
        additional_claims: dict = None
    ) -> str:
        """
        Generate a JWT access token.

        Args:
            user_id: Unique user identifier
            roles: List of user roles
            mfa_verified: Whether MFA was completed
            additional_claims: Additional claims to include

        Returns:
            JWT token string
        """
        now = datetime.now(timezone.utc)

        payload = {
            "user_id": user_id,
            "roles": roles,
            "mfa_verified": mfa_verified,
            "iat": now,
            "exp": now + self.session_timeout
        }

        if additional_claims:
            payload.update(additional_claims)

        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[dict]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded payload if valid, None if invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def refresh_token(self, token: str) -> Optional[str]:
        """
        Refresh a valid token with new expiration.

        Args:
            token: Current JWT token

        Returns:
            New token with extended expiration, or None if invalid
        """
        payload = self.verify_token(token)
        if payload is None:
            return None

        return self.generate_token(
            user_id=payload["user_id"],
            roles=payload["roles"],
            mfa_verified=payload.get("mfa_verified", False)
        )


class RoleBasedAccessControl:
    """
    Role-based access control for medical data.

    Defines permissions for different clinical roles.
    """

    # Default role permissions mapping
    DEFAULT_PERMISSIONS = {
        "admin": [
            "manage_users", "view_audit_logs", "configure_system",
            "view_images", "view_reports", "view_patient_info"
        ],
        "radiologist": [
            "view_images", "view_reports", "approve_reports",
            "view_patient_info", "use_ai_analysis"
        ],
        "physician": [
            "view_images", "view_reports", "view_patient_info",
            "use_ai_analysis", "use_communication_agent"
        ],
        "nurse": [
            "view_patient_info", "use_communication_agent",
            "schedule_appointments"
        ],
        "clinician": [
            "view_images", "view_reports", "view_patient_info"
        ],
        "viewer": [
            "view_reports"
        ],
        "ai_system": [
            "process_phi", "generate_reports", "log_activity"
        ]
    }

    def __init__(self, permissions: dict = None):
        """
        Initialize RBAC.

        Args:
            permissions: Custom permissions mapping, or None for defaults
        """
        self.permissions = permissions or self.DEFAULT_PERMISSIONS.copy()

    def has_permission(self, roles: list[str], permission: str) -> bool:
        """
        Check if any of the roles has the required permission.

        Args:
            roles: User's roles
            permission: Required permission

        Returns:
            True if permission granted, False otherwise
        """
        for role in roles:
            role_permissions = self.permissions.get(role, [])
            if permission in role_permissions:
                return True
        return False

    def get_permissions(self, roles: list[str]) -> set[str]:
        """
        Get all permissions for a set of roles.

        Args:
            roles: User's roles

        Returns:
            Set of all permissions
        """
        all_permissions = set()
        for role in roles:
            role_permissions = self.permissions.get(role, [])
            all_permissions.update(role_permissions)
        return all_permissions
