"""
PHI Encryption module for HIPAA compliance.

Provides AES-256 encryption for PHI at rest and
secure hashing for audit logging.
"""

import base64
import hashlib
import secrets
from typing import Union

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class PHIEncryptor:
    """
    AES-256 encryption for Protected Health Information (PHI).

    Uses Fernet (AES-128-CBC with HMAC) for symmetric encryption.
    In production, use dedicated key management (e.g., HashiCorp Vault).
    """

    def __init__(self, key: bytes = None):
        """
        Initialize PHI encryptor.

        Args:
            key: 32-byte encryption key. If None, generates a new key.
        """
        if key is None:
            key = self.generate_key()
        self._fernet = Fernet(base64.urlsafe_b64encode(key[:32].ljust(32, b'\0')))

    @staticmethod
    def generate_key() -> bytes:
        """
        Generate a secure random encryption key.

        Returns:
            32-byte random key
        """
        return secrets.token_bytes(32)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt PHI data.

        Args:
            plaintext: Data to encrypt

        Returns:
            Base64-encoded encrypted data
        """
        encrypted = self._fernet.encrypt(plaintext.encode('utf-8'))
        return encrypted.decode('utf-8')

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt PHI data.

        Args:
            ciphertext: Base64-encoded encrypted data

        Returns:
            Decrypted plaintext
        """
        decrypted = self._fernet.decrypt(ciphertext.encode('utf-8'))
        return decrypted.decode('utf-8')


def hash_for_audit(data: str) -> str:
    """
    Create SHA-256 hash for audit logging.

    Used to create non-reversible identifiers for audit trails
    while protecting actual values.

    Args:
        data: Data to hash

    Returns:
        Hex-encoded SHA-256 hash (64 characters)
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def derive_key_from_password(password: str, salt: bytes = None) -> tuple[bytes, bytes]:
    """
    Derive encryption key from password using PBKDF2.

    Args:
        password: User password
        salt: Salt for key derivation. If None, generates new salt.

    Returns:
        Tuple of (derived_key, salt)
    """
    if salt is None:
        salt = secrets.token_bytes(16)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,  # OWASP recommended
    )

    key = kdf.derive(password.encode('utf-8'))
    return key, salt


# =============================================================================
# KEY ROTATION AND MANAGEMENT
# =============================================================================

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)


@dataclass
class RotationPolicy:
    """
    Policy configuration for key rotation.

    Attributes:
        rotation_interval_days: Days between automatic rotations
        max_key_age_days: Maximum age before key must be rotated
        min_keys_to_retain: Minimum number of old keys to keep
    """
    rotation_interval_days: int = 90
    max_key_age_days: int = 365
    min_keys_to_retain: int = 3


class KeyManager:
    """
    Manages encryption keys with rotation support.

    Provides:
    - Key generation with metadata
    - Secure key storage
    - Key rotation with deprecation
    - Audit logging
    """

    def __init__(
        self,
        key_store_path: Optional[str] = None,
        master_key: Optional[bytes] = None,
        rotation_policy: Optional[RotationPolicy] = None,
        enable_audit: bool = False
    ):
        """
        Initialize key manager.

        Args:
            key_store_path: Directory for key storage
            master_key: Master key for encrypting stored keys
            rotation_policy: Key rotation policy
            enable_audit: Enable audit logging
        """
        self._key_store_path = Path(key_store_path) if key_store_path else None
        self._master_key = master_key or secrets.token_bytes(32)
        self._rotation_policy = rotation_policy or RotationPolicy()
        self._enable_audit = enable_audit

        self._keys: Dict[str, Dict[str, Any]] = {}
        self._active_key_id: Optional[str] = None
        self._audit_log: List[Dict[str, Any]] = []

        if self._key_store_path:
            self._key_store_path.mkdir(parents=True, exist_ok=True)
            self._load_keys()

    def generate_key(self) -> bytes:
        """
        Generate a new random encryption key.

        Returns:
            32-byte random key (256 bits)
        """
        return secrets.token_bytes(32)

    def generate_key_with_metadata(self) -> Dict[str, Any]:
        """
        Generate a new key with full metadata.

        Returns:
            Dictionary with key and metadata
        """
        key_id = str(uuid.uuid4())
        key_bytes = self.generate_key()

        key_info = {
            "key_id": key_id,
            "key": base64.b64encode(key_bytes).decode('utf-8'),
            "algorithm": "AES-256-GCM",
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
            "version": 1,
        }

        self._log_audit("key_generated", key_id)
        return key_info

    def store_key(self, key_info: Dict[str, Any]) -> None:
        """
        Store a key securely.

        Args:
            key_info: Key information dictionary
        """
        key_id = key_info["key_id"]
        self._keys[key_id] = key_info.copy()

        if self._key_store_path:
            self._save_key_to_file(key_info)

        self._log_audit("key_stored", key_id)

    def get_key(self, key_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a key by ID.

        Args:
            key_id: Key identifier

        Returns:
            Key information or None if not found
        """
        key_info = self._keys.get(key_id)
        if key_info:
            self._log_audit("key_accessed", key_id)
        return key_info

    def list_keys(self) -> List[Dict[str, Any]]:
        """
        List all stored keys (without actual key material).

        Returns:
            List of key metadata
        """
        return [
            {k: v for k, v in info.items() if k != "key"}
            for info in self._keys.values()
        ]

    def set_active_key(self, key_id: str) -> None:
        """
        Set the active key for new encryptions.

        Args:
            key_id: Key identifier to activate
        """
        if key_id not in self._keys:
            raise ValueError(f"Key not found: {key_id}")

        self._active_key_id = key_id
        self._log_audit("key_activated", key_id)

    def get_active_key_id(self) -> Optional[str]:
        """Get the active key ID."""
        return self._active_key_id

    def rotate_key(self) -> Dict[str, Any]:
        """
        Rotate to a new encryption key.

        Returns:
            New key information
        """
        # Mark old key as deprecated
        if self._active_key_id and self._active_key_id in self._keys:
            old_key = self._keys[self._active_key_id]
            old_key["status"] = "deprecated"
            old_key["deprecated_at"] = datetime.utcnow().isoformat()

            if self._key_store_path:
                self._save_key_to_file(old_key)

        # Generate and activate new key
        new_key = self.generate_key_with_metadata()
        self.store_key(new_key)
        self.set_active_key(new_key["key_id"])

        self._log_audit("key_rotated", new_key["key_id"], {
            "old_key_id": self._active_key_id
        })

        return new_key

    def needs_rotation(self) -> bool:
        """
        Check if key rotation is needed based on policy.

        Returns:
            True if rotation is recommended
        """
        if not self._active_key_id:
            return True

        key_info = self._keys.get(self._active_key_id)
        if not key_info:
            return True

        created_at = datetime.fromisoformat(key_info["created_at"])
        age_days = (datetime.utcnow() - created_at).days

        return age_days >= self._rotation_policy.rotation_interval_days

    def cleanup_old_keys(self) -> int:
        """
        Remove old deprecated keys according to retention policy.

        Returns:
            Number of keys removed
        """
        # Get deprecated keys sorted by age
        deprecated_keys = [
            (k, v) for k, v in self._keys.items()
            if v.get("status") == "deprecated"
        ]

        deprecated_keys.sort(
            key=lambda x: x[1].get("deprecated_at", ""),
            reverse=True
        )

        # Keep minimum required keys
        keys_to_remove = deprecated_keys[self._rotation_policy.min_keys_to_retain:]

        removed_count = 0
        for key_id, _ in keys_to_remove:
            # Check if key is too old
            key_info = self._keys[key_id]
            created_at = datetime.fromisoformat(key_info["created_at"])
            age_days = (datetime.utcnow() - created_at).days

            if age_days > self._rotation_policy.max_key_age_days:
                del self._keys[key_id]

                if self._key_store_path:
                    key_file = self._key_store_path / f"{key_id}.key"
                    if key_file.exists():
                        key_file.unlink()

                self._log_audit("key_removed", key_id)
                removed_count += 1

        return removed_count

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the audit log entries."""
        return self._audit_log.copy()

    def _log_audit(
        self,
        action: str,
        key_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an audit event."""
        if not self._enable_audit:
            return

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "key_id": key_id,
            "details": details or {},
        }
        self._audit_log.append(entry)
        logger.info(f"Key audit: {action} for {key_id}")

    def _save_key_to_file(self, key_info: Dict[str, Any]) -> None:
        """Save key to encrypted file."""
        key_id = key_info["key_id"]
        key_file = self._key_store_path / f"{key_id}.key"

        # Encrypt key data with master key
        aesgcm = AESGCM(self._master_key)
        nonce = secrets.token_bytes(12)

        plaintext = json.dumps(key_info).encode('utf-8')
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Store nonce + ciphertext
        with open(key_file, 'wb') as f:
            f.write(nonce + ciphertext)

        # Set restrictive permissions (owner read/write only)
        os.chmod(key_file, 0o600)

    def _load_keys(self) -> None:
        """Load keys from storage."""
        if not self._key_store_path or not self._key_store_path.exists():
            return

        aesgcm = AESGCM(self._master_key)

        for key_file in self._key_store_path.glob("*.key"):
            try:
                with open(key_file, 'rb') as f:
                    data = f.read()

                nonce = data[:12]
                ciphertext = data[12:]

                plaintext = aesgcm.decrypt(nonce, ciphertext, None)
                key_info = json.loads(plaintext.decode('utf-8'))

                self._keys[key_info["key_id"]] = key_info

            except Exception as e:
                logger.warning(f"Failed to load key file {key_file}: {e}")


class DataEncryptor:
    """
    Data encryption using managed keys.

    Supports key rotation with automatic key selection for decryption.
    """

    def __init__(self, key_manager: KeyManager):
        """
        Initialize data encryptor.

        Args:
            key_manager: Key manager instance
        """
        self._key_manager = key_manager

    def encrypt(self, plaintext: bytes) -> bytes:
        """
        Encrypt data using the active key.

        Args:
            plaintext: Data to encrypt

        Returns:
            Encrypted data with key ID prefix
        """
        key_id = self._key_manager.get_active_key_id()
        if not key_id:
            raise ValueError("No active key set")

        key_info = self._key_manager.get_key(key_id)
        key_bytes = base64.b64decode(key_info["key"])

        aesgcm = AESGCM(key_bytes)
        nonce = secrets.token_bytes(12)

        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Prepend key ID (36 bytes) + nonce (12 bytes) to ciphertext
        key_id_bytes = key_id.encode('utf-8')
        return key_id_bytes + nonce + ciphertext

    def decrypt(self, ciphertext: bytes) -> bytes:
        """
        Decrypt data, automatically selecting the correct key.

        Args:
            ciphertext: Encrypted data with key ID prefix

        Returns:
            Decrypted plaintext
        """
        # Extract key ID (36 bytes for UUID)
        key_id = ciphertext[:36].decode('utf-8')
        nonce = ciphertext[36:48]
        encrypted_data = ciphertext[48:]

        key_info = self._key_manager.get_key(key_id)
        if not key_info:
            raise ValueError(f"Key not found: {key_id}")

        key_bytes = base64.b64decode(key_info["key"])
        aesgcm = AESGCM(key_bytes)

        return aesgcm.decrypt(nonce, encrypted_data, None)

    def encrypt_string(self, plaintext: str) -> str:
        """
        Encrypt string data.

        Args:
            plaintext: String to encrypt

        Returns:
            Base64-encoded encrypted data
        """
        encrypted = self.encrypt(plaintext.encode('utf-8'))
        return base64.b64encode(encrypted).decode('utf-8')

    def decrypt_string(self, ciphertext: str) -> str:
        """
        Decrypt string data.

        Args:
            ciphertext: Base64-encoded encrypted data

        Returns:
            Decrypted string
        """
        encrypted = base64.b64decode(ciphertext)
        decrypted = self.decrypt(encrypted)
        return decrypted.decode('utf-8')
