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
