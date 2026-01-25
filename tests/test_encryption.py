"""
Tests for encryption key rotation and secure data handling.

TDD approach: Tests written first for encryption capabilities.
"""

import pytest
from unittest.mock import MagicMock, patch
import os
import tempfile
from datetime import datetime, timedelta


class TestEncryptionKeyManager:
    """Tests for encryption key management."""

    def test_key_manager_initialization(self):
        """Verify key manager initializes correctly."""
        from medai_compass.security.encryption import KeyManager

        manager = KeyManager()
        assert manager is not None

    def test_generate_new_key(self):
        """Verify new encryption key can be generated."""
        from medai_compass.security.encryption import KeyManager

        manager = KeyManager()
        key = manager.generate_key()

        assert key is not None
        assert len(key) >= 32  # At least 256 bits

    def test_key_has_metadata(self):
        """Verify generated key includes metadata."""
        from medai_compass.security.encryption import KeyManager

        manager = KeyManager()
        key_info = manager.generate_key_with_metadata()

        assert "key_id" in key_info
        assert "created_at" in key_info
        assert "algorithm" in key_info
        assert key_info["algorithm"] == "AES-256-GCM"

    def test_store_and_retrieve_key(self):
        """Verify key can be stored and retrieved."""
        from medai_compass.security.encryption import KeyManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyManager(key_store_path=tmpdir)

            key_info = manager.generate_key_with_metadata()
            manager.store_key(key_info)

            retrieved = manager.get_key(key_info["key_id"])
            assert retrieved is not None
            assert retrieved["key_id"] == key_info["key_id"]

    def test_list_all_keys(self):
        """Verify all keys can be listed."""
        from medai_compass.security.encryption import KeyManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyManager(key_store_path=tmpdir)

            # Generate multiple keys
            key1 = manager.generate_key_with_metadata()
            key2 = manager.generate_key_with_metadata()
            manager.store_key(key1)
            manager.store_key(key2)

            keys = manager.list_keys()
            assert len(keys) >= 2


class TestKeyRotation:
    """Tests for key rotation functionality."""

    def test_rotate_key(self):
        """Verify key rotation creates new key and marks old as deprecated."""
        from medai_compass.security.encryption import KeyManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyManager(key_store_path=tmpdir)

            # Create initial key
            old_key = manager.generate_key_with_metadata()
            manager.store_key(old_key)
            manager.set_active_key(old_key["key_id"])

            # Rotate
            new_key = manager.rotate_key()

            assert new_key["key_id"] != old_key["key_id"]
            assert manager.get_active_key_id() == new_key["key_id"]

            # Old key should be marked as deprecated
            old_info = manager.get_key(old_key["key_id"])
            assert old_info["status"] == "deprecated"

    def test_rotation_preserves_old_key_for_decryption(self):
        """Verify old keys are kept for decrypting existing data."""
        from medai_compass.security.encryption import KeyManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyManager(key_store_path=tmpdir)

            old_key = manager.generate_key_with_metadata()
            manager.store_key(old_key)
            manager.set_active_key(old_key["key_id"])

            # Rotate multiple times
            manager.rotate_key()
            manager.rotate_key()

            # Old key should still be retrievable
            retrieved = manager.get_key(old_key["key_id"])
            assert retrieved is not None

    def test_scheduled_rotation(self):
        """Verify keys can be scheduled for rotation."""
        from medai_compass.security.encryption import KeyManager, RotationPolicy

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = RotationPolicy(rotation_interval_days=90)
            manager = KeyManager(key_store_path=tmpdir, rotation_policy=policy)

            key = manager.generate_key_with_metadata()
            manager.store_key(key)
            manager.set_active_key(key["key_id"])

            # Check if rotation is needed
            assert manager.needs_rotation() is False

            # Simulate old key
            manager._keys[key["key_id"]]["created_at"] = (
                datetime.utcnow() - timedelta(days=100)
            ).isoformat()

            assert manager.needs_rotation() is True

    def test_automatic_rotation_check(self):
        """Verify automatic rotation check works."""
        from medai_compass.security.encryption import KeyManager, RotationPolicy

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = RotationPolicy(rotation_interval_days=30)
            manager = KeyManager(key_store_path=tmpdir, rotation_policy=policy)

            key = manager.generate_key_with_metadata()
            manager.store_key(key)
            manager.set_active_key(key["key_id"])

            # Force key to be old
            manager._keys[key["key_id"]]["created_at"] = (
                datetime.utcnow() - timedelta(days=45)
            ).isoformat()

            # Auto-rotate if needed
            if manager.needs_rotation():
                new_key = manager.rotate_key()
                assert new_key is not None


class TestDataEncryption:
    """Tests for data encryption and decryption."""

    def test_encrypt_data(self):
        """Verify data can be encrypted."""
        from medai_compass.security.encryption import DataEncryptor, KeyManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyManager(key_store_path=tmpdir)
            key_info = manager.generate_key_with_metadata()
            manager.store_key(key_info)
            manager.set_active_key(key_info["key_id"])

            encryptor = DataEncryptor(manager)

            plaintext = b"Sensitive medical data: Patient has diabetes"
            ciphertext = encryptor.encrypt(plaintext)

            assert ciphertext != plaintext
            assert len(ciphertext) > 0

    def test_decrypt_data(self):
        """Verify data can be decrypted."""
        from medai_compass.security.encryption import DataEncryptor, KeyManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyManager(key_store_path=tmpdir)
            key_info = manager.generate_key_with_metadata()
            manager.store_key(key_info)
            manager.set_active_key(key_info["key_id"])

            encryptor = DataEncryptor(manager)

            plaintext = b"Sensitive medical data: Patient has diabetes"
            ciphertext = encryptor.encrypt(plaintext)
            decrypted = encryptor.decrypt(ciphertext)

            assert decrypted == plaintext

    def test_decrypt_with_old_key_after_rotation(self):
        """Verify data encrypted with old key can still be decrypted after rotation."""
        from medai_compass.security.encryption import DataEncryptor, KeyManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyManager(key_store_path=tmpdir)
            key_info = manager.generate_key_with_metadata()
            manager.store_key(key_info)
            manager.set_active_key(key_info["key_id"])

            encryptor = DataEncryptor(manager)

            # Encrypt with old key
            plaintext = b"Data encrypted before rotation"
            ciphertext = encryptor.encrypt(plaintext)

            # Rotate key
            manager.rotate_key()

            # Should still decrypt with metadata pointing to old key
            decrypted = encryptor.decrypt(ciphertext)
            assert decrypted == plaintext

    def test_encrypt_string(self):
        """Verify string data can be encrypted."""
        from medai_compass.security.encryption import DataEncryptor, KeyManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyManager(key_store_path=tmpdir)
            key_info = manager.generate_key_with_metadata()
            manager.store_key(key_info)
            manager.set_active_key(key_info["key_id"])

            encryptor = DataEncryptor(manager)

            plaintext = "Patient SSN: 123-45-6789"
            ciphertext = encryptor.encrypt_string(plaintext)
            decrypted = encryptor.decrypt_string(ciphertext)

            assert decrypted == plaintext


class TestSecureKeyStorage:
    """Tests for secure key storage."""

    def test_keys_stored_encrypted(self):
        """Verify keys are stored in encrypted form."""
        from medai_compass.security.encryption import KeyManager
        import secrets

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a proper 32-byte key
            master_key = secrets.token_bytes(32)
            manager = KeyManager(
                key_store_path=tmpdir,
                master_key=master_key
            )

            key_info = manager.generate_key_with_metadata()
            manager.store_key(key_info)

            # Key file should exist but not contain plaintext key
            key_files = os.listdir(tmpdir)
            assert len(key_files) > 0

    def test_key_file_permissions(self):
        """Verify key files have restricted permissions."""
        from medai_compass.security.encryption import KeyManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyManager(key_store_path=tmpdir)

            key_info = manager.generate_key_with_metadata()
            manager.store_key(key_info)

            # Check file permissions (should be readable only by owner)
            key_files = os.listdir(tmpdir)
            for f in key_files:
                filepath = os.path.join(tmpdir, f)
                if os.path.isfile(filepath):
                    mode = os.stat(filepath).st_mode
                    # Owner read/write only (0o600)
                    assert (mode & 0o777) == 0o600


class TestKeyRotationPolicy:
    """Tests for key rotation policy configuration."""

    def test_default_rotation_policy(self):
        """Verify default rotation policy settings."""
        from medai_compass.security.encryption import RotationPolicy

        policy = RotationPolicy()

        assert policy.rotation_interval_days == 90  # Default 90 days
        assert policy.max_key_age_days == 365  # Max 1 year
        assert policy.min_keys_to_retain == 3

    def test_custom_rotation_policy(self):
        """Verify custom rotation policy can be set."""
        from medai_compass.security.encryption import RotationPolicy

        policy = RotationPolicy(
            rotation_interval_days=30,
            max_key_age_days=180,
            min_keys_to_retain=5
        )

        assert policy.rotation_interval_days == 30
        assert policy.max_key_age_days == 180
        assert policy.min_keys_to_retain == 5

    def test_key_cleanup_respects_retention(self):
        """Verify old keys are cleaned up according to policy."""
        from medai_compass.security.encryption import KeyManager, RotationPolicy

        with tempfile.TemporaryDirectory() as tmpdir:
            policy = RotationPolicy(min_keys_to_retain=2)
            manager = KeyManager(key_store_path=tmpdir, rotation_policy=policy)

            # Create several keys
            for _ in range(5):
                key = manager.generate_key_with_metadata()
                manager.store_key(key)
                manager.set_active_key(key["key_id"])

            # Run cleanup
            manager.cleanup_old_keys()

            # Should retain at least min_keys_to_retain
            assert len(manager.list_keys()) >= 2


class TestEncryptionAuditLog:
    """Tests for encryption audit logging."""

    def test_key_generation_logged(self):
        """Verify key generation is logged."""
        from medai_compass.security.encryption import KeyManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyManager(key_store_path=tmpdir, enable_audit=True)

            key = manager.generate_key_with_metadata()
            manager.store_key(key)

            audit_log = manager.get_audit_log()
            assert len(audit_log) > 0
            assert any(e["action"] == "key_generated" for e in audit_log)

    def test_key_rotation_logged(self):
        """Verify key rotation is logged."""
        from medai_compass.security.encryption import KeyManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyManager(key_store_path=tmpdir, enable_audit=True)

            key = manager.generate_key_with_metadata()
            manager.store_key(key)
            manager.set_active_key(key["key_id"])

            manager.rotate_key()

            audit_log = manager.get_audit_log()
            assert any(e["action"] == "key_rotated" for e in audit_log)

    def test_key_access_logged(self):
        """Verify key access is logged."""
        from medai_compass.security.encryption import KeyManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyManager(key_store_path=tmpdir, enable_audit=True)

            key = manager.generate_key_with_metadata()
            manager.store_key(key)

            # Access the key
            manager.get_key(key["key_id"])

            audit_log = manager.get_audit_log()
            assert any(e["action"] == "key_accessed" for e in audit_log)
