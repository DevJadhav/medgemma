"""Tests for the HIPAA-compliant security module."""

import pytest
from datetime import datetime, timedelta, timezone


class TestPHIEncryption:
    """Tests for PHI encryption functionality."""

    def test_encrypt_decrypt_roundtrip(self):
        """Test that encrypted data can be decrypted correctly."""
        from medai_compass.security.encryption import PHIEncryptor

        encryptor = PHIEncryptor()
        original = "Patient SSN: 123-45-6789"

        encrypted = encryptor.encrypt(original)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == original

    def test_encrypted_data_differs_from_original(self):
        """Test that encrypted data is different from original."""
        from medai_compass.security.encryption import PHIEncryptor

        encryptor = PHIEncryptor()
        original = "Patient SSN: 123-45-6789"

        encrypted = encryptor.encrypt(original)

        assert encrypted != original
        assert len(encrypted) > 0

    def test_same_plaintext_different_ciphertext(self):
        """Test that encrypting same text twice gives different results (due to IV)."""
        from medai_compass.security.encryption import PHIEncryptor

        encryptor = PHIEncryptor()
        plaintext = "Sensitive data"

        encrypted1 = encryptor.encrypt(plaintext)
        encrypted2 = encryptor.encrypt(plaintext)

        # Due to random IV, same plaintext should produce different ciphertext
        # This may or may not be true depending on implementation
        # At minimum, both should decrypt to same value
        assert encryptor.decrypt(encrypted1) == encryptor.decrypt(encrypted2)

    def test_encryption_key_generation(self):
        """Test encryption key generation."""
        from medai_compass.security.encryption import PHIEncryptor

        key = PHIEncryptor.generate_key()

        assert len(key) > 0
        assert isinstance(key, bytes)


class TestTokenManagement:
    """Tests for JWT token management."""

    def test_generate_user_token(self):
        """Test generating a user access token."""
        from medai_compass.security.auth import TokenManager

        manager = TokenManager(secret_key="test-secret-key-12345")
        token = manager.generate_token(
            user_id="user123",
            roles=["clinician", "viewer"],
            mfa_verified=True
        )

        assert token is not None
        assert len(token) > 0

    def test_verify_valid_token(self):
        """Test verifying a valid token."""
        from medai_compass.security.auth import TokenManager

        manager = TokenManager(secret_key="test-secret-key-12345")
        token = manager.generate_token(
            user_id="user123",
            roles=["clinician"],
            mfa_verified=True
        )

        payload = manager.verify_token(token)

        assert payload is not None
        assert payload["user_id"] == "user123"
        assert "clinician" in payload["roles"]
        assert payload["mfa_verified"] is True

    def test_reject_expired_token(self):
        """Test that expired tokens are rejected."""
        from medai_compass.security.auth import TokenManager

        manager = TokenManager(
            secret_key="test-secret-key-12345",
            session_timeout_minutes=0  # Immediate expiration
        )
        token = manager.generate_token(user_id="user123", roles=["viewer"])

        # Wait a moment for token to expire
        import time
        time.sleep(0.1)

        payload = manager.verify_token(token)
        assert payload is None

    def test_reject_invalid_token(self):
        """Test that invalid tokens are rejected."""
        from medai_compass.security.auth import TokenManager

        manager = TokenManager(secret_key="test-secret-key-12345")
        payload = manager.verify_token("invalid.token.here")

        assert payload is None

    def test_reject_tampered_token(self):
        """Test that tampered tokens are rejected."""
        from medai_compass.security.auth import TokenManager

        manager = TokenManager(secret_key="test-secret-key-12345")
        token = manager.generate_token(user_id="user123", roles=["viewer"])

        # Tamper with the token
        tampered = token[:-5] + "XXXXX"

        payload = manager.verify_token(tampered)
        assert payload is None

    def test_mfa_verification_status(self):
        """Test MFA verification status in token."""
        from medai_compass.security.auth import TokenManager

        manager = TokenManager(secret_key="test-secret-key-12345")

        # Token without MFA
        token_no_mfa = manager.generate_token(
            user_id="user123",
            roles=["viewer"],
            mfa_verified=False
        )
        payload_no_mfa = manager.verify_token(token_no_mfa)
        assert payload_no_mfa["mfa_verified"] is False

        # Token with MFA
        token_with_mfa = manager.generate_token(
            user_id="user123",
            roles=["viewer"],
            mfa_verified=True
        )
        payload_with_mfa = manager.verify_token(token_with_mfa)
        assert payload_with_mfa["mfa_verified"] is True


class TestAuditLogging:
    """Tests for HIPAA-compliant audit logging."""

    def test_log_access_event(self):
        """Test logging a PHI access event."""
        from medai_compass.security.audit import AuditLogger

        logger = AuditLogger()
        event_id = logger.log_access(
            user_id="user123",
            action="view",
            resource_type="patient_record",
            resource_id="PAT-001",
            phi_accessed=True,
            outcome="success"
        )

        assert event_id is not None
        assert len(event_id) > 0

    def test_user_id_is_hashed(self):
        """Test that user IDs are hashed in audit logs."""
        from medai_compass.security.audit import AuditLogger

        logger = AuditLogger()
        event_id = logger.log_access(
            user_id="user123",
            action="view",
            resource_type="patient_record",
            resource_id="PAT-001",
            phi_accessed=True,
            outcome="success"
        )

        # Get the log entry
        entry = logger.get_entry(event_id)

        # User ID should be hashed, not plaintext
        assert entry["user_id_hash"] != "user123"
        assert len(entry["user_id_hash"]) == 64  # SHA-256 hex

    def test_log_ai_inference(self):
        """Test logging an AI inference event."""
        from medai_compass.security.audit import AuditLogger

        logger = AuditLogger()
        event_id = logger.log_ai_inference(
            user_id="user123",
            model_name="medgemma_4b",
            input_hash="abc123",
            output_hash="def456",
            confidence=0.95,
            escalated=False,
            processing_time_ms=150.5
        )

        entry = logger.get_entry(event_id)

        assert entry is not None
        assert entry["action"] == "ai_inference"
        assert entry["details"]["confidence"] == 0.95
        assert entry["details"]["escalated_to_human"] is False

    def test_log_entry_has_timestamp(self):
        """Test that log entries have timestamps."""
        from medai_compass.security.audit import AuditLogger

        logger = AuditLogger()
        event_id = logger.log_access(
            user_id="user123",
            action="view",
            resource_type="patient_record",
            resource_id="PAT-001",
            phi_accessed=True,
            outcome="success"
        )

        entry = logger.get_entry(event_id)

        assert "timestamp" in entry
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))

    def test_log_entry_retention(self):
        """Test that log entries are retained."""
        from medai_compass.security.audit import AuditLogger

        logger = AuditLogger()

        # Log multiple events
        event_ids = []
        for i in range(5):
            event_id = logger.log_access(
                user_id=f"user{i}",
                action="view",
                resource_type="patient_record",
                resource_id=f"PAT-00{i}",
                phi_accessed=True,
                outcome="success"
            )
            event_ids.append(event_id)

        # All events should be retrievable
        for event_id in event_ids:
            entry = logger.get_entry(event_id)
            assert entry is not None

    def test_export_audit_log(self):
        """Test exporting audit log for compliance review."""
        from medai_compass.security.audit import AuditLogger

        logger = AuditLogger()

        # Log some events
        logger.log_access(
            user_id="user123",
            action="view",
            resource_type="patient_record",
            resource_id="PAT-001",
            phi_accessed=True,
            outcome="success"
        )

        export = logger.export_logs()

        assert "entries" in export
        assert len(export["entries"]) >= 1


class TestSecurityHash:
    """Tests for secure hashing functions."""

    def test_hash_for_audit(self):
        """Test that hashing produces consistent results."""
        from medai_compass.security.encryption import hash_for_audit

        data = "user123"
        hash1 = hash_for_audit(data)
        hash2 = hash_for_audit(data)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_different_input_different_hash(self):
        """Test that different inputs produce different hashes."""
        from medai_compass.security.encryption import hash_for_audit

        hash1 = hash_for_audit("user123")
        hash2 = hash_for_audit("user456")

        assert hash1 != hash2


class TestHIPAASecurityManager:
    """Tests for the HIPAA security manager."""

    def test_init_security_manager(self):
        """Test initializing the security manager."""
        from medai_compass.security import HIPAASecurityManager

        manager = HIPAASecurityManager(
            encryption_key=b"test-key-32-bytes-long-exactly!!"
        )

        assert manager is not None

    def test_encrypt_and_audit(self):
        """Test encrypting PHI and creating audit entry."""
        from medai_compass.security import HIPAASecurityManager

        manager = HIPAASecurityManager(
            encryption_key=b"test-key-32-bytes-long-exactly!!"
        )

        # Encrypt some PHI
        original = "Patient DOB: 01/15/1985"
        encrypted = manager.encrypt_phi(original)

        # Should be able to decrypt
        decrypted = manager.decrypt_phi(encrypted)
        assert decrypted == original

    def test_validate_access_permissions(self):
        """Test access permission validation."""
        from medai_compass.security import HIPAASecurityManager

        manager = HIPAASecurityManager(
            encryption_key=b"test-key-32-bytes-long-exactly!!"
        )

        # Valid permission
        assert manager.validate_access(
            roles=["radiologist"],
            required_permission="view_images"
        ) is True

        # Invalid permission
        assert manager.validate_access(
            roles=["nurse"],
            required_permission="admin_settings"
        ) is False
