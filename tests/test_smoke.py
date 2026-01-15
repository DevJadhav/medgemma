"""Smoke tests to verify environment setup."""

import pytest


class TestEnvironmentSetup:
    """Verify the development environment is correctly configured."""

    def test_python_version(self):
        """Verify Python version is 3.10+."""
        import sys
        assert sys.version_info >= (3, 10), "Python 3.10+ required"

    def test_pytest_working(self):
        """Verify pytest is functioning correctly."""
        assert True

    def test_numpy_available(self):
        """Verify numpy is installed and working."""
        import numpy as np
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6

    def test_torch_available(self):
        """Verify PyTorch is installed (may not have GPU)."""
        try:
            import torch
            tensor = torch.tensor([1.0, 2.0, 3.0])
            assert tensor.sum().item() == 6.0
        except ImportError:
            pytest.skip("PyTorch not installed yet")

    def test_pydantic_available(self):
        """Verify Pydantic is installed for data validation."""
        try:
            from pydantic import BaseModel

            class TestModel(BaseModel):
                name: str
                value: int

            model = TestModel(name="test", value=42)
            assert model.name == "test"
        except ImportError:
            pytest.skip("Pydantic not installed yet")

    def test_cryptography_available(self):
        """Verify cryptography library is available for HIPAA compliance."""
        try:
            from cryptography.fernet import Fernet
            key = Fernet.generate_key()
            assert len(key) > 0
        except ImportError:
            pytest.skip("Cryptography not installed yet")


class TestProjectStructure:
    """Verify project structure is correctly set up."""

    def test_medai_compass_package_exists(self):
        """Verify main package can be imported."""
        try:
            import medai_compass
            assert hasattr(medai_compass, "__version__")
        except ImportError:
            pytest.skip("Package not installed yet - run 'pip install -e .'")

    def test_conftest_fixtures_available(self, sample_text_without_phi):
        """Verify conftest fixtures are accessible."""
        assert "Chest X-ray" in sample_text_without_phi

    def test_mock_model_fixtures(self, mock_medgemma_model):
        """Verify mock model fixtures work correctly."""
        assert mock_medgemma_model is not None
        result = mock_medgemma_model.generate()
        assert result is not None
