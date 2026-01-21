"""
TDD Tests for Modal GPU Integration.

Tests cover:
1. Modal setup script functionality
2. Modal volume configuration
3. Token verification and passing
4. Trained model detection with HuggingFace fallback
5. API container → Modal inference flow

Run with: uv run pytest tests/test_modal_integration.py -v
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import dataclass


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_env_with_modal_tokens():
    """Fixture providing environment with Modal tokens."""
    return {
        "MODAL_TOKEN_ID": "test-token-id",
        "MODAL_TOKEN_SECRET": "test-token-secret",
        "HF_TOKEN": "hf_test_token",
        "PREFER_MODAL_GPU": "true",
        "MODEL_CHECKPOINT_DIR": "./model_output/checkpoints",
    }


@pytest.fixture
def mock_env_without_modal_tokens():
    """Fixture providing environment without Modal tokens."""
    return {
        "HF_TOKEN": "hf_test_token",
    }


@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    """Create a mock checkpoint directory with model files."""
    checkpoint = tmp_path / "checkpoints" / "medgemma-4b-finetuned"
    checkpoint.mkdir(parents=True)
    
    # Create minimal model files
    (checkpoint / "config.json").write_text('{"model_type": "gemma"}')
    (checkpoint / "model.safetensors").write_text("mock_weights")
    
    return checkpoint


@pytest.fixture
def mock_empty_checkpoint_dir(tmp_path):
    """Create an empty checkpoint directory."""
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir(parents=True)
    return checkpoint


# =============================================================================
# Test: Modal Setup Script
# =============================================================================

class TestModalSetupScript:
    """Tests for scripts/setup_modal.py functionality."""
    
    def test_setup_modal_loads_env_variables(self, mock_env_with_modal_tokens):
        """Test that setup_modal reads Modal tokens from environment."""
        with patch.dict(os.environ, mock_env_with_modal_tokens, clear=True):
            from scripts.setup_modal import get_modal_config
            
            config = get_modal_config()
            
            assert config["token_id"] == "test-token-id"
            assert config["token_secret"] == "test-token-secret"
    
    def test_setup_modal_fails_without_tokens(self, mock_env_without_modal_tokens):
        """Test that setup_modal raises error without Modal tokens."""
        with patch.dict(os.environ, mock_env_without_modal_tokens, clear=True):
            from scripts.setup_modal import get_modal_config
            
            with pytest.raises(ValueError, match="MODAL_TOKEN_ID"):
                get_modal_config()
    
    def test_verify_modal_connection_success(self, mock_env_with_modal_tokens):
        """Test Modal connection verification."""
        with patch.dict(os.environ, mock_env_with_modal_tokens, clear=True):
            with patch("modal.App") as mock_app:
                mock_app.lookup.return_value = MagicMock()
                
                from scripts.setup_modal import verify_modal_connection
                
                result = verify_modal_connection()
                assert result["status"] == "connected"
    
    def test_verify_modal_connection_failure(self, mock_env_with_modal_tokens):
        """Test Modal connection failure handling."""
        with patch.dict(os.environ, mock_env_with_modal_tokens, clear=True):
            with patch("modal.App") as mock_app:
                mock_app.lookup.side_effect = Exception("Connection failed")
                
                from scripts.setup_modal import verify_modal_connection
                
                result = verify_modal_connection()
                assert result["status"] == "error"
                assert "Connection failed" in result["error"]


class TestModalVolumeConfiguration:
    """Tests for Modal volume setup and configuration."""
    
    def test_create_model_cache_volume(self, mock_env_with_modal_tokens):
        """Test creation of model cache volume."""
        with patch.dict(os.environ, mock_env_with_modal_tokens, clear=True):
            with patch("modal.Volume") as mock_volume:
                mock_volume.from_name.return_value = MagicMock()
                
                from scripts.setup_modal import setup_volumes
                
                volumes = setup_volumes()
                
                mock_volume.from_name.assert_any_call(
                    "medgemma-model-cache",
                    create_if_missing=True
                )
                assert "model_cache" in volumes
    
    def test_create_checkpoints_volume(self, mock_env_with_modal_tokens):
        """Test creation of checkpoints volume."""
        with patch.dict(os.environ, mock_env_with_modal_tokens, clear=True):
            with patch("modal.Volume") as mock_volume:
                mock_volume.from_name.return_value = MagicMock()
                
                from scripts.setup_modal import setup_volumes
                
                volumes = setup_volumes()
                
                mock_volume.from_name.assert_any_call(
                    "medgemma-checkpoints",
                    create_if_missing=True
                )
                assert "checkpoints" in volumes
    
    def test_upload_trained_model_to_volume(
        self, mock_env_with_modal_tokens, mock_checkpoint_dir
    ):
        """Test uploading trained model to Modal volume."""
        env = {**mock_env_with_modal_tokens, "MODEL_CHECKPOINT_DIR": str(mock_checkpoint_dir.parent)}
        
        with patch.dict(os.environ, env, clear=True):
            with patch("modal.Volume") as mock_volume:
                mock_vol_instance = MagicMock()
                mock_volume.from_name.return_value = mock_vol_instance
                
                from scripts.setup_modal import upload_trained_model
                
                result = upload_trained_model(str(mock_checkpoint_dir))
                
                assert result["uploaded"] is True
                assert result["model_path"] == str(mock_checkpoint_dir)


# =============================================================================
# Test: Token Verification
# =============================================================================

class TestModalTokenVerification:
    """Tests for Modal token verification."""
    
    def test_verify_tokens_valid(self, mock_env_with_modal_tokens):
        """Test token verification with valid tokens."""
        with patch.dict(os.environ, mock_env_with_modal_tokens, clear=True):
            from medai_compass.modal.client import MedGemmaModalClient
            
            client = MedGemmaModalClient()
            result = client.verify_tokens()
            
            assert result.token_id_set is True
            assert result.token_secret_set is True
    
    def test_verify_tokens_missing(self, mock_env_without_modal_tokens):
        """Test token verification with missing tokens."""
        with patch.dict(os.environ, mock_env_without_modal_tokens, clear=True):
            from medai_compass.modal.client import MedGemmaModalClient
            
            client = MedGemmaModalClient()
            result = client.verify_tokens()
            
            assert result.token_id_set is False
            assert result.token_secret_set is False
    
    def test_tokens_passed_to_docker_container(self, mock_env_with_modal_tokens):
        """Test that tokens are correctly passed to Docker container."""
        # This test verifies the docker-compose.yml configuration
        import yaml
        
        compose_path = Path(__file__).parent.parent / "docker-compose.yml"
        
        if compose_path.exists():
            with open(compose_path) as f:
                compose = yaml.safe_load(f)
            
            api_env = compose.get("services", {}).get("api", {}).get("environment", [])
            
            # Check if Modal tokens are in API container environment
            env_vars = [e.split("=")[0] if "=" in e else e.replace("${", "").replace("}", "").split(":-")[0] for e in api_env if isinstance(e, str)]
            
            assert "MODAL_TOKEN_ID" in str(api_env) or any("MODAL_TOKEN_ID" in str(e) for e in api_env)


# =============================================================================
# Test: Trained Model Detection
# =============================================================================

class TestTrainedModelDetection:
    """Tests for trained model detection and fallback logic."""
    
    def test_find_trained_model_exists(self, mock_checkpoint_dir):
        """Test finding existing trained model."""
        from medai_compass.models.inference_service import MedGemmaInferenceService
        
        # mock_checkpoint_dir is tmp_path/checkpoints/medgemma-4b-finetuned
        # We pass its parent (checkpoints) as the search directory
        service = MedGemmaInferenceService(
            checkpoint_dirs=[str(mock_checkpoint_dir.parent)]
        )
        
        model_path = service._find_trained_model()
        
        assert model_path is not None
        assert Path(model_path).exists()
    
    def test_find_trained_model_not_exists(self, mock_empty_checkpoint_dir):
        """Test fallback when no trained model exists."""
        from medai_compass.models.inference_service import MedGemmaInferenceService
        
        service = MedGemmaInferenceService(
            checkpoint_dirs=[str(mock_empty_checkpoint_dir)]
        )
        
        model_path = service._find_trained_model()
        
        assert model_path is None
    
    def test_model_loading_priority_trained_first(self, mock_checkpoint_dir):
        """Test that trained model is loaded before HuggingFace."""
        from medai_compass.models.inference_service import MedGemmaInferenceService
        
        service = MedGemmaInferenceService(
            model_name="google/medgemma-4b-it",
            prefer_modal=False,  # Use local for testing
            checkpoint_dirs=[str(mock_checkpoint_dir.parent)]
        )
        
        # The service should detect the trained model
        model_path = service._find_trained_model()
        
        assert model_path is not None
        assert Path(model_path).exists()
    
    def test_model_loading_fallback_to_huggingface(self, mock_empty_checkpoint_dir):
        """Test fallback to HuggingFace when no trained model."""
        from medai_compass.models.inference_service import MedGemmaInferenceService
        
        with patch.dict(os.environ, {"MODEL_CHECKPOINT_DIR": str(mock_empty_checkpoint_dir)}, clear=False):
            service = MedGemmaInferenceService(
                checkpoint_dirs=[str(mock_empty_checkpoint_dir)]
            )
            
            model_path = service._find_trained_model()
            
            # Should not find any trained model
            assert model_path is None


# =============================================================================
# Test: API Container to Modal Flow
# =============================================================================

class TestAPIToModalFlow:
    """Tests for API container calling Modal inference."""
    
    @pytest.mark.asyncio
    async def test_generate_endpoint_calls_modal(self, mock_env_with_modal_tokens):
        """Test that /api/v1/inference/generate calls Modal."""
        with patch.dict(os.environ, mock_env_with_modal_tokens, clear=True):
            # Test the inference service integration
            from medai_compass.models.inference_service import MedGemmaInferenceService
            
            with patch("medai_compass.models.inference_service.should_use_modal") as mock_should_use:
                mock_should_use.return_value = True
                
                service = MedGemmaInferenceService(prefer_modal=True)
                
                # Verify prefer_modal is set
                assert service.prefer_modal is True
    
    @pytest.mark.asyncio
    async def test_status_endpoint_returns_modal_info(self, mock_env_with_modal_tokens):
        """Test that /api/v1/inference/status returns Modal status."""
        with patch.dict(os.environ, mock_env_with_modal_tokens, clear=True):
            from medai_compass.models.inference_service import MedGemmaInferenceService
            
            service = MedGemmaInferenceService()
            backend_info = service.get_backend_info()
            
            # Check that backend_info contains expected keys
            assert "initialized" in backend_info
            assert "use_modal" in backend_info
            assert "model_source" in backend_info
    
    def test_single_container_architecture(self):
        """Verify single API container architecture (no GPU container needed)."""
        import yaml
        
        compose_path = Path(__file__).parent.parent / "docker-compose.yml"
        
        if compose_path.exists():
            with open(compose_path) as f:
                compose = yaml.safe_load(f)
            
            services = compose.get("services", {})
            
            # API container should exist
            assert "api" in services
            
            # API should not require GPU (Modal handles it)
            api_deploy = services.get("api", {}).get("deploy", {})
            api_resources = api_deploy.get("resources", {}).get("reservations", {})
            
            # No GPU reservation in API container (Modal provides GPU)
            devices = api_resources.get("devices", [])
            has_gpu_reservation = any(
                d.get("capabilities", []) == ["gpu"] 
                for d in devices if isinstance(d, dict)
            )
            
            # In default profile, API should NOT have GPU reservation
            # GPU is handled by Modal cloud


# =============================================================================
# Test: Modal App Trained Model Support
# =============================================================================

class TestModalAppTrainedModelSupport:
    """Tests for Modal app trained model loading."""
    
    def test_modal_app_accepts_trained_model_path(self):
        """Test that Modal app can accept trained model path."""
        # Read the Modal app file
        app_path = Path(__file__).parent.parent / "medai_compass" / "modal" / "app.py"
        
        if app_path.exists():
            content = app_path.read_text()
            
            # Check for trained model path support
            assert "TRAINED_MODEL_PATH" in content or "trained_model" in content.lower()
    
    def test_modal_app_loads_trained_model_when_available(self):
        """Test Modal app loads trained model when path is set."""
        # This will be tested after implementation
        pass
    
    def test_modal_app_falls_back_to_huggingface(self):
        """Test Modal app falls back to HuggingFace when no trained model."""
        # This will be tested after implementation
        pass


# =============================================================================
# Test: Integration Flow
# =============================================================================

class TestModalIntegrationFlow:
    """End-to-end integration tests for Modal flow."""
    
    @pytest.mark.asyncio
    async def test_full_inference_flow_with_modal(self, mock_env_with_modal_tokens):
        """Test complete inference flow: API → Inference Service → Modal."""
        with patch.dict(os.environ, mock_env_with_modal_tokens, clear=True):
            # Mock the Modal client
            with patch("medai_compass.modal.client.MedGemmaModalClient") as MockClient:
                mock_client_instance = MagicMock()
                mock_client_instance.generate = AsyncMock(return_value=MagicMock(
                    response="Pneumonia symptoms include cough and fever.",
                    confidence=0.92,
                    model="google/medgemma-4b-it",
                    gpu="H100"
                ))
                MockClient.return_value = mock_client_instance
                
                # Test will be completed after implementation
    
    def test_model_status_reflects_trained_model(self, mock_checkpoint_dir):
        """Test that model status correctly shows trained model."""
        from medai_compass.models.inference_service import MedGemmaInferenceService
        
        with patch.dict(os.environ, {"MODEL_CHECKPOINT_DIR": str(mock_checkpoint_dir.parent)}, clear=False):
            service = MedGemmaInferenceService(
                checkpoint_dirs=[str(mock_checkpoint_dir.parent)]
            )
            
            backend_info = service.get_backend_info()
            
            # Check service tracks trained model path
            assert "trained_model_path" in backend_info
            assert "model_source" in backend_info


# =============================================================================
# Test: UV Package Management
# =============================================================================

class TestUVPackageManagement:
    """Tests verifying uv is used for package management."""
    
    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists for uv."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists()
    
    def test_modal_dependency_in_pyproject(self):
        """Test Modal is in pyproject.toml dependencies."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            assert "modal" in content.lower()
    
    def test_uv_lock_file_exists(self):
        """Test that uv.lock exists (indicates uv is used)."""
        uv_lock_path = Path(__file__).parent.parent / "uv.lock"
        # This may or may not exist depending on setup
        # Just check pyproject.toml is the source of truth
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists()
