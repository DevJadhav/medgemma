"""Infrastructure tests for Phase 1: Ray, MLflow, Modal, and artifact storage.

These tests follow TDD principles - written before implementation.
Tests are designed to verify:
1. Ray cluster configuration and connectivity
2. MLflow experiment tracking setup
3. Modal GPU configuration for training (8x H100) and inference (1x H100)
4. MinIO artifact storage integration
5. Model selection between MedGemma 4B IT and MedGemma 27B IT
"""

import os
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock
import yaml


# =============================================================================
# Configuration Tests
# =============================================================================

class TestRayClusterConfiguration:
    """Test Ray cluster configuration files and settings."""
    
    @pytest.fixture
    def ray_config_path(self) -> Path:
        """Path to Ray cluster configuration."""
        return Path(__file__).parent.parent / "config" / "ray_cluster.yaml"
    
    def test_ray_config_file_exists(self, ray_config_path: Path):
        """Ray cluster configuration file must exist."""
        assert ray_config_path.exists(), f"Ray config not found at {ray_config_path}"
    
    def test_ray_config_has_required_sections(self, ray_config_path: Path):
        """Ray config must have head_node, worker_nodes, and resources sections."""
        with open(ray_config_path) as f:
            config = yaml.safe_load(f)
        
        required_sections = ["cluster_name", "head_node", "worker_nodes", "resources"]
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"
    
    def test_ray_config_gpu_resources(self, ray_config_path: Path):
        """Ray config must define GPU resources for training."""
        with open(ray_config_path) as f:
            config = yaml.safe_load(f)
        
        # Check GPU resources are defined
        resources = config.get("resources", {})
        assert "GPU" in resources or "accelerator_type" in str(config), \
            "GPU resources must be defined for training"
    
    def test_ray_config_model_profiles(self, ray_config_path: Path):
        """Ray config must have profiles for both 4B and 27B models."""
        with open(ray_config_path) as f:
            config = yaml.safe_load(f)
        
        model_profiles = config.get("model_profiles", {})
        assert "medgemma_4b" in model_profiles, "Missing MedGemma 4B profile"
        assert "medgemma_27b" in model_profiles, "Missing MedGemma 27B profile"
    
    def test_ray_config_27b_requires_multi_gpu(self, ray_config_path: Path):
        """27B model profile must specify multi-GPU requirements."""
        with open(ray_config_path) as f:
            config = yaml.safe_load(f)
        
        profile_27b = config.get("model_profiles", {}).get("medgemma_27b", {})
        gpu_count = profile_27b.get("gpu_count", 1)
        assert gpu_count >= 8, f"27B model requires 8+ GPUs for training, got {gpu_count}"


class TestMLflowConfiguration:
    """Test MLflow tracking server configuration."""
    
    @pytest.fixture
    def mlflow_config_path(self) -> Path:
        """Path to MLflow configuration."""
        return Path(__file__).parent.parent / "config" / "mlflow_config.yaml"
    
    def test_mlflow_config_file_exists(self, mlflow_config_path: Path):
        """MLflow configuration file must exist."""
        assert mlflow_config_path.exists(), f"MLflow config not found at {mlflow_config_path}"
    
    def test_mlflow_config_has_required_sections(self, mlflow_config_path: Path):
        """MLflow config must have server, backend, and artifact sections."""
        with open(mlflow_config_path) as f:
            config = yaml.safe_load(f)
        
        required_sections = ["server", "backend_store", "artifact_store", "experiments"]
        for section in required_sections:
            assert section in config, f"Missing required section: {section}"
    
    def test_mlflow_backend_uses_postgres(self, mlflow_config_path: Path):
        """MLflow backend store should use PostgreSQL."""
        with open(mlflow_config_path) as f:
            config = yaml.safe_load(f)
        
        backend = config.get("backend_store", {})
        assert "postgresql" in backend.get("uri", "").lower() or \
               backend.get("type") == "postgresql", \
               "MLflow backend should use PostgreSQL"
    
    def test_mlflow_artifact_uses_minio(self, mlflow_config_path: Path):
        """MLflow artifact store should use MinIO/S3."""
        with open(mlflow_config_path) as f:
            config = yaml.safe_load(f)
        
        artifact = config.get("artifact_store", {})
        assert "s3" in artifact.get("uri", "").lower() or \
               artifact.get("type") in ["s3", "minio"], \
               "MLflow artifacts should use MinIO/S3"
    
    def test_mlflow_has_medgemma_experiments(self, mlflow_config_path: Path):
        """MLflow config should define experiments for MedGemma training."""
        with open(mlflow_config_path) as f:
            config = yaml.safe_load(f)
        
        experiments = config.get("experiments", [])
        experiment_names = [e.get("name", "") for e in experiments]
        
        assert any("medgemma" in name.lower() for name in experiment_names), \
            "Should have MedGemma-related experiments defined"


class TestModalConfiguration:
    """Test Modal GPU configuration for training and inference."""
    
    @pytest.fixture
    def modal_config_path(self) -> Path:
        """Path to Modal configuration."""
        return Path(__file__).parent.parent / "config" / "modal_config.yaml"
    
    def test_modal_config_file_exists(self, modal_config_path: Path):
        """Modal configuration file must exist."""
        assert modal_config_path.exists(), f"Modal config not found at {modal_config_path}"
    
    def test_modal_config_has_training_profile(self, modal_config_path: Path):
        """Modal config must have training profile with 8x H100."""
        with open(modal_config_path) as f:
            config = yaml.safe_load(f)
        
        training = config.get("training", {})
        assert training.get("gpu_type") == "H100", "Training should use H100 GPUs"
        assert training.get("gpu_count") == 8, "Training should use 8x H100 GPUs"
    
    def test_modal_config_has_inference_profile(self, modal_config_path: Path):
        """Modal config must have inference profile with 1x H100."""
        with open(modal_config_path) as f:
            config = yaml.safe_load(f)
        
        inference = config.get("inference", {})
        assert inference.get("gpu_type") == "H100", "Inference should use H100 GPU"
        assert inference.get("gpu_count") == 1, "Inference should use 1x H100 GPU"
    
    def test_modal_config_model_selection(self, modal_config_path: Path):
        """Modal config must support both MedGemma 4B and 27B models."""
        with open(modal_config_path) as f:
            config = yaml.safe_load(f)
        
        models = config.get("models", {})
        assert "medgemma_4b_it" in models, "Missing MedGemma 4B IT configuration"
        assert "medgemma_27b_it" in models, "Missing MedGemma 27B IT configuration"
    
    def test_modal_config_has_volume_mounts(self, modal_config_path: Path):
        """Modal config should define volume mounts for model caching."""
        with open(modal_config_path) as f:
            config = yaml.safe_load(f)
        
        volumes = config.get("volumes", [])
        assert len(volumes) > 0, "Modal config should define volume mounts"


# =============================================================================
# Model Selection Tests
# =============================================================================

class TestModelSelection:
    """Test model selection functionality for MedGemma variants."""
    
    @pytest.fixture
    def model_config_path(self) -> Path:
        """Path to model configuration."""
        return Path(__file__).parent.parent / "config" / "models.yaml"
    
    def test_model_config_file_exists(self, model_config_path: Path):
        """Model configuration file must exist."""
        assert model_config_path.exists(), f"Model config not found at {model_config_path}"
    
    def test_model_config_has_4b_model(self, model_config_path: Path):
        """Configuration for MedGemma 4B IT must exist."""
        with open(model_config_path) as f:
            config = yaml.safe_load(f)
        
        models = config.get("models", {})
        assert "medgemma_4b_it" in models
        
        model_4b = models["medgemma_4b_it"]
        assert model_4b.get("hf_model_id") == "google/medgemma-4b-it"
        assert model_4b.get("vram_requirement_gb") <= 16  # Should fit on single GPU with quantization
    
    def test_model_config_has_27b_model(self, model_config_path: Path):
        """Configuration for MedGemma 27B IT must exist."""
        with open(model_config_path) as f:
            config = yaml.safe_load(f)
        
        models = config.get("models", {})
        assert "medgemma_27b_it" in models
        
        model_27b = models["medgemma_27b_it"]
        assert model_27b.get("hf_model_id") == "google/medgemma-27b-it"
        assert model_27b.get("vram_requirement_gb") >= 60  # Full precision requirement
    
    def test_model_config_has_training_settings(self, model_config_path: Path):
        """Each model should have training-specific settings."""
        with open(model_config_path) as f:
            config = yaml.safe_load(f)
        
        for model_name, model_config in config.get("models", {}).items():
            training = model_config.get("training", {})
            assert "batch_size" in training, f"{model_name} missing batch_size"
            assert "learning_rate" in training, f"{model_name} missing learning_rate"
            assert "lora_r" in training, f"{model_name} missing LoRA rank"


# =============================================================================
# Docker Infrastructure Tests
# =============================================================================

class TestDockerInfrastructure:
    """Test Docker configuration for training infrastructure."""
    
    @pytest.fixture
    def training_dockerfile_path(self) -> Path:
        """Path to training Dockerfile."""
        return Path(__file__).parent.parent / "docker" / "training.Dockerfile"
    
    @pytest.fixture
    def docker_compose_path(self) -> Path:
        """Path to docker-compose.yml."""
        return Path(__file__).parent.parent / "docker-compose.yml"
    
    def test_training_dockerfile_exists(self, training_dockerfile_path: Path):
        """Training Dockerfile must exist."""
        assert training_dockerfile_path.exists(), \
            f"Training Dockerfile not found at {training_dockerfile_path}"
    
    def test_training_dockerfile_has_cuda(self, training_dockerfile_path: Path):
        """Training Dockerfile must use CUDA base image."""
        content = training_dockerfile_path.read_text()
        assert "cuda" in content.lower(), "Training Dockerfile should use CUDA base image"
    
    def test_training_dockerfile_has_ray(self, training_dockerfile_path: Path):
        """Training Dockerfile must install Ray."""
        content = training_dockerfile_path.read_text()
        assert "ray" in content.lower(), "Training Dockerfile should install Ray"
    
    def test_training_dockerfile_uses_uv(self, training_dockerfile_path: Path):
        """Training Dockerfile must use uv for package management."""
        content = training_dockerfile_path.read_text()
        assert "uv" in content.lower(), "Training Dockerfile should use uv for package management"
    
    def test_docker_compose_has_ray_head(self, docker_compose_path: Path):
        """Docker Compose must have ray-head service."""
        content = docker_compose_path.read_text()
        assert "ray-head" in content, "Docker Compose should have ray-head service"
    
    def test_docker_compose_has_ray_worker(self, docker_compose_path: Path):
        """Docker Compose must have ray-worker service."""
        content = docker_compose_path.read_text()
        assert "ray-worker" in content, "Docker Compose should have ray-worker service"
    
    def test_docker_compose_has_mlflow(self, docker_compose_path: Path):
        """Docker Compose must have MLflow service."""
        content = docker_compose_path.read_text()
        assert "mlflow" in content, "Docker Compose should have mlflow service"


# =============================================================================
# Setup Script Tests
# =============================================================================

class TestSetupScripts:
    """Test infrastructure setup scripts."""
    
    @pytest.fixture
    def scripts_dir(self) -> Path:
        """Path to scripts directory."""
        return Path(__file__).parent.parent / "scripts"
    
    def test_setup_infrastructure_script_exists(self, scripts_dir: Path):
        """Infrastructure setup script must exist."""
        script_path = scripts_dir / "setup_infrastructure.sh"
        assert script_path.exists(), f"Setup script not found at {script_path}"
    
    def test_setup_script_is_executable(self, scripts_dir: Path):
        """Setup script must be executable."""
        script_path = scripts_dir / "setup_infrastructure.sh"
        assert os.access(script_path, os.X_OK), "Setup script should be executable"
    
    def test_start_ray_script_exists(self, scripts_dir: Path):
        """Ray cluster start script must exist."""
        script_path = scripts_dir / "start_ray_cluster.sh"
        assert script_path.exists(), f"Ray start script not found at {script_path}"
    
    def test_start_mlflow_script_exists(self, scripts_dir: Path):
        """MLflow start script must exist."""
        script_path = scripts_dir / "start_mlflow_server.sh"
        assert script_path.exists(), f"MLflow start script not found at {script_path}"


# =============================================================================
# Integration Tests (Require Infrastructure Running)
# =============================================================================

@pytest.mark.integration
class TestRayIntegration:
    """Integration tests for Ray cluster connectivity."""
    
    @pytest.fixture
    def ray_address(self) -> str:
        """Get Ray head address from environment."""
        return os.environ.get("RAY_ADDRESS", "ray://localhost:10001")
    
    @pytest.mark.slow
    def test_ray_cluster_connection(self, ray_address: str):
        """Test connection to Ray cluster."""
        pytest.importorskip("ray")
        import ray
        
        try:
            ray.init(address=ray_address, ignore_reinit_error=True)
            assert ray.is_initialized(), "Ray should be initialized"
            
            # Check cluster resources
            resources = ray.cluster_resources()
            assert resources.get("CPU", 0) > 0, "Ray cluster should have CPU resources"
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_ray_gpu_resources(self, ray_address: str):
        """Test GPU resources are available in Ray cluster."""
        pytest.importorskip("ray")
        import ray
        
        try:
            ray.init(address=ray_address, ignore_reinit_error=True)
            resources = ray.cluster_resources()
            assert resources.get("GPU", 0) > 0, "Ray cluster should have GPU resources"
        finally:
            if ray.is_initialized():
                ray.shutdown()


@pytest.mark.integration
class TestMLflowIntegration:
    """Integration tests for MLflow tracking server."""
    
    @pytest.fixture
    def mlflow_tracking_uri(self) -> str:
        """Get MLflow tracking URI from environment."""
        return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    @pytest.mark.slow
    def test_mlflow_server_connection(self, mlflow_tracking_uri: str):
        """Test connection to MLflow tracking server."""
        mlflow = pytest.importorskip("mlflow")
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Try to create a test experiment
        experiment_name = "test_connection_experiment"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            assert experiment_id is not None
            
            # Clean up
            mlflow.delete_experiment(experiment_id)
        except Exception as e:
            pytest.fail(f"Failed to connect to MLflow: {e}")
    
    @pytest.mark.slow
    def test_mlflow_artifact_logging(self, mlflow_tracking_uri: str, tmp_path: Path):
        """Test artifact logging to MinIO."""
        mlflow = pytest.importorskip("mlflow")
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Create a test artifact
        test_file = tmp_path / "test_artifact.txt"
        test_file.write_text("Test artifact content")
        
        with mlflow.start_run():
            mlflow.log_artifact(str(test_file))
            run_id = mlflow.active_run().info.run_id
        
        # Verify artifact was logged
        artifacts = mlflow.artifacts.list_artifacts(run_id)
        assert len(artifacts) > 0, "Artifact should be logged"


@pytest.mark.integration
class TestMinIOIntegration:
    """Integration tests for MinIO artifact storage."""
    
    @pytest.fixture
    def minio_client(self):
        """Create MinIO client."""
        minio = pytest.importorskip("minio")
        
        endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        
        return minio.Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )
    
    @pytest.mark.slow
    def test_minio_connection(self, minio_client):
        """Test connection to MinIO."""
        # List buckets to verify connection
        try:
            buckets = minio_client.list_buckets()
            assert buckets is not None
        except Exception as e:
            pytest.fail(f"Failed to connect to MinIO: {e}")
    
    @pytest.mark.slow
    def test_minio_mlflow_bucket_exists(self, minio_client):
        """Test that MLflow artifacts bucket exists."""
        bucket_name = os.environ.get("MLFLOW_ARTIFACTS_BUCKET", "mlflow-artifacts")
        
        assert minio_client.bucket_exists(bucket_name), \
            f"MLflow artifacts bucket '{bucket_name}' should exist"


# =============================================================================
# Modal Integration Tests
# =============================================================================

@pytest.mark.integration
class TestModalIntegration:
    """Integration tests for Modal GPU deployment."""
    
    @pytest.fixture
    def modal_available(self) -> bool:
        """Check if Modal is available."""
        token_id = os.environ.get("MODAL_TOKEN_ID")
        token_secret = os.environ.get("MODAL_TOKEN_SECRET")
        
        if not token_id or not token_secret:
            pytest.skip("Modal tokens not configured")
        
        try:
            import modal  # noqa: F401
            return True
        except ImportError:
            pytest.skip("Modal package not installed")
    
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_modal_app_deployed(self, modal_available):
        """Test Modal app is deployed and accessible."""
        import modal
        
        try:
            # Try to look up the training app
            cls = modal.Cls.lookup("medai-compass-training", "MedGemmaTrainer")
            assert cls is not None, "MedGemma trainer should be deployed"
        except modal.exception.NotFoundError:
            pytest.fail("Modal training app not deployed")
    
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_modal_inference_available(self, modal_available):
        """Test Modal inference function is available."""
        import modal
        
        try:
            cls = modal.Cls.lookup("medai-compass", "MedGemmaInference")
            assert cls is not None, "MedGemma inference should be deployed"
        except modal.exception.NotFoundError:
            pytest.fail("Modal inference app not deployed")


# =============================================================================
# Model Selection Integration Tests
# =============================================================================

class TestModelSelectionLogic:
    """Test model selection logic without requiring actual models."""
    
    def test_select_4b_model_by_name(self):
        """Test selecting 4B model by name."""
        from medai_compass.training.model_selector import select_model
        
        config = select_model("medgemma-4b")
        assert config["model_id"] == "google/medgemma-4b-it"
        assert config["gpu_count"] == 1  # Single GPU for inference
    
    def test_select_27b_model_by_name(self):
        """Test selecting 27B model by name."""
        from medai_compass.training.model_selector import select_model
        
        config = select_model("medgemma-27b")
        assert config["model_id"] == "google/medgemma-27b-it"
        assert config["training_gpu_count"] == 8  # 8x H100 for training
    
    def test_select_model_with_invalid_name_raises(self):
        """Test that invalid model name raises error."""
        from medai_compass.training.model_selector import select_model, ModelNotFoundError
        
        with pytest.raises(ModelNotFoundError):
            select_model("invalid-model-name")
    
    def test_get_training_config_4b(self):
        """Test getting training config for 4B model."""
        from medai_compass.training.model_selector import get_training_config
        
        config = get_training_config("medgemma-4b")
        assert config["batch_size"] > 0
        assert config["learning_rate"] > 0
        assert "lora_r" in config
    
    def test_get_training_config_27b(self):
        """Test getting training config for 27B model."""
        from medai_compass.training.model_selector import get_training_config
        
        config = get_training_config("medgemma-27b")
        assert config["batch_size"] > 0
        assert config["gpu_count"] == 8
        assert "deepspeed_config" in config  # 27B requires DeepSpeed
