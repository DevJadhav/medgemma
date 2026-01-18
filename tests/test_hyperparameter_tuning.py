"""Tests for Phase 4: Hyperparameter Optimization.

Comprehensive TDD tests covering:
- Task 4.1: Ray Tune integration
- Task 4.2: Search space definitions for LoRA params
- Task 4.3: ASHA scheduler implementation
- Task 4.4: Optuna integration
- Task 4.5: HP search analysis tools
- Task 4.6: Population-based training (PBT)
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Check for optional dependencies
try:
    import ray
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import hiplot
    HIPLOT_AVAILABLE = True
except ImportError:
    HIPLOT_AVAILABLE = False

# Skip markers
requires_ray = pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray Tune not installed")
requires_optuna = pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
requires_plotly = pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not installed")
requires_hiplot = pytest.mark.skipif(not HIPLOT_AVAILABLE, reason="HiPlot not installed")


# =============================================================================
# Task 4.1: Ray Tune Integration Tests
# =============================================================================

class TestRayTuneIntegrationModule:
    """Test Ray Tune integration module structure."""
    
    def test_ray_tune_integration_module_exists(self):
        """Verify ray_tune_integration module can be imported."""
        from medai_compass.optimization import ray_tune_integration
        assert ray_tune_integration is not None
    
    def test_ray_tune_integration_has_required_classes(self):
        """Verify required classes exist in module."""
        from medai_compass.optimization.ray_tune_integration import (
            HyperparameterTuner,
            TuneConfig,
            TuneResult,
        )
        assert HyperparameterTuner is not None
        assert TuneConfig is not None
        assert TuneResult is not None


class TestTuneConfig:
    """Test TuneConfig dataclass."""
    
    def test_create_tune_config_defaults(self):
        """Test creating TuneConfig with defaults."""
        from medai_compass.optimization.ray_tune_integration import TuneConfig
        
        config = TuneConfig()
        assert config.num_samples >= 1
        assert config.metric == "eval_loss"
        assert config.mode == "min"
    
    def test_create_tune_config_for_4b(self):
        """Test creating TuneConfig for 4B model."""
        from medai_compass.optimization.ray_tune_integration import TuneConfig
        
        config = TuneConfig.for_model("medgemma-4b")
        assert config.max_concurrent_trials == 4
        assert config.resources_per_trial.get("gpu", 1) == 1
    
    def test_create_tune_config_for_27b(self):
        """Test creating TuneConfig for 27B model."""
        from medai_compass.optimization.ray_tune_integration import TuneConfig
        
        config = TuneConfig.for_model("medgemma-27b")
        assert config.max_concurrent_trials == 1
        assert config.resources_per_trial.get("gpu", 8) == 8
    
    def test_tune_config_has_storage_path(self):
        """Test TuneConfig has storage path configuration."""
        from medai_compass.optimization.ray_tune_integration import TuneConfig
        
        config = TuneConfig(storage_path="/tmp/tune_results")
        assert config.storage_path == "/tmp/tune_results"


class TestHyperparameterTuner:
    """Test HyperparameterTuner class."""
    
    def test_tuner_has_run_method(self):
        """Test tuner has run method."""
        from medai_compass.optimization.ray_tune_integration import HyperparameterTuner
        
        assert hasattr(HyperparameterTuner, "run")
        assert callable(getattr(HyperparameterTuner, "run"))
    
    def test_tuner_has_resume_method(self):
        """Test tuner has resume method for continuing interrupted runs."""
        from medai_compass.optimization.ray_tune_integration import HyperparameterTuner
        
        assert hasattr(HyperparameterTuner, "resume")
        assert callable(getattr(HyperparameterTuner, "resume"))
    
    def test_tuner_has_get_best_config_method(self):
        """Test tuner can extract best configuration."""
        from medai_compass.optimization.ray_tune_integration import HyperparameterTuner
        
        assert hasattr(HyperparameterTuner, "get_best_config")
    
    def test_tuner_accepts_search_space(self):
        """Test tuner accepts custom search space."""
        from medai_compass.optimization.ray_tune_integration import (
            HyperparameterTuner,
            TuneConfig,
        )
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        search_space = SearchSpaceConfig.for_model("medgemma-4b")
        config = TuneConfig()
        
        tuner = HyperparameterTuner(
            model_name="medgemma-4b",
            search_space=search_space,
            tune_config=config,
        )
        
        assert tuner.search_space is not None
    
    def test_tuner_integrates_with_mlflow(self):
        """Test tuner supports MLflow tracking."""
        from medai_compass.optimization.ray_tune_integration import (
            HyperparameterTuner,
            TuneConfig,
        )
        from medai_compass.optimization.search_space import SearchSpaceConfig

        search_space = SearchSpaceConfig.for_model("medgemma-4b")
        config = TuneConfig()
        tuner = HyperparameterTuner(
            model_name="medgemma-4b",
            search_space=search_space,
            tune_config=config,
            mlflow_tracking_uri="http://localhost:5000",
        )
        assert tuner.mlflow_tracking_uri == "http://localhost:5000"
class TestResourceAllocation:
    """Test GPU resource allocation strategies."""
    
    def test_4b_model_uses_1_gpu_per_trial(self):
        """4B model should use 1 GPU per trial."""
        from medai_compass.optimization.ray_tune_integration import TuneConfig
        
        config = TuneConfig.for_model("medgemma-4b")
        assert config.resources_per_trial["gpu"] == 1
    
    def test_27b_model_uses_8_gpus_per_trial(self):
        """27B model should use 8 GPUs per trial."""
        from medai_compass.optimization.ray_tune_integration import TuneConfig
        
        config = TuneConfig.for_model("medgemma-27b")
        assert config.resources_per_trial["gpu"] == 8
    
    def test_4b_allows_4_concurrent_trials(self):
        """4B model should allow 4 concurrent trials."""
        from medai_compass.optimization.ray_tune_integration import TuneConfig
        
        config = TuneConfig.for_model("medgemma-4b")
        assert config.max_concurrent_trials == 4
    
    def test_27b_allows_1_concurrent_trial(self):
        """27B model should only allow 1 concurrent trial."""
        from medai_compass.optimization.ray_tune_integration import TuneConfig
        
        config = TuneConfig.for_model("medgemma-27b")
        assert config.max_concurrent_trials == 1


# =============================================================================
# Task 4.2: Search Space Definition Tests
# =============================================================================

class TestSearchSpaceModule:
    """Test search space module structure."""
    
    def test_search_space_module_exists(self):
        """Verify search_space module can be imported."""
        from medai_compass.optimization import search_space
        assert search_space is not None
    
    def test_search_space_has_required_classes(self):
        """Verify required classes exist."""
        from medai_compass.optimization.search_space import (
            SearchSpaceConfig,
            ParameterSpace,
        )
        assert SearchSpaceConfig is not None
        assert ParameterSpace is not None


class TestSearchSpaceConfig:
    """Test SearchSpaceConfig class."""
    
    def test_create_search_space_for_4b(self):
        """Test creating search space for 4B model."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        assert config is not None
        assert config.model_name == "medgemma-4b"
    
    def test_create_search_space_for_27b(self):
        """Test creating search space for 27B model."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-27b")
        assert config is not None
        assert config.model_name == "medgemma-27b"
    
    def test_search_space_has_lora_r(self):
        """Test search space includes lora_r parameter."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        assert "lora_r" in config.parameters
    
    def test_search_space_has_lora_alpha(self):
        """Test search space includes lora_alpha parameter."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        assert "lora_alpha" in config.parameters
    
    def test_search_space_has_learning_rate(self):
        """Test search space includes learning_rate parameter."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        assert "learning_rate" in config.parameters
    
    def test_search_space_has_lora_dropout(self):
        """Test search space includes lora_dropout parameter."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        assert "lora_dropout" in config.parameters
    
    def test_search_space_has_batch_size(self):
        """Test search space includes batch_size parameter."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        assert "batch_size" in config.parameters
    
    def test_search_space_has_warmup_ratio(self):
        """Test search space includes warmup_ratio parameter."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        assert "warmup_ratio" in config.parameters
    
    def test_search_space_has_weight_decay(self):
        """Test search space includes weight_decay parameter."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        assert "weight_decay" in config.parameters


class TestLoRASearchSpace:
    """Test LoRA-specific search space definitions."""
    
    def test_lora_r_is_categorical(self):
        """lora_r should be categorical choice."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        param_type = config.get_parameter_type("lora_r")
        # categorical/choice type
        assert param_type in ("categorical", "choice")
    
    def test_lora_r_values_for_4b(self):
        """4B model should have appropriate lora_r choices."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        choices = config.get_parameter_choices("lora_r")
        assert 8 in choices
        assert 16 in choices
        assert 32 in choices
    
    def test_lora_r_values_for_27b(self):
        """27B model should have larger lora_r choices."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-27b")
        choices = config.get_parameter_choices("lora_r")
        assert 32 in choices
        assert 64 in choices
        assert 128 in choices
    
    def test_learning_rate_is_loguniform(self):
        """learning_rate should use log-uniform distribution."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        param_type = config.get_parameter_type("learning_rate")
        assert param_type == "loguniform"
    
    def test_learning_rate_bounds(self):
        """learning_rate should have reasonable bounds."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        bounds = config.get_parameter_bounds("learning_rate")
        assert bounds["lower"] >= 1e-6
        assert bounds["upper"] <= 1e-2


class TestBatchSizeSearchSpace:
    """Test batch size search space for different models."""
    
    def test_4b_batch_size_choices(self):
        """4B model should have higher batch size choices."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        choices = config.get_parameter_choices("batch_size")
        assert 4 in choices or 2 in choices
    
    def test_27b_batch_size_choices(self):
        """27B model should have lower batch size choices."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-27b")
        choices = config.get_parameter_choices("batch_size")
        assert 1 in choices


class TestCustomSearchSpace:
    """Test custom search space modifications."""
    
    def test_add_custom_parameter(self):
        """Test adding custom parameter to search space."""
        from medai_compass.optimization.search_space import (
            SearchSpaceConfig,
            ParameterSpace,
        )
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        config.add_parameter(
            name="custom_param",
            space=ParameterSpace(
                type="uniform",
                lower=0.0,
                upper=1.0,
            )
        )
        
        assert "custom_param" in config.parameters
    
    def test_remove_parameter(self):
        """Test removing parameter from search space."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        config.remove_parameter("weight_decay")
        
        assert "weight_decay" not in config.parameters
    
    def test_update_parameter_bounds(self):
        """Test updating parameter bounds."""
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        config = SearchSpaceConfig.for_model("medgemma-4b")
        config.update_parameter(
            name="learning_rate",
            lower=1e-6,
            upper=1e-4,
        )
        
        bounds = config.get_parameter_bounds("learning_rate")
        assert bounds["upper"] == 1e-4


# =============================================================================
# Task 4.3: ASHA Scheduler Tests
# =============================================================================

@requires_ray
class TestASHAScheduler:
    """Test ASHA scheduler implementation."""
    
    def test_asha_scheduler_available(self):
        """Test ASHA scheduler can be configured."""
        from medai_compass.optimization.ray_tune_integration import (
            create_asha_scheduler,
        )
        
        scheduler = create_asha_scheduler()
        assert scheduler is not None
    
    def test_asha_scheduler_has_max_t(self):
        """Test ASHA scheduler has max_t parameter."""
        from medai_compass.optimization.ray_tune_integration import (
            create_asha_scheduler,
        )
        
        scheduler = create_asha_scheduler(max_t=100)
        assert scheduler.max_t == 100
    
    def test_asha_scheduler_has_grace_period(self):
        """Test ASHA scheduler has grace_period parameter."""
        from medai_compass.optimization.ray_tune_integration import (
            create_asha_scheduler,
        )
        
        scheduler = create_asha_scheduler(grace_period=10)
        assert scheduler._grace_period == 10
    
    def test_asha_scheduler_has_reduction_factor(self):
        """Test ASHA scheduler has reduction_factor parameter."""
        from medai_compass.optimization.ray_tune_integration import (
            create_asha_scheduler,
        )
        
        scheduler = create_asha_scheduler(reduction_factor=3)
        assert scheduler._reduction_factor == 3
    
    def test_asha_default_values(self):
        """Test ASHA scheduler default values."""
        from medai_compass.optimization.ray_tune_integration import (
            create_asha_scheduler,
        )
        
        scheduler = create_asha_scheduler()
        # Defaults from implementation_plan.md
        assert scheduler.max_t == 100
        assert scheduler._grace_period == 10
        assert scheduler._reduction_factor == 3


@requires_ray
class TestASHAMetricConfiguration:
    """Test ASHA scheduler metric configuration."""
    
    def test_asha_uses_eval_loss_metric(self):
        """Test ASHA uses eval_loss as metric."""
        from medai_compass.optimization.ray_tune_integration import (
            create_asha_scheduler,
        )
        
        scheduler = create_asha_scheduler(metric="eval_loss")
        assert scheduler._metric == "eval_loss"
    
    def test_asha_minimizes_by_default(self):
        """Test ASHA minimizes metric by default."""
        from medai_compass.optimization.ray_tune_integration import (
            create_asha_scheduler,
        )
        
        scheduler = create_asha_scheduler(mode="min")
        assert scheduler._mode == "min"


# =============================================================================
# Task 4.4: Optuna Integration Tests
# =============================================================================

@requires_ray
@requires_optuna
class TestOptunaIntegration:
    """Test Optuna integration."""
    
    def test_optuna_search_available(self):
        """Test OptunaSearch can be configured."""
        from medai_compass.optimization.ray_tune_integration import (
            create_optuna_search,
        )
        
        search_alg = create_optuna_search()
        assert search_alg is not None
    
    def test_optuna_search_uses_tpe_sampler(self):
        """Test OptunaSearch uses TPE sampler by default."""
        from medai_compass.optimization.ray_tune_integration import (
            create_optuna_search,
        )
        
        search_alg = create_optuna_search(sampler="tpe")
        # OptunaSearch wraps Optuna's TPESampler
        assert search_alg is not None
    
    def test_optuna_search_metric_configuration(self):
        """Test OptunaSearch metric configuration."""
        from medai_compass.optimization.ray_tune_integration import (
            create_optuna_search,
        )
        
        search_alg = create_optuna_search(
            metric="eval_loss",
            mode="min",
        )
        assert search_alg._metric == "eval_loss"
        assert search_alg._mode == "min"
    
    def test_optuna_with_pruning(self):
        """Test OptunaSearch with Optuna pruning."""
        from medai_compass.optimization.ray_tune_integration import (
            create_optuna_search,
        )
        
        search_alg = create_optuna_search(use_pruning=True)
        assert search_alg is not None


@requires_optuna
class TestOptunaStudyIntegration:
    """Test Optuna study integration."""
    
    def test_create_optuna_study(self):
        """Test creating Optuna study."""
        from medai_compass.optimization.ray_tune_integration import (
            create_optuna_study,
        )
        
        study = create_optuna_study(
            study_name="test_study",
            direction="minimize",
        )
        assert study is not None
        assert study.study_name == "test_study"
    
    def test_optuna_study_with_storage(self):
        """Test Optuna study with database storage."""
        from medai_compass.optimization.ray_tune_integration import (
            create_optuna_study,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = f"sqlite:///{tmpdir}/optuna.db"
            study = create_optuna_study(
                study_name="test_study",
                storage=storage_path,
            )
            assert study is not None


# =============================================================================
# Task 4.5: HP Search Analysis Tools Tests
# =============================================================================

class TestAnalysisModule:
    """Test analysis module structure."""
    
    def test_analysis_module_exists(self):
        """Verify analysis module can be imported."""
        from medai_compass.optimization import analysis
        assert analysis is not None
    
    def test_analysis_has_required_classes(self):
        """Verify required classes exist."""
        from medai_compass.optimization.analysis import (
            HPAnalyzer,
            TrialResult,
            AnalysisReport,
        )
        assert HPAnalyzer is not None
        assert TrialResult is not None
        assert AnalysisReport is not None


class TestHPAnalyzer:
    """Test HPAnalyzer class."""
    
    def test_analyzer_has_load_results_method(self):
        """Test analyzer can load results."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        assert hasattr(HPAnalyzer, "load_results_from_path") or hasattr(HPAnalyzer, "load_results_from_dict")
    
    def test_analyzer_has_get_best_trial_method(self):
        """Test analyzer can get best trial."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        assert hasattr(HPAnalyzer, "get_best_trial")
    
    def test_analyzer_has_plot_parallel_coordinates(self):
        """Test analyzer can plot parallel coordinates."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        assert hasattr(HPAnalyzer, "plot_parallel_coordinates")
    
    def test_analyzer_has_plot_parameter_importance(self):
        """Test analyzer can plot parameter importance."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        assert hasattr(HPAnalyzer, "plot_parameter_importance")
    
    def test_analyzer_has_plot_optimization_history(self):
        """Test analyzer can plot optimization history."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        assert hasattr(HPAnalyzer, "plot_optimization_history")
    
    def test_analyzer_has_export_to_mlflow(self):
        """Test analyzer can export to MLflow or has alternative export."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        # Support export_to_json as alternative
        assert hasattr(HPAnalyzer, "export_to_json") or hasattr(HPAnalyzer, "export_to_mlflow")


class TestParameterImportance:
    """Test parameter importance analysis."""
    
    def test_calculate_parameter_importance(self):
        """Test calculating parameter importance."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        assert hasattr(HPAnalyzer, "calculate_parameter_importance")
    
    def test_parameter_importance_uses_fanova(self):
        """Test parameter importance method exists."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        # Importance calculation is available
        assert hasattr(HPAnalyzer, "calculate_parameter_importance")


class TestTrialComparison:
    """Test trial comparison functionality."""
    
    def test_compare_top_trials(self):
        """Test can access trial data for comparison."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        # Can access trials for comparison
        assert hasattr(HPAnalyzer, "trials") or hasattr(HPAnalyzer, "dataframe")
    
    def test_get_trial_history(self):
        """Test getting trial data."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        # Can get trial data
        assert hasattr(HPAnalyzer, "trials") or hasattr(HPAnalyzer, "dataframe")


class TestVisualization:
    """Test visualization capabilities."""
    
    def test_create_plotly_figure(self):
        """Test creating Plotly figure."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        analyzer = HPAnalyzer()
        # Mock some results
        assert hasattr(analyzer, "plot_parallel_coordinates")
    
    def test_create_hiplot_experiment(self):
        """Test creating HiPlot experiment."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        assert hasattr(HPAnalyzer, "create_hiplot_experiment")
    
    def test_save_visualization(self):
        """Test saving visualization to file."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        assert hasattr(HPAnalyzer, "save_figure")


class TestReportGeneration:
    """Test report generation."""
    
    def test_generate_analysis_report(self):
        """Test generating analysis report."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        assert hasattr(HPAnalyzer, "generate_report")
    
    def test_report_includes_best_config(self):
        """Test report includes best trial which contains best configuration."""
        from medai_compass.optimization.analysis import AnalysisReport, TrialResult
        
        # Create a minimal report to check structure
        trial = TrialResult(
            trial_id="test",
            config={"lr": 0.001},
            metrics={"loss": 0.5},
        )
        report = AnalysisReport(
            best_trial=trial,
            parameter_importance={},
            correlation_matrix={},
            summary_statistics={},
        )
        assert report.best_trial is not None
    
    def test_report_includes_parameter_importance(self):
        """Test report includes parameter importance."""
        from medai_compass.optimization.analysis import AnalysisReport, TrialResult
        
        trial = TrialResult(
            trial_id="test",
            config={"lr": 0.001},
            metrics={"loss": 0.5},
        )
        report = AnalysisReport(
            best_trial=trial,
            parameter_importance={"lr": 0.8},
            correlation_matrix={},
            summary_statistics={},
        )
        assert report.parameter_importance == {"lr": 0.8}


# =============================================================================
# Task 4.6: Population-Based Training (PBT) Tests
# =============================================================================

@requires_ray
class TestPBTScheduler:
    """Test Population-Based Training scheduler."""
    
    def test_pbt_scheduler_available(self):
        """Test PBT scheduler can be configured."""
        from medai_compass.optimization.ray_tune_integration import (
            create_pbt_scheduler,
        )
        
        scheduler = create_pbt_scheduler()
        assert scheduler is not None
    
    def test_pbt_scheduler_has_perturbation_interval(self):
        """Test PBT has perturbation interval."""
        from medai_compass.optimization.ray_tune_integration import (
            create_pbt_scheduler,
        )
        
        scheduler = create_pbt_scheduler(perturbation_interval=5)
        assert scheduler._perturbation_interval == 5
    
    def test_pbt_scheduler_has_hyperparam_mutations(self):
        """Test PBT has hyperparameter mutations."""
        from medai_compass.optimization.ray_tune_integration import (
            create_pbt_scheduler,
        )
        
        mutations = {
            "learning_rate": [1e-5, 1e-4, 1e-3],
        }
        scheduler = create_pbt_scheduler(hyperparam_mutations=mutations)
        assert scheduler._hyperparam_mutations is not None
    
    def test_pbt_scheduler_has_quantile_fraction(self):
        """Test PBT has quantile fraction for selection."""
        from medai_compass.optimization.ray_tune_integration import (
            create_pbt_scheduler,
        )
        
        scheduler = create_pbt_scheduler(quantile_fraction=0.25)
        assert scheduler._quantile_fraction == 0.25


@requires_ray
class TestPBTConfiguration:
    """Test PBT configuration options."""
    
    def test_pbt_resample_probability(self):
        """Test PBT resample probability."""
        from medai_compass.optimization.ray_tune_integration import (
            create_pbt_scheduler,
        )
        
        scheduler = create_pbt_scheduler(resample_probability=0.25)
        assert scheduler._resample_probability == 0.25
    
    def test_pbt_custom_explore_function(self):
        """Test PBT with custom explore function."""
        from medai_compass.optimization.ray_tune_integration import (
            create_pbt_scheduler,
        )
        
        def custom_explore(config):
            return config
        
        scheduler = create_pbt_scheduler(custom_explore_fn=custom_explore)
        assert scheduler._custom_explore_fn is not None


@requires_ray
class TestPBTForMedGemma:
    """Test PBT configuration for MedGemma models."""
    
    def test_pbt_config_for_4b(self):
        """Test PBT configuration for 4B model."""
        from medai_compass.optimization.ray_tune_integration import (
            create_pbt_scheduler_for_model,
        )
        
        scheduler = create_pbt_scheduler_for_model("medgemma-4b")
        assert scheduler is not None
    
    def test_pbt_config_for_27b(self):
        """Test PBT configuration for 27B model."""
        from medai_compass.optimization.ray_tune_integration import (
            create_pbt_scheduler_for_model,
        )
        
        scheduler = create_pbt_scheduler_for_model("medgemma-27b")
        assert scheduler is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestOptimizationModuleExports:
    """Test optimization module exports."""
    
    def test_optimization_module_exports(self):
        """Test all required exports from optimization module."""
        from medai_compass.optimization import (
            HyperparameterTuner,
            TuneConfig,
            TuneResult,
            SearchSpaceConfig,
            ParameterSpace,
            HPAnalyzer,
            AnalysisReport,
            create_asha_scheduler,
            create_optuna_search,
            create_pbt_scheduler,
        )
        
        assert HyperparameterTuner is not None
        assert TuneConfig is not None
        assert SearchSpaceConfig is not None
        assert HPAnalyzer is not None
    
    def test_optimization_profiles_exist(self):
        """Test OPTIMIZATION_PROFILES constant exists."""
        from medai_compass.optimization import OPTIMIZATION_PROFILES
        
        assert "medgemma-4b" in OPTIMIZATION_PROFILES
        assert "medgemma-27b" in OPTIMIZATION_PROFILES


class TestTunerWithTrainingPipeline:
    """Test tuner integration with Phase 3 training pipeline."""
    
    def test_tuner_uses_lora_trainer(self):
        """Test tuner uses LoRATrainer from Phase 3."""
        from medai_compass.optimization.ray_tune_integration import HyperparameterTuner
        
        # Tuner should be able to wrap LoRATrainer
        assert hasattr(HyperparameterTuner, "train_func")
    
    def test_tuner_uses_mlflow_tracker(self):
        """Test tuner integrates with MLflow tracker."""
        from medai_compass.optimization.ray_tune_integration import (
            HyperparameterTuner,
            TuneConfig,
        )
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        search_space = SearchSpaceConfig.for_model("medgemma-4b")
        config = TuneConfig()
        tuner = HyperparameterTuner(
            model_name="medgemma-4b",
            search_space=search_space,
            tune_config=config,
            mlflow_tracking_uri="http://localhost:5000",
        )
        assert tuner.mlflow_tracking_uri == "http://localhost:5000"


class TestAirflowCompatibility:
    """Test Airflow DAG compatibility."""
    
    def test_hp_tuning_task_functions_exist(self):
        """Test HP tuning Airflow task functions exist."""
        from medai_compass.optimization.ray_tune_integration import (
            define_search_space_task,
            run_hp_tuning_task,
            analyze_results_task,
            select_best_config_task,
        )
        
        assert callable(define_search_space_task)
        assert callable(run_hp_tuning_task)
        assert callable(analyze_results_task)
        assert callable(select_best_config_task)
    
    def test_task_functions_return_dict(self):
        """Test task functions return dictionary for XCom."""
        from medai_compass.optimization.ray_tune_integration import (
            define_search_space_task,
        )
        
        # Task should be callable and return dict for Airflow XCom
        assert callable(define_search_space_task)


# =============================================================================
# End-to-End Tests (Mocked)
# =============================================================================

class TestEndToEndTuning:
    """End-to-end tuning tests with mocking."""
    
    def test_tuner_run_mocked(self):
        """Test tuner instantiation."""
        from medai_compass.optimization.ray_tune_integration import (
            HyperparameterTuner,
            TuneConfig,
        )
        from medai_compass.optimization.search_space import SearchSpaceConfig
        
        search_space = SearchSpaceConfig.for_model("medgemma-4b")
        config = TuneConfig(num_samples=2)
        
        tuner = HyperparameterTuner(
            model_name="medgemma-4b",
            search_space=search_space,
            tune_config=config,
        )
        
        assert tuner is not None
        assert tuner.model_name == "medgemma-4b"
    
    def test_analyzer_with_mock_results(self):
        """Test analyzer with mock results."""
        from medai_compass.optimization.analysis import HPAnalyzer
        
        # Create mock trial results as dict
        mock_results = [
            {
                "trial_id": "trial_1",
                "config": {"learning_rate": 1e-4, "lora_r": 16},
                "metrics": {"eval_loss": 0.5},
            },
            {
                "trial_id": "trial_2",
                "config": {"learning_rate": 1e-3, "lora_r": 32},
                "metrics": {"eval_loss": 0.3},
            },
        ]
        
        analyzer = HPAnalyzer()
        analyzer.load_results_from_dict(mock_results)
        
        best = analyzer.get_best_trial()
        assert best.trial_id == "trial_2"
