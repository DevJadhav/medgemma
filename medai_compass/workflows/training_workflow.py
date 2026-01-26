"""
Training Workflow using Ray Workflows.

Provides end-to-end ML pipeline as a workflow DAG.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStepResult:
    """Result from a workflow step."""

    status: str  # success, failed, skipped
    output: Dict[str, Any]
    error: Optional[str] = None


class TrainingWorkflow:
    """
    End-to-end training workflow using Ray Workflows.

    Pipeline steps:
    1. Download data
    2. Preprocess data
    3. Train model
    4. Evaluate model
    5. Deploy model (if evaluation passes)
    """

    def __init__(self, cfg: "DictConfig"):
        """
        Initialize training workflow.

        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg
        self.workflow_id = f"{cfg.project.name}-workflow"

    def download_data(self) -> WorkflowStepResult:
        """Download and prepare data."""
        try:
            logger.info("Downloading data...")

            # Use Modal data download if configured
            if self.cfg.compute.backend == "modal":
                from medai_compass.modal.data_download import download_datasets

                result = download_datasets(
                    datasets=["medqa", "pubmedqa", "medmcqa"],
                    output_path="/data/combined_medical",
                )
            else:
                # Local data preparation
                result = {"status": "success", "path": self.cfg.data.path}

            return WorkflowStepResult(
                status="success",
                output={"data_path": result.get("path", self.cfg.data.path)},
            )

        except Exception as e:
            logger.error(f"Data download failed: {e}")
            return WorkflowStepResult(
                status="failed",
                output={},
                error=str(e),
            )

    def preprocess_data(self, data_result: WorkflowStepResult) -> WorkflowStepResult:
        """Preprocess data for training."""
        if data_result.status != "success":
            return WorkflowStepResult(
                status="skipped",
                output={},
                error="Previous step failed",
            )

        try:
            logger.info("Preprocessing data...")

            from medai_compass.data.ray_data_pipeline import create_ray_data_pipeline

            pipeline = create_ray_data_pipeline(self.cfg)
            dataset = pipeline.get_train_dataset()

            return WorkflowStepResult(
                status="success",
                output={
                    "processed_path": data_result.output["data_path"],
                    "num_examples": getattr(dataset, "count", lambda: 0)(),
                },
            )

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return WorkflowStepResult(
                status="failed",
                output={},
                error=str(e),
            )

    def train_model(self, preprocess_result: WorkflowStepResult) -> WorkflowStepResult:
        """Train the model."""
        if preprocess_result.status != "success":
            return WorkflowStepResult(
                status="skipped",
                output={},
                error="Previous step failed",
            )

        try:
            logger.info("Training model...")

            if self.cfg.compute.backend == "modal":
                # Use Modal training
                from medai_compass.modal.training_app import train

                result = train.remote(
                    model_name=self.cfg.model.name,
                    max_steps=self.cfg.training.args.max_steps,
                )
            else:
                # Use local/Ray training
                from medai_compass.training.ray_trainer import MedGemmaTrainer

                trainer = MedGemmaTrainer(self.cfg)
                result = trainer.train()

            return WorkflowStepResult(
                status="success",
                output={
                    "checkpoint_path": result.get("checkpoint_path", "/checkpoints/final"),
                    "metrics": result.get("metrics", {}),
                },
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return WorkflowStepResult(
                status="failed",
                output={},
                error=str(e),
            )

    def evaluate_model(self, train_result: WorkflowStepResult) -> WorkflowStepResult:
        """Evaluate the trained model."""
        if train_result.status != "success":
            return WorkflowStepResult(
                status="skipped",
                output={},
                error="Previous step failed",
            )

        try:
            logger.info("Evaluating model...")

            # Run evaluation benchmarks
            from medai_compass.evaluation import run_evaluation

            eval_result = run_evaluation(
                model_path=train_result.output["checkpoint_path"],
                benchmarks=["medqa", "pubmedqa"],
            )

            # Check quality gates
            thresholds = self.cfg.evaluation.quality_thresholds
            passed = all(
                eval_result.get(k, 0) >= v
                for k, v in thresholds.items()
                if hasattr(thresholds, k)
            )

            return WorkflowStepResult(
                status="success",
                output={
                    "metrics": eval_result,
                    "quality_gate_passed": passed,
                },
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return WorkflowStepResult(
                status="failed",
                output={},
                error=str(e),
            )

    def deploy_model(
        self,
        train_result: WorkflowStepResult,
        eval_result: WorkflowStepResult,
    ) -> WorkflowStepResult:
        """Deploy the model if evaluation passed."""
        if eval_result.status != "success":
            return WorkflowStepResult(
                status="skipped",
                output={},
                error="Evaluation step failed",
            )

        if not eval_result.output.get("quality_gate_passed", False):
            return WorkflowStepResult(
                status="skipped",
                output={},
                error="Quality gate not passed",
            )

        try:
            logger.info("Deploying model...")

            # Deploy to inference service
            if self.cfg.compute.backend == "modal":
                from medai_compass.modal.app import deploy_inference

                endpoint = deploy_inference(
                    model_path=train_result.output["checkpoint_path"],
                )
            else:
                from medai_compass.serving.ray_serve_app import deploy_model

                endpoint = deploy_model(
                    model_path=train_result.output["checkpoint_path"],
                )

            return WorkflowStepResult(
                status="success",
                output={"endpoint": endpoint},
            )

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return WorkflowStepResult(
                status="failed",
                output={},
                error=str(e),
            )

    def run(self) -> Dict[str, Any]:
        """
        Run the complete training workflow.

        Returns:
            Workflow results
        """
        logger.info(f"Starting workflow: {self.workflow_id}")

        # Execute workflow steps
        data_result = self.download_data()
        preprocess_result = self.preprocess_data(data_result)
        train_result = self.train_model(preprocess_result)
        eval_result = self.evaluate_model(train_result)
        deploy_result = self.deploy_model(train_result, eval_result)

        return {
            "workflow_id": self.workflow_id,
            "steps": {
                "download": data_result.status,
                "preprocess": preprocess_result.status,
                "train": train_result.status,
                "evaluate": eval_result.status,
                "deploy": deploy_result.status,
            },
            "final_status": deploy_result.status,
            "outputs": {
                "checkpoint": train_result.output.get("checkpoint_path"),
                "metrics": eval_result.output.get("metrics"),
                "endpoint": deploy_result.output.get("endpoint"),
            },
        }


def create_training_workflow(cfg: "DictConfig") -> TrainingWorkflow:
    """Create training workflow from config."""
    return TrainingWorkflow(cfg)


def run_training_pipeline(cfg: "DictConfig") -> Dict[str, Any]:
    """
    Run complete training pipeline.

    This is the main entry point for the training workflow.

    Args:
        cfg: Hydra configuration

    Returns:
        Pipeline results
    """
    workflow = create_training_workflow(cfg)
    return workflow.run()
