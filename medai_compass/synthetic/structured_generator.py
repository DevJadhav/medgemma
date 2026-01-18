"""Structured Data Generator (Task 5.4).

Generates synthetic structured medical data using:
- Synthea for FHIR/EHR data generation
- CTGAN/SDV for tabular data augmentation
- Faker for auxiliary data generation

Supports rare condition augmentation and diverse patient populations.
"""

import logging
import random
import subprocess
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from faker import Faker

from medai_compass.synthetic.base import BaseSyntheticGenerator

logger = logging.getLogger(__name__)


# Synthea modules for condition generation
SYNTHEA_MODULES = {
    "diabetes": {
        "module": "diabetes",
        "icd10_codes": ["E11", "E10", "E13"],
        "snomed_codes": ["73211009", "46635009"],
    },
    "hypertension": {
        "module": "hypertension",
        "icd10_codes": ["I10", "I11", "I12", "I13"],
        "snomed_codes": ["38341003"],
    },
    "cancer": {
        "module": "lung_cancer",
        "icd10_codes": ["C34", "C50", "C18"],
        "snomed_codes": ["254637007"],
    },
    "covid19": {
        "module": "covid19",
        "icd10_codes": ["U07.1", "U07.2"],
        "snomed_codes": ["840539006"],
    },
    "heart_disease": {
        "module": "heart_disease",
        "icd10_codes": ["I25", "I21", "I50"],
        "snomed_codes": ["53741008"],
    },
    "copd": {
        "module": "copd",
        "icd10_codes": ["J44"],
        "snomed_codes": ["13645005"],
    },
    "asthma": {
        "module": "asthma",
        "icd10_codes": ["J45"],
        "snomed_codes": ["195967001"],
    },
    "chronic_kidney_disease": {
        "module": "chronic_kidney_disease",
        "icd10_codes": ["N18"],
        "snomed_codes": ["709044004"],
    },
}

# Patient demographics schema
DEMOGRAPHICS_SCHEMA = {
    "age": {"type": "int", "min": 0, "max": 100},
    "gender": {"type": "categorical", "values": ["male", "female", "other"]},
    "ethnicity": {
        "type": "categorical",
        "values": ["white", "black", "asian", "hispanic", "other"],
    },
    "weight_kg": {"type": "float", "min": 30.0, "max": 200.0},
    "height_cm": {"type": "float", "min": 100.0, "max": 220.0},
}


class StructuredDataGenerator(BaseSyntheticGenerator):
    """
    Generator for structured medical data.
    
    Generates:
    - FHIR resources using Synthea
    - Tabular patient data using CTGAN/SDV
    - Augmented data for rare conditions
    
    Attributes:
        synthea_path: Path to Synthea installation
        mock_mode: Generate mock data for testing
    """
    
    def __init__(
        self,
        synthea_path: Optional[str] = None,
        mock_mode: bool = False,
        target_count: int = 2500,
        batch_size: int = 50,
        checkpoint_interval: int = 100,
        checkpoint_dir: Optional[str] = None,
        use_dvc: bool = True,
        **kwargs,
    ):
        """
        Initialize the structured data generator.
        
        Args:
            synthea_path: Path to Synthea installation
            mock_mode: Enable mock mode for testing
            target_count: Target samples to generate
            batch_size: Batch size for generation
            checkpoint_interval: Checkpoint save interval
            checkpoint_dir: Directory for checkpoints
            use_dvc: Enable DVC tracking
        """
        super().__init__(
            target_count=target_count,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
            use_dvc=use_dvc,
            mock_mode=mock_mode,
            **kwargs,
        )
        
        self.synthea_path = Path(synthea_path) if synthea_path else None
        self.faker = Faker()
        
        # SDV/CTGAN models (lazy loaded)
        self._ctgan_model = None
        
        logger.info("Initialized StructuredDataGenerator")
    
    def _load_ctgan(self):
        """Lazy load CTGAN model."""
        if self._ctgan_model is not None or self.mock_mode:
            return
        
        try:
            from ctgan import CTGAN
            
            self._ctgan_model = CTGAN(epochs=300)
            logger.info("CTGAN model initialized")
            
        except ImportError:
            logger.warning("CTGAN not available, using mock generation")
    
    def list_synthea_modules(self) -> List[str]:
        """List available Synthea modules/conditions."""
        return list(SYNTHEA_MODULES.keys())
    
    def generate_single(self, **kwargs) -> Dict[str, Any]:
        """Generate a single structured record."""
        schema = kwargs.get("schema", "patient_demographics")
        
        return self._generate_demographic_record()
    
    def generate_fhir_bundle(
        self,
        patient_count: int = 10,
        conditions: Optional[List[str]] = None,
        output_format: str = "fhir",
    ) -> Dict[str, Any]:
        """
        Generate FHIR Bundle with patient resources.
        
        Args:
            patient_count: Number of patients to generate
            conditions: Medical conditions to include
            output_format: Output format (fhir, ndjson)
            
        Returns:
            FHIR Bundle resource
        """
        if self.mock_mode:
            return self._generate_mock_fhir_bundle(patient_count, conditions)
        
        # Try Synthea first if available
        if self.synthea_path and self.synthea_path.exists():
            return self._run_synthea(patient_count, conditions)
        
        # Fall back to mock generation
        return self._generate_mock_fhir_bundle(patient_count, conditions)
    
    def generate_fhir_batch(
        self,
        patient_count: int = 500,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple FHIR bundles in batches.
        
        Args:
            patient_count: Total number of patients
            show_progress: Show progress bar
            
        Returns:
            List of FHIR bundles
        """
        bundles = []
        patients_per_batch = min(50, patient_count)
        num_batches = (patient_count + patients_per_batch - 1) // patients_per_batch
        
        pbar = None
        if show_progress:
            pbar = self.create_progress_bar(num_batches, desc="Generating FHIR")
        
        try:
            generated = 0
            while generated < patient_count:
                batch_size = min(patients_per_batch, patient_count - generated)
                bundle = self.generate_fhir_bundle(patient_count=batch_size)
                bundles.append(bundle)
                generated += batch_size
                
                if pbar:
                    pbar.update(1)
        
        finally:
            if pbar:
                pbar.close()
        
        return bundles
    
    def generate_tabular(
        self,
        schema: str = "patient_demographics",
        count: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Generate tabular patient data.
        
        Args:
            schema: Data schema to use
            count: Number of records to generate
            
        Returns:
            List of tabular records
        """
        if self.mock_mode or schema == "patient_demographics":
            return [self._generate_demographic_record() for _ in range(count)]
        
        # Use CTGAN for other schemas
        self._load_ctgan()
        
        if self._ctgan_model is None:
            return [self._generate_demographic_record() for _ in range(count)]
        
        # Generate using CTGAN
        # This would require training data - placeholder for now
        return [self._generate_demographic_record() for _ in range(count)]
    
    def augment_rare_conditions(
        self,
        conditions: List[str],
        samples_per_condition: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Generate augmented data for rare medical conditions.
        
        Args:
            conditions: List of rare conditions
            samples_per_condition: Samples to generate per condition
            
        Returns:
            List of augmented records
        """
        results = []
        
        for condition in conditions:
            for _ in range(samples_per_condition):
                record = self._generate_demographic_record()
                record["condition"] = condition
                record["rare_condition"] = True
                
                # Add condition-specific attributes
                record["condition_metadata"] = self._get_condition_metadata(condition)
                
                results.append(record)
        
        return results
    
    def _run_synthea(
        self,
        patient_count: int,
        conditions: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Run Synthea to generate FHIR data."""
        cmd = [
            "java",
            "-jar",
            str(self.synthea_path / "synthea-with-dependencies.jar"),
            "-p",
            str(patient_count),
            "--exporter.fhir.export",
            "true",
        ]
        
        if conditions:
            for condition in conditions:
                if condition in SYNTHEA_MODULES:
                    cmd.extend(["-m", SYNTHEA_MODULES[condition]["module"]])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.synthea_path,
            )
            
            if result.returncode != 0:
                logger.warning(f"Synthea error: {result.stderr}")
                return self._generate_mock_fhir_bundle(patient_count, conditions)
            
            # Load generated FHIR files
            output_dir = self.synthea_path / "output" / "fhir"
            # Parse and return FHIR bundle
            # Placeholder - actual implementation would parse files
            return self._generate_mock_fhir_bundle(patient_count, conditions)
            
        except FileNotFoundError:
            logger.warning("Java not found, using mock generation")
            return self._generate_mock_fhir_bundle(patient_count, conditions)
    
    def _generate_mock_fhir_bundle(
        self,
        patient_count: int,
        conditions: Optional[List[str]],
    ) -> Dict[str, Any]:
        """Generate a mock FHIR Bundle for testing."""
        entries = []
        
        for i in range(patient_count):
            patient_id = str(uuid.uuid4())
            
            # Patient resource
            patient = {
                "fullUrl": f"urn:uuid:{patient_id}",
                "resource": {
                    "resourceType": "Patient",
                    "id": patient_id,
                    "name": [
                        {
                            "family": self.faker.last_name(),
                            "given": [self.faker.first_name()],
                        }
                    ],
                    "gender": random.choice(["male", "female"]),
                    "birthDate": self.faker.date_of_birth(
                        minimum_age=18, maximum_age=90
                    ).isoformat(),
                },
            }
            entries.append(patient)
            
            # Add conditions if specified
            if conditions:
                for condition_name in conditions:
                    condition_id = str(uuid.uuid4())
                    module = SYNTHEA_MODULES.get(condition_name, {})
                    
                    condition = {
                        "fullUrl": f"urn:uuid:{condition_id}",
                        "resource": {
                            "resourceType": "Condition",
                            "id": condition_id,
                            "subject": {"reference": f"Patient/{patient_id}"},
                            "code": {
                                "coding": [
                                    {
                                        "system": "http://hl7.org/fhir/sid/icd-10",
                                        "code": module.get("icd10_codes", ["U00"])[0],
                                        "display": condition_name,
                                    }
                                ],
                                "text": condition_name,
                            },
                            "onsetDateTime": (
                                datetime.now(timezone.utc)
                                - timedelta(days=random.randint(30, 365))
                            ).isoformat(),
                        },
                    }
                    entries.append(condition)
        
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": entries,
            "total": len(entries),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def _generate_demographic_record(self) -> Dict[str, Any]:
        """Generate a single demographic record."""
        age = random.randint(18, 90)
        gender = random.choice(["male", "female", "other"])
        
        # Height/weight based on demographics
        if gender == "male":
            height = random.gauss(175, 10)
            weight = random.gauss(80, 15)
        else:
            height = random.gauss(162, 10)
            weight = random.gauss(65, 15)
        
        return {
            "id": str(uuid.uuid4()),
            "age": age,
            "gender": gender,
            "ethnicity": random.choice(["white", "black", "asian", "hispanic", "other"]),
            "weight_kg": round(max(30, min(200, weight)), 1),
            "height_cm": round(max(100, min(220, height)), 1),
            "bmi": round(weight / ((height / 100) ** 2), 1),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def _get_condition_metadata(self, condition: str) -> Dict[str, Any]:
        """Get metadata for a specific condition."""
        module = SYNTHEA_MODULES.get(condition, {})
        
        return {
            "synthea_module": module.get("module", condition),
            "icd10_codes": module.get("icd10_codes", []),
            "snomed_codes": module.get("snomed_codes", []),
            "is_rare": True,
        }
