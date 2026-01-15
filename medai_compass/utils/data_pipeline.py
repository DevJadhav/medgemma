"""
Synthetic EHR Data Generator using Synthea-generated FHIR bundles.

This module provides utilities for generating and loading synthetic patient data
for testing and development purposes.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class SyntheticPatient:
    """Represents a synthetic patient record."""
    patient_id: str
    name: str
    birth_date: str
    gender: str
    address: Optional[dict] = None
    conditions: list[dict] | None = None
    medications: list[dict] | None = None
    observations: list[dict] | None = None
    encounters: list[dict] | None = None


class SyntheaDataGenerator:
    """
    Generate synthetic EHR data using Synthea or load pre-generated FHIR bundles.
    
    For M3 Mac development, we use pre-generated sample data or mock data
    since Synthea requires Java runtime.
    """
    
    def __init__(self, data_dir: str = "data/synthetic"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_mock_patients(self, count: int = 10) -> list[SyntheticPatient]:
        """
        Generate mock patient data for local testing.
        
        Args:
            count: Number of patients to generate
            
        Returns:
            List of SyntheticPatient objects
        """
        patients = []
        
        # Sample conditions for variety
        sample_conditions = [
            {"code": "38341003", "display": "Hypertension"},
            {"code": "44054006", "display": "Type 2 Diabetes"},
            {"code": "195662009", "display": "Acute Viral Pharyngitis"},
            {"code": "233604007", "display": "Pneumonia"},
            {"code": "59621000", "display": "Essential Hypertension"},
        ]
        
        # Sample medications
        sample_medications = [
            {"code": "197361", "display": "Lisinopril 10 MG"},
            {"code": "860975", "display": "Metformin 500 MG"},
            {"code": "308189", "display": "Omeprazole 20 MG"},
            {"code": "197446", "display": "Atorvastatin 20 MG"},
        ]
        
        for i in range(count):
            patient = SyntheticPatient(
                patient_id=f"synth-{i:04d}",
                name=f"Patient {i}",
                birth_date=f"{1950 + (i % 50)}-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                gender="male" if i % 2 == 0 else "female",
                address={
                    "city": ["Boston", "Cambridge", "Somerville", "Brookline"][i % 4],
                    "state": "MA",
                    "postalCode": f"0{2100 + i:04d}"
                },
                conditions=[sample_conditions[i % len(sample_conditions)]],
                medications=[sample_medications[i % len(sample_medications)]],
                observations=[
                    {
                        "code": "8302-2",
                        "display": "Body Height",
                        "value": 160 + (i % 30),
                        "unit": "cm"
                    },
                    {
                        "code": "29463-7",
                        "display": "Body Weight",
                        "value": 60 + (i % 40),
                        "unit": "kg"
                    },
                    {
                        "code": "8480-6",
                        "display": "Systolic Blood Pressure",
                        "value": 110 + (i % 30),
                        "unit": "mmHg"
                    }
                ],
                encounters=[
                    {
                        "type": "outpatient",
                        "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                        "reason": sample_conditions[i % len(sample_conditions)]["display"]
                    }
                ]
            )
            patients.append(patient)
            
        return patients
    
    def save_as_fhir_bundle(self, patients: list[SyntheticPatient], output_file: str = "bundle.json") -> Path:
        """
        Convert synthetic patients to FHIR R4 Bundle format.
        
        Args:
            patients: List of SyntheticPatient objects
            output_file: Output filename
            
        Returns:
            Path to the saved bundle file
        """
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": []
        }
        
        for patient in patients:
            # Patient resource
            patient_resource = {
                "resource": {
                    "resourceType": "Patient",
                    "id": patient.patient_id,
                    "name": [{"text": patient.name}],
                    "birthDate": patient.birth_date,
                    "gender": patient.gender
                }
            }
            if patient.address:
                patient_resource["resource"]["address"] = [patient.address]
            bundle["entry"].append(patient_resource)
            
            # Condition resources
            if patient.conditions:
                for condition in patient.conditions:
                    bundle["entry"].append({
                        "resource": {
                            "resourceType": "Condition",
                            "id": f"{patient.patient_id}-cond-{condition['code']}",
                            "subject": {"reference": f"Patient/{patient.patient_id}"},
                            "code": {
                                "coding": [{
                                    "system": "http://snomed.info/sct",
                                    "code": condition["code"],
                                    "display": condition["display"]
                                }]
                            }
                        }
                    })
            
            # Observation resources
            if patient.observations:
                for obs in patient.observations:
                    bundle["entry"].append({
                        "resource": {
                            "resourceType": "Observation",
                            "id": f"{patient.patient_id}-obs-{obs['code']}",
                            "subject": {"reference": f"Patient/{patient.patient_id}"},
                            "code": {
                                "coding": [{
                                    "system": "http://loinc.org",
                                    "code": obs["code"],
                                    "display": obs["display"]
                                }]
                            },
                            "valueQuantity": {
                                "value": obs["value"],
                                "unit": obs["unit"]
                            }
                        }
                    })
        
        output_path = self.data_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(bundle, f, indent=2)
            
        return output_path
    
    def load_fhir_bundle(self, bundle_path: str) -> dict[str, Any]:
        """
        Load a FHIR bundle from file.
        
        Args:
            bundle_path: Path to the FHIR bundle JSON file
            
        Returns:
            Parsed FHIR bundle as dictionary
        """
        with open(bundle_path, 'r') as f:
            return json.load(f)
    
    def extract_patients_from_bundle(self, bundle: dict[str, Any]) -> list[dict]:
        """
        Extract patient resources from a FHIR bundle.
        
        Args:
            bundle: Parsed FHIR bundle
            
        Returns:
            List of Patient resources
        """
        patients = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                patients.append(resource)
        return patients


class DataLoadingPipeline:
    """
    Pipeline for loading various medical data formats.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dicom_dataset(self, dataset_path: str) -> list[Path]:
        """
        Load DICOM files from a dataset directory.
        
        Args:
            dataset_path: Path to directory containing DICOM files
            
        Returns:
            List of paths to DICOM files
        """
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            return []
            
        dicom_files = []
        for ext in ["*.dcm", "*.DCM", "*.dicom"]:
            dicom_files.extend(dataset_dir.rglob(ext))
            
        return sorted(dicom_files)
    
    def create_dataset_manifest(self, dataset_name: str, files: list[Path]) -> dict:
        """
        Create a manifest file for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            files: List of file paths in the dataset
            
        Returns:
            Manifest dictionary
        """
        manifest = {
            "dataset_name": dataset_name,
            "file_count": len(files),
            "files": [str(f) for f in files],
            "created_at": "2026-01-15T11:00:00Z"
        }
        
        manifest_path = self.data_dir / f"{dataset_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        return manifest


# Convenience function for quick data generation
def generate_sample_data(output_dir: str = "data/synthetic", patient_count: int = 10) -> Path:
    """
    Generate sample synthetic data for development/testing.
    
    Args:
        output_dir: Directory to save generated data
        patient_count: Number of patients to generate
        
    Returns:
        Path to generated FHIR bundle
    """
    generator = SyntheaDataGenerator(output_dir)
    patients = generator.generate_mock_patients(patient_count)
    return generator.save_as_fhir_bundle(patients, "sample_patients.json")
