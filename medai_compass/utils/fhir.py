"""FHIR client utilities for MedAI Compass.

Provides:
- FHIRClient for EHR data retrieval
- Helper functions for creating FHIR resources
- Patient context aggregation
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


class FHIRConnectionError(Exception):
    """Raised when FHIR server connection fails."""
    pass


class FHIRClient:
    """
    FHIR R4 client for EHR integration.
    
    Provides methods to:
    - Retrieve patient information
    - Get conditions, medications, allergies
    - Aggregate patient context for AI analysis
    """
    
    def __init__(
        self, 
        base_url: str,
        auth_token: str | None = None,
        timeout: int = 30
    ):
        """
        Initialize FHIR client.
        
        Args:
            base_url: FHIR server base URL (e.g., http://localhost:8080/fhir)
            auth_token: Optional authorization token
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout
        
    def _get_headers(self) -> dict[str, str]:
        """Get request headers including auth if configured."""
        headers = {
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        }
        if self.auth_token:
            headers["Authorization"] = self.auth_token
        return headers
    
    def _get_resource(self, resource_type: str, resource_id: str) -> dict[str, Any]:
        """
        Get a specific FHIR resource by ID.
        
        Args:
            resource_type: FHIR resource type (e.g., 'Patient')
            resource_id: Resource ID
            
        Returns:
            FHIR resource as dictionary
        """
        import httpx
        
        url = f"{self.base_url}/{resource_type}/{resource_id}"
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, headers=self._get_headers())
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError as e:
            raise FHIRConnectionError(f"Failed to connect to FHIR server: {e}")
        except httpx.HTTPStatusError as e:
            raise FHIRConnectionError(f"FHIR server error: {e}")
    
    def _search_resources(
        self, 
        resource_type: str, 
        params: dict[str, str]
    ) -> list[dict[str, Any]]:
        """
        Search for FHIR resources.
        
        Args:
            resource_type: FHIR resource type
            params: Search parameters
            
        Returns:
            List of matching FHIR resources
        """
        import httpx
        
        url = f"{self.base_url}/{resource_type}"
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(url, params=params, headers=self._get_headers())
                response.raise_for_status()
                bundle = response.json()
                
                # Extract resources from Bundle
                entries = bundle.get("entry", [])
                return [entry.get("resource", {}) for entry in entries]
        except httpx.ConnectError as e:
            raise FHIRConnectionError(f"Failed to connect to FHIR server: {e}")
        except httpx.HTTPStatusError as e:
            raise FHIRConnectionError(f"FHIR server error: {e}")
    
    def get_patient(self, patient_id: str) -> dict[str, Any]:
        """
        Get patient demographic information.
        
        Args:
            patient_id: Patient resource ID
            
        Returns:
            Patient FHIR resource
        """
        return self._get_resource("Patient", patient_id)
    
    def get_patient_conditions(self, patient_id: str) -> list[dict[str, Any]]:
        """
        Get patient's active conditions/diagnoses.
        
        Args:
            patient_id: Patient resource ID
            
        Returns:
            List of Condition resources
        """
        return self._search_resources(
            "Condition",
            {"patient": patient_id, "clinical-status": "active"}
        )
    
    def get_patient_medications(self, patient_id: str) -> list[dict[str, Any]]:
        """
        Get patient's active medications.
        
        Args:
            patient_id: Patient resource ID
            
        Returns:
            List of MedicationRequest resources
        """
        return self._search_resources(
            "MedicationRequest",
            {"patient": patient_id, "status": "active"}
        )
    
    def get_patient_allergies(self, patient_id: str) -> list[dict[str, Any]]:
        """
        Get patient's allergy intolerances.
        
        Args:
            patient_id: Patient resource ID
            
        Returns:
            List of AllergyIntolerance resources
        """
        return self._search_resources(
            "AllergyIntolerance",
            {"patient": patient_id}
        )
    
    def get_patient_context(self, patient_id: str) -> dict[str, Any]:
        """
        Aggregate all relevant patient context for AI analysis.
        
        Args:
            patient_id: Patient resource ID
            
        Returns:
            Aggregated context including patient, conditions, medications, allergies
        """
        return {
            "patient": self.get_patient(patient_id),
            "conditions": self.get_patient_conditions(patient_id),
            "medications": self.get_patient_medications(patient_id),
            "allergies": self.get_patient_allergies(patient_id),
        }


def create_diagnostic_report(
    patient_id: str,
    study_id: str,
    findings: list[str],
    impression: str,
    performer_id: str,
    status: str = "preliminary",
    modality: str = "DX"
) -> dict[str, Any]:
    """
    Create a FHIR DiagnosticReport resource for AI-generated analysis.
    
    Args:
        patient_id: Patient resource ID
        study_id: ImagingStudy resource ID
        findings: List of findings texts
        impression: Overall impression/conclusion
        performer_id: ID of the performer (AI system)
        status: Report status (preliminary or final)
        modality: Imaging modality code
        
    Returns:
        FHIR DiagnosticReport resource
    """
    report_id = str(uuid4())
    
    return {
        "resourceType": "DiagnosticReport",
        "id": report_id,
        "status": status,
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Radiology"
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {
                    "system": "http://loinc.org",
                    "code": "18748-4",
                    "display": "Diagnostic imaging study"
                }
            ]
        },
        "subject": {
            "reference": f"Patient/{patient_id}"
        },
        "issued": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "performer": [
            {
                "reference": f"Device/{performer_id}",
                "display": "MedAI Compass AI System"
            }
        ],
        "imagingStudy": [
            {
                "reference": f"ImagingStudy/{study_id}"
            }
        ],
        "conclusion": impression,
        "presentedForm": [
            {
                "contentType": "text/plain",
                "data": _encode_findings(findings)
            }
        ],
        "extension": [
            {
                "url": "http://medai-compass.com/fhir/extensions/ai-generated",
                "valueBoolean": True
            },
            {
                "url": "http://medai-compass.com/fhir/extensions/ai-model",
                "valueString": "MedGemma 27B"
            }
        ]
    }


def _encode_findings(findings: list[str]) -> str:
    """Base64 encode findings for FHIR presentedForm."""
    import base64
    
    text = "\n".join(f"- {finding}" for finding in findings)
    return base64.b64encode(text.encode()).decode()
