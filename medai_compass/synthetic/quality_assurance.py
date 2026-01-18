"""Synthetic Data Quality Assurance (Task 5.5).

Quality validation for synthetic medical data:
- PHI detection using existing guardrails
- Medical terminology validation
- Clinical consistency checks
- Quality score calculation
- Batch validation with reports

Leverages existing PHI detection from medai_compass.pipelines.phi_detection.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm

logger = logging.getLogger(__name__)


# Medical terminology patterns for validation
MEDICAL_TERMS = {
    "conditions": [
        r"\b(diabetes|hypertension|pneumonia|cardiomegaly|edema)\b",
        r"\b(myocardial infarction|heart failure|atrial fibrillation)\b",
        r"\b(cancer|carcinoma|melanoma|lymphoma|leukemia)\b",
        r"\b(stroke|epilepsy|parkinson|alzheimer|dementia)\b",
        r"\b(asthma|copd|emphysema|bronchitis)\b",
    ],
    "medications": [
        r"\b(aspirin|metformin|lisinopril|atorvastatin)\b",
        r"\b(metoprolol|amlodipine|omeprazole|levothyroxine)\b",
        r"\b(acetaminophen|ibuprofen|naproxen)\b",
        r"\b(antibiotic|antiviral|antifungal|anti-inflammatory)\b",
    ],
    "procedures": [
        r"\b(surgery|biopsy|catheterization|endoscopy)\b",
        r"\b(mri|ct scan|x-ray|ultrasound|echocardiogram)\b",
        r"\b(blood test|urinalysis|culture)\b",
    ],
    "anatomy": [
        r"\b(heart|lung|liver|kidney|brain|pancreas)\b",
        r"\b(artery|vein|muscle|bone|nerve)\b",
        r"\b(chest|abdomen|thorax|pelvis)\b",
    ],
}

# Clinical inconsistency rules
CLINICAL_RULES = [
    {
        "name": "pregnancy_gender",
        "description": "Pregnancy diagnosis requires female patient",
        "condition": lambda r: (
            r.get("diagnosis", "").lower() == "pregnancy"
            and r.get("patient", {}).get("gender", "").lower() == "male"
        ),
        "message": "Male patient cannot have pregnancy diagnosis",
    },
    {
        "name": "pediatric_age",
        "description": "Pediatric conditions require appropriate age",
        "condition": lambda r: (
            "pediatric" in r.get("diagnosis", "").lower()
            and r.get("patient", {}).get("age", 0) > 18
        ),
        "message": "Pediatric diagnosis in adult patient",
    },
    {
        "name": "geriatric_age",
        "description": "Geriatric conditions require appropriate age",
        "condition": lambda r: (
            "geriatric" in r.get("diagnosis", "").lower()
            and r.get("patient", {}).get("age", 100) < 65
        ),
        "message": "Geriatric diagnosis in non-elderly patient",
    },
]


@dataclass
class PHICheckResult:
    """Result of PHI detection check."""
    
    contains_phi: bool
    phi_types: List[str] = field(default_factory=list)
    phi_locations: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    risk_level: str = "none"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contains_phi": self.contains_phi,
            "phi_types": self.phi_types,
            "phi_locations": self.phi_locations,
            "risk_level": self.risk_level,
        }


@dataclass
class TerminologyResult:
    """Result of terminology validation."""
    
    is_valid: bool
    medical_terms_found: List[str] = field(default_factory=list)
    term_categories: Dict[str, int] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class ConsistencyResult:
    """Result of clinical consistency check."""
    
    is_consistent: bool
    inconsistencies: List[str] = field(default_factory=list)
    failed_rules: List[str] = field(default_factory=list)


@dataclass
class QAReport:
    """Quality assurance report for synthetic data."""
    
    total_records: int
    passed_records: int
    failed_records: int
    average_quality_score: float
    phi_detections: int = 0
    terminology_failures: int = 0
    consistency_failures: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)
    phi_check_results: List[Dict[str, Any]] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class SyntheticDataQA:
    """
    Quality assurance for synthetic medical data.
    
    Performs:
    - PHI detection using existing guardrails
    - Medical terminology validation
    - Clinical consistency checking
    - Quality score calculation
    
    Attributes:
        strict_phi: Fail records with any PHI detected
        min_quality_score: Minimum quality score threshold
    """
    
    # PHI patterns (from existing guardrails)
    PHI_PATTERNS = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "mrn": r"\b(MRN|mrn)[:\s]?\d{6,10}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "name": r"\b(Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",
        "address": r"\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr)\b",
    }
    
    def __init__(
        self,
        strict_phi: bool = True,
        min_quality_score: float = 0.7,
    ):
        """
        Initialize the QA system.
        
        Args:
            strict_phi: Fail records with any PHI
            min_quality_score: Minimum quality threshold
        """
        self.strict_phi = strict_phi
        self.min_quality_score = min_quality_score
        
        # Compile patterns
        self._phi_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PHI_PATTERNS.items()
        }
        
        self._medical_patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in MEDICAL_TERMS.items()
        }
        
        logger.info("Initialized SyntheticDataQA")
    
    def check_phi(self, text: str) -> Dict[str, Any]:
        """
        Check text for PHI (Protected Health Information).
        
        Args:
            text: Text to check
            
        Returns:
            PHI check result as dictionary
        """
        phi_types = []
        phi_locations = {}
        
        for phi_type, pattern in self._phi_patterns.items():
            matches = list(pattern.finditer(text))
            
            if matches:
                phi_types.append(phi_type)
                phi_locations[phi_type] = [
                    (m.start(), m.end()) for m in matches
                ]
        
        contains_phi = len(phi_types) > 0
        
        # Determine risk level
        if not contains_phi:
            risk_level = "none"
        elif "ssn" in phi_types:
            risk_level = "critical"
        elif len(phi_types) > 1 or "mrn" in phi_types:
            risk_level = "high"
        else:
            risk_level = "medium"
        
        return {
            "contains_phi": contains_phi,
            "phi_types": phi_types,
            "phi_locations": phi_locations,
            "risk_level": risk_level,
        }
    
    def validate_terminology(self, text: str) -> Dict[str, Any]:
        """
        Validate medical terminology in text.
        
        Args:
            text: Text to validate
            
        Returns:
            Terminology validation result
        """
        terms_found = []
        term_categories = {}
        
        for category, patterns in self._medical_patterns.items():
            category_count = 0
            
            for pattern in patterns:
                matches = pattern.findall(text)
                if matches:
                    terms_found.extend(matches)
                    category_count += len(matches)
            
            if category_count > 0:
                term_categories[category] = category_count
        
        # Valid if at least one medical term found
        is_valid = len(terms_found) > 0
        
        # Calculate confidence based on term diversity
        categories_present = len(term_categories)
        confidence = min(1.0, categories_present / 3)  # Normalize by 3 categories
        
        return {
            "is_valid": is_valid,
            "medical_terms_found": list(set(terms_found)),
            "term_categories": term_categories,
            "confidence": confidence,
        }
    
    def check_clinical_consistency(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check clinical consistency of a record.
        
        Args:
            record: Record to check
            
        Returns:
            Consistency check result
        """
        inconsistencies = []
        failed_rules = []
        
        for rule in CLINICAL_RULES:
            try:
                if rule["condition"](record):
                    inconsistencies.append(rule["message"])
                    failed_rules.append(rule["name"])
            except (KeyError, TypeError):
                # Rule doesn't apply to this record
                continue
        
        return {
            "is_consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies,
            "failed_rules": failed_rules,
        }
    
    def calculate_quality_score(
        self,
        record: Dict[str, Any],
    ) -> float:
        """
        Calculate overall quality score for a record.
        
        Args:
            record: Record to score
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        scores = []
        
        # Get text from record
        text = record.get("text", "")
        if not text and isinstance(record, dict):
            # Try to construct text from other fields
            text = " ".join(str(v) for v in record.values() if isinstance(v, str))
        
        # PHI check (40% weight)
        phi_result = self.check_phi(text)
        phi_score = 0.0 if phi_result["contains_phi"] else 1.0
        scores.append(("phi", phi_score, 0.4))
        
        # Terminology check (30% weight)
        term_result = self.validate_terminology(text)
        term_score = term_result["confidence"] if term_result["is_valid"] else 0.0
        scores.append(("terminology", term_score, 0.3))
        
        # Consistency check (30% weight)
        consistency_result = self.check_clinical_consistency(record)
        consistency_score = 1.0 if consistency_result["is_consistent"] else 0.0
        scores.append(("consistency", consistency_score, 0.3))
        
        # Weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        
        return round(total_score, 3)
    
    def validate_batch(
        self,
        records: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate a batch of records.
        
        Args:
            records: List of records to validate
            show_progress: Show progress bar
            
        Returns:
            Validation report
        """
        total = len(records)
        passed = 0
        scores = []
        phi_detections = 0
        
        pbar = None
        if show_progress:
            pbar = tqdm(total=total, desc="Validating", unit="records")
        
        try:
            for record in records:
                score = self.calculate_quality_score(record)
                scores.append(score)
                
                # Check if passed
                if score >= self.min_quality_score:
                    passed += 1
                
                # Count PHI detections
                text = record.get("text", "")
                if self.check_phi(text)["contains_phi"]:
                    phi_detections += 1
                
                if pbar:
                    pbar.update(1)
        
        finally:
            if pbar:
                pbar.close()
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            "total_records": total,
            "passed_records": passed,
            "failed_records": total - passed,
            "average_quality_score": round(avg_score, 3),
            "phi_detections": phi_detections,
            "pass_rate": round(passed / total, 3) if total > 0 else 0.0,
        }
    
    def generate_report(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive QA report.
        
        Args:
            records: Records to analyze
            
        Returns:
            Full QA report
        """
        # Validate batch
        batch_results = self.validate_batch(records, show_progress=False)
        
        # Detailed analysis
        phi_results = []
        quality_scores = []
        terminology_failures = 0
        consistency_failures = 0
        
        for record in records:
            # PHI check
            text = record.get("text", "")
            phi_result = self.check_phi(text)
            phi_results.append(phi_result)
            
            # Quality score
            score = self.calculate_quality_score(record)
            quality_scores.append(score)
            
            # Terminology
            if not self.validate_terminology(text)["is_valid"]:
                terminology_failures += 1
            
            # Consistency
            if not self.check_clinical_consistency(record)["is_consistent"]:
                consistency_failures += 1
        
        # Generate recommendations
        recommendations = []
        
        if batch_results["phi_detections"] > 0:
            recommendations.append(
                f"Remove {batch_results['phi_detections']} records with PHI detected"
            )
        
        if terminology_failures > len(records) * 0.1:
            recommendations.append(
                "Improve medical terminology coverage in generated text"
            )
        
        if consistency_failures > 0:
            recommendations.append(
                f"Fix {consistency_failures} clinically inconsistent records"
            )
        
        if batch_results["average_quality_score"] < 0.8:
            recommendations.append(
                "Consider regenerating low-quality samples to improve overall quality"
            )
        
        return {
            "summary": batch_results,
            "phi_check_results": phi_results,
            "quality_scores": quality_scores,
            "terminology_failures": terminology_failures,
            "consistency_failures": consistency_failures,
            "recommendations": recommendations,
        }
