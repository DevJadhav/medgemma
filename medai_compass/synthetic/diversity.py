"""Diversity Analysis Tools (Task 5.7).

Analyzes diversity of synthetic medical data:
- Demographic distribution analysis
- Medical condition coverage
- Specialty coverage analysis
- Gap identification
- Diversity scoring

Ensures balanced synthetic datasets for training.
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Target distributions for balanced datasets
DEFAULT_TARGET_DISTRIBUTIONS = {
    "gender": {
        "male": 0.49,
        "female": 0.50,
        "other": 0.01,
    },
    "age_groups": {
        "18-30": 0.20,
        "31-50": 0.30,
        "51-70": 0.30,
        "70+": 0.20,
    },
    "ethnicity": {
        "white": 0.60,
        "black": 0.13,
        "hispanic": 0.18,
        "asian": 0.06,
        "other": 0.03,
    },
}


@dataclass
class DemographicDistribution:
    """Distribution of demographic attributes."""
    
    age_distribution: Dict[str, int] = field(default_factory=dict)
    gender_distribution: Dict[str, int] = field(default_factory=dict)
    ethnicity_distribution: Dict[str, int] = field(default_factory=dict)
    total_records: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "age_distribution": self.age_distribution,
            "gender_distribution": self.gender_distribution,
            "ethnicity_distribution": self.ethnicity_distribution,
            "total_records": self.total_records,
        }


@dataclass
class CoverageGap:
    """Identified coverage gap in data."""
    
    category: str
    attribute: str
    expected: float
    actual: float
    deficit: float
    severity: str  # low, medium, high
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "attribute": self.attribute,
            "expected": self.expected,
            "actual": self.actual,
            "deficit": self.deficit,
            "severity": self.severity,
        }


@dataclass
class DiversityReport:
    """Comprehensive diversity analysis report."""
    
    demographics: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, int] = field(default_factory=dict)
    specialties: Dict[str, int] = field(default_factory=dict)
    diversity_score: float = 0.0
    coverage_gaps: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "demographics": self.demographics,
            "conditions": self.conditions,
            "specialties": self.specialties,
            "diversity_score": self.diversity_score,
            "coverage_gaps": self.coverage_gaps,
            "recommendations": self.recommendations,
        }


class DiversityAnalyzer:
    """
    Analyzer for synthetic data diversity.
    
    Analyzes:
    - Demographic distributions (age, gender, ethnicity)
    - Medical condition coverage
    - Specialty coverage
    - Identifies coverage gaps
    - Calculates diversity scores
    
    Attributes:
        target_distributions: Target distributions for balanced data
    """
    
    def __init__(
        self,
        target_distributions: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Initialize the diversity analyzer.
        
        Args:
            target_distributions: Target distributions for each category
        """
        self.target_distributions = target_distributions or DEFAULT_TARGET_DISTRIBUTIONS
        
        logger.info("Initialized DiversityAnalyzer")
    
    def analyze_demographics(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze demographic distribution of records.
        
        Args:
            records: List of records with patient data
            
        Returns:
            Demographic distribution analysis
        """
        age_dist = defaultdict(int)
        gender_dist = defaultdict(int)
        ethnicity_dist = defaultdict(int)
        
        for record in records:
            patient = record.get("patient", record)
            
            # Age
            age = patient.get("age")
            if age is not None:
                age_group = self._get_age_group(age)
                age_dist[age_group] += 1
            
            # Gender
            gender = patient.get("gender", "unknown")
            gender_dist[gender.lower()] += 1
            
            # Ethnicity
            ethnicity = patient.get("ethnicity", "unknown")
            ethnicity_dist[ethnicity.lower()] += 1
        
        return {
            "age_distribution": dict(age_dist),
            "gender_distribution": dict(gender_dist),
            "ethnicity_distribution": dict(ethnicity_dist),
            "total_records": len(records),
        }
    
    def analyze_conditions(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Analyze medical condition distribution.
        
        Args:
            records: List of records
            
        Returns:
            Condition distribution (condition -> count)
        """
        condition_counts = Counter()
        
        for record in records:
            condition = record.get("condition")
            if condition:
                condition_counts[condition.lower()] += 1
            
            # Also check diagnosis field
            diagnosis = record.get("diagnosis")
            if diagnosis:
                condition_counts[diagnosis.lower()] += 1
        
        return dict(condition_counts)
    
    def analyze_specialty_coverage(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Analyze specialty coverage in records.
        
        Args:
            records: List of records
            
        Returns:
            Specialty distribution (specialty -> count)
        """
        specialty_counts = Counter()
        
        for record in records:
            specialty = record.get("specialty")
            if specialty:
                specialty_counts[specialty.lower()] += 1
        
        return dict(specialty_counts)
    
    def identify_gaps(
        self,
        records: List[Dict[str, Any]],
        target_distribution: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Identify coverage gaps compared to target distribution.
        
        Args:
            records: List of records
            target_distribution: Target distribution to compare against
            
        Returns:
            Dictionary of identified gaps
        """
        target = target_distribution or self.target_distributions
        gaps = {}
        
        # Analyze current distributions
        demographics = self.analyze_demographics(records)
        total = len(records)
        
        if total == 0:
            return {"error": "No records to analyze"}
        
        # Check gender distribution
        gender_gaps = {"missing": [], "underrepresented": []}
        gender_dist = demographics["gender_distribution"]
        
        for gender, target_pct in target.get("gender", {}).items():
            actual_count = gender_dist.get(gender, 0)
            actual_pct = actual_count / total if total > 0 else 0
            
            if actual_count == 0:
                gender_gaps["missing"].append(gender)
            elif actual_pct < target_pct * 0.5:  # Less than 50% of target
                gender_gaps["underrepresented"].append(
                    {"attribute": gender, "actual": actual_pct, "target": target_pct}
                )
        
        if gender_gaps["missing"] or gender_gaps["underrepresented"]:
            gaps["gender"] = gender_gaps
        
        # Check age distribution
        age_gaps = {"missing": [], "underrepresented": []}
        age_dist = demographics["age_distribution"]
        
        for age_group, target_pct in target.get("age_groups", {}).items():
            actual_count = age_dist.get(age_group, 0)
            actual_pct = actual_count / total if total > 0 else 0
            
            if actual_count == 0:
                age_gaps["missing"].append(age_group)
            elif actual_pct < target_pct * 0.5:
                age_gaps["underrepresented"].append(
                    {"attribute": age_group, "actual": actual_pct, "target": target_pct}
                )
        
        if age_gaps["missing"] or age_gaps["underrepresented"]:
            gaps["age_groups"] = age_gaps
        
        # Check ethnicity distribution
        ethnicity_gaps = {"missing": [], "underrepresented": []}
        ethnicity_dist = demographics["ethnicity_distribution"]
        
        for ethnicity, target_pct in target.get("ethnicity", {}).items():
            actual_count = ethnicity_dist.get(ethnicity, 0)
            actual_pct = actual_count / total if total > 0 else 0
            
            if actual_count == 0:
                ethnicity_gaps["missing"].append(ethnicity)
            elif actual_pct < target_pct * 0.5:
                ethnicity_gaps["underrepresented"].append(
                    {"attribute": ethnicity, "actual": actual_pct, "target": target_pct}
                )
        
        if ethnicity_gaps["missing"] or ethnicity_gaps["underrepresented"]:
            gaps["ethnicity"] = ethnicity_gaps
        
        return gaps
    
    def calculate_diversity_score(
        self,
        records: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate overall diversity score.
        
        Score based on:
        - Distribution evenness (entropy-based)
        - Coverage of categories
        - Alignment with target distributions
        
        Args:
            records: List of records
            
        Returns:
            Diversity score between 0.0 and 1.0
        """
        if not records:
            return 0.0
        
        scores = []
        
        # Gender diversity
        demographics = self.analyze_demographics(records)
        gender_dist = demographics["gender_distribution"]
        gender_score = self._calculate_distribution_score(
            gender_dist,
            self.target_distributions.get("gender", {}),
        )
        scores.append(gender_score)
        
        # Age diversity
        age_dist = demographics["age_distribution"]
        age_score = self._calculate_distribution_score(
            age_dist,
            self.target_distributions.get("age_groups", {}),
        )
        scores.append(age_score)
        
        # Ethnicity diversity
        ethnicity_dist = demographics["ethnicity_distribution"]
        ethnicity_score = self._calculate_distribution_score(
            ethnicity_dist,
            self.target_distributions.get("ethnicity", {}),
        )
        scores.append(ethnicity_score)
        
        # Average score
        return round(sum(scores) / len(scores), 3) if scores else 0.0
    
    def generate_report(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive diversity report.
        
        Args:
            records: List of records
            
        Returns:
            Full diversity report
        """
        # Analyze all aspects
        demographics = self.analyze_demographics(records)
        conditions = self.analyze_conditions(records)
        specialties = self.analyze_specialty_coverage(records)
        gaps = self.identify_gaps(records)
        diversity_score = self.calculate_diversity_score(records)
        
        # Generate recommendations
        recommendations = []
        
        if gaps:
            for category, category_gaps in gaps.items():
                if category_gaps.get("missing"):
                    missing = ", ".join(category_gaps["missing"])
                    recommendations.append(
                        f"Add samples for missing {category}: {missing}"
                    )
                
                if category_gaps.get("underrepresented"):
                    for item in category_gaps["underrepresented"]:
                        recommendations.append(
                            f"Increase {category} representation for {item['attribute']} "
                            f"(current: {item['actual']:.1%}, target: {item['target']:.1%})"
                        )
        
        if diversity_score < 0.5:
            recommendations.append(
                "Overall diversity score is low. Consider rebalancing the dataset."
            )
        
        if len(conditions) < 5:
            recommendations.append(
                "Limited condition coverage. Add more diverse medical conditions."
            )
        
        return {
            "demographics": demographics,
            "conditions": conditions,
            "specialties": specialties,
            "diversity_score": diversity_score,
            "coverage_gaps": gaps,
            "recommendations": recommendations,
        }
    
    def _get_age_group(self, age: int) -> str:
        """Map age to age group."""
        if age < 18:
            return "0-17"
        elif age <= 30:
            return "18-30"
        elif age <= 50:
            return "31-50"
        elif age <= 70:
            return "51-70"
        else:
            return "70+"
    
    def _calculate_distribution_score(
        self,
        actual_dist: Dict[str, int],
        target_dist: Dict[str, float],
    ) -> float:
        """
        Calculate how well actual distribution matches target.
        
        Uses normalized difference from target.
        """
        if not actual_dist or not target_dist:
            return 0.0
        
        total = sum(actual_dist.values())
        if total == 0:
            return 0.0
        
        # Calculate normalized actual distribution
        actual_pcts = {k: v / total for k, v in actual_dist.items()}
        
        # Calculate difference from target
        differences = []
        for key, target_pct in target_dist.items():
            actual_pct = actual_pcts.get(key, 0)
            diff = abs(actual_pct - target_pct)
            differences.append(diff)
        
        # Score = 1 - average difference (bounded 0-1)
        avg_diff = sum(differences) / len(differences) if differences else 0
        score = max(0, 1 - avg_diff * 2)  # Scale difference impact
        
        return round(score, 3)
