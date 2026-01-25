"""PHI Detection Pipeline Filter.

Strict PHI filtering for data pipeline - records containing PHI are
completely removed (not redacted) to ensure HIPAA compliance.

Reuses existing PHI detector from medai_compass.guardrails.phi_detection.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from medai_compass.guardrails.phi_detection import (
    detect_phi,
    validate_no_phi,
    PHI_PATTERNS,
    PHIDetector,
)

logger = logging.getLogger(__name__)


@dataclass
class PHIFilterConfig:
    """Configuration for PHI filtering."""
    
    # Strict mode - completely remove records with PHI (default per user requirement)
    strict: bool = True
    
    # Scan nested fields in records
    scan_nested: bool = True
    
    # Log PHI detections for audit
    log_detections: bool = True
    
    # Audit log path
    audit_log_path: Optional[str] = None
    
    # PHI types to detect (None = all)
    phi_types: Optional[List[str]] = None


@dataclass
class PHIFilterResult:
    """Result of scanning a record for PHI."""
    
    contains_phi: bool
    phi_types: List[str] = field(default_factory=list)
    phi_locations: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    risk_level: str = "none"  # none, medium, high, critical
    
    def __post_init__(self):
        # Set risk level based on PHI types
        if not self.contains_phi:
            self.risk_level = "none"
        elif "ssn" in self.phi_types:
            self.risk_level = "critical"
        elif len(self.phi_types) > 1 or "mrn" in self.phi_types:
            self.risk_level = "high"
        else:
            self.risk_level = "medium"


class PHIPipelineFilter:
    """
    PHI filter for data pipeline with strict filtering mode.
    
    In strict mode (default), records containing any PHI are completely
    removed from the dataset. This ensures HIPAA compliance by preventing
    PHI from entering the training data.
    
    Attributes:
        strict: If True, filter (remove) records with PHI
        audit_log: List of audit log entries for PHI detections
    """
    
    def __init__(
        self,
        strict: bool = True,
        log_detections: bool = True,
        config: Optional[PHIFilterConfig] = None,
    ):
        """
        Initialize the PHI filter.
        
        Args:
            strict: If True, completely remove records with PHI (default)
            log_detections: If True, log all PHI detections for audit
            config: Optional configuration object
        """
        self.config = config or PHIFilterConfig(
            strict=strict,
            log_detections=log_detections,
        )
        
        # Use config values
        self.strict = self.config.strict
        self.log_detections = self.config.log_detections
        
        # Initialize underlying detector
        self._detector = PHIDetector()
        
        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized PHIPipelineFilter (strict={self.strict})")
    
    def scan(self, text: str) -> PHIFilterResult:
        """
        Scan text for PHI.
        
        Args:
            text: Text to scan
            
        Returns:
            PHIFilterResult with detection details
        """
        detected = detect_phi(text)
        
        contains_phi = len(detected) > 0
        phi_types = list(detected.keys())
        
        # Get locations (simplified - just record which types found)
        phi_locations = {}
        for phi_type, instances in detected.items():
            # Find approximate locations
            locations = []
            for instance in instances:
                try:
                    start = text.find(str(instance))
                    if start >= 0:
                        locations.append((start, start + len(str(instance))))
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error locating PHI instance: {e}")
            phi_locations[phi_type] = locations
        
        return PHIFilterResult(
            contains_phi=contains_phi,
            phi_types=phi_types,
            phi_locations=phi_locations,
        )
    
    def filter_record(
        self,
        record: Dict[str, Any],
        record_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Filter a single record for PHI.
        
        In strict mode, returns None if PHI is found (record is filtered out).
        
        Args:
            record: Record to check
            record_id: Optional record ID for logging
            
        Returns:
            Original record if clean, None if PHI found (in strict mode)
        """
        # Get record ID for logging
        if record_id is None:
            record_id = record.get("id") or record.get("patient_id") or "unknown"
        
        # Collect all text from record
        texts_to_scan = self._extract_texts(record)
        
        # Scan all texts
        all_phi_types: Set[str] = set()
        contains_phi = False
        
        for text in texts_to_scan:
            result = self.scan(text)
            if result.contains_phi:
                contains_phi = True
                all_phi_types.update(result.phi_types)
        
        # Log if PHI detected
        if contains_phi and self.log_detections:
            self._log_detection(record_id, list(all_phi_types))
        
        # In strict mode, filter out records with PHI
        if contains_phi and self.strict:
            logger.debug(f"Filtering record {record_id}: PHI types {all_phi_types}")
            return None
        
        return record
    
    def _extract_texts(self, record: Dict[str, Any], prefix: str = "") -> List[str]:
        """Extract all text values from a record, including nested fields."""
        texts = []
        
        for key, value in record.items():
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, dict) and self.config.scan_nested:
                # Recursively extract from nested dicts
                texts.extend(self._extract_texts(value, f"{prefix}{key}."))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict) and self.config.scan_nested:
                        texts.extend(self._extract_texts(item, f"{prefix}{key}[{i}]."))
        
        return texts
    
    def _log_detection(self, record_id: str, phi_types: List[str]):
        """Log a PHI detection for audit."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "record_id": record_id,
            "phi_types": phi_types,
            "action": "filtered" if self.strict else "detected",
        }
        
        self.audit_log.append(entry)
        
        # Also log to file if configured
        if self.config.audit_log_path:
            try:
                log_path = Path(self.config.audit_log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                logger.warning(f"Failed to write audit log: {e}")
    
    def filter_batch(
        self,
        records: List[Dict[str, Any]],
        return_stats: bool = False,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Filter a batch of records for PHI.
        
        Args:
            records: List of records to filter
            return_stats: If True, also return filtering statistics
            
        Returns:
            Filtered records, and optionally statistics
        """
        filtered_records = []
        phi_types_found: Set[str] = set()
        filtered_count = 0
        
        for i, record in enumerate(records):
            record_id = record.get("id") or f"record_{i}"
            result = self.filter_record(record, record_id)
            
            if result is not None:
                filtered_records.append(result)
            else:
                filtered_count += 1
                # Get PHI types from last scan
                if self.audit_log:
                    last_entry = self.audit_log[-1]
                    phi_types_found.update(last_entry.get("phi_types", []))
        
        if return_stats:
            stats = {
                "total": len(records),
                "filtered": filtered_count,
                "kept": len(filtered_records),
                "phi_types_found": list(phi_types_found),
            }
            return filtered_records, stats
        
        return filtered_records
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of audit log."""
        if not self.audit_log:
            return {"total_detections": 0, "phi_types": {}}
        
        phi_type_counts: Dict[str, int] = {}
        for entry in self.audit_log:
            for phi_type in entry.get("phi_types", []):
                phi_type_counts[phi_type] = phi_type_counts.get(phi_type, 0) + 1
        
        return {
            "total_detections": len(self.audit_log),
            "phi_types": phi_type_counts,
            "first_detection": self.audit_log[0]["timestamp"],
            "last_detection": self.audit_log[-1]["timestamp"],
        }
