"""Synthetic Data Pipeline for Medical AI Training.

Phase 5 implementation providing synthetic medical data generation
using MedGemma 27B IT as the primary generator with batch processing,
progress tracking, checkpointing, and DVC versioning.

Modules:
- base: Foundation classes for all generators
- text_generator: Clinical text generation (Task 5.1)
- dialogue_generator: Patient-provider dialogues (Task 5.2)
- image_generator: Medical image synthesis (Task 5.3)
- structured_generator: Synthea/FHIR structured data (Task 5.4)
- quality_assurance: Data quality validation (Task 5.5)
- versioning: DVC-integrated versioning (Task 5.6)
- diversity: Diversity analysis tools (Task 5.7)
"""

from medai_compass.synthetic.base import BaseSyntheticGenerator
from medai_compass.synthetic.text_generator import ClinicalTextGenerator
from medai_compass.synthetic.dialogue_generator import DialogueGenerator
from medai_compass.synthetic.image_generator import ImageSynthesizer, ImageTrainingPipeline
from medai_compass.synthetic.structured_generator import StructuredDataGenerator
from medai_compass.synthetic.quality_assurance import SyntheticDataQA
from medai_compass.synthetic.versioning import SyntheticDataVersionManager
from medai_compass.synthetic.diversity import DiversityAnalyzer

__all__ = [
    "BaseSyntheticGenerator",
    "ClinicalTextGenerator",
    "DialogueGenerator",
    "ImageSynthesizer",
    "ImageTrainingPipeline",
    "StructuredDataGenerator",
    "SyntheticDataQA",
    "SyntheticDataVersionManager",
    "DiversityAnalyzer",
]
