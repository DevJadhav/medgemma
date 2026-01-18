"""Clinical Text Generator (Task 5.1).

Generates synthetic clinical text using MedGemma 27B IT including:
- Clinical notes (progress notes, discharge summaries)
- Radiology reports
- Pathology reports
- Operative notes

Uses MedGemma 27B IT explicitly as per requirements.
"""

import logging
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from medai_compass.synthetic.base import BaseSyntheticGenerator, GenerationConfig

logger = logging.getLogger(__name__)


# Medical specialty templates
SPECIALTY_TEMPLATES = {
    "cardiology": {
        "conditions": [
            "acute myocardial infarction",
            "congestive heart failure",
            "atrial fibrillation",
            "hypertension",
            "coronary artery disease",
            "aortic stenosis",
            "mitral regurgitation",
            "pericarditis",
        ],
        "symptoms": [
            "chest pain",
            "shortness of breath",
            "palpitations",
            "syncope",
            "edema",
            "fatigue",
        ],
        "tests": [
            "ECG",
            "echocardiogram",
            "stress test",
            "cardiac catheterization",
            "BNP",
            "troponin",
        ],
        "medications": [
            "aspirin",
            "beta-blockers",
            "ACE inhibitors",
            "statins",
            "anticoagulants",
            "diuretics",
        ],
    },
    "oncology": {
        "conditions": [
            "breast cancer",
            "lung cancer",
            "colorectal cancer",
            "prostate cancer",
            "lymphoma",
            "leukemia",
            "melanoma",
        ],
        "symptoms": [
            "fatigue",
            "weight loss",
            "pain",
            "night sweats",
            "lymphadenopathy",
        ],
        "tests": [
            "CT scan",
            "PET scan",
            "biopsy",
            "tumor markers",
            "MRI",
            "bone scan",
        ],
        "medications": [
            "chemotherapy",
            "immunotherapy",
            "targeted therapy",
            "hormone therapy",
            "radiation therapy",
        ],
    },
    "radiology": {
        "modalities": [
            "chest X-ray",
            "CT scan",
            "MRI",
            "ultrasound",
            "mammography",
            "PET scan",
        ],
        "findings_normal": [
            "no acute cardiopulmonary abnormality",
            "normal examination",
            "unremarkable study",
            "no significant abnormality detected",
        ],
        "findings_abnormal": [
            "opacity in the right lower lobe",
            "cardiomegaly",
            "pleural effusion",
            "pulmonary nodule",
            "consolidation",
        ],
    },
    "neurology": {
        "conditions": [
            "stroke",
            "epilepsy",
            "multiple sclerosis",
            "Parkinson's disease",
            "migraine",
            "dementia",
            "neuropathy",
        ],
        "symptoms": [
            "headache",
            "weakness",
            "numbness",
            "seizures",
            "tremor",
            "memory loss",
            "dizziness",
        ],
        "tests": [
            "MRI brain",
            "CT head",
            "EEG",
            "nerve conduction studies",
            "lumbar puncture",
        ],
    },
    "emergency": {
        "presentations": [
            "chest pain",
            "shortness of breath",
            "abdominal pain",
            "trauma",
            "altered mental status",
            "syncope",
            "fever",
        ],
        "acuity_levels": [
            "critical",
            "emergent",
            "urgent",
            "less urgent",
            "non-urgent",
        ],
    },
}

# Note type templates
NOTE_TYPE_TEMPLATES = {
    "progress_note": {
        "sections": ["subjective", "objective", "assessment", "plan"],
        "format": "SOAP",
    },
    "discharge_summary": {
        "sections": [
            "admission_date",
            "discharge_date",
            "diagnoses",
            "hospital_course",
            "discharge_medications",
            "follow_up",
        ],
    },
    "consultation": {
        "sections": [
            "reason_for_consultation",
            "history",
            "examination",
            "impression",
            "recommendations",
        ],
    },
    "operative_note": {
        "sections": [
            "preoperative_diagnosis",
            "postoperative_diagnosis",
            "procedure",
            "findings",
            "complications",
        ],
    },
    "admission_note": {
        "sections": [
            "chief_complaint",
            "history_present_illness",
            "past_medical_history",
            "medications",
            "allergies",
            "physical_exam",
            "assessment",
            "plan",
        ],
    },
}


class ClinicalTextGenerator(BaseSyntheticGenerator):
    """
    Generator for synthetic clinical text using MedGemma 27B IT.
    
    Generates various types of clinical documentation:
    - Progress notes (SOAP format)
    - Discharge summaries
    - Consultation notes
    - Operative notes
    - Radiology reports
    
    Attributes:
        model_name: Model to use (default: google/medgemma-27b-text-it)
        device: Device for inference
        mock_mode: Generate mock data for testing
    """
    
    # Default to MedGemma 27B IT
    DEFAULT_MODEL = "google/medgemma-27b-text-it"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        mock_mode: bool = False,
        target_count: int = 2500,
        batch_size: int = 50,
        checkpoint_interval: int = 100,
        checkpoint_dir: Optional[str] = None,
        use_dvc: bool = True,
        **kwargs,
    ):
        """
        Initialize the clinical text generator.
        
        Args:
            model_name: Model name (default: MedGemma 27B IT)
            device: Device for inference (auto, cpu, cuda)
            mock_mode: Enable mock mode for testing
            target_count: Target samples to generate
            batch_size: Batch size for generation
            checkpoint_interval: Checkpoint save interval
            checkpoint_dir: Directory for checkpoints
            use_dvc: Enable DVC checkpoint tracking
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
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        
        # Model and tokenizer (lazy loaded)
        self._model = None
        self._tokenizer = None
        
        logger.info(f"Initialized ClinicalTextGenerator with {self.model_name}")
    
    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._model is not None or self.mock_mode:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading model {self.model_name}...")
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device if device == "cuda" else None,
            )
            
            if device == "cpu":
                self._model = self._model.to(device)
            
            logger.info(f"Model loaded on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def list_specialties(self) -> List[str]:
        """List available medical specialties."""
        return list(SPECIALTY_TEMPLATES.keys())
    
    def list_note_types(self) -> List[str]:
        """List available note types."""
        return list(NOTE_TYPE_TEMPLATES.keys())
    
    def generate_single(self, **kwargs) -> Dict[str, Any]:
        """Generate a single clinical note."""
        specialty = kwargs.get("specialty", "cardiology")
        note_type = kwargs.get("note_type", "progress_note")
        
        return self.generate_clinical_note(
            specialty=specialty,
            note_type=note_type,
        )
    
    def generate_clinical_note(
        self,
        specialty: str = "cardiology",
        note_type: str = "progress_note",
        patient_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a synthetic clinical note.
        
        Args:
            specialty: Medical specialty
            note_type: Type of clinical note
            patient_context: Optional patient context
            
        Returns:
            Generated clinical note as dictionary
        """
        if self.mock_mode:
            return self._generate_mock_note(specialty, note_type, patient_context)
        
        self._load_model()
        
        # Build prompt
        prompt = self._build_note_prompt(specialty, note_type, patient_context)
        
        # Generate text
        text = self._generate_text(prompt)
        
        return {
            "id": str(uuid.uuid4()),
            "text": text,
            "specialty": specialty,
            "note_type": note_type,
            "model": self.model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "patient_context": patient_context,
        }
    
    def generate_radiology_report(
        self,
        modality: str = "chest_xray",
        findings_type: str = "normal",
        patient_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a synthetic radiology report.
        
        Args:
            modality: Imaging modality
            findings_type: Type of findings (normal, abnormal)
            patient_context: Optional patient context
            
        Returns:
            Generated radiology report
        """
        if self.mock_mode:
            return self._generate_mock_radiology_report(
                modality, findings_type, patient_context
            )
        
        self._load_model()
        
        prompt = self._build_radiology_prompt(modality, findings_type, patient_context)
        text = self._generate_text(prompt)
        
        return {
            "id": str(uuid.uuid4()),
            "text": text,
            "modality": modality,
            "findings_type": findings_type,
            "model": self.model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def _build_note_prompt(
        self,
        specialty: str,
        note_type: str,
        patient_context: Optional[Dict[str, Any]],
    ) -> str:
        """Build a prompt for clinical note generation."""
        template = SPECIALTY_TEMPLATES.get(specialty, SPECIALTY_TEMPLATES["cardiology"])
        note_template = NOTE_TYPE_TEMPLATES.get(note_type, NOTE_TYPE_TEMPLATES["progress_note"])
        
        # Select random elements for variety
        condition = random.choice(template.get("conditions", ["general condition"]))
        
        prompt = f"""Generate a synthetic {note_type.replace('_', ' ')} for a {specialty} patient.

Condition: {condition}
Format: {note_template.get('format', 'Standard')}
Sections to include: {', '.join(note_template['sections'])}

Generate a realistic, de-identified clinical note:"""
        
        if patient_context:
            context_str = ", ".join(f"{k}: {v}" for k, v in patient_context.items())
            prompt = f"Patient context: {context_str}\n\n" + prompt
        
        return prompt
    
    def _build_radiology_prompt(
        self,
        modality: str,
        findings_type: str,
        patient_context: Optional[Dict[str, Any]],
    ) -> str:
        """Build a prompt for radiology report generation."""
        template = SPECIALTY_TEMPLATES["radiology"]
        
        findings_key = f"findings_{findings_type}"
        findings = template.get(findings_key, template["findings_normal"])
        selected_finding = random.choice(findings)
        
        prompt = f"""Generate a synthetic radiology report for a {modality.replace('_', ' ')}.

Expected findings: {selected_finding}

Include sections:
- Clinical indication
- Technique
- Findings
- Impression

Generate a realistic, de-identified radiology report:"""
        
        return prompt
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the loaded model."""
        inputs = self._tokenizer(prompt, return_tensors="pt")
        
        if self._model.device.type == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from the output
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        
        return text
    
    def _generate_mock_note(
        self,
        specialty: str,
        note_type: str,
        patient_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a mock clinical note for testing."""
        template = SPECIALTY_TEMPLATES.get(specialty, SPECIALTY_TEMPLATES["cardiology"])
        note_template = NOTE_TYPE_TEMPLATES.get(note_type, NOTE_TYPE_TEMPLATES["progress_note"])
        
        # Build mock note text
        condition = random.choice(template.get("conditions", ["condition"]))
        symptoms = random.sample(
            template.get("symptoms", ["symptom"]),
            min(2, len(template.get("symptoms", ["symptom"]))),
        )
        
        sections = []
        for section in note_template["sections"]:
            if section == "subjective":
                sections.append(
                    f"Subjective: Patient reports {', '.join(symptoms)}."
                )
            elif section == "objective":
                sections.append(
                    f"Objective: Vital signs stable. Physical exam unremarkable."
                )
            elif section in ["assessment", "impression"]:
                sections.append(f"Assessment: {condition}")
            elif section == "plan":
                meds = template.get("medications", ["medication"])
                sections.append(
                    f"Plan: Continue {random.choice(meds)}. Follow up in 2 weeks."
                )
            else:
                sections.append(f"{section.replace('_', ' ').title()}: [Mock content]")
        
        text = "\n\n".join(sections)
        
        return {
            "id": str(uuid.uuid4()),
            "text": text,
            "specialty": specialty,
            "note_type": note_type,
            "model": self.model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "patient_context": patient_context,
            "mock": True,
        }
    
    def _generate_mock_radiology_report(
        self,
        modality: str,
        findings_type: str,
        patient_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a mock radiology report for testing."""
        template = SPECIALTY_TEMPLATES["radiology"]
        
        findings_key = f"findings_{findings_type}"
        findings = template.get(findings_key, template["findings_normal"])
        selected_finding = random.choice(findings)
        
        text = f"""RADIOLOGY REPORT

Examination: {modality.replace('_', ' ').title()}

Clinical Indication: Routine examination

Technique: Standard protocol

Findings: {selected_finding}

Impression: {selected_finding}"""
        
        return {
            "id": str(uuid.uuid4()),
            "text": text,
            "modality": modality,
            "findings_type": findings_type,
            "model": self.model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mock": True,
        }
