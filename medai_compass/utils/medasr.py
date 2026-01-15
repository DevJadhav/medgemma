"""
MedASR Integration for Clinical Dictation.

This module provides integration with Google's MedASR model for
clinical speech recognition and dictation workflows.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"
    M4A = "m4a"


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""
    text: str
    confidence: float
    duration_seconds: float
    word_timestamps: list[dict] = field(default_factory=list)
    medical_terms_detected: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    model_used: str = "medasr"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DictationSession:
    """Active dictation session."""
    session_id: str
    patient_id: Optional[str] = None
    encounter_id: Optional[str] = None
    document_type: str = "progress_note"
    transcriptions: list[TranscriptionResult] = field(default_factory=list)
    status: str = "active"  # active, paused, completed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class MedASRWrapper:
    """
    Wrapper for MedASR speech recognition model.
    
    MedASR is specialized for medical terminology and clinical dictation.
    On M3 Mac, this uses mock mode. Real inference requires Modal/GPU.
    """
    
    # Common medical abbreviations for post-processing
    MEDICAL_ABBREVIATIONS = {
        "prn": "PRN (as needed)",
        "tid": "TID (three times daily)",
        "bid": "BID (twice daily)",
        "qid": "QID (four times daily)",
        "stat": "STAT (immediately)",
        "po": "PO (by mouth)",
        "iv": "IV (intravenous)",
        "im": "IM (intramuscular)",
        "sq": "SQ (subcutaneous)",
        "dx": "diagnosis",
        "hx": "history",
        "px": "physical exam",
        "rx": "prescription",
        "tx": "treatment",
        "sx": "symptoms",
        "fx": "fracture",
        "sob": "shortness of breath",
        "cp": "chest pain",
        "ha": "headache",
        "n/v": "nausea/vomiting",
        "r/o": "rule out",
        "f/u": "follow up",
        "w/u": "workup"
    }
    
    def __init__(
        self,
        model_name: str = "google/medasr",
        device: str = "auto",
        use_gpu: bool = False
    ):
        self.model_name = model_name
        self.device = device
        self.use_gpu = use_gpu
        self.model = None
        self.processor = None
        self._mock_mode = not use_gpu  # Use mock on M3 Mac
        
    def load_model(self) -> bool:
        """
        Load the MedASR model.
        
        Returns:
            True if model loaded successfully
        """
        if self._mock_mode:
            # Mock mode for M3 Mac development
            return True
            
        try:
            # Real model loading would happen here
            # from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            pass
        except Exception as e:
            print(f"Failed to load MedASR model: {e}")
            return False
            
        return True
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        return_timestamps: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            return_timestamps: Whether to return word-level timestamps
            
        Returns:
            TranscriptionResult with transcription
        """
        start_time = datetime.now()
        
        # Validate audio file
        audio_file = Path(audio_path)
        if not audio_file.exists():
            return TranscriptionResult(
                text="",
                confidence=0.0,
                duration_seconds=0.0,
                processing_time_ms=0.0
            )
            
        if self._mock_mode:
            # Mock transcription for development
            result = self._mock_transcribe(audio_path, return_timestamps)
        else:
            # Real transcription
            result = self._real_transcribe(audio_path, language, return_timestamps)
            
        result.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Post-process for medical terms
        result.text = self._post_process_medical_text(result.text)
        result.medical_terms_detected = self._detect_medical_terms(result.text)
        
        return result
    
    def _mock_transcribe(self, audio_path: str, return_timestamps: bool) -> TranscriptionResult:
        """Mock transcription for development."""
        # Simulate transcription based on file name
        mock_transcriptions = {
            "chest_pain": "Patient presents with acute onset chest pain, rated 7 out of 10. Pain is substernal, radiating to left arm. No shortness of breath. History of hypertension. Vitals stable.",
            "follow_up": "Follow up visit for diabetes management. Patient reports good compliance with metformin. Fasting glucose levels averaging 120. No hypoglycemic episodes. Continue current regimen.",
            "default": "Clinical dictation recorded. Patient encounter documented for medical records. Please review and sign."
        }
        
        # Determine which mock to use
        audio_name = Path(audio_path).stem.lower()
        for key, text in mock_transcriptions.items():
            if key in audio_name:
                return TranscriptionResult(
                    text=text,
                    confidence=0.92,
                    duration_seconds=15.0,
                    word_timestamps=self._generate_mock_timestamps(text) if return_timestamps else []
                )
                
        return TranscriptionResult(
            text=mock_transcriptions["default"],
            confidence=0.88,
            duration_seconds=5.0,
            word_timestamps=[]
        )
    
    def _real_transcribe(
        self,
        audio_path: str,
        language: str,
        return_timestamps: bool
    ) -> TranscriptionResult:
        """Real transcription using MedASR model."""
        # This would use the actual model
        # For now, return empty result
        return TranscriptionResult(
            text="",
            confidence=0.0,
            duration_seconds=0.0
        )
    
    def _generate_mock_timestamps(self, text: str) -> list[dict]:
        """Generate mock word timestamps."""
        words = text.split()
        timestamps = []
        current_time = 0.0
        
        for word in words:
            word_duration = len(word) * 0.05 + 0.1  # Rough estimate
            timestamps.append({
                "word": word,
                "start": current_time,
                "end": current_time + word_duration
            })
            current_time += word_duration + 0.1  # Add pause
            
        return timestamps
    
    def _post_process_medical_text(self, text: str) -> str:
        """Post-process transcription for medical formatting."""
        # Capitalize medical abbreviations
        processed = text
        
        # Fix common medical abbreviation capitalizations
        for abbrev in self.MEDICAL_ABBREVIATIONS.keys():
            # Match whole word only
            import re
            pattern = rf'\b{re.escape(abbrev)}\b'
            processed = re.sub(pattern, abbrev.upper(), processed, flags=re.IGNORECASE)
            
        return processed
    
    def _detect_medical_terms(self, text: str) -> list[str]:
        """Detect medical terms in transcription."""
        detected = []
        text_lower = text.lower()
        
        # Check for medical abbreviations
        for abbrev, full_form in self.MEDICAL_ABBREVIATIONS.items():
            if abbrev in text_lower:
                detected.append(full_form)
                
        # Check for common medical terms
        medical_terms = [
            "hypertension", "diabetes", "pneumonia", "cardiac", "pulmonary",
            "hepatic", "renal", "neurological", "oncology", "radiology",
            "pathology", "surgery", "anesthesia", "infectious", "autoimmune"
        ]
        
        for term in medical_terms:
            if term in text_lower:
                detected.append(term)
                
        return list(set(detected))


class DictationService:
    """
    Service for managing clinical dictation workflows.
    
    Integrates MedASR transcription with document generation.
    """
    
    def __init__(self, asr_model: Optional[MedASRWrapper] = None):
        self.asr = asr_model or MedASRWrapper()
        self.active_sessions: dict[str, DictationSession] = {}
        
    def start_session(
        self,
        patient_id: Optional[str] = None,
        encounter_id: Optional[str] = None,
        document_type: str = "progress_note"
    ) -> DictationSession:
        """
        Start a new dictation session.
        
        Args:
            patient_id: Associated patient ID
            encounter_id: Associated encounter ID
            document_type: Type of document being dictated
            
        Returns:
            New DictationSession
        """
        session_id = f"dict-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        session = DictationSession(
            session_id=session_id,
            patient_id=patient_id,
            encounter_id=encounter_id,
            document_type=document_type
        )
        
        self.active_sessions[session_id] = session
        return session
    
    def add_audio_to_session(
        self,
        session_id: str,
        audio_path: str
    ) -> Optional[TranscriptionResult]:
        """
        Add audio recording to session and transcribe.
        
        Args:
            session_id: Session to add audio to
            audio_path: Path to audio file
            
        Returns:
            TranscriptionResult or None if session not found
        """
        if session_id not in self.active_sessions:
            return None
            
        session = self.active_sessions[session_id]
        
        # Transcribe audio
        result = self.asr.transcribe(audio_path)
        
        # Add to session
        session.transcriptions.append(result)
        
        return result
    
    def get_session_transcript(self, session_id: str) -> str:
        """
        Get combined transcript from all audio in session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Combined transcript text
        """
        if session_id not in self.active_sessions:
            return ""
            
        session = self.active_sessions[session_id]
        
        return "\n\n".join([t.text for t in session.transcriptions])
    
    def complete_session(self, session_id: str) -> Optional[DictationSession]:
        """
        Complete a dictation session.
        
        Args:
            session_id: Session to complete
            
        Returns:
            Completed session or None
        """
        if session_id not in self.active_sessions:
            return None
            
        session = self.active_sessions[session_id]
        session.status = "completed"
        
        return session
    
    def generate_document_from_session(
        self,
        session_id: str,
        template: str = "soap"
    ) -> dict:
        """
        Generate a clinical document from dictation session.
        
        Args:
            session_id: Session ID
            template: Document template (soap, progress, discharge)
            
        Returns:
            Generated document structure
        """
        transcript = self.get_session_transcript(session_id)
        session = self.active_sessions.get(session_id)
        
        if not session:
            return {"error": "Session not found"}
            
        # Generate structured document based on template
        if template == "soap":
            document = self._generate_soap_note(transcript, session)
        elif template == "progress":
            document = self._generate_progress_note(transcript, session)
        else:
            document = self._generate_generic_note(transcript, session)
            
        return document
    
    def _generate_soap_note(self, transcript: str, session: DictationSession) -> dict:
        """Generate SOAP note structure from transcript."""
        # In production, this would use MedGemma to structure the note
        return {
            "document_type": "SOAP Note",
            "patient_id": session.patient_id,
            "encounter_id": session.encounter_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "sections": {
                "subjective": "Patient-reported symptoms and history from dictation.",
                "objective": "Physical examination findings and vital signs.",
                "assessment": "Clinical assessment and diagnosis.",
                "plan": "Treatment plan and follow-up instructions."
            },
            "raw_transcript": transcript,
            "status": "draft"
        }
    
    def _generate_progress_note(self, transcript: str, session: DictationSession) -> dict:
        """Generate progress note from transcript."""
        return {
            "document_type": "Progress Note",
            "patient_id": session.patient_id,
            "encounter_id": session.encounter_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "content": transcript,
            "status": "draft"
        }
    
    def _generate_generic_note(self, transcript: str, session: DictationSession) -> dict:
        """Generate generic clinical note."""
        return {
            "document_type": session.document_type,
            "patient_id": session.patient_id,
            "encounter_id": session.encounter_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "content": transcript,
            "status": "draft"
        }


# Convenience function for quick transcription
def transcribe_audio(audio_path: str) -> TranscriptionResult:
    """
    Transcribe a single audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        TranscriptionResult
    """
    asr = MedASRWrapper()
    return asr.transcribe(audio_path)
