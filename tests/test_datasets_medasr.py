"""
Tests for the datasets and MedASR modules.
"""

import json
import tempfile
from pathlib import Path

import pytest

from medai_compass.utils.datasets import (
    DATASETS,
    DatasetDownloader,
    DatasetInfo,
    DatasetLoader,
    download_sample_data,
)
from medai_compass.utils.medasr import (
    AudioFormat,
    DictationService,
    DictationSession,
    MedASRWrapper,
    TranscriptionResult,
    transcribe_audio,
)


class TestDatasetInfo:
    """Tests for DatasetInfo dataclass."""
    
    def test_create_dataset_info(self):
        """Test creating dataset info."""
        info = DatasetInfo(
            name="TestDataset",
            description="Test description",
            url="https://example.com",
            size_gb=10.0,
            license="MIT",
            citation="Test citation"
        )
        
        assert info.name == "TestDataset"
        assert info.requires_auth is False


class TestDatasetDownloader:
    """Tests for DatasetDownloader class."""
    
    def test_init(self, tmp_path):
        """Test downloader initialization."""
        downloader = DatasetDownloader(str(tmp_path / "datasets"))
        assert (tmp_path / "datasets").exists()
        
    def test_list_available_datasets(self, tmp_path):
        """Test listing available datasets."""
        downloader = DatasetDownloader(str(tmp_path))
        datasets = downloader.list_available_datasets()
        
        assert len(datasets) >= 4
        assert any(d.name == "ChestX-ray14" for d in datasets)
        
    def test_get_dataset_info(self, tmp_path):
        """Test getting specific dataset info."""
        downloader = DatasetDownloader(str(tmp_path))
        
        info = downloader.get_dataset_info("chestxray14")
        
        assert info is not None
        assert info.name == "ChestX-ray14"
        
    def test_download_chestxray14_sample(self, tmp_path):
        """Test downloading ChestX-ray14 sample."""
        downloader = DatasetDownloader(str(tmp_path))
        
        sample_path = downloader.download_chestxray14_sample(sample_size=10)
        
        assert sample_path.exists()
        assert (sample_path / "manifest.json").exists()
        
    def test_get_camelyon16_instructions(self, tmp_path):
        """Test getting CAMELYON16 download instructions."""
        downloader = DatasetDownloader(str(tmp_path))
        
        instructions = downloader.get_camelyon16_instructions()
        
        assert "dataset" in instructions
        assert "instructions" in instructions
        assert len(instructions["instructions"]) > 0


class TestDatasetLoader:
    """Tests for DatasetLoader class."""
    
    def test_init(self, tmp_path):
        """Test loader initialization."""
        loader = DatasetLoader(str(tmp_path))
        assert tmp_path.exists()
        
    def test_get_label_statistics_empty(self, tmp_path):
        """Test getting statistics from empty labels."""
        loader = DatasetLoader(str(tmp_path))
        
        stats = loader.get_label_statistics({})
        
        assert stats["total_images"] == 0


class TestDownloadSampleData:
    """Tests for download_sample_data function."""
    
    def test_download_sample_data(self, tmp_path):
        """Test downloading sample data."""
        result = download_sample_data(str(tmp_path))
        
        assert "chestxray14" in result
        assert "camelyon16" in result
        assert "available_datasets" in result


class TestMedASRWrapper:
    """Tests for MedASRWrapper class."""
    
    def test_init(self):
        """Test wrapper initialization."""
        wrapper = MedASRWrapper()
        
        assert wrapper.model_name == "google/medasr"
        assert wrapper._mock_mode is True  # No GPU
        
    def test_load_model(self):
        """Test loading model (mock mode)."""
        wrapper = MedASRWrapper()
        
        result = wrapper.load_model()
        
        assert result is True
        
    def test_transcribe_nonexistent_file(self, tmp_path):
        """Test transcribing non-existent file."""
        wrapper = MedASRWrapper()
        
        result = wrapper.transcribe(str(tmp_path / "nonexistent.wav"))
        
        assert result.text == ""
        assert result.confidence == 0.0
        
    def test_transcribe_mock_file(self, tmp_path):
        """Test transcribing with mock file."""
        wrapper = MedASRWrapper()
        
        # Create a mock audio file
        audio_file = tmp_path / "chest_pain_dictation.wav"
        audio_file.write_bytes(b"mock audio data")
        
        result = wrapper.transcribe(str(audio_file))
        
        assert len(result.text) > 0
        assert result.confidence > 0.5
        assert "chest pain" in result.text.lower()
        
    def test_post_process_medical_text(self):
        """Test medical text post-processing."""
        wrapper = MedASRWrapper()
        
        text = "take medication tid with food prn"
        processed = wrapper._post_process_medical_text(text)
        
        assert "TID" in processed
        assert "PRN" in processed
        
    def test_detect_medical_terms(self):
        """Test medical term detection."""
        wrapper = MedASRWrapper()
        
        terms = wrapper._detect_medical_terms("Patient has hypertension and diabetes")
        
        assert "hypertension" in terms
        assert "diabetes" in terms


class TestDictationSession:
    """Tests for DictationSession dataclass."""
    
    def test_create_session(self):
        """Test creating a dictation session."""
        session = DictationSession(
            session_id="test-001",
            patient_id="P001"
        )
        
        assert session.session_id == "test-001"
        assert session.status == "active"
        assert len(session.transcriptions) == 0


class TestDictationService:
    """Tests for DictationService class."""
    
    def test_init(self):
        """Test service initialization."""
        service = DictationService()
        
        assert service.asr is not None
        assert len(service.active_sessions) == 0
        
    def test_start_session(self):
        """Test starting a dictation session."""
        service = DictationService()
        
        session = service.start_session(
            patient_id="P001",
            document_type="progress_note"
        )
        
        assert session.patient_id == "P001"
        assert session.document_type == "progress_note"
        assert session.session_id in service.active_sessions
        
    def test_add_audio_to_session(self, tmp_path):
        """Test adding audio to session."""
        service = DictationService()
        session = service.start_session(patient_id="P001")
        
        # Create mock audio file
        audio_file = tmp_path / "test_audio.wav"
        audio_file.write_bytes(b"mock audio")
        
        result = service.add_audio_to_session(session.session_id, str(audio_file))
        
        assert result is not None
        assert len(session.transcriptions) == 1
        
    def test_add_audio_invalid_session(self, tmp_path):
        """Test adding audio to invalid session."""
        service = DictationService()
        
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"mock")
        
        result = service.add_audio_to_session("invalid-session", str(audio_file))
        
        assert result is None
        
    def test_get_session_transcript(self, tmp_path):
        """Test getting session transcript."""
        service = DictationService()
        session = service.start_session(patient_id="P001")
        
        # Add mock audio
        audio_file = tmp_path / "follow_up.wav"
        audio_file.write_bytes(b"mock")
        service.add_audio_to_session(session.session_id, str(audio_file))
        
        transcript = service.get_session_transcript(session.session_id)
        
        assert len(transcript) > 0
        
    def test_complete_session(self):
        """Test completing a session."""
        service = DictationService()
        session = service.start_session(patient_id="P001")
        
        completed = service.complete_session(session.session_id)
        
        assert completed.status == "completed"
        
    def test_generate_document_from_session(self, tmp_path):
        """Test generating document from session."""
        service = DictationService()
        session = service.start_session(
            patient_id="P001",
            encounter_id="ENC001"
        )
        
        # Add mock transcription
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"mock")
        service.add_audio_to_session(session.session_id, str(audio_file))
        
        document = service.generate_document_from_session(
            session.session_id,
            template="soap"
        )
        
        assert "document_type" in document
        assert "SOAP Note" in document["document_type"]


class TestTranscribeAudio:
    """Tests for transcribe_audio convenience function."""
    
    def test_transcribe_audio(self, tmp_path):
        """Test transcribing audio file."""
        audio_file = tmp_path / "default_audio.wav"
        audio_file.write_bytes(b"mock audio")
        
        result = transcribe_audio(str(audio_file))
        
        assert isinstance(result, TranscriptionResult)
