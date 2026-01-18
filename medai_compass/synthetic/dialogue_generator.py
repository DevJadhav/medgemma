"""Patient Dialogue Generator (Task 5.2).

Generates synthetic patient-provider dialogues using MedGemma 27B IT:
- Initial consultations
- Symptom assessments
- Follow-up visits
- Medication discussions
- Discharge instructions

Multi-turn conversation support with context tracking.
"""

import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from medai_compass.synthetic.base import BaseSyntheticGenerator

logger = logging.getLogger(__name__)


# Dialogue scenario templates
DIALOGUE_SCENARIOS = {
    "initial_consultation": {
        "description": "First visit with a new patient",
        "typical_turns": 8,
        "provider_goals": [
            "gather chief complaint",
            "take history",
            "assess symptoms",
            "develop initial plan",
        ],
        "patient_concerns": [
            "explain symptoms",
            "share medical history",
            "ask questions about diagnosis",
        ],
    },
    "symptom_assessment": {
        "description": "Focused assessment of specific symptoms",
        "typical_turns": 6,
        "provider_goals": [
            "characterize symptoms",
            "identify red flags",
            "determine severity",
        ],
        "symptom_categories": [
            "chest pain",
            "shortness of breath",
            "abdominal pain",
            "headache",
            "fatigue",
            "dizziness",
        ],
    },
    "follow_up_visit": {
        "description": "Follow-up after treatment or diagnosis",
        "typical_turns": 6,
        "provider_goals": [
            "assess treatment response",
            "review medications",
            "adjust plan if needed",
        ],
        "patient_updates": [
            "symptom improvement",
            "side effects",
            "new concerns",
        ],
    },
    "medication_discussion": {
        "description": "Discussion about medications",
        "typical_turns": 6,
        "topics": [
            "new prescription",
            "side effects",
            "adherence concerns",
            "refill request",
        ],
    },
    "discharge_instructions": {
        "description": "Post-hospitalization discharge education",
        "typical_turns": 8,
        "elements": [
            "diagnosis summary",
            "medication instructions",
            "activity restrictions",
            "warning signs",
            "follow-up appointments",
        ],
    },
    "emergency_triage": {
        "description": "Emergency department triage conversation",
        "typical_turns": 4,
        "priorities": [
            "identify chief complaint",
            "assess acuity",
            "gather vital information",
        ],
    },
}

# Sample dialogue turns for mock generation
MOCK_DIALOGUE_TURNS = {
    "provider": [
        "Good morning. What brings you in today?",
        "Can you tell me more about when this started?",
        "How would you rate the pain on a scale of 1 to 10?",
        "Have you experienced any other symptoms?",
        "Are you currently taking any medications?",
        "Do you have any allergies?",
        "Let me examine you and then we'll discuss the next steps.",
        "Based on my assessment, I recommend the following...",
        "Do you have any questions about the plan?",
        "Please follow up with me in two weeks.",
    ],
    "patient": [
        "I've been having chest pain for the past few days.",
        "It started about a week ago and has been getting worse.",
        "I would say it's about a 6 out of 10.",
        "Yes, I've also been feeling short of breath.",
        "I take medication for high blood pressure.",
        "No known allergies.",
        "Okay, thank you doctor.",
        "What should I do if it gets worse?",
        "When should I come back?",
        "Thank you for your help.",
    ],
}


class DialogueGenerator(BaseSyntheticGenerator):
    """
    Generator for patient-provider dialogues using MedGemma 27B IT.
    
    Generates multi-turn conversations for various medical scenarios:
    - Initial consultations
    - Symptom assessments
    - Follow-up visits
    - Medication discussions
    - Discharge instructions
    
    Attributes:
        model_name: Model to use (default: google/medgemma-27b-text-it)
        mock_mode: Generate mock dialogues for testing
    """
    
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
        Initialize the dialogue generator.
        
        Args:
            model_name: Model name (default: MedGemma 27B IT)
            device: Device for inference
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
        
        self._model = None
        self._tokenizer = None
        
        logger.info(f"Initialized DialogueGenerator with {self.model_name}")
    
    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._model is not None or self.mock_mode:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading model {self.model_name}...")
            
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
    
    def list_scenarios(self) -> List[str]:
        """List available dialogue scenarios."""
        return list(DIALOGUE_SCENARIOS.keys())
    
    def generate_single(self, **kwargs) -> Dict[str, Any]:
        """Generate a single dialogue."""
        scenario = kwargs.get("scenario", "initial_consultation")
        num_turns = kwargs.get("num_turns", 6)
        
        return self.generate_conversation(
            scenario=scenario,
            num_turns=num_turns,
            patient_context=kwargs.get("patient_context"),
        )
    
    def generate_turn(
        self,
        context: str,
        speaker: str = "patient",
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a single dialogue turn.
        
        Args:
            context: Context for the turn
            speaker: Speaker (patient or provider)
            conversation_history: Previous turns
            
        Returns:
            Generated turn as dictionary
        """
        if self.mock_mode:
            return self._generate_mock_turn(context, speaker)
        
        self._load_model()
        
        prompt = self._build_turn_prompt(context, speaker, conversation_history)
        utterance = self._generate_text(prompt)
        
        return {
            "speaker": speaker,
            "utterance": utterance,
            "context": context,
        }
    
    def generate_conversation(
        self,
        scenario: str = "initial_consultation",
        num_turns: int = 6,
        patient_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete multi-turn conversation.
        
        Args:
            scenario: Dialogue scenario type
            num_turns: Number of turns to generate
            patient_context: Optional patient context
            
        Returns:
            Generated conversation as dictionary
        """
        if self.mock_mode:
            return self._generate_mock_conversation(scenario, num_turns, patient_context)
        
        self._load_model()
        
        scenario_info = DIALOGUE_SCENARIOS.get(
            scenario, DIALOGUE_SCENARIOS["initial_consultation"]
        )
        
        turns = []
        conversation_history = []
        
        for i in range(num_turns):
            # Alternate between provider and patient
            speaker = "provider" if i % 2 == 0 else "patient"
            
            # Build context for this turn
            turn_context = self._build_conversation_context(
                scenario_info, i, patient_context
            )
            
            # Generate turn
            prompt = self._build_turn_prompt(turn_context, speaker, conversation_history)
            utterance = self._generate_text(prompt)
            
            turn = {
                "turn_number": i + 1,
                "speaker": speaker,
                "utterance": utterance,
            }
            
            turns.append(turn)
            conversation_history.append(turn)
        
        return {
            "id": str(uuid.uuid4()),
            "scenario": scenario,
            "turns": turns,
            "patient_context": patient_context,
            "model": self.model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def _build_turn_prompt(
        self,
        context: str,
        speaker: str,
        history: Optional[List[Dict[str, str]]],
    ) -> str:
        """Build prompt for generating a dialogue turn."""
        prompt = f"Generate a realistic {speaker} response in a medical conversation.\n\n"
        prompt += f"Context: {context}\n\n"
        
        if history:
            prompt += "Conversation so far:\n"
            for turn in history[-4:]:  # Last 4 turns for context
                prompt += f"{turn['speaker'].title()}: {turn['utterance']}\n"
            prompt += "\n"
        
        prompt += f"{speaker.title()}: "
        
        return prompt
    
    def _build_conversation_context(
        self,
        scenario_info: Dict[str, Any],
        turn_number: int,
        patient_context: Optional[Dict[str, Any]],
    ) -> str:
        """Build context for a conversation turn."""
        context_parts = [scenario_info["description"]]
        
        if patient_context:
            context_parts.append(
                f"Patient: {patient_context.get('age', 'unknown age')}, "
                f"{patient_context.get('gender', 'unknown gender')}"
            )
            if "chief_complaint" in patient_context:
                context_parts.append(
                    f"Chief complaint: {patient_context['chief_complaint']}"
                )
        
        if "provider_goals" in scenario_info and turn_number < len(scenario_info["provider_goals"]):
            context_parts.append(
                f"Current goal: {scenario_info['provider_goals'][turn_number // 2]}"
            )
        
        return ". ".join(context_parts)
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the loaded model."""
        inputs = self._tokenizer(prompt, return_tensors="pt")
        
        if self._model.device.type == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.8,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        
        # Stop at newline (single turn)
        if "\n" in text:
            text = text.split("\n")[0].strip()
        
        return text
    
    def _generate_mock_turn(
        self,
        context: str,
        speaker: str,
    ) -> Dict[str, Any]:
        """Generate a mock dialogue turn for testing."""
        turns = MOCK_DIALOGUE_TURNS.get(speaker, MOCK_DIALOGUE_TURNS["patient"])
        utterance = random.choice(turns)
        
        return {
            "speaker": speaker,
            "utterance": utterance,
            "context": context,
            "mock": True,
        }
    
    def _generate_mock_conversation(
        self,
        scenario: str,
        num_turns: int,
        patient_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate a mock conversation for testing."""
        turns = []
        
        for i in range(num_turns):
            speaker = "provider" if i % 2 == 0 else "patient"
            speaker_turns = MOCK_DIALOGUE_TURNS.get(speaker, [])
            
            # Cycle through mock turns
            idx = (i // 2) % len(speaker_turns) if speaker_turns else 0
            utterance = speaker_turns[idx] if speaker_turns else "..."
            
            turns.append({
                "turn_number": i + 1,
                "speaker": speaker,
                "utterance": utterance,
            })
        
        return {
            "id": str(uuid.uuid4()),
            "scenario": scenario,
            "turns": turns,
            "patient_context": patient_context,
            "model": self.model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mock": True,
        }
