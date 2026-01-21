"""Modal GPU Deployment for MedGemma Inference.

This module defines the Modal app and GPU functions for running
MedGemma models on cloud H100 GPUs.

Supports:
- Trained/fine-tuned models (priority)
- HuggingFace MedGemma models (fallback)

Deployment:
    uv run modal deploy medai_compass/modal/app.py

Local Testing:
    uv run modal run medai_compass/modal/app.py

Environment Variables:
    TRAINED_MODEL_PATH: Path to trained model in volume (optional)
    MEDGEMMA_MODEL: HuggingFace model ID (default: google/medgemma-4b-it)
"""

import os
from pathlib import Path
from typing import Optional

# Only import modal if available - this file is meant to be run by Modal
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    modal = None

# Constants
HUGGINGFACE_MODELS = {
    "4b": "google/medgemma-4b-it",
    "27b": "google/medgemma-27b-it",
}

DEFAULT_MODEL = "google/medgemma-27b-it"
TRAINED_MODEL_MOUNT_PATH = "/models/trained"

if MODAL_AVAILABLE:
    # Define the Modal app
    app = modal.App("medai-compass")
    
    # Create the model image with all dependencies
    medgemma_image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "bitsandbytes>=0.42.0",
        "huggingface_hub>=0.20.0",
        "pillow>=10.0.0",
        "numpy>=1.26.0",
        "peft>=0.10.0",  # For LoRA adapter loading
    ).env({
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/transformers",
    })
    
    # Volumes for caching and trained models
    model_cache = modal.Volume.from_name("medgemma-model-cache", create_if_missing=True)
    checkpoints_volume = modal.Volume.from_name("medgemma-checkpoints", create_if_missing=True)
    
    @app.cls(
        image=medgemma_image,
        gpu="H100",
        volumes={
            "/root/.cache/huggingface": model_cache,
            TRAINED_MODEL_MOUNT_PATH: checkpoints_volume,
        },
        timeout=600,
        scaledown_window=300,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    @modal.concurrent(max_inputs=4)
    class MedGemmaInference:
        """Modal class for MedGemma inference on H100 GPU.
        
        Supports loading trained/fine-tuned models from volume or
        falling back to HuggingFace models.
        """
        
        model: Optional[object] = None
        tokenizer: Optional[object] = None
        processor: Optional[object] = None
        model_source: Optional[str] = None  # "trained" or "huggingface"
        model_path: Optional[str] = None
        
        def _find_trained_model(self) -> Optional[str]:
            """Find trained model in the checkpoints volume.
            
            Returns:
                Path to trained model if found, None otherwise.
            """
            trained_path = Path(TRAINED_MODEL_MOUNT_PATH)
            
            # Check for explicit path from environment
            env_path = os.environ.get("TRAINED_MODEL_PATH")
            if env_path:
                full_path = trained_path / env_path if not env_path.startswith("/") else Path(env_path)
                if full_path.exists() and self._is_valid_checkpoint(full_path):
                    return str(full_path)
            
            # Search for latest checkpoint in standard directories
            search_paths = [
                trained_path / "best",
                trained_path / "latest", 
                trained_path / "final",
                trained_path,
            ]
            
            for search_path in search_paths:
                if search_path.exists():
                    if self._is_valid_checkpoint(search_path):
                        return str(search_path)
                    
                    # Look for subdirectories with checkpoints
                    for subdir in sorted(search_path.iterdir(), reverse=True):
                        if subdir.is_dir() and self._is_valid_checkpoint(subdir):
                            return str(subdir)
            
            return None
        
        def _is_valid_checkpoint(self, path: Path) -> bool:
            """Check if path contains a valid model checkpoint.
            
            Args:
                path: Path to check.
                
            Returns:
                True if valid checkpoint, False otherwise.
            """
            path = Path(path)
            
            # Check for PyTorch model files
            pytorch_files = [
                "pytorch_model.bin",
                "model.safetensors",
                "pytorch_model.bin.index.json",
                "model.safetensors.index.json",
            ]
            
            # Check for config files
            config_files = ["config.json", "adapter_config.json"]
            
            has_model = any((path / f).exists() for f in pytorch_files)
            has_config = any((path / f).exists() for f in config_files)
            
            return has_model or has_config
        
        @modal.enter()
        def load_model(self):
            """Load model on container startup.
            
            Priority:
            1. Trained/fine-tuned model from checkpoints volume
            2. HuggingFace MedGemma model (fallback)
            """
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
            
            # Try to find trained model first
            trained_model_path = self._find_trained_model()
            
            if trained_model_path:
                print(f"Found trained model at: {trained_model_path}")
                model_path = trained_model_path
                self.model_source = "trained"
            else:
                model_path = os.environ.get("MEDGEMMA_MODEL", DEFAULT_MODEL)
                self.model_source = "huggingface"
                print(f"No trained model found, using HuggingFace: {model_path}")
            
            self.model_path = model_path
            print(f"Loading model: {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Check for LoRA adapter
            adapter_config = Path(model_path) / "adapter_config.json" if self.model_source == "trained" else None
            
            if adapter_config and adapter_config.exists():
                # Load base model first, then adapter
                from peft import PeftModel
                
                base_model_name = os.environ.get("MEDGEMMA_MODEL", DEFAULT_MODEL)
                print(f"Loading base model: {base_model_name}")
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
                
                print(f"Loading LoRA adapter from: {model_path}")
                self.model = PeftModel.from_pretrained(base_model, model_path)
            else:
                # Load model in full precision on H100
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True
                )
            
            # Load processor for multimodal
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
            except Exception:
                # Fallback to HuggingFace processor for trained models
                if self.model_source == "trained":
                    try:
                        self.processor = AutoProcessor.from_pretrained(
                            os.environ.get("MEDGEMMA_MODEL", DEFAULT_MODEL),
                            trust_remote_code=True
                        )
                    except Exception:
                        self.processor = None
                else:
                    self.processor = None
            
            print(f"Model loaded successfully on {torch.cuda.get_device_name()}")
            print(f"Model source: {self.model_source}")
            print(f"Model path: {self.model_path}")
            model_cache.commit()
        
        @modal.method()
        def generate(
            self,
            prompt: str,
            max_tokens: int = 512,
            temperature: float = 0.1,
            system_prompt: Optional[str] = None
        ) -> dict:
            """
            Generate text response from prompt.
            
            Args:
                prompt: User prompt
                max_tokens: Maximum tokens to generate
                temperature: Sampling temperature
                system_prompt: Optional system prompt
                
            Returns:
                Dict with response, confidence, and metadata
            """
            import torch
            
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Tokenize
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # Extract confidence (simplified)
            confidence = self._extract_confidence(response)
            
            return {
                "response": response,
                "confidence": confidence,
                "model": self.model_path or os.environ.get("MEDGEMMA_MODEL", DEFAULT_MODEL),
                "model_source": self.model_source,
                "gpu": "H100",
                "tokens_generated": len(outputs[0]) - inputs["input_ids"].shape[1]
            }
        
        @modal.method()
        def analyze_image(
            self,
            image_bytes: bytes,
            prompt: str,
            max_tokens: int = 1024
        ) -> dict:
            """
            Analyze medical image with text prompt.
            
            Args:
                image_bytes: Image as bytes (PNG/JPEG)
                prompt: Analysis prompt
                max_tokens: Maximum tokens to generate
                
            Returns:
                Dict with response and metadata
            """
            import torch
            from PIL import Image
            import io
            
            if self.processor is None:
                return {
                    "error": "Model does not support multimodal inference",
                    "response": None
                }
            
            # Load image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Use chat template format for MedGemma/Gemma3 multimodal models
            # This is the proper format as per HuggingFace documentation
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist. Provide detailed, accurate medical image analysis."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            # Process inputs using chat template
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=torch.bfloat16)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Get only the generated tokens (exclude input)
            generation = outputs[0][input_len:]
            response = self.processor.decode(generation, skip_special_tokens=True)
            
            return {
                "response": response,
                "confidence": self._extract_confidence(response),
                "model": self.model_path or os.environ.get("MEDGEMMA_MODEL", DEFAULT_MODEL),
                "model_source": self.model_source,
                "gpu": "H100"
            }
        
        @modal.method()
        def batch_generate(
            self,
            prompts: list[str],
            max_tokens: int = 256,
            temperature: float = 0.1
        ) -> list[dict]:
            """
            Generate responses for multiple prompts.
            
            Args:
                prompts: List of prompts
                max_tokens: Maximum tokens per response
                temperature: Sampling temperature
                
            Returns:
                List of response dicts
            """
            results = []
            for prompt in prompts:
                result = self.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                results.append(result)
            return results
        
        @modal.method()
        def health_check(self) -> dict:
            """Check model health and GPU status."""
            import torch
            
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "model_path": self.model_path,
                "model_source": self.model_source,
                "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
            }
        
        @modal.method()
        def get_model_info(self) -> dict:
            """Get detailed model information.
            
            Returns:
                Dict with model source, path, and configuration details.
            """
            import torch
            
            model_config = {}
            if self.model is not None:
                try:
                    model_config = {
                        "num_parameters": sum(p.numel() for p in self.model.parameters()),
                        "dtype": str(next(self.model.parameters()).dtype),
                        "device": str(next(self.model.parameters()).device),
                    }
                except Exception:
                    pass
            
            return {
                "model_path": self.model_path,
                "model_source": self.model_source,
                "is_trained_model": self.model_source == "trained",
                "model_loaded": self.model is not None,
                "tokenizer_loaded": self.tokenizer is not None,
                "processor_loaded": self.processor is not None,
                "multimodal_capable": self.processor is not None,
                "model_config": model_config,
                "gpu": {
                    "name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
                }
            }
        
        def _extract_confidence(self, response: str) -> float:
            """Extract confidence score from response."""
            import re
            
            # Look for explicit confidence mentions
            patterns = [
                r'confidence[:\s]+(\d+(?:\.\d+)?)\s*%',
                r'(\d+(?:\.\d+)?)\s*%\s*(?:confident|certainty)',
                r'certainty[:\s]+(\d+(?:\.\d+)?)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response.lower())
                if match:
                    value = float(match.group(1))
                    return value / 100 if value > 1 else value
            
            # Default confidence based on response quality indicators
            if any(word in response.lower() for word in ["cannot", "unable", "uncertain"]):
                return 0.5
            
            return 0.85  # Default confidence
    
    
    # Standalone functions for testing
    @app.function(
        image=medgemma_image,
        gpu="H100",
        volumes={"/root/.cache/huggingface": model_cache},
        timeout=300,
    )
    def test_inference(prompt: str = "What are the symptoms of pneumonia?") -> dict:
        """Test function for quick inference testing."""
        inference = MedGemmaInference()
        inference.load_model()
        return inference.generate(prompt)
    
    
    @app.local_entrypoint()
    def main():
        """Local entrypoint for testing."""
        print("Testing MedGemma Modal inference...")
        result = test_inference.remote("What are the common symptoms of pneumonia?")
        print(f"Response: {result['response'][:500]}...")
        print(f"Confidence: {result['confidence']}")
        print(f"GPU: {result['gpu']}")


# For non-Modal environments, provide a stub
if not MODAL_AVAILABLE:
    class MedGemmaInference:
        """Stub class when Modal is not available."""
        
        def __init__(self):
            raise ImportError(
                "Modal is not installed. Install with: pip install modal\n"
                "Or use local GPU inference instead."
            )
