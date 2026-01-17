"""Modal GPU Deployment for MedGemma Inference.

This module defines the Modal app and GPU functions for running
MedGemma models on cloud H100 GPUs.

Deployment:
    modal deploy medai_compass/modal/app.py

Local Testing:
    modal run medai_compass/modal/app.py
"""

import os
from typing import Optional

# Only import modal if available - this file is meant to be run by Modal
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False
    modal = None

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
    ).env({
        "HF_HOME": "/root/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface/transformers",
    })
    
    # Volume for caching models
    model_cache = modal.Volume.from_name("medgemma-model-cache", create_if_missing=True)
    
    @app.cls(
        image=medgemma_image,
        gpu=modal.gpu.H100(count=1),
        volumes={"/root/.cache/huggingface": model_cache},
        timeout=600,
        container_idle_timeout=300,
        secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
    )
    class MedGemmaInference:
        """Modal class for MedGemma inference on H100 GPU."""
        
        model: Optional[object] = None
        tokenizer: Optional[object] = None
        processor: Optional[object] = None
        
        @modal.enter()
        def load_model(self):
            """Load model on container startup."""
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
            
            model_name = os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it")
            
            print(f"Loading model: {model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model in full precision on H100
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            # Load processor for multimodal
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            except Exception:
                self.processor = None
            
            print(f"Model loaded successfully on {torch.cuda.get_device_name()}")
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
                "model": os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it"),
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
            
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "response": response,
                "confidence": self._extract_confidence(response),
                "model": os.environ.get("MEDGEMMA_MODEL", "google/medgemma-4b-it"),
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
                "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
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
        gpu=modal.gpu.H100(count=1),
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
