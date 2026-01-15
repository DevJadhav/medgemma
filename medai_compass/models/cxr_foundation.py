"""CXR Foundation model wrapper for chest X-ray analysis.

Provides:
- Language-aligned embedding extraction (768-dim)
- Zero-shot classification with text labels
"""

from typing import Optional
import numpy as np

# Lazy imports
torch = None


def _lazy_import_torch():
    """Lazily import torch."""
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch


class CXRFoundationWrapper:
    """
    Wrapper for Google's CXR Foundation model.
    
    Provides contrastive language-image pretraining for chest X-rays,
    enabling zero-shot classification with arbitrary text labels.
    """
    
    def __init__(
        self,
        model_name: str = "google/cxr-foundation",
        device: Optional[str] = None
    ):
        """
        Initialize CXR Foundation wrapper.
        
        Args:
            model_name: Model identifier
            device: Device to use (cuda/cpu/auto)
        """
        _lazy_import_torch()
        
        self.model_name = model_name
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model from torch hub
        self.model = torch.hub.load(
            "google/health-ai-developer-foundations",
            "cxr_foundation",
            pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Model dimensions
        self.image_size = 896
        self.embedding_dim = 768
    
    def preprocess(self, image: np.ndarray) -> "torch.Tensor":
        """
        Preprocess chest X-ray for CXR Foundation.
        
        Args:
            image: Image array (H, W, 3) or (H, W)
            
        Returns:
            Preprocessed tensor
        """
        from PIL import Image
        from torchvision import transforms
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.shape[2] == 1:
            image = np.concatenate([image, image, image], axis=-1)
        
        transform = transforms.Compose([
            transforms.Resize((896, 896)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        
        pil_image = Image.fromarray(image)
        tensor = transform(pil_image).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract language-aligned embedding from chest X-ray.
        
        Args:
            image: Chest X-ray image array
            
        Returns:
            768-dimensional embedding vector
        """
        tensor = self.preprocess(image)
        
        with torch.no_grad():
            embedding = self.model(tensor)
        
        return embedding.detach().cpu().numpy().squeeze()
    
    def classify_zero_shot(
        self,
        image: np.ndarray,
        labels: list[str],
        prompt_template: str = "chest x-ray showing {}"
    ) -> dict[str, float]:
        """
        Perform zero-shot classification with text labels.
        
        Args:
            image: Chest X-ray image
            labels: List of condition labels
            prompt_template: Template for label prompts
            
        Returns:
            Dict mapping labels to probabilities
        """
        # Get image embedding
        image_tensor = self.preprocess(image)
        
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_tensor)
            
            # Encode text labels
            text_prompts = [prompt_template.format(label) for label in labels]
            text_embeddings = self.model.encode_text(text_prompts)
            
            # Compute similarities
            image_embedding = image_embedding.detach().cpu().numpy()
            text_embeddings = text_embeddings.detach().cpu().numpy()
        
        # Normalize embeddings
        image_norm = image_embedding / np.linalg.norm(image_embedding, axis=-1, keepdims=True)
        text_norm = text_embeddings / np.linalg.norm(text_embeddings, axis=-1, keepdims=True)
        
        # Compute cosine similarities
        similarities = np.dot(image_norm, text_norm.T).squeeze()
        
        # Convert to probabilities with softmax
        exp_sim = np.exp(similarities * 100)  # Temperature scaling
        probs = exp_sim / exp_sim.sum()
        
        return {label: float(prob) for label, prob in zip(labels, probs)}
