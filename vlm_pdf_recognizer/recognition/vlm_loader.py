"""
VLM model loader using Ollama API with GPU preference and CPU fallback.

This module provides singleton-based Ollama client for glm-ocr model
with automatic hardware detection and health checking.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any
import logging

import requests

logger = logging.getLogger(__name__)

# Ollama API defaults
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL_NAME = "glm-ocr"


@dataclass
class VLMConfig:
    """Configuration for Ollama VLM model.

    Attributes:
        model_name: Ollama model name (default: glm-ocr)
        ollama_host: Ollama server URL (default: http://localhost:11434)
        device: Detected device ("gpu" or "cpu")
        vram_gb: Available GPU VRAM in GB (if GPU available)
        num_gpu: Number of GPU layers to use (-1=auto, 0=CPU only)
        temperature: Generation temperature (0.0 for deterministic)
        num_predict: Max tokens to generate
    """

    model_name: str = DEFAULT_MODEL_NAME
    ollama_host: str = DEFAULT_OLLAMA_HOST
    device: str = ""  # Populated by detection
    vram_gb: Optional[float] = None
    num_gpu: int = -1  # -1 = auto (Ollama decides), 0 = CPU only
    temperature: float = 0.0
    num_predict: int = 256

    def __post_init__(self):
        """Detect hardware and configure GPU/CPU preference."""
        if not self.device:
            self._detect_hardware()

    def _detect_hardware(self):
        """Detect available hardware (GPU/CPU) and set device preference."""
        try:
            import torch

            if torch.cuda.is_available():
                self.device = "gpu"
                self.vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.num_gpu = -1  # Let Ollama use all GPU layers
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} with {self.vram_gb:.2f}GB VRAM")
                logger.info("Ollama will prioritize GPU for inference")
            else:
                self.device = "cpu"
                self.num_gpu = 0  # Force CPU only
                logger.info("No GPU detected, Ollama will use CPU")

        except ImportError:
            # torch not installed - try nvidia-smi as fallback
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    vram_mb = float(result.stdout.strip().split('\n')[0])
                    self.vram_gb = vram_mb / 1024
                    self.device = "gpu"
                    self.num_gpu = -1
                    logger.info(f"GPU detected via nvidia-smi: {self.vram_gb:.2f}GB VRAM")
                else:
                    self.device = "cpu"
                    self.num_gpu = 0
                    logger.info("No GPU detected (nvidia-smi failed), using CPU")
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                self.device = "cpu"
                self.num_gpu = 0
                logger.info("No GPU detection available, defaulting to CPU")

    def to_log_dict(self) -> dict:
        """Convert to dictionary for structured logging."""
        return {
            "model": self.model_name,
            "ollama_host": self.ollama_host,
            "device": self.device,
            "vram_gb": self.vram_gb,
            "num_gpu": self.num_gpu,
        }


class OllamaClient:
    """HTTP client wrapper for Ollama API."""

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST):
        self.host = host.rstrip('/')
        self.session = requests.Session()

    def check_health(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.ConnectionError:
            return False
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def check_model_available(self, model_name: str) -> bool:
        """Check if the specified model is available in Ollama."""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                return model_name in models
            return False
        except Exception as e:
            logger.warning(f"Model availability check failed: {e}")
            return False

    def generate(self, model: str, prompt: str, images: list = None,
                 temperature: float = 0.0, num_predict: int = 256,
                 num_gpu: int = -1) -> str:
        """Call Ollama generate API.

        Args:
            model: Model name
            prompt: Text prompt
            images: List of base64-encoded images
            temperature: Generation temperature
            num_predict: Max tokens to generate
            num_gpu: Number of GPU layers (-1=auto, 0=CPU)

        Returns:
            Generated text response

        Raises:
            RuntimeError: If API call fails
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_gpu": num_gpu,
            }
        }

        if images:
            payload["images"] = images

        try:
            response = self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=120  # 2 min timeout for inference
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama API error (status {response.status_code}): {response.text}"
                )

            result = response.json()
            return result.get("response", "")

        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama server at {self.host}. "
                f"Please ensure Ollama is running: ollama serve"
            )
        except requests.Timeout:
            raise RuntimeError("Ollama inference timed out (>120s)")
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Ollama API call failed: {e}")


class VLMLoader:
    """Singleton VLM loader using Ollama API for glm-ocr model.

    This class manages the Ollama client connection and model availability
    with automatic hardware detection for GPU/CPU preference.
    """

    _instance: Optional['VLMLoader'] = None
    _client: Optional[OllamaClient] = None
    _config: Optional[VLMConfig] = None

    def __new__(cls):
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'VLMLoader':
        """Get singleton instance of VLMLoader."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, config: Optional[VLMConfig] = None) -> Tuple[OllamaClient, None, str, str]:
        """Initialize Ollama client and verify model availability.

        Args:
            config: Optional VLMConfig. If None, creates default config with auto-detection.

        Returns:
            Tuple of (client, None, device, model_name)
            Note: tokenizer is None (not needed for Ollama API)

        Raises:
            RuntimeError: If Ollama server is not reachable or model not available
        """
        if config is None:
            config = VLMConfig()

        self._config = config

        logger.info(f"Initializing Ollama client for model: {config.model_name}")
        logger.info(f"Hardware config: {config.to_log_dict()}")

        # Create Ollama client
        self._client = OllamaClient(host=config.ollama_host)

        # Check Ollama server health
        if not self._client.check_health():
            raise RuntimeError(
                f"Ollama server is not running at {config.ollama_host}. "
                f"Please start it with: ollama serve"
            )

        logger.info(f"Ollama server is running at {config.ollama_host}")

        # Check model availability
        if not self._client.check_model_available(config.model_name):
            logger.warning(
                f"Model '{config.model_name}' not found in Ollama. "
                f"Attempting to pull..."
            )
            # Try to pull the model
            try:
                pull_response = self._client.session.post(
                    f"{config.ollama_host}/api/pull",
                    json={"name": config.model_name, "stream": False},
                    timeout=600  # 10 min for model download
                )
                if pull_response.status_code != 200:
                    raise RuntimeError(
                        f"Failed to pull model '{config.model_name}': {pull_response.text}"
                    )
                logger.info(f"Model '{config.model_name}' pulled successfully")
            except requests.Timeout:
                raise RuntimeError(
                    f"Model pull timed out. Please manually run: ollama pull {config.model_name}"
                )
        else:
            logger.info(f"Model '{config.model_name}' is available")

        # Log GPU/CPU configuration
        if config.device == "gpu":
            logger.info(f"GPU mode: {config.vram_gb:.2f}GB VRAM, num_gpu={config.num_gpu}")
        else:
            logger.info(f"CPU mode: num_gpu=0")

        return self._client, None, config.device, config.model_name

    def reload_model(self) -> Tuple[OllamaClient, None, str, str]:
        """Force reconnect to Ollama server.

        Returns:
            Tuple of (client, None, device, model_name)
        """
        if self._config is None:
            raise RuntimeError("Cannot reload: no config available")

        logger.info("Reconnecting to Ollama server...")
        return self.load_model(self._config)

    def get_model(self) -> Tuple[OllamaClient, None, str, str]:
        """Get Ollama client, initializing if necessary."""
        if self._client is None:
            return self.load_model()
        return self._client, None, self._config.device, self._config.model_name

    @property
    def config(self) -> Optional[VLMConfig]:
        """Get current configuration."""
        return self._config
