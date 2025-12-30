"""
VLM model loader with hardware-adaptive loading strategy.

This module provides singleton-based VLM loading with automatic hardware detection
and quantization fallback for CPU environments.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class VLMConfig:
    """Configuration for VLM model loading and hardware detection.

    Attributes:
        model_name: HuggingFace model identifier
        device: Selected device ("cuda" or "cpu")
        precision: Model precision ("BF16", "FP16", "INT8", "INT4", "BF16_CPU")
        vram_gb: Available GPU VRAM in GB (if device=="cuda")
        ram_gb: Available system RAM in GB (if device=="cpu")
        quantization_fallback: Whether to attempt INT8→INT4→unquantized fallback on CPU OOM
        cache_dir: HuggingFace cache directory override (None uses default ~/.cache/huggingface/)
    """

    model_name: str = "OpenGVLab/InternVL3_5-2B"
    device: str = ""  # Populated by detectio
    precision: str = ""  # Populated by detection
    vram_gb: Optional[float] = None
    ram_gb: Optional[float] = None
    quantization_fallback: bool = True
    cache_dir: Optional[str] = None

    def __post_init__(self):
        """Detect hardware and set device/precision if not already set."""
        if not self.device:
            self._detect_hardware()

    def _detect_hardware(self):
        """Detect available hardware (GPU/CPU) and set device/precision."""
        try:
            import torch
            import psutil

            if torch.cuda.is_available():
                self.device = "cuda"
                self.vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

                # Select precision based on available VRAM (2B model requires less VRAM)
                if self.vram_gb >= 8:
                    self.precision = "BF16"  # Best quality for 8GB+ VRAM (2B model)
                elif self.vram_gb >= 6:
                    self.precision = "FP16"  # Fallback for 6-8GB VRAM
                elif self.vram_gb >= 4:
                    self.precision = "INT8"  # 8-bit quantization for 4-6GB VRAM
                else:
                    self.precision = "INT4"  # 4-bit quantization for <4GB VRAM

                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} with {self.vram_gb:.2f}GB VRAM")
                logger.info(f"Selected precision: {self.precision}")
            else:
                self.device = "cpu"
                self.ram_gb = psutil.virtual_memory().available / 1e9

                # CPU defaults to INT8 quantization
                self.precision = "INT8"
                logger.info(f"No GPU detected, using CPU with {self.ram_gb:.2f}GB available RAM")
                logger.info(f"Selected precision: {self.precision} (quantized)")

        except ImportError as e:
            logger.error(f"Failed to import required libraries for hardware detection: {e}")
            # Fallback to CPU without quantization
            self.device = "cpu"
            self.precision = "BF16_CPU"
            logger.warning("Falling back to CPU without quantization (may require significant RAM)")

    def to_log_dict(self) -> dict:
        """Convert to dictionary for structured logging.

        Returns:
            Dictionary containing model name, device, precision, and memory info
        """
        return {
            "model": self.model_name,
            "device": self.device,
            "precision": self.precision,
            "vram_gb": self.vram_gb,
            "ram_gb": self.ram_gb,
        }


class VLMLoader:
    """Singleton VLM loader with hardware-adaptive model loading.

    This class manages the loading and caching of the InternVL vision-language model
    with automatic hardware detection and quantization fallback.
    """

    _instance: Optional['VLMLoader'] = None
    _model: Optional[Any] = None
    _tokenizer: Optional[Any] = None
    _config: Optional[VLMConfig] = None

    def __new__(cls):
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'VLMLoader':
        """Get singleton instance of VLMLoader.

        Returns:
            Singleton VLMLoader instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, config: Optional[VLMConfig] = None) -> Tuple[Any, Any, str, str]:
        """Load VLM model with hardware-adaptive strategy.

        Args:
            config: Optional VLMConfig. If None, creates default config with auto-detection

        Returns:
            Tuple of (model, tokenizer, device, precision)

        Raises:
            RuntimeError: If model loading fails after all retry attempts
        """
        if config is None:
            config = VLMConfig()

        self._config = config

        logger.info(f"Loading VLM model: {config.model_name}")
        logger.info(f"Hardware config: {config.to_log_dict()}")

        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError as e:
            raise RuntimeError(f"Failed to import required libraries: {e}. Please install: pip install torch transformers")

        # Load model based on device and precision
        if config.device == "cuda":
            self._model, self._tokenizer = self._load_gpu_model(config)
        else:
            self._model, self._tokenizer = self._load_cpu_model(config)

        logger.info(f"VLM model loaded successfully: {config.device} / {config.precision}")
        return self._model, self._tokenizer, config.device, config.precision

    def _load_gpu_model(self, config: VLMConfig) -> Tuple[Any, Any]:
        """Load model on GPU with BF16/FP16/INT8/INT4 precision.

        Args:
            config: VLM configuration

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            RuntimeError: If GPU loading fails
        """
        from transformers import AutoModel, AutoTokenizer
        import torch

        logger.info(f"Loading 2B model on GPU with {config.precision} precision...")

        try:
            # Load with quantization if INT8/INT4
            if config.precision == "INT8":
                logger.info("Loading with 8-bit quantization for GPU...")
                model = AutoModel.from_pretrained(
                    config.model_name,
                    dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    cache_dir=config.cache_dir
                ).eval()
            elif config.precision == "INT4":
                logger.info("Loading with 4-bit quantization for GPU...")
                model = AutoModel.from_pretrained(
                    config.model_name,
                    dtype=torch.bfloat16,
                    load_in_4bit=True,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    cache_dir=config.cache_dir
                ).eval()
            else:
                # BF16 or FP16 (full precision)
                if config.precision == "BF16":
                    torch_dtype = torch.bfloat16
                elif config.precision == "FP16":
                    torch_dtype = torch.float16
                else:
                    torch_dtype = torch.bfloat16  # Default to BF16

                model = AutoModel.from_pretrained(
                    config.model_name,
                    dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    cache_dir=config.cache_dir
                ).eval().cuda()

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                use_fast=False,
                cache_dir=config.cache_dir
            )

            logger.info(f"2B model loaded on GPU: {torch.cuda.get_device_name(0)}, {config.vram_gb:.2f}GB VRAM")
            return model, tokenizer

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"GPU OOM error with {config.precision}: {e}")

                # Try fallback quantization on GPU first
                if config.precision == "BF16":
                    logger.warning("Retrying with FP16...")
                    config.precision = "FP16"
                    return self._load_gpu_model(config)
                elif config.precision == "FP16":
                    logger.warning("Retrying with INT8 quantization...")
                    config.precision = "INT8"
                    return self._load_gpu_model(config)
                elif config.precision == "INT8":
                    logger.warning("Retrying with INT4 quantization...")
                    config.precision = "INT4"
                    return self._load_gpu_model(config)
                else:
                    # Final fallback to CPU
                    logger.warning("All GPU attempts failed, falling back to CPU with quantization...")
                    config.device = "cpu"
                    config.precision = "INT8"
                    return self._load_cpu_model(config)
            else:
                raise RuntimeError(f"Failed to load model on GPU: {e}")

    def _load_cpu_model(self, config: VLMConfig) -> Tuple[Any, Any]:
        """Load model on CPU with quantization fallback (INT8 → INT4 → unquantized).

        Args:
            config: VLM configuration

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            RuntimeError: If all CPU loading attempts fail
        """
        from transformers import AutoModel, AutoTokenizer
        import torch

        logger.info(f"Loading model on CPU with {config.precision} quantization...")

        # Try INT8 first
        if config.precision == "INT8" and config.quantization_fallback:
            try:
                model = AutoModel.from_pretrained(
                    config.model_name,
                    dtype=torch.bfloat16,
                    load_in_8bit=True,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    cache_dir=config.cache_dir
                ).eval()

                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_name,
                    trust_remote_code=True,
                    use_fast=False,
                    cache_dir=config.cache_dir
                )

                logger.info(f"CPU model loaded with INT8 quantization, {config.ram_gb:.2f}GB RAM available")
                return model, tokenizer

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"INT8 OOM error: {e}")
                    logger.warning("Falling back to INT4 quantization...")
                    config.precision = "INT4"
                else:
                    logger.error(f"INT8 loading failed (non-OOM): {e}")
                    raise RuntimeError(f"Failed to load INT8 model: {e}")

        # Try INT4 if INT8 failed or precision is INT4
        if config.precision == "INT4" and config.quantization_fallback:
            try:
                model = AutoModel.from_pretrained(
                    config.model_name,
                    dtype=torch.bfloat16,
                    load_in_4bit=True,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    cache_dir=config.cache_dir
                ).eval()

                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_name,
                    trust_remote_code=True,
                    use_fast=False,
                    cache_dir=config.cache_dir
                )

                logger.info(f"CPU model loaded with INT4 quantization, {config.ram_gb:.2f}GB RAM available")
                return model, tokenizer

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"INT4 OOM error: {e}")
                    logger.warning("Falling back to unquantized model (requires significant RAM)...")
                    config.precision = "BF16_CPU"
                else:
                    logger.error(f"INT4 loading failed (non-OOM): {e}")
                    raise RuntimeError(f"Failed to load INT4 model: {e}")

        # Final fallback: unquantized on CPU
        try:
            model = AutoModel.from_pretrained(
                config.model_name,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=config.cache_dir
            ).eval()

            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                use_fast=False,
                cache_dir=config.cache_dir
            )

            logger.warning(f"CPU model loaded WITHOUT quantization (may be slow), {config.ram_gb:.2f}GB RAM available")
            return model, tokenizer

        except Exception as e:
            raise RuntimeError(f"Failed to load model on CPU (all quantization levels failed): {e}")

    def reload_model(self) -> Tuple[Any, Any, str, str]:
        """Force reload model after OOM or other errors.

        This method releases existing model from memory and reloads with current config.

        Returns:
            Tuple of (model, tokenizer, device, precision)

        Raises:
            RuntimeError: If no config available or reload fails
        """
        if self._config is None:
            raise RuntimeError("Cannot reload model: no config available")

        logger.info("Reloading VLM model after error...")

        # Release existing model
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # Clear GPU cache if using CUDA
        if self._config.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear GPU cache: {e}")

        # Reload with current config
        return self.load_model(self._config)

    def get_model(self) -> Tuple[Any, Any, str, str]:
        """Get loaded model, loading if necessary.

        Returns:
            Tuple of (model, tokenizer, device, precision)
        """
        if self._model is None or self._tokenizer is None:
            return self.load_model()
        return self._model, self._tokenizer, self._config.device, self._config.precision
