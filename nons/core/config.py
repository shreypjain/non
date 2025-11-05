"""
Configuration management for NoN (Network of Networks).

This module provides configuration classes and default settings for
network-level, layer-level, and node-level behavior, including
environment variable detection and default value management.
"""

import os
from typing import Optional, Dict, Any
from .types import (
    ModelConfig,
    LayerConfig,
    NetworkConfig,
    RateLimitConfig,
    ModelProvider,
)


# Default configurations
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4000


def get_default_model_config(provider: Optional[ModelProvider] = None) -> ModelConfig:
    """
    Get default model configuration with environment variable overrides.

    Args:
        provider: Optional provider to use (defaults to OpenAI if available)

    Returns:
        ModelConfig with appropriate defaults
    """
    # Determine provider from environment or default
    if provider is None:
        # Check for API keys to determine default provider
        if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
            provider = ModelProvider.GOOGLE
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = ModelProvider.ANTHROPIC
        elif os.getenv("OPENAI_API_KEY"):
            provider = ModelProvider.OPENAI
        else:
            # Default to Google if no keys found
            provider = ModelProvider.GOOGLE

    # Set model name based on provider
    if provider == ModelProvider.OPENAI:
        model_name = os.getenv("NON_DEFAULT_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    elif provider == ModelProvider.ANTHROPIC:
        model_name = os.getenv("NON_DEFAULT_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    elif provider == ModelProvider.GOOGLE:
        model_name = os.getenv("NON_DEFAULT_GOOGLE_MODEL", DEFAULT_GOOGLE_MODEL)
    else:
        model_name = "unknown"

    # Get other defaults from environment
    temperature = float(os.getenv("NON_DEFAULT_TEMPERATURE", DEFAULT_TEMPERATURE))
    max_tokens = int(os.getenv("NON_DEFAULT_MAX_TOKENS", DEFAULT_MAX_TOKENS))

    return ModelConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_default_layer_config() -> LayerConfig:
    """
    Get default layer configuration with environment variable overrides.

    Returns:
        LayerConfig with appropriate defaults
    """
    return LayerConfig(
        timeout_seconds=float(os.getenv("NON_LAYER_TIMEOUT", 30.0)),
        max_retries=int(os.getenv("NON_MAX_RETRIES", 3)),
        retry_delay_seconds=float(os.getenv("NON_RETRY_DELAY", 1.0)),
        min_success_threshold=float(os.getenv("NON_MIN_SUCCESS_THRESHOLD", 1.0)),
    )


def get_default_network_config() -> NetworkConfig:
    """
    Get default network configuration with environment variable overrides.

    Returns:
        NetworkConfig with appropriate defaults
    """
    return NetworkConfig(
        max_concurrent_layers=int(os.getenv("NON_MAX_CONCURRENT_LAYERS", 1)),
        global_timeout_seconds=float(os.getenv("NON_GLOBAL_TIMEOUT", 300.0)),
        enable_tracing=os.getenv("NON_ENABLE_TRACING", "true").lower() == "true",
        enable_metrics=os.getenv("NON_ENABLE_METRICS", "true").lower() == "true",
    )


def get_rate_limit_config(provider: ModelProvider) -> RateLimitConfig:
    """
    Get rate limit configuration for a specific provider.

    Args:
        provider: The model provider

    Returns:
        RateLimitConfig for the provider
    """
    provider_upper = provider.value.upper()

    return RateLimitConfig(
        requests_per_minute=int(
            os.getenv(f"NON_{provider_upper}_REQUESTS_PER_MINUTE", 60)
        ),
        tokens_per_minute=int(
            os.getenv(f"NON_{provider_upper}_TOKENS_PER_MINUTE", 150000)
        ),
        concurrent_requests=int(
            os.getenv(f"NON_{provider_upper}_CONCURRENT_REQUESTS", 10)
        ),
    )


def get_api_key(provider: ModelProvider) -> Optional[str]:
    """
    Get API key for a specific provider from environment variables.

    Args:
        provider: The model provider

    Returns:
        API key if found, None otherwise
    """
    if provider == ModelProvider.OPENAI:
        return os.getenv("OPENAI_API_KEY")
    elif provider == ModelProvider.ANTHROPIC:
        return os.getenv("ANTHROPIC_API_KEY")
    elif provider == ModelProvider.GOOGLE:
        return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    else:
        return None


def validate_api_keys() -> Dict[ModelProvider, bool]:
    """
    Validate that required API keys are available.

    Returns:
        Dictionary mapping providers to key availability
    """
    return {
        ModelProvider.OPENAI: get_api_key(ModelProvider.OPENAI) is not None,
        ModelProvider.ANTHROPIC: get_api_key(ModelProvider.ANTHROPIC) is not None,
        ModelProvider.GOOGLE: get_api_key(ModelProvider.GOOGLE) is not None,
    }


class ConfigManager:
    """
    Centralized configuration manager for NoN system.

    Provides methods to get and update configurations at different levels
    (network, layer, node) with environment variable overrides.
    """

    def __init__(self):
        self._model_configs: Dict[str, ModelConfig] = {}
        self._layer_configs: Dict[str, LayerConfig] = {}
        self._network_config: Optional[NetworkConfig] = None
        self._rate_limit_configs: Dict[ModelProvider, RateLimitConfig] = {}

    def get_model_config(
        self, config_name: str = "default", provider: Optional[ModelProvider] = None
    ) -> ModelConfig:
        """Get model configuration by name."""
        if config_name not in self._model_configs:
            self._model_configs[config_name] = get_default_model_config(provider)
        return self._model_configs[config_name]

    def set_model_config(self, config_name: str, config: ModelConfig) -> None:
        """Set model configuration by name."""
        self._model_configs[config_name] = config

    def get_layer_config(self, config_name: str = "default") -> LayerConfig:
        """Get layer configuration by name."""
        if config_name not in self._layer_configs:
            self._layer_configs[config_name] = get_default_layer_config()
        return self._layer_configs[config_name]

    def set_layer_config(self, config_name: str, config: LayerConfig) -> None:
        """Set layer configuration by name."""
        self._layer_configs[config_name] = config

    def get_network_config(self) -> NetworkConfig:
        """Get network configuration."""
        if self._network_config is None:
            self._network_config = get_default_network_config()
        return self._network_config

    def set_network_config(self, config: NetworkConfig) -> None:
        """Set network configuration."""
        self._network_config = config

    def get_rate_limit_config(self, provider: ModelProvider) -> RateLimitConfig:
        """Get rate limit configuration for provider."""
        if provider not in self._rate_limit_configs:
            self._rate_limit_configs[provider] = get_rate_limit_config(provider)
        return self._rate_limit_configs[provider]

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate current configuration and return status.

        Returns:
            Dictionary with validation results
        """
        api_keys = validate_api_keys()
        available_providers = [p for p, available in api_keys.items() if available]

        return {
            "api_keys": api_keys,
            "available_providers": available_providers,
            "has_valid_provider": len(available_providers) > 0,
            "network_config": self.get_network_config().model_dump(),
            "default_model_config": self.get_model_config().model_dump(),
        }


# Global configuration manager instance
_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    return _config_manager
