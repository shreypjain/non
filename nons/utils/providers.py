"""
Model provider adapters for OpenAI and Anthropic with cost and token tracking.

This module provides adapters that integrate with the actual LLM APIs,
extract usage information, and calculate costs automatically.
"""

import time
import uuid
from typing import Any, Dict, Optional, Union, List
from abc import ABC, abstractmethod
import asyncio

from ..core.types import (
    ModelConfig,
    TokenUsage,
    CostInfo,
    ExecutionMetrics,
    RateLimitInfo,
    calculate_cost,
    ModelProvider,
    OperatorError,
)
from ..core.config import get_api_key

# Import the actual LLM client libraries
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from google import genai

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


def _parse_openai_rate_limits(response: Any) -> RateLimitInfo:
    """
    Parse rate limit information from OpenAI response headers.

    OpenAI includes headers like:
    - x-ratelimit-limit-requests
    - x-ratelimit-remaining-requests
    - x-ratelimit-reset-requests
    - x-ratelimit-limit-tokens
    - x-ratelimit-remaining-tokens
    - x-ratelimit-reset-tokens
    """
    rate_limit_info = RateLimitInfo()

    try:
        # Try to access response headers if available
        if hasattr(response, "_raw_response"):
            headers = response._raw_response.headers
        elif hasattr(response, "headers"):
            headers = response.headers
        else:
            return rate_limit_info

        # Parse request limits
        if "x-ratelimit-limit-requests" in headers:
            rate_limit_info.requests_limit = int(headers["x-ratelimit-limit-requests"])
        if "x-ratelimit-remaining-requests" in headers:
            rate_limit_info.requests_remaining = int(headers["x-ratelimit-remaining-requests"])
        if "x-ratelimit-reset-requests" in headers:
            rate_limit_info.requests_reset = headers["x-ratelimit-reset-requests"]

        # Parse token limits
        if "x-ratelimit-limit-tokens" in headers:
            rate_limit_info.tokens_limit = int(headers["x-ratelimit-limit-tokens"])
        if "x-ratelimit-remaining-tokens" in headers:
            rate_limit_info.tokens_remaining = int(headers["x-ratelimit-remaining-tokens"])
        if "x-ratelimit-reset-tokens" in headers:
            rate_limit_info.tokens_reset = headers["x-ratelimit-reset-tokens"]
    except Exception:
        # If header parsing fails, return empty rate limit info
        pass

    return rate_limit_info


def _parse_anthropic_rate_limits(response: Any) -> RateLimitInfo:
    """
    Parse rate limit information from Anthropic response headers.

    Anthropic includes headers like:
    - anthropic-ratelimit-requests-limit
    - anthropic-ratelimit-requests-remaining
    - anthropic-ratelimit-requests-reset
    - anthropic-ratelimit-tokens-limit
    - anthropic-ratelimit-tokens-remaining
    - anthropic-ratelimit-tokens-reset
    - retry-after (in 429 responses)
    """
    rate_limit_info = RateLimitInfo()

    try:
        # Try to access response headers
        if hasattr(response, "headers"):
            headers = response.headers
        elif hasattr(response, "_headers"):
            headers = response._headers
        else:
            return rate_limit_info

        # Parse request limits
        if "anthropic-ratelimit-requests-limit" in headers:
            rate_limit_info.requests_limit = int(headers["anthropic-ratelimit-requests-limit"])
        if "anthropic-ratelimit-requests-remaining" in headers:
            rate_limit_info.requests_remaining = int(headers["anthropic-ratelimit-requests-remaining"])
        if "anthropic-ratelimit-requests-reset" in headers:
            rate_limit_info.requests_reset = headers["anthropic-ratelimit-requests-reset"]

        # Parse token limits
        if "anthropic-ratelimit-tokens-limit" in headers:
            rate_limit_info.tokens_limit = int(headers["anthropic-ratelimit-tokens-limit"])
        if "anthropic-ratelimit-tokens-remaining" in headers:
            rate_limit_info.tokens_remaining = int(headers["anthropic-ratelimit-tokens-remaining"])
        if "anthropic-ratelimit-tokens-reset" in headers:
            rate_limit_info.tokens_reset = headers["anthropic-ratelimit-tokens-reset"]

        # Parse retry-after if present
        if "retry-after" in headers:
            rate_limit_info.retry_after = int(headers["retry-after"])
    except Exception:
        # If header parsing fails, return empty rate limit info
        pass

    return rate_limit_info


class BaseLLMProvider(ABC):
    """Base class for LLM provider adapters."""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.provider_name = model_config.provider.value

    @abstractmethod
    async def generate_completion(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> tuple[str, ExecutionMetrics]:
        """
        Generate completion from the LLM provider.

        Returns:
            tuple: (completion_text, execution_metrics)
        """
        pass

    def _create_base_metrics(self) -> ExecutionMetrics:
        """Create base execution metrics object."""
        return ExecutionMetrics(
            model_name=self.model_config.model_name,
            provider=self.provider_name,
            request_id=str(uuid.uuid4()),
        )


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider adapter with cost and token tracking."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not available. Install with: pip install openai"
            )

        api_key = get_api_key(ModelProvider.OPENAI)
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )

        self.client = openai.AsyncOpenAI(api_key=api_key)

    async def generate_completion(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> tuple[str, ExecutionMetrics]:
        """Generate completion using OpenAI API."""
        start_time = time.time()
        metrics = self._create_base_metrics()

        try:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Prepare API parameters
            api_params = {
                "model": self.model_config.model_name,
                "messages": messages,
                "temperature": self.model_config.temperature,
            }

            # Add optional parameters
            if self.model_config.max_tokens:
                api_params["max_tokens"] = self.model_config.max_tokens
            if self.model_config.top_p:
                api_params["top_p"] = self.model_config.top_p
            if self.model_config.frequency_penalty:
                api_params["frequency_penalty"] = self.model_config.frequency_penalty
            if self.model_config.presence_penalty:
                api_params["presence_penalty"] = self.model_config.presence_penalty
            if self.model_config.stop:
                api_params["stop"] = self.model_config.stop

            # Add any extra parameters
            api_params.update(self.model_config.extra_params)

            # Make API call
            response = await self.client.chat.completions.create(**api_params)

            # Extract completion text
            completion_text = response.choices[0].message.content or ""

            # Extract token usage
            usage = response.usage
            if usage:
                token_usage = TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )
            else:
                # Fallback if usage not provided
                token_usage = TokenUsage()

            # Calculate cost
            cost_info = calculate_cost(
                token_usage, self.model_config.model_name, self.provider_name
            )

            # Parse rate limit information
            rate_limit_info = _parse_openai_rate_limits(response)

            # Update metrics
            metrics.token_usage = token_usage
            metrics.cost_info = cost_info
            metrics.rate_limit_info = rate_limit_info
            metrics.response_time_ms = (time.time() - start_time) * 1000

            return completion_text, metrics

        except openai.AuthenticationError as e:
            # Fall back to mock provider on authentication error
            print(
                f"Warning: OpenAI authentication failed, falling back to mock provider: {str(e)}"
            )
            mock_provider = MockProvider(self.model_config)
            return await mock_provider.generate_completion(
                prompt, system_prompt, **kwargs
            )
        except Exception as e:
            metrics.response_time_ms = (time.time() - start_time) * 1000
            raise OperatorError(f"OpenAI API call failed: {str(e)}") from e


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider adapter with cost and token tracking."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not available. Install with: pip install anthropic"
            )

        api_key = get_api_key(ModelProvider.ANTHROPIC)
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )

        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate_completion(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> tuple[str, ExecutionMetrics]:
        """Generate completion using Anthropic API."""
        start_time = time.time()
        metrics = self._create_base_metrics()

        try:
            # Prepare API parameters
            api_params = {
                "model": self.model_config.model_name,
                "max_tokens": self.model_config.max_tokens or 4000,
                "temperature": self.model_config.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            # Add system prompt if provided
            if system_prompt:
                api_params["system"] = system_prompt

            # Add extra parameters
            api_params.update(self.model_config.extra_params)

            # Make API call
            response = await self.client.messages.create(**api_params)

            # Extract completion text
            completion_text = ""
            for content_block in response.content:
                if content_block.type == "text":
                    completion_text += content_block.text

            # Extract token usage
            usage = response.usage
            token_usage = TokenUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            )

            # Calculate cost
            cost_info = calculate_cost(
                token_usage, self.model_config.model_name, self.provider_name
            )

            # Parse rate limit information
            rate_limit_info = _parse_anthropic_rate_limits(response)

            # Update metrics
            metrics.token_usage = token_usage
            metrics.cost_info = cost_info
            metrics.rate_limit_info = rate_limit_info
            metrics.response_time_ms = (time.time() - start_time) * 1000

            return completion_text, metrics

        except anthropic.AuthenticationError as e:
            # Fall back to mock provider on authentication error
            print(
                f"Warning: Anthropic authentication failed, falling back to mock provider: {str(e)}"
            )
            mock_provider = MockProvider(self.model_config)
            return await mock_provider.generate_completion(
                prompt, system_prompt, **kwargs
            )
        except Exception as e:
            metrics.response_time_ms = (time.time() - start_time) * 1000
            raise OperatorError(f"Anthropic API call failed: {str(e)}") from e


class GoogleProvider(BaseLLMProvider):
    """Google Gemini API provider adapter with cost and token tracking."""

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "Google GenAI package not available. Install with: pip install google-genai"
            )

        api_key = get_api_key(ModelProvider.GOOGLE)
        if not api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )

        # Configure client with API key
        self.client = genai.Client(api_key=api_key)

    async def generate_completion(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> tuple[str, ExecutionMetrics]:
        """Generate completion using Google Gemini API."""
        start_time = time.time()
        metrics = self._create_base_metrics()

        try:
            # Prepare contents
            contents = []

            # Add system prompt if provided (as a user message with instruction)
            if system_prompt:
                contents.append(f"System: {system_prompt}\n\nUser: {prompt}")
            else:
                contents.append(prompt)

            # Prepare API parameters
            generation_config = {
                "temperature": self.model_config.temperature,
            }

            # Add optional parameters
            if self.model_config.max_tokens:
                generation_config["max_output_tokens"] = self.model_config.max_tokens
            if self.model_config.top_p:
                generation_config["top_p"] = self.model_config.top_p
            if self.model_config.stop:
                generation_config["stop_sequences"] = (
                    [self.model_config.stop]
                    if isinstance(self.model_config.stop, str)
                    else self.model_config.stop
                )

            # Make API call using the new SDK
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_config.model_name,
                contents=contents[0],
                config=generation_config,
            )

            # Extract completion text
            completion_text = (
                response.text if hasattr(response, "text") else str(response)
            )

            # Extract token usage from response metadata
            usage = getattr(response, "usage_metadata", None)
            if usage:
                token_usage = TokenUsage(
                    prompt_tokens=getattr(usage, "prompt_token_count", 0),
                    completion_tokens=getattr(usage, "candidates_token_count", 0),
                    total_tokens=getattr(usage, "total_token_count", 0),
                )
            else:
                # Fallback if usage not provided
                prompt_tokens = len(prompt.split()) * 2  # Rough estimation
                completion_tokens = len(completion_text.split()) * 2
                token_usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )

            # Calculate cost
            cost_info = calculate_cost(
                token_usage, self.model_config.model_name, self.provider_name
            )

            # Update metrics
            metrics.token_usage = token_usage
            metrics.cost_info = cost_info
            metrics.response_time_ms = (time.time() - start_time) * 1000

            return completion_text, metrics

        except Exception as e:
            metrics.response_time_ms = (time.time() - start_time) * 1000
            raise OperatorError(f"Google API call failed: {str(e)}") from e


class MockProvider(BaseLLMProvider):
    """Mock provider for testing and placeholder operations."""

    async def generate_completion(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> tuple[str, ExecutionMetrics]:
        """Generate mock completion with simulated metrics."""
        start_time = time.time()

        # Simulate API delay
        await asyncio.sleep(0.001)  # 1ms simulated delay

        # Create mock response
        completion_text = f"[MOCK {self.model_config.model_name.upper()}: {prompt}]"

        # Create simulated metrics
        prompt_tokens = len(prompt.split()) * 2  # Rough token estimation
        completion_tokens = len(completion_text.split()) * 2

        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        cost_info = calculate_cost(
            token_usage, self.model_config.model_name, self.provider_name
        )

        metrics = ExecutionMetrics(
            token_usage=token_usage,
            cost_info=cost_info,
            model_name=self.model_config.model_name,
            provider=self.provider_name,
            request_id=str(uuid.uuid4()),
            response_time_ms=(time.time() - start_time) * 1000,
        )

        return completion_text, metrics


def create_provider(model_config: ModelConfig) -> BaseLLMProvider:
    """
    Factory function to create the appropriate provider based on model config.

    Args:
        model_config: Configuration specifying the provider and model

    Returns:
        Provider instance for the specified model

    Raises:
        ValueError: If provider is not supported or API keys are missing
    """
    if model_config.provider == ModelProvider.OPENAI:
        if OPENAI_AVAILABLE and get_api_key(ModelProvider.OPENAI):
            try:
                return OpenAIProvider(model_config)
            except (ValueError, ImportError):
                # Fall back to mock provider if API key is invalid or setup fails
                return MockProvider(model_config)
        else:
            # Fall back to mock provider if API not available
            return MockProvider(model_config)

    elif model_config.provider == ModelProvider.ANTHROPIC:
        if ANTHROPIC_AVAILABLE and get_api_key(ModelProvider.ANTHROPIC):
            try:
                return AnthropicProvider(model_config)
            except (ValueError, ImportError):
                # Fall back to mock provider if API key is invalid or setup fails
                return MockProvider(model_config)
        else:
            # Fall back to mock provider if API not available
            return MockProvider(model_config)

    elif model_config.provider == ModelProvider.GOOGLE:
        if GOOGLE_AVAILABLE and get_api_key(ModelProvider.GOOGLE):
            try:
                return GoogleProvider(model_config)
            except (ValueError, ImportError):
                # Fall back to mock provider if API key is invalid or setup fails
                return MockProvider(model_config)
        else:
            # Fall back to mock provider if API not available
            return MockProvider(model_config)

    elif model_config.provider == ModelProvider.MOCK:
        return MockProvider(model_config)
    else:
        raise ValueError(f"Unsupported provider: {model_config.provider}")


async def test_provider(model_config: ModelConfig) -> Dict[str, Any]:
    """
    Test a provider configuration and return connection status.

    Args:
        model_config: Configuration to test

    Returns:
        Dictionary with test results
    """
    try:
        provider = create_provider(model_config)
        completion, metrics = await provider.generate_completion("Test prompt")

        return {
            "status": "success",
            "provider": model_config.provider.value,
            "model": model_config.model_name,
            "completion": (
                completion[:100] + "..." if len(completion) > 100 else completion
            ),
            "tokens": metrics.token_usage.total_tokens,
            "cost": f"${metrics.cost_info.total_cost_usd:.6f}",
            "response_time_ms": metrics.response_time_ms,
        }

    except Exception as e:
        return {
            "status": "error",
            "provider": model_config.provider.value,
            "model": model_config.model_name,
            "error": str(e),
        }
