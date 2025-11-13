#!/usr/bin/env python3
"""
Test Provider:Model String Parsing

Tests the ModelConfig.from_string() method and NON_DEFAULT_MODEL environment
variable with the new provider:model_name format.
"""

import os
import sys

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nons.core.types import ModelConfig, ModelProvider
from nons.core.config import get_default_model_config


def test_provider_string_parsing():
    """Test ModelConfig.from_string() with provider:model_name format."""
    print("=" * 70)
    print("TEST: ModelConfig.from_string() Provider:Model Parsing")
    print("=" * 70)
    print()

    # Test 1: Explicit provider:model_name format
    print("Test 1: Explicit provider:model_name format")
    config = ModelConfig.from_string("anthropic:claude-sonnet-4-5-20250929")
    print(f"  Input: 'anthropic:claude-sonnet-4-5-20250929'")
    print(f"  Provider: {config.provider}")
    print(f"  Model name: {config.model_name}")
    assert config.provider == ModelProvider.ANTHROPIC
    assert config.model_name == "claude-sonnet-4-5-20250929"
    print("  ✓ Anthropic explicit format works")
    print()

    # Test 2: OpenAI explicit format
    print("Test 2: OpenAI explicit format")
    config = ModelConfig.from_string("openai:gpt-4o-mini", temperature=0.9)
    print(f"  Input: 'openai:gpt-4o-mini', temperature=0.9")
    print(f"  Provider: {config.provider}")
    print(f"  Model name: {config.model_name}")
    print(f"  Temperature: {config.temperature}")
    assert config.provider == ModelProvider.OPENAI
    assert config.model_name == "gpt-4o-mini"
    assert config.temperature == 0.9
    print("  ✓ OpenAI explicit format works with custom temperature")
    print()

    # Test 3: Google explicit format (using "google" prefix)
    print("Test 3: Google explicit format")
    config = ModelConfig.from_string("google:gemini-2.0-flash")
    print(f"  Input: 'google:gemini-2.0-flash'")
    print(f"  Provider: {config.provider}")
    print(f"  Model name: {config.model_name}")
    assert config.provider == ModelProvider.GOOGLE
    assert config.model_name == "gemini-2.0-flash"
    print("  ✓ Google explicit format works")
    print()

    # Test 4: Google with "gemini" prefix (alternative)
    print("Test 4: Google with 'gemini' prefix")
    config = ModelConfig.from_string("gemini:gemini-2.0-flash")
    print(f"  Input: 'gemini:gemini-2.0-flash'")
    print(f"  Provider: {config.provider}")
    print(f"  Model name: {config.model_name}")
    assert config.provider == ModelProvider.GOOGLE
    assert config.model_name == "gemini-2.0-flash"
    print("  ✓ 'gemini:' prefix maps to Google provider")
    print()

    # Test 5: Auto-detection (no provider prefix)
    print("Test 5: Auto-detection without provider prefix")
    config = ModelConfig.from_string("claude-sonnet-4-5-20250929")
    print(f"  Input: 'claude-sonnet-4-5-20250929'")
    print(f"  Provider: {config.provider}")
    print(f"  Model name: {config.model_name}")
    assert config.provider == ModelProvider.ANTHROPIC
    assert config.model_name == "claude-sonnet-4-5-20250929"
    print("  ✓ Auto-detection works for Anthropic")
    print()

    # Test 6: Auto-detection for OpenAI
    print("Test 6: Auto-detection for OpenAI models")
    config = ModelConfig.from_string("gpt-4o-mini")
    print(f"  Input: 'gpt-4o-mini'")
    print(f"  Provider: {config.provider}")
    print(f"  Model name: {config.model_name}")
    assert config.provider == ModelProvider.OPENAI
    assert config.model_name == "gpt-4o-mini"
    print("  ✓ Auto-detection works for OpenAI")
    print()

    # Test 7: Auto-detection for OpenAI o1 models
    print("Test 7: Auto-detection for o1 models")
    config = ModelConfig.from_string("o1-preview")
    print(f"  Input: 'o1-preview'")
    print(f"  Provider: {config.provider}")
    print(f"  Model name: {config.model_name}")
    assert config.provider == ModelProvider.OPENAI
    assert config.model_name == "o1-preview"
    print("  ✓ Auto-detection works for o1 models")
    print()

    # Test 8: With fallback configuration
    print("Test 8: With fallback configuration")
    config = ModelConfig.from_string(
        "anthropic:claude-sonnet-4-5-20250929",
        max_latency_ms=5000,
        fallback_on_rate_limit=True,
        fallback_models=[
            ModelConfig.from_string("openai:gpt-4o-mini")
        ]
    )
    print(f"  Primary: anthropic:claude-sonnet-4-5-20250929")
    print(f"  Max latency: {config.max_latency_ms}ms")
    print(f"  Fallback models: {len(config.fallback_models)} configured")
    print(f"  Fallback model: {config.fallback_models[0].provider.value}:{config.fallback_models[0].model_name}")
    assert config.max_latency_ms == 5000
    assert config.fallback_on_rate_limit == True
    assert len(config.fallback_models) == 1
    assert config.fallback_models[0].provider == ModelProvider.OPENAI
    print("  ✓ Fallback configuration works with from_string()")
    print()

    print("=" * 70)
    print("ALL MODELCONFIG.FROM_STRING() TESTS PASSED")
    print("=" * 70)
    print()


def test_env_var_with_provider_prefix():
    """Test NON_DEFAULT_MODEL with provider:model_name format."""
    print("=" * 70)
    print("TEST: NON_DEFAULT_MODEL with Provider Prefix")
    print("=" * 70)
    print()

    # Test 1: Using provider:model_name in env var
    print("Test 1: NON_DEFAULT_MODEL with 'anthropic:' prefix")
    os.environ["NON_DEFAULT_MODEL"] = "anthropic:claude-sonnet-4-5-20250929"
    config = get_default_model_config()
    print(f"  Env var: {os.environ['NON_DEFAULT_MODEL']}")
    print(f"  Provider: {config.provider}")
    print(f"  Model name: {config.model_name}")
    assert config.provider == ModelProvider.ANTHROPIC
    assert config.model_name == "claude-sonnet-4-5-20250929"
    print("  ✓ Environment variable with provider prefix works")
    print()

    # Test 2: OpenAI with prefix
    print("Test 2: NON_DEFAULT_MODEL with 'openai:' prefix")
    os.environ["NON_DEFAULT_MODEL"] = "openai:gpt-4o-mini"
    config = get_default_model_config()
    print(f"  Env var: {os.environ['NON_DEFAULT_MODEL']}")
    print(f"  Provider: {config.provider}")
    print(f"  Model name: {config.model_name}")
    assert config.provider == ModelProvider.OPENAI
    assert config.model_name == "gpt-4o-mini"
    print("  ✓ OpenAI with provider prefix works")
    print()

    # Test 3: Google with prefix
    print("Test 3: NON_DEFAULT_MODEL with 'google:' prefix")
    os.environ["NON_DEFAULT_MODEL"] = "google:gemini-2.0-flash"
    config = get_default_model_config()
    print(f"  Env var: {os.environ['NON_DEFAULT_MODEL']}")
    print(f"  Provider: {config.provider}")
    print(f"  Model name: {config.model_name}")
    assert config.provider == ModelProvider.GOOGLE
    assert config.model_name == "gemini-2.0-flash"
    print("  ✓ Google with provider prefix works")
    print()

    # Test 4: Still supports auto-detection without prefix
    print("Test 4: NON_DEFAULT_MODEL without prefix (auto-detection)")
    os.environ["NON_DEFAULT_MODEL"] = "claude-sonnet-4-5-20250929"
    config = get_default_model_config()
    print(f"  Env var: {os.environ['NON_DEFAULT_MODEL']}")
    print(f"  Provider: {config.provider}")
    print(f"  Model name: {config.model_name}")
    assert config.provider == ModelProvider.ANTHROPIC
    assert config.model_name == "claude-sonnet-4-5-20250929"
    print("  ✓ Auto-detection still works without prefix")
    print()

    # Clean up
    if "NON_DEFAULT_MODEL" in os.environ:
        del os.environ["NON_DEFAULT_MODEL"]

    print("=" * 70)
    print("ALL ENVIRONMENT VARIABLE TESTS PASSED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    try:
        test_provider_string_parsing()
        test_env_var_with_provider_prefix()

        print("\n" + "=" * 70)
        print("SUCCESS: ALL TESTS PASSED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ ModelConfig.from_string() parses provider:model_name format")
        print("  ✓ Auto-detection works without provider prefix")
        print("  ✓ All providers (anthropic, openai, google, gemini) supported")
        print("  ✓ Custom parameters (temperature, max_tokens, etc.) work")
        print("  ✓ Fallback configuration works with from_string()")
        print("  ✓ NON_DEFAULT_MODEL environment variable supports both formats")
        print()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
