#!/usr/bin/env python3
"""
Test Default Configuration

Tests the simplified NON_DEFAULT_MODEL environment variable to ensure
it correctly sets a single default model across the package.
"""

import os
import sys

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nons.core.config import get_default_model_config
from nons.core.types import ModelProvider


def test_default_model_env_var():
    """Test that NON_DEFAULT_MODEL is correctly parsed."""
    print("=" * 70)
    print("TEST: NON_DEFAULT_MODEL Environment Variable")
    print("=" * 70)
    print()

    # Test 1: Anthropic model
    print("Test 1: Setting NON_DEFAULT_MODEL=claude-sonnet-4-5-20250929")
    os.environ["NON_DEFAULT_MODEL"] = "claude-sonnet-4-5-20250929"
    config = get_default_model_config()

    print(f"  Provider detected: {config.provider}")
    print(f"  Model name: {config.model_name}")

    assert config.provider == ModelProvider.ANTHROPIC, f"Expected ANTHROPIC, got {config.provider}"
    assert config.model_name == "claude-sonnet-4-5-20250929"
    print("  ✓ Anthropic model correctly detected")
    print()

    # Test 2: OpenAI model
    print("Test 2: Setting NON_DEFAULT_MODEL=gpt-4o-mini")
    os.environ["NON_DEFAULT_MODEL"] = "gpt-4o-mini"
    config = get_default_model_config()

    print(f"  Provider detected: {config.provider}")
    print(f"  Model name: {config.model_name}")

    assert config.provider == ModelProvider.OPENAI, f"Expected OPENAI, got {config.provider}"
    assert config.model_name == "gpt-4o-mini"
    print("  ✓ OpenAI model correctly detected")
    print()

    # Test 3: Google model
    print("Test 3: Setting NON_DEFAULT_MODEL=gemini-2.0-flash")
    os.environ["NON_DEFAULT_MODEL"] = "gemini-2.0-flash"
    config = get_default_model_config()

    print(f"  Provider detected: {config.provider}")
    print(f"  Model name: {config.model_name}")

    assert config.provider == ModelProvider.GOOGLE, f"Expected GOOGLE, got {config.provider}"
    assert config.model_name == "gemini-2.0-flash"
    print("  ✓ Google model correctly detected")
    print()

    # Test 4: OpenAI o1 model
    print("Test 4: Setting NON_DEFAULT_MODEL=o1-preview")
    os.environ["NON_DEFAULT_MODEL"] = "o1-preview"
    config = get_default_model_config()

    print(f"  Provider detected: {config.provider}")
    print(f"  Model name: {config.model_name}")

    assert config.provider == ModelProvider.OPENAI, f"Expected OPENAI, got {config.provider}"
    assert config.model_name == "o1-preview"
    print("  ✓ OpenAI o1 model correctly detected")
    print()

    # Test 5: Clear env var and test fallback to per-provider defaults
    print("Test 5: Clearing NON_DEFAULT_MODEL (fallback to per-provider defaults)")
    if "NON_DEFAULT_MODEL" in os.environ:
        del os.environ["NON_DEFAULT_MODEL"]

    # Set a per-provider default
    os.environ["NON_DEFAULT_ANTHROPIC_MODEL"] = "claude-sonnet-4-5-20250929"
    config = get_default_model_config(provider=ModelProvider.ANTHROPIC)

    print(f"  Provider: {config.provider}")
    print(f"  Model name: {config.model_name}")

    assert config.model_name == "claude-sonnet-4-5-20250929"
    print("  ✓ Per-provider default correctly used as fallback")
    print()

    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print()
    print("Summary:")
    print("  ✓ NON_DEFAULT_MODEL sets global default model")
    print("  ✓ Provider auto-detection works for common model names")
    print("  ✓ Backward compatibility maintained with per-provider defaults")
    print()


if __name__ == "__main__":
    try:
        test_default_model_env_var()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
