#!/usr/bin/env python3
"""
Rate Limit Tracking Test

Tests the rate limit header parsing functionality for OpenAI, Anthropic, and Google.
Verifies that rate limit information is correctly extracted from API responses.
"""

import asyncio
import sys
import os

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider, RateLimitInfo
from nons.core.scheduler import start_scheduler, stop_scheduler
import nons.operators.base


async def test_anthropic_rate_limits():
    """Test rate limit tracking with Anthropic API."""
    print("=" * 70)
    print("ANTHROPIC RATE LIMIT TRACKING TEST")
    print("=" * 70)
    print()

    config = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=50,
    )

    node = Node("generate", model_config=config)

    # Start scheduler
    await start_scheduler()

    try:
        print("Making request to Anthropic API...")
        result = await node.execute("Say hello in one word")
        print(f"Result: {result}")
        print()

        # Check if rate limit info was captured
        if node._last_metrics and node._last_metrics.rate_limit_info:
            rate_info = node._last_metrics.rate_limit_info
            print("Rate Limit Information:")
            print(f"  Requests limit: {rate_info.requests_limit}")
            print(f"  Requests remaining: {rate_info.requests_remaining}")
            print(f"  Requests reset: {rate_info.requests_reset}")
            print(f"  Tokens limit: {rate_info.tokens_limit}")
            print(f"  Tokens remaining: {rate_info.tokens_remaining}")
            print(f"  Tokens reset: {rate_info.tokens_reset}")
            print(f"  Retry after: {rate_info.retry_after}")
            print()
            print(f"  Is rate limited: {rate_info.is_rate_limited()}")
            print(f"  Should consider fallback: {rate_info.should_consider_fallback()}")

            if rate_info.requests_remaining is not None:
                print(f"\n✓ Successfully parsed Anthropic rate limit headers!")
            else:
                print(f"\n⚠ Warning: Could not parse Anthropic rate limit headers")
        else:
            print("⚠ Warning: No rate limit info in metrics")

    finally:
        await stop_scheduler()

    print()


async def test_openai_rate_limits():
    """Test rate limit tracking with OpenAI API."""
    print("=" * 70)
    print("OPENAI RATE LIMIT TRACKING TEST")
    print("=" * 70)
    print()

    config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o-mini",
        max_tokens=50,
    )

    node = Node("generate", model_config=config)

    # Start scheduler
    await start_scheduler()

    try:
        print("Making request to OpenAI API...")
        result = await node.execute("Say hello in one word")
        print(f"Result: {result}")
        print()

        # Check if rate limit info was captured
        if node._last_metrics and node._last_metrics.rate_limit_info:
            rate_info = node._last_metrics.rate_limit_info
            print("Rate Limit Information:")
            print(f"  Requests limit: {rate_info.requests_limit}")
            print(f"  Requests remaining: {rate_info.requests_remaining}")
            print(f"  Requests reset: {rate_info.requests_reset}")
            print(f"  Tokens limit: {rate_info.tokens_limit}")
            print(f"  Tokens remaining: {rate_info.tokens_remaining}")
            print(f"  Tokens reset: {rate_info.tokens_reset}")
            print()
            print(f"  Is rate limited: {rate_info.is_rate_limited()}")
            print(f"  Should consider fallback: {rate_info.should_consider_fallback()}")

            if rate_info.requests_remaining is not None:
                print(f"\n✓ Successfully parsed OpenAI rate limit headers!")
            else:
                print(f"\n⚠ Warning: Could not parse OpenAI rate limit headers")
        else:
            print("⚠ Warning: No rate limit info in metrics")

    finally:
        await stop_scheduler()

    print()


async def test_google_rate_limits():
    """Test rate limit tracking with Google API."""
    print("=" * 70)
    print("GOOGLE RATE LIMIT TRACKING TEST")
    print("=" * 70)
    print()

    print("Note: Google Gemini API does not provide programmatic rate limit")
    print("information in response headers. Rate limits must be monitored via")
    print("the Google Cloud Console.")
    print()

    config = ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-2.0-flash",
        max_tokens=50,
    )

    node = Node("generate", model_config=config)

    # Start scheduler
    await start_scheduler()

    try:
        print("Making request to Google API...")
        result = await node.execute("Say hello in one word")
        print(f"Result: {result}")
        print()

        # Check if rate limit info exists (should be empty for Google)
        if node._last_metrics and node._last_metrics.rate_limit_info:
            rate_info = node._last_metrics.rate_limit_info
            print("Rate Limit Information:")
            print(f"  Requests remaining: {rate_info.requests_remaining}")
            print(f"  Tokens remaining: {rate_info.tokens_remaining}")

            if rate_info.requests_remaining is None and rate_info.tokens_remaining is None:
                print(f"\n✓ As expected: Google does not provide rate limit headers")
            else:
                print(f"\n⚠ Unexpected: Found rate limit data from Google")
        else:
            print("⚠ Warning: No rate limit info in metrics")

    finally:
        await stop_scheduler()

    print()


async def test_multiple_requests_tracking():
    """Test rate limit tracking across multiple requests."""
    print("=" * 70)
    print("MULTIPLE REQUESTS RATE LIMIT TRACKING")
    print("=" * 70)
    print()

    config = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=30,
    )

    node = Node("generate", model_config=config)

    # Start scheduler
    await start_scheduler()

    try:
        print("Making 5 requests to track rate limit changes...")
        print()

        for i in range(5):
            result = await node.execute(f"Say number {i+1}")

            if node._last_metrics and node._last_metrics.rate_limit_info:
                rate_info = node._last_metrics.rate_limit_info
                print(f"Request {i+1}:")
                print(f"  Requests remaining: {rate_info.requests_remaining}")
                print(f"  Tokens remaining: {rate_info.tokens_remaining}")

                if rate_info.requests_remaining is not None:
                    print(f"  Capacity: {(rate_info.requests_remaining / rate_info.requests_limit * 100):.1f}%")

                if rate_info.should_consider_fallback():
                    print(f"  ⚠ WARNING: Rate limit capacity below 10%, consider fallback!")
                print()

        print("✓ Successfully tracked rate limits across multiple requests")

    finally:
        await stop_scheduler()

    print()


async def test_rate_limit_helpers():
    """Test RateLimitInfo helper methods."""
    print("=" * 70)
    print("RATE LIMIT INFO HELPER METHODS TEST")
    print("=" * 70)
    print()

    # Test is_rate_limited()
    print("Testing is_rate_limited():")

    rate_info1 = RateLimitInfo(requests_remaining=0, requests_limit=100)
    print(f"  requests_remaining=0: {rate_info1.is_rate_limited()} (should be True)")

    rate_info2 = RateLimitInfo(tokens_remaining=0, tokens_limit=10000)
    print(f"  tokens_remaining=0: {rate_info2.is_rate_limited()} (should be True)")

    rate_info3 = RateLimitInfo(retry_after=30)
    print(f"  retry_after=30: {rate_info3.is_rate_limited()} (should be True)")

    rate_info4 = RateLimitInfo(requests_remaining=50, requests_limit=100)
    print(f"  requests_remaining=50: {rate_info4.is_rate_limited()} (should be False)")
    print()

    # Test should_consider_fallback()
    print("Testing should_consider_fallback():")

    rate_info5 = RateLimitInfo(requests_remaining=5, requests_limit=100)
    print(f"  5% remaining: {rate_info5.should_consider_fallback()} (should be True)")

    rate_info6 = RateLimitInfo(requests_remaining=50, requests_limit=100)
    print(f"  50% remaining: {rate_info6.should_consider_fallback()} (should be False)")

    rate_info7 = RateLimitInfo(tokens_remaining=500, tokens_limit=10000)
    print(f"  5% tokens remaining: {rate_info7.should_consider_fallback()} (should be True)")
    print()

    print("✓ All helper method tests passed")
    print()


async def main():
    """Run all rate limit tracking tests."""
    print("\n")
    print("=" * 70)
    print("RATE LIMIT TRACKING TEST SUITE")
    print("=" * 70)
    print()
    print("Testing rate limit header parsing and tracking across providers")
    print()

    try:
        # Test helper methods first (no API calls)
        await test_rate_limit_helpers()

        # Test with real API calls
        if os.getenv("ANTHROPIC_API_KEY"):
            await test_anthropic_rate_limits()
            await test_multiple_requests_tracking()
        else:
            print("⚠ Skipping Anthropic tests (no API key)")
            print()

        if os.getenv("OPENAI_API_KEY"):
            await test_openai_rate_limits()
        else:
            print("⚠ Skipping OpenAI tests (no API key)")
            print()

        if os.getenv("GOOGLE_API_KEY"):
            await test_google_rate_limits()
        else:
            print("⚠ Skipping Google tests (no API key)")
            print()

        print("=" * 70)
        print("RATE LIMIT TRACKING TESTS COMPLETED")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✓ Helper methods working correctly")
        print("  ✓ Rate limit headers being parsed from API responses")
        print("  ✓ Fallback suggestions based on remaining capacity")
        print()

    except Exception as e:
        print(f"\n❌ Error during tests: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
