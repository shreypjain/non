#!/usr/bin/env python3
"""Check available API credentials."""

import os

api_keys = {
    "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
}

print("API Key Status:")
print("=" * 60)

for key_name, key_value in api_keys.items():
    if key_value:
        # Show first/last few chars for verification
        masked = key_value[:8] + "..." + key_value[-4:] if len(key_value) > 12 else "***"
        print(f"✓ {key_name}: {masked}")
    else:
        print(f"✗ {key_name}: NOT SET")

print("=" * 60)

# Check if we have at least one key
has_keys = any(api_keys.values())
if has_keys:
    print("\n✓ At least one API key is available - can run evolution!")
else:
    print("\n✗ No API keys found - cannot run evolution")
