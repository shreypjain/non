# NoN Utilities

Provider adapters, helper functions, and utility modules that support the core NoN functionality.

## 🔌 Provider Adapters (`providers.py`)

The provider system enables seamless integration with multiple LLM providers while maintaining a unified interface.

### Supported Providers

- **OpenAI**: GPT-4, GPT-3.5-turbo, and other OpenAI models
- **Anthropic**: Claude 3 family (Haiku, Sonnet, Opus)
- **Google**: Gemini 2.5 Flash and other Google AI models
- **Mock**: Testing provider for development and CI/CD

## 🚀 Basic Provider Usage

### 1. Automatic Provider Selection
```python
from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider

# Provider is automatically selected based on model config
node = Node('generate', ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-haiku-20240307"
))

result = await node.execute(prompt="Hello, world!")
```

### 2. Direct Provider Usage
```python
from nons.utils.providers import create_provider
from nons.core.types import ModelConfig, ModelProvider

# Create provider directly
config = ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=150
)

provider = create_provider(config.provider, config.model_name)
response, metrics = await provider.generate_completion("Write a haiku", config)

print(f"Response: {response}")
print(f"Tokens used: {metrics.token_usage.total_tokens}")
print(f"Cost: ${metrics.cost_info.cost_usd:.6f}")
```

## 🔧 Provider Configuration Examples

### 3. OpenAI Configuration
```python
from nons.core.types import ModelConfig, ModelProvider

# GPT-4 configuration
gpt4_config = ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4",
    temperature=0.3,
    max_tokens=500,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1
)

# GPT-3.5 for faster, cheaper operations
gpt35_config = ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=200
)
```

### 4. Anthropic Configuration
```python
# Claude 3 Opus for complex reasoning
opus_config = ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-opus-20240229",
    temperature=0.2,
    max_tokens=1000
)

# Claude 3 Haiku for fast processing
haiku_config = ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-haiku-20240307",
    temperature=0.7,
    max_tokens=300
)

# Claude 3 Sonnet for balanced performance
sonnet_config = ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-sonnet-20240229",
    temperature=0.5,
    max_tokens=500
)
```

### 5. Google AI Configuration
```python
# Gemini 2.0 Flash for fast processing
gemini_config = ModelConfig(
    provider=ModelProvider.GOOGLE,
    model_name="gemini-2.0-flash",
    temperature=0.6,
    max_tokens=400
)

# Gemini Pro for more complex tasks
gemini_pro_config = ModelConfig(
    provider=ModelProvider.GOOGLE,
    model_name="gemini-pro",
    temperature=0.4,
    max_tokens=800
)
```

## 🎯 Multi-Provider Strategies

### 6. Provider Fallback Chain
```python
import asyncio
from nons.utils.providers import create_provider
from nons.core.types import ModelProvider

async def provider_fallback(prompt: str):
    """Try multiple providers in order until one succeeds."""
    providers = [
        (ModelProvider.ANTHROPIC, "claude-3-haiku-20240307"),
        (ModelProvider.OPENAI, "gpt-3.5-turbo"),
        (ModelProvider.GOOGLE, "gemini-2.0-flash")
    ]

    for provider_type, model_name in providers:
        try:
            config = ModelConfig(
                provider=provider_type,
                model_name=model_name,
                temperature=0.7
            )

            provider = create_provider(provider_type, model_name)
            response, metrics = await provider.generate_completion(prompt, config)

            print(f"✅ Success with {provider_type}: {response}")
            return response, metrics

        except Exception as e:
            print(f"❌ Failed with {provider_type}: {e}")
            continue

    raise Exception("All providers failed")

# Usage
result = await provider_fallback("What is the capital of France?")
```

### 7. Provider Load Balancing
```python
import random
from typing import List, Tuple

class LoadBalancer:
    def __init__(self, provider_configs: List[Tuple[ModelProvider, str]]):
        self.provider_configs = provider_configs
        self.provider_stats = {config: {"requests": 0, "errors": 0}
                             for config in provider_configs}

    def select_provider(self) -> Tuple[ModelProvider, str]:
        """Select provider based on least recent usage."""
        # Simple round-robin selection
        return min(self.provider_stats.keys(),
                  key=lambda x: self.provider_stats[x]["requests"])

    async def execute_with_load_balancing(self, prompt: str):
        provider_type, model_name = self.select_provider()

        try:
            config = ModelConfig(
                provider=provider_type,
                model_name=model_name
            )

            provider = create_provider(provider_type, model_name)
            response, metrics = await provider.generate_completion(prompt, config)

            # Update stats
            self.provider_stats[(provider_type, model_name)]["requests"] += 1

            return response, metrics

        except Exception as e:
            self.provider_stats[(provider_type, model_name)]["errors"] += 1
            raise e

# Usage
load_balancer = LoadBalancer([
    (ModelProvider.ANTHROPIC, "claude-3-haiku-20240307"),
    (ModelProvider.OPENAI, "gpt-3.5-turbo"),
    (ModelProvider.GOOGLE, "gemini-2.0-flash")
])

# Execute multiple requests with load balancing
for i in range(5):
    result = await load_balancer.execute_with_load_balancing(f"Request {i}")
    print(f"Request {i} completed")
```

### 8. Cost-Optimized Provider Selection
```python
class CostOptimizer:
    def __init__(self):
        # Cost per 1K tokens (approximate)
        self.costs = {
            (ModelProvider.ANTHROPIC, "claude-3-haiku-20240307"): 0.00025,
            (ModelProvider.OPENAI, "gpt-3.5-turbo"): 0.0015,
            (ModelProvider.GOOGLE, "gemini-2.0-flash"): 0.0005,
            (ModelProvider.ANTHROPIC, "claude-3-opus-20240229"): 0.015
        }

    def select_cheapest_provider(self, task_complexity: str = "simple"):
        """Select provider based on cost and task complexity."""
        if task_complexity == "simple":
            # For simple tasks, use cheapest options
            candidates = [
                (ModelProvider.ANTHROPIC, "claude-3-haiku-20240307"),
                (ModelProvider.GOOGLE, "gemini-2.0-flash")
            ]
        elif task_complexity == "complex":
            # For complex tasks, include more capable models
            candidates = [
                (ModelProvider.ANTHROPIC, "claude-3-sonnet-20240229"),
                (ModelProvider.OPENAI, "gpt-4")
            ]
        else:
            # Medium complexity
            candidates = [
                (ModelProvider.OPENAI, "gpt-3.5-turbo"),
                (ModelProvider.ANTHROPIC, "claude-3-haiku-20240307")
            ]

        # Select cheapest from candidates
        return min(candidates, key=lambda x: self.costs.get(x, float('inf')))

    async def cost_optimized_execution(self, prompt: str, task_complexity: str = "simple"):
        provider_type, model_name = self.select_cheapest_provider(task_complexity)

        config = ModelConfig(
            provider=provider_type,
            model_name=model_name,
            temperature=0.7
        )

        provider = create_provider(provider_type, model_name)
        response, metrics = await provider.generate_completion(prompt, config)

        cost = metrics.cost_info.cost_usd
        print(f"Used {provider_type}:{model_name} - Cost: ${cost:.6f}")

        return response, metrics

# Usage
optimizer = CostOptimizer()

# Simple task - will use cheapest provider
simple_result = await optimizer.cost_optimized_execution(
    "What is 2+2?",
    task_complexity="simple"
)

# Complex task - will use more capable provider
complex_result = await optimizer.cost_optimized_execution(
    "Analyze the economic implications of renewable energy adoption",
    task_complexity="complex"
)
```

## 🧪 Testing with Mock Provider

### 9. Mock Provider for Development
```python
from nons.core.types import ModelProvider

# Mock provider for testing and development
mock_config = ModelConfig(
    provider=ModelProvider.MOCK,
    model_name="test-model",
    temperature=0.5
)

# Create node with mock provider
mock_node = Node('generate', mock_config)

# Execute - will return predictable mock responses
result = await mock_node.execute(prompt="Test prompt")
print(f"Mock result: {result}")
```

### 10. Custom Mock Responses
```python
from nons.utils.providers import create_provider

# Create mock provider with custom responses
provider = create_provider(ModelProvider.MOCK, "custom-mock")

# Mock providers can be configured for specific test scenarios
config = ModelConfig(
    provider=ModelProvider.MOCK,
    model_name="custom-mock"
)

# Test different scenarios
test_cases = [
    "Generate creative content",
    "Analyze data patterns",
    "Classify sentiment"
]

for test_case in test_cases:
    response, metrics = await provider.generate_completion(test_case, config)
    print(f"Test: {test_case}")
    print(f"Response: {response}")
    print(f"Tokens: {metrics.token_usage.total_tokens}")
    print("---")
```

## 🔐 Environment and Security

### 11. API Key Management
```python
import os
from nons.core.types import ModelProvider

# API keys are automatically loaded from environment variables
api_keys = {
    ModelProvider.OPENAI: os.getenv("OPENAI_API_KEY"),
    ModelProvider.ANTHROPIC: os.getenv("ANTHROPIC_API_KEY"),
    ModelProvider.GOOGLE: os.getenv("GOOGLE_API_KEY")
}

# Check which providers are available
available_providers = [
    provider for provider, key in api_keys.items()
    if key is not None
]

print(f"Available providers: {available_providers}")

# Create configurations only for available providers
configs = []
if ModelProvider.ANTHROPIC in available_providers:
    configs.append(ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307"
    ))

if ModelProvider.OPENAI in available_providers:
    configs.append(ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo"
    ))
```

### 12. Provider Health Checks
```python
async def provider_health_check():
    """Check if providers are healthy and responsive."""
    test_prompt = "Hello"

    providers_to_test = [
        (ModelProvider.ANTHROPIC, "claude-3-haiku-20240307"),
        (ModelProvider.OPENAI, "gpt-3.5-turbo"),
        (ModelProvider.GOOGLE, "gemini-2.0-flash")
    ]

    health_status = {}

    for provider_type, model_name in providers_to_test:
        try:
            config = ModelConfig(
                provider=provider_type,
                model_name=model_name,
                max_tokens=10  # Minimal request
            )

            provider = create_provider(provider_type, model_name)
            start_time = time.time()

            response, metrics = await provider.generate_completion(test_prompt, config)

            end_time = time.time()
            response_time = end_time - start_time

            health_status[f"{provider_type}:{model_name}"] = {
                "status": "healthy",
                "response_time": response_time,
                "cost": metrics.cost_info.cost_usd
            }

        except Exception as e:
            health_status[f"{provider_type}:{model_name}"] = {
                "status": "unhealthy",
                "error": str(e)
            }

    # Print health report
    print("🏥 PROVIDER HEALTH CHECK")
    print("=" * 40)
    for provider, status in health_status.items():
        if status["status"] == "healthy":
            print(f"✅ {provider}: {status['response_time']:.2f}s, ${status['cost']:.6f}")
        else:
            print(f"❌ {provider}: {status['error']}")

    return health_status

# Run health check
health_report = await provider_health_check()
```

## 🔗 Integration with Core Components

The utilities seamlessly integrate with all NoN components:

- **Nodes**: Automatic provider selection and configuration
- **Networks**: Multi-provider workflows with fallback
- **Scheduler**: Provider-aware request scheduling
- **Observability**: Provider metrics and cost tracking

## 🎯 Best Practices

1. **API Key Security**: Always use environment variables for API keys
2. **Cost Monitoring**: Track costs across different providers
3. **Fallback Strategies**: Implement provider fallbacks for reliability
4. **Performance Testing**: Regular health checks for all providers
5. **Load Balancing**: Distribute requests across providers

## 🔗 Next Steps

- Learn about [Core Components](../core/README.md)
- Explore [Operators](../operators/README.md)
- Check [Observability](../observability/README.md)
- See [Complete Examples](../../examples/)