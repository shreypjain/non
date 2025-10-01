# NoN (Network of Networks) - Core Package

Welcome to the core NoN package! This directory contains all the essential components for building compound AI systems using networks of interconnected language model operators.

## 🏗️ Package Structure

```
nons/
├── core/           # Core system components
├── operators/      # Operator implementations and registry
├── observability/ # Monitoring, tracing, and metrics
└── utils/         # Utilities and provider adapters
```

## 🚀 Quick Installation

```bash
# Clone the repository
git clone https://github.com/shreypjain/non
cd non

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## 🔑 Setup API Keys

```bash
# Set your API keys as environment variables
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

## 📚 Progressive Examples

### Example 1: Hello World - Your First Network

```python
import asyncio
from nons.core.network import NoN

# Import operators to register them
import nons.operators.base

async def hello_world():
    # Create the simplest possible network
    network = NoN.from_operators(['generate'])

    result = await network.forward("Say hello to the world!")
    print(f"AI Response: {result}")

# Run it
asyncio.run(hello_world())
```

### Example 2: Sequential Processing Pipeline

```python
import asyncio
from nons.core.network import NoN
import nons.operators.base

async def text_pipeline():
    # Create a 3-step processing pipeline
    network = NoN.from_operators([
        'transform',   # Step 1: Clean and format
        'classify',    # Step 2: Categorize content
        'generate'     # Step 3: Generate response
    ])

    messy_text = "this is some MESSY text that needs processing!!!"
    result = await network.forward(messy_text)

    print(f"Original: {messy_text}")
    print(f"Processed: {result}")

asyncio.run(text_pipeline())
```

### Example 3: Parallel Processing Power

```python
import asyncio
from nons.core.network import NoN
import nons.operators.base

async def parallel_analysis():
    # Create network with parallel analysis layer
    network = NoN.from_operators([
        'transform',                          # Preprocessing
        ['classify', 'extract', 'condense'],  # Parallel analysis
        'synthesize'                          # Combine results
    ])

    document = """
    The quarterly sales report shows a 15% increase in revenue.
    Key factors include strong performance in the mobile sector
    and successful launch of our new product line.
    """

    result = await network.forward(document)
    print(f"Analysis Result: {result}")

asyncio.run(parallel_analysis())
```

### Example 4: Node Multiplication for Scale

```python
import asyncio
from nons.core.network import NoN
from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider
import nons.operators.base

async def scaled_processing():
    # Create a base node
    config = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        temperature=0.7
    )

    generator = Node('generate', config)

    # Use multiplication operator for parallel scaling
    parallel_generators = generator * 3  # Creates 3 parallel instances

    network = NoN.from_operators([
        'transform',           # Input preparation
        parallel_generators,   # 3 parallel generators
        'synthesize'          # Combine outputs
    ])

    prompt = "Generate creative ideas for sustainable transportation"
    result = await network.forward(prompt)

    print(f"Scaled Result: {result}")

asyncio.run(scaled_processing())
```

### Example 5: Multi-Provider Network

```python
import asyncio
from nons.core.network import NoN
from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider
import nons.operators.base

async def multi_provider_network():
    # Create nodes with different providers
    openai_node = Node('generate', ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4",
        temperature=0.1
    ))

    anthropic_node = Node('generate', ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-sonnet-20240229",
        temperature=0.7
    ))

    google_node = Node('generate', ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-2.0-flash",
        temperature=0.5
    ))

    # Network using different providers
    network = NoN.from_operators([
        'transform',                              # Preprocessing
        [openai_node, anthropic_node, google_node], # Multi-provider analysis
        'synthesize'                              # Combine perspectives
    ])

    question = "What are the key challenges in AI safety?"
    result = await network.forward(question)

    print(f"Multi-Provider Analysis: {result}")

asyncio.run(multi_provider_network())
```

### Example 6: Advanced Configuration with Error Handling

```python
import asyncio
from nons.core.network import NoN
from nons.core.types import LayerConfig, ErrorPolicy, ModelConfig, ModelProvider
import nons.operators.base

async def robust_network():
    # Configure resilient error handling
    resilient_config = LayerConfig(
        error_policy=ErrorPolicy.RETRY_WITH_BACKOFF,
        max_retries=3,
        timeout_seconds=30,
        min_success_threshold=0.7
    )

    # Configure fast model for preprocessing
    fast_config = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        temperature=0.1,
        max_tokens=100
    )

    network = NoN.from_operators([
        'transform',                    # Fast preprocessing
        ['classify', 'extract'] * 2,   # Multiple parallel analyzers
        'validate',                     # Quality check
        'generate'                      # Final output
    ], layer_config=resilient_config, model_config=fast_config)

    complex_input = """
    Analyze this complex business scenario and provide recommendations:
    Company XYZ is facing declining sales, increased competition,
    and supply chain disruptions. What strategic options should they consider?
    """

    result = await network.forward(complex_input)
    print(f"Strategic Analysis: {result}")

asyncio.run(robust_network())
```

### Example 7: Full Observability and Cost Tracking

```python
import asyncio
from nons.core.network import NoN
from nons.observability.integration import get_observability
import nons.operators.base

async def monitored_network():
    # Create network with built-in observability
    network = NoN.from_operators([
        'transform',
        ['classify', 'extract', 'condense'],
        'synthesize'
    ])

    # Execute with monitoring
    result = await network.forward("Analyze market trends in renewable energy")

    # Get observability data
    obs = get_observability()
    stats = obs.get_stats()

    print(f"Result: {result}")
    print(f"\n📊 Observability Stats:")
    print(f"Total Spans: {stats['tracing']['total_spans']}")
    print(f"Log Entries: {stats['logging']['total_entries']}")
    print(f"Metric Points: {stats['metrics']['total_points']}")

    # Export all data (ready for database storage)
    all_data = obs.export_all_data()
    print(f"\n💾 Exportable Data:")
    print(f"Spans: {len(all_data['spans'])}")
    print(f"Logs: {len(all_data['logs'])}")
    print(f"Metrics: {len(all_data['metrics'])}")

asyncio.run(monitored_network())
```

## 🔗 Next Steps

- Explore the [API Reference](../docs/api-reference.md)
- Read the [Architecture Guide](../docs/architecture.md)
- Check out [Advanced Features](../docs/advanced-features.md)
- Run the [examples](../examples/) directory

## 🆘 Need Help?

- 📖 Documentation: Check the `/docs` directory
- 🐛 Issues: Report on GitHub
- 💬 Discussions: GitHub Discussions