# NoN Operators

Operators are the fundamental building blocks of NoN networks. They define transformations and computations that can be composed into complex AI workflows.

## 🎯 Available Operators

NoN comes with 10 built-in operators that cover the most common AI operations:

| Operator | Purpose | Example Use |
|----------|---------|-------------|
| `transform` | Change format/style | Convert JSON to text, clean data |
| `generate` | Create new content | Write stories, code, responses |
| `classify` | Categorize content | Sentiment analysis, topic classification |
| `extract` | Pull specific info | Extract names, dates, key facts |
| `condense` | Summarize content | Create summaries, abstracts |
| `expand` | Add detail/context | Elaborate on points, add examples |
| `compare` | Find differences | Compare documents, analyze similarities |
| `validate` | Check correctness | Fact-check, quality assessment |
| `route` | Determine next action | Workflow routing, decision making |
| `synthesize` | Combine inputs | Merge multiple perspectives |

## 🚀 Basic Usage Examples

### 1. Single Operator
```python
import asyncio
from nons.core.node import Node
import nons.operators.base

async def single_operator():
    # Create a node with the generate operator
    node = Node('generate')

    result = await node.execute(
        prompt="Write a creative story about a robot learning to paint"
    )
    print(result)

asyncio.run(single_operator())
```

### 2. Transform Operations
```python
async def transform_examples():
    transform_node = Node('transform')

    # Text transformation
    result1 = await transform_node.execute(
        text="this is messy text!!!",
        target_format="clean professional format"
    )

    # Format conversion
    result2 = await transform_node.execute(
        text='{"name": "John", "age": 30}',
        target_format="readable sentence"
    )

    print(f"Cleaned: {result1}")
    print(f"Converted: {result2}")
```

### 3. Classification and Extraction
```python
async def analysis_operators():
    classify_node = Node('classify')
    extract_node = Node('extract')

    text = """
    I absolutely love this new smartphone! The camera quality is amazing
    and the battery life exceeds my expectations. However, the price
    point of $899 might be too high for some users.
    """

    # Classify sentiment
    sentiment = await classify_node.execute(
        text=text,
        categories=["positive", "negative", "neutral"]
    )

    # Extract key information
    details = await extract_node.execute(
        text=text,
        target_info="product features, price, and sentiment indicators"
    )

    print(f"Sentiment: {sentiment}")
    print(f"Details: {details}")
```

### 4. Content Processing Pipeline
```python
async def content_pipeline():
    from nons.core.network import NoN

    # Create a content processing network
    network = NoN.from_operators([
        'condense',    # Summarize input
        'expand',      # Add detailed context
        'validate'     # Check quality
    ])

    long_article = """
    Artificial intelligence is rapidly transforming industries across the globe.
    From healthcare to finance, AI technologies are enabling unprecedented
    levels of automation and insight generation. Machine learning algorithms
    can now process vast amounts of data to identify patterns that humans
    might miss. However, this technological advancement also raises important
    questions about employment, privacy, and ethical considerations...
    [imagine this continues for several paragraphs]
    """

    result = await network.forward(long_article)
    print(f"Processed Article: {result}")
```

## 🔧 Creating Custom Operators

You can easily create your own operators using the `@operator` decorator:

### Simple Custom Operator
```python
from nons.operators.registry import operator
from nons.core.types import InputSchema, OutputSchema, OperatorMetadata

@operator(
    input_schema=InputSchema(
        required_params=["text", "language"],
        optional_params=["formality"],
        param_types={"text": "str", "language": "str", "formality": "str"}
    ),
    output_schema=OutputSchema(
        return_type="str",
        description="Translated text"
    ),
    metadata=OperatorMetadata(
        name="translate",
        description="Translate text to specified language",
        examples=["translate('Hello', 'Spanish') -> 'Hola'"],
        tags=["language", "translation"]
    )
)
async def translate(text: str, language: str, formality: str = "neutral") -> str:
    """Translate text to the specified language."""
    # Your implementation here - this would call a translation service
    # For demo purposes, we'll use the LLM
    from nons.utils.providers import create_provider
    from nons.core.config import get_default_model_config

    config = get_default_model_config()
    provider = create_provider(config.provider, config.model_name)

    prompt = f"""
    Translate the following text to {language} with {formality} formality:

    Text: {text}

    Provide only the translation, no explanations.
    """

    response, _ = await provider.generate_completion(prompt, config)
    return response.strip()
```

### Advanced Custom Operator with Structured Output
```python
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class AnalysisResult:
    sentiment: str
    confidence: float
    key_topics: List[str]
    summary: str

@operator(
    input_schema=InputSchema(
        required_params=["text"],
        optional_params=["include_topics"],
        param_types={"text": "str", "include_topics": "bool"}
    ),
    output_schema=OutputSchema(
        return_type="AnalysisResult",
        description="Comprehensive text analysis"
    ),
    metadata=OperatorMetadata(
        name="analyze_comprehensive",
        description="Perform comprehensive text analysis including sentiment and topics",
        examples=[],
        tags=["analysis", "sentiment", "topics"]
    )
)
async def analyze_comprehensive(text: str, include_topics: bool = True) -> AnalysisResult:
    """Perform comprehensive analysis of text."""
    from nons.utils.providers import create_provider
    from nons.core.config import get_default_model_config
    import json

    config = get_default_model_config()
    provider = create_provider(config.provider, config.model_name)

    prompt = f"""
    Analyze the following text and provide a JSON response with:
    - sentiment: "positive", "negative", or "neutral"
    - confidence: float between 0 and 1
    - key_topics: list of main topics (if include_topics is True)
    - summary: brief summary in one sentence

    Text: {text}
    Include topics: {include_topics}

    Respond only with valid JSON.
    """

    response, _ = await provider.generate_completion(prompt, config)

    try:
        data = json.loads(response.strip())
        return AnalysisResult(
            sentiment=data["sentiment"],
            confidence=data["confidence"],
            key_topics=data.get("key_topics", []),
            summary=data["summary"]
        )
    except json.JSONDecodeError:
        # Fallback for non-JSON responses
        return AnalysisResult(
            sentiment="neutral",
            confidence=0.5,
            key_topics=[],
            summary=response.strip()
        )
```

### Using Custom Operators
```python
async def use_custom_operators():
    # Register your custom operators by importing them
    # (they're automatically registered when the module is imported)

    # Create nodes with custom operators
    translate_node = Node('translate')
    analysis_node = Node('analyze_comprehensive')

    # Use translation
    spanish_text = await translate_node.execute(
        text="Hello, how are you today?",
        language="Spanish",
        formality="formal"
    )

    # Use comprehensive analysis
    analysis = await analysis_node.execute(
        text="I love programming with Python! It's so intuitive and powerful.",
        include_topics=True
    )

    print(f"Translation: {spanish_text}")
    print(f"Analysis: {analysis}")
```

## 🔍 Operator Registry

### Listing Available Operators
```python
from nons.operators.registry import list_operators, get_operator_info

# List all registered operators
operators = list_operators()
print("Available operators:", operators)

# Get detailed info about an operator
info = get_operator_info('generate')
print(f"Generate operator info: {info}")
```

### Dynamic Operator Discovery
```python
async def discover_operators():
    from nons.operators.registry import list_operators, get_operator

    # Dynamically use all available operators
    for op_name in list_operators():
        try:
            operator = get_operator(op_name)
            print(f"Operator: {op_name}")
            print(f"Description: {operator.metadata.description}")
            print("---")
        except Exception as e:
            print(f"Error with {op_name}: {e}")
```

## 🎨 Operator Composition Patterns

### 1. Preprocessing Chain
```python
async def preprocessing_chain():
    network = NoN.from_operators([
        'transform',   # Clean and format
        'validate',    # Check quality
        'condense'     # Summarize if too long
    ])

    messy_input = "THIS IS VERY MESSY TEXT!!! with bad formatting..."
    clean_output = await network.forward(messy_input)
    return clean_output
```

### 2. Analysis Pipeline
```python
async def analysis_pipeline():
    network = NoN.from_operators([
        'extract',                    # Pull key facts
        ['classify', 'condense'],     # Parallel: categorize and summarize
        'synthesize'                  # Combine insights
    ])

    document = "Your business document content here..."
    insights = await network.forward(document)
    return insights
```

### 3. Generation and Refinement
```python
async def generation_refinement():
    network = NoN.from_operators([
        'generate',    # Initial content creation
        'expand',      # Add detail and examples
        'validate',    # Quality check
        'transform'    # Final formatting
    ])

    brief = "Write about sustainable energy"
    polished_content = await network.forward(brief)
    return polished_content
```

## 🔗 Next Steps

- Learn about [Core Components](../core/README.md)
- Explore [Observability](../observability/README.md)
- Check [Provider Utilities](../utils/README.md)
- See [Complete Examples](../../examples/)