"""
RLM Network - Plan-Execute-Verify-Refine Loop

This network implements the RLM loop:
1. Planner generates Python code strategy
2. Execute in RLMOperator
   - If error: Call Fixer once, retry execution
   - If success: Continue
3. Verifier returns confidence 0-1
4. If confidence > 0.8 or max iterations: Return
5. Else: Refine with feedback and repeat

Key Patterns Demonstrated:
1. Multi-node network with specialized roles
2. Iterative refinement loop
3. Automatic error fixing
4. Confidence-based stopping criteria
5. Full execution trace tracking

Architecture:
- Planner Node: Generates Python code for processing
- Fixer Node: Fixes code errors
- Verifier Node: Checks answer quality and returns confidence
- RLM Operator: Executes code with llm_query() injection
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider
from .rlm_operator import RLMOperator, RLMExecutionResult


class RLMIteration(BaseModel):
    """Single iteration of RLM loop"""

    iteration: int
    code: str
    execution_result: Dict[str, Any]
    had_error: bool
    fixed_code: Optional[str] = None
    confidence: float
    verifier_reasoning: str


class RLMResult(BaseModel):
    """Final result from RLM network"""

    success: bool
    final_output: Any
    iterations: List[RLMIteration]
    total_llm_calls: int
    total_iterations: int
    final_confidence: float
    stop_reason: str  # "confidence_threshold", "max_iterations", "error"


class RLMNetwork:
    """
    Network implementing Plan-Execute-Verify-Refine loop for RLM.

    Architecture:
    - Planner Node: Node('generate') with planning system prompt
    - Fixer Node: Node('generate') with fixing system prompt
    - Verifier Node: Node('generate') with verification system prompt
    - RLM Operator: Executes code with llm_query()

    All nodes use gpt-5.2 for cost efficiency.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        confidence_threshold: float = 0.8,
        max_llm_calls_per_execution: int = 50
    ):
        """
        Initialize RLM Network.

        Args:
            max_iterations: Maximum refinement iterations
            confidence_threshold: Stop if confidence exceeds this
            max_llm_calls_per_execution: Max LLM calls in single code execution
        """
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        # Create model config (gpt-4o-mini for all nodes)
        self.model_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000
        )

        # Initialize RLM operator
        self.rlm_operator = RLMOperator(
            model_config=self.model_config,
            max_llm_calls=max_llm_calls_per_execution
        )

        # Create nodes
        self.planner_node = self._create_planner_node()
        self.fixer_node = self._create_fixer_node()
        self.verifier_node = self._create_verifier_node()

        # Track iterations
        self.iterations: List[RLMIteration] = []

    def _create_planner_node(self) -> Node:
        """
        Create Planner node with system prompt.

        Planner generates Python code that:
        - Uses context variable (stored in REPL)
        - Calls llm_query(prompt, context_chunk) for semantic operations
        - Stores final result in 'result' variable
        - Uses print() for intermediate outputs
        """
        system_prompt = """You are a Python code planner for processing long documents.

Your task: Generate Python code that solves the given task using the available context.

Available in execution environment:
- context: The full document/data (can be very long, 10M+ tokens)
- llm_query(prompt, context_chunk): Async function to query LLM on chunks
  - Use this for hard semantic operations only after lighter heuristics
  - Pass small context_chunk (not entire context) to stay under token limits
  - Returns string response from LLM

Code requirements:
1. Store final answer in variable named 'result'
2. Use print() for intermediate outputs/debugging
3. Process context in chunks if needed (e.g., split by sections, paragraphs)
4. Before calling llm_query(), apply deterministic heuristics to each chunk such as regex matches, character counts, and sampling the first 10-15 words.
   - For example, run a regex to capture "$ digits" when searching for revenue or using parallel regex search for all types of "AI" questions.
   - Measure the chunk length, word counts, or presence of keywords to rank chunks.
   - Use these filtering steps to narrow down candidate chunks to a handful before any LM call.
5. Use llm_query() only after heuristics indicate a chunk is likely relevant; mention the heuristic evidence when deciding to call the LM.
6. Use regular Python for filtering, counting, and aggregation.
7. Be efficient—minimize LM calls (max 50 per execution) and avoid redundant prompts by caching or reusing results within the REPL.

Example pattern showing the heuristic-first approach:
```python
chunks = context.split('\\n---\\n')
candidates = []
pattern = re.compile(r'Q3 2022.*Revenue:\\s*\\$([0-9,.]+)M')

for chunk in chunks:
    header = chunk.strip().split('\\n')[0]
    snippet = ' '.join(chunk.split()[:15])
    if pattern.search(chunk) or 'Q3 2022' in header:
        candidates.append((chunk, snippet))

results = []
for chunk, snippet in candidates[:5]:
    print(f\"Heuristic candidates: {snippet}\")
    response = await llm_query(\"Does this chunk have the data we need?\", chunk)
    if \"yes\" in response.lower():
        results.append(chunk)

result = len(results)
print(f\"Processed {len(candidates)} heuristic candidates and found {result} relevant chunks\") 
```

Generate ONLY the Python code, no explanations."""
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider
from .rlm_operator import RLMOperator, RLMExecutionResult


class RLMIteration(BaseModel):
    """Single iteration of RLM loop"""

    iteration: int
    code: str
    execution_result: Dict[str, Any]
    had_error: bool
    fixed_code: Optional[str] = None
    confidence: float
    verifier_reasoning: str


class RLMResult(BaseModel):
    """Final result from RLM network"""

    success: bool
    final_output: Any
    iterations: List[RLMIteration]
    total_llm_calls: int
    total_iterations: int
    final_confidence: float
    stop_reason: str  # "confidence_threshold", "max_iterations", "error"


class RLMNetwork:
    """
    Network implementing Plan-Execute-Verify-Refine loop for RLM.

    Architecture:
    - Planner Node: Node('generate') with planning system prompt
    - Fixer Node: Node('generate') with fixing system prompt
    - Verifier Node: Node('generate') with verification system prompt
    - RLM Operator: Executes code with llm_query()

    All nodes use gpt-5.2 for cost efficiency.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        confidence_threshold: float = 0.8,
        max_llm_calls_per_execution: int = 50
    ):
        """
        Initialize RLM Network.

        Args:
            max_iterations: Maximum refinement iterations
            confidence_threshold: Stop if confidence exceeds this
            max_llm_calls_per_execution: Max LLM calls in single code execution
        """
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        # Create model config (gpt-4o-mini for all nodes)
        self.model_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000
        )

        # Initialize RLM operator
        self.rlm_operator = RLMOperator(
            model_config=self.model_config,
            max_llm_calls=max_llm_calls_per_execution
        )

        # Create nodes
        self.planner_node = self._create_planner_node()
        self.fixer_node = self._create_fixer_node()
        self.verifier_node = self._create_verifier_node()

        # Track iterations
        self.iterations: List[RLMIteration] = []

    def _create_planner_node(self) -> Node:
        """
        Create Planner node with system prompt.

        Planner generates Python code that:
        - Uses context variable (stored in REPL)
        - Calls llm_query(prompt, context_chunk) for semantic operations
        - Stores final result in 'result' variable
        - Uses print() for intermediate outputs
        """
        system_prompt = """You are a Python code planner for processing long documents.

Your task: Generate Python code that solves the given task using the available context.

Available in execution environment:
- context: The full document/data (can be very long, 10M+ tokens)
- llm_query(prompt, context_chunk): Async function to query LLM on chunks
  - Use this for hard semantic operations only after lighter heuristics
  - Pass small context_chunk (not entire context) to stay under token limits
  - Returns string response from LLM

Code requirements:
1. Store final answer in variable named 'result'
2. Use print() for intermediate outputs/debugging
3. Process context in chunks if needed (e.g., split by sections, paragraphs)
4. Before calling llm_query(), apply deterministic heuristics to each chunk such as regex matches, character counts, and sampling the first 10-15 words.
   - For example, run a regex to capture "$ digits" when searching for revenue or using parallel regex search for all types of "AI" questions.
   - Measure the chunk length, word counts, or presence of keywords to rank chunks.
   - Use these filtering steps to narrow down candidate chunks to a handful before any LM call.
5. Use llm_query() only after heuristics indicate a chunk is likely relevant; mention the heuristic evidence when deciding to call the LM.
6. Use regular Python for filtering, counting, and aggregation.
7. Be efficient—minimize LM calls (max 50 per execution) and avoid redundant prompts by caching or reusing results within the REPL.

Example pattern showing the heuristic-first approach:
```python
chunks = context.split('\\n---\\n')
candidates = []
pattern = re.compile(r'Q3 2022.*Revenue:\\s*\\$([0-9,.]+)M')

for chunk in chunks:
    header = chunk.strip().split('\\n')[0]
    snippet = ' '.join(chunk.split()[:15])
    if pattern.search(chunk) or 'Q3 2022' in header:
        candidates.append((chunk, snippet))

results = []
for chunk, snippet in candidates[:5]:
    print(f\"Heuristic candidates: {snippet}\")
    response = await llm_query(\"Does this chunk have the data we need?\", chunk)
    if \"yes\" in response.lower():
        results.append(chunk)

result = len(results)
print(f\"Processed {len(candidates)} heuristic candidates and found {result} relevant chunks\") 
```

Generate ONLY the Python code, no explanations."""
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider
from .rlm_operator import RLMOperator, RLMExecutionResult


class RLMIteration(BaseModel):
    """Single iteration of RLM loop"""

    iteration: int
    code: str
    execution_result: Dict[str, Any]
    had_error: bool
    fixed_code: Optional[str] = None
    confidence: float
    verifier_reasoning: str


class RLMResult(BaseModel):
    """Final result from RLM network"""

    success: bool
    final_output: Any
    iterations: List[RLMIteration]
    total_llm_calls: int
    total_iterations: int
    final_confidence: float
    stop_reason: str  # "confidence_threshold", "max_iterations", "error"


class RLMNetwork:
    """
    Network implementing Plan-Execute-Verify-Refine loop for RLM.

    Architecture:
    - Planner Node: Node('generate') with planning system prompt
    - Fixer Node: Node('generate') with fixing system prompt
    - Verifier Node: Node('generate') with verification system prompt
    - RLM Operator: Executes code with llm_query()

    All nodes use gpt-5.2 for cost efficiency.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        confidence_threshold: float = 0.8,
        max_llm_calls_per_execution: int = 50
    ):
        """
        Initialize RLM Network.

        Args:
            max_iterations: Maximum refinement iterations
            confidence_threshold: Stop if confidence exceeds this
            max_llm_calls_per_execution: Max LLM calls in single code execution
        """
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        # Create model config (gpt-4o-mini for all nodes)
        self.model_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000
        )

        # Initialize RLM operator
        self.rlm_operator = RLMOperator(
            model_config=self.model_config,
            max_llm_calls=max_llm_calls_per_execution
        )

        # Create nodes
        self.planner_node = self._create_planner_node()
        self.fixer_node = self._create_fixer_node()
        self.verifier_node = self._create_verifier_node()

        # Track iterations
        self.iterations: List[RLMIteration] = []

    def _create_planner_node(self) -> Node:
        """
        Create Planner node with system prompt.

        Planner generates Python code that:
        - Uses context variable (stored in REPL)
        - Calls llm_query(prompt, context_chunk) for semantic operations
        - Stores final result in 'result' variable
        - Uses print() for intermediate outputs
        """
        system_prompt = """You are a Python code planner for processing long documents.

Your task: Generate Python code that solves the given task using the available context.

Available in execution environment:
- context: The full document/data (can be very long, 10M+ tokens)
- llm_query(prompt, context_chunk): Async function to query LLM on chunks
  - Use this for hard semantic operations only after lighter heuristics
  - Pass small context_chunk (not entire context) to stay under token limits
  - Returns string response from LLM

Code requirements:
1. Store final answer in variable named 'result'
2. Use print() for intermediate outputs/debugging
3. Process context in chunks if needed (e.g., split by sections, paragraphs)
4. Before calling llm_query(), apply deterministic heuristics to each chunk such as regex matches, character counts, and sampling the first 10-15 words.
   - For example, run a regex to capture "$ digits" when searching for revenue or using parallel regex search for all types of "AI" questions.
   - Measure the chunk length, word counts, or presence of keywords to rank chunks.
   - Use these filtering steps to narrow down candidate chunks to a handful before any LM call.
5. Use llm_query() only after heuristics indicate a chunk is likely relevant; mention the heuristic evidence when deciding to call the LM.
6. Use regular Python for filtering, counting, and aggregation.
7. Be efficient—minimize LM calls (max 50 per execution) and avoid redundant prompts by caching or reusing results within the REPL.

Example pattern showing the heuristic-first approach:
```python
chunks = context.split('\\n---\\n')
candidates = []
pattern = re.compile(r'Q3 2022.*Revenue:\\s*\\$([0-9,.]+)M')

for chunk in chunks:
    header = chunk.strip().split('\\n')[0]
    snippet = ' '.join(chunk.split()[:15])
    if pattern.search(chunk) or 'Q3 2022' in header:
        candidates.append((chunk, snippet))

results = []
for chunk, snippet in candidates[:5]:
    print(f\"Heuristic candidates: {snippet}\")
    response = await llm_query(\"Does this chunk have the data we need?\", chunk)
    if \"yes\" in response.lower():
        results.append(chunk)

result = len(results)
print(f\"Processed {len(candidates)} heuristic candidates and found {result} relevant chunks\") 
```

Generate ONLY the Python code, no explanations."""

from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider
from .rlm_operator import RLMOperator, RLMExecutionResult


class RLMIteration(BaseModel):
    """Single iteration of RLM loop"""

    iteration: int
    code: str
    execution_result: Dict[str, Any]
    had_error: bool
    fixed_code: Optional[str] = None
    confidence: float
    verifier_reasoning: str


class RLMResult(BaseModel):
    """Final result from RLM network"""

    success: bool
    final_output: Any
    iterations: List[RLMIteration]
    total_llm_calls: int
    total_iterations: int
    final_confidence: float
    stop_reason: str  # "confidence_threshold", "max_iterations", "error"


class RLMNetwork:
    """
    Network implementing Plan-Execute-Verify-Refine loop for RLM.

    Architecture:
    - Planner Node: Node('generate') with planning system prompt
    - Fixer Node: Node('generate') with fixing system prompt
    - Verifier Node: Node('generate') with verification system prompt
    - RLM Operator: Executes code with llm_query()

    All nodes use gpt-5.2 for cost efficiency.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        confidence_threshold: float = 0.8,
        max_llm_calls_per_execution: int = 50
    ):
        """
        Initialize RLM Network.

        Args:
            max_iterations: Maximum refinement iterations
            confidence_threshold: Stop if confidence exceeds this
            max_llm_calls_per_execution: Max LLM calls in single code execution
        """
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        # Create model config (gpt-4o-mini for all nodes)
        self.model_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000
        )

        # Initialize RLM operator
        self.rlm_operator = RLMOperator(
            model_config=self.model_config,
            max_llm_calls=max_llm_calls_per_execution
        )

        # Create nodes
        self.planner_node = self._create_planner_node()
        self.fixer_node = self._create_fixer_node()
        self.verifier_node = self._create_verifier_node()

        # Track iterations
        self.iterations: List[RLMIteration] = []

    def _create_planner_node(self) -> Node:
        """
        Create Planner node with system prompt.

        Planner generates Python code that:
        - Uses context variable (stored in REPL)
        - Calls llm_query(prompt, context_chunk) for semantic operations
        - Stores final result in 'result' variable
        - Uses print() for intermediate outputs
        """
        system_prompt = """You are a Python code planner for processing long documents.

Your task: Generate Python code that solves the given task using the available context.

Available in execution environment:
- context: The full document/data (can be very long, 10M+ tokens)
- llm_query(prompt, context_chunk): Async function to query LLM on chunks
  - Use this for hard semantic operations only after lighter heuristics
  - Pass small context_chunk (not entire context) to stay under token limits
  - Returns string response from LLM

Code requirements:
1. Store final answer in variable named 'result'
2. Use print() for intermediate outputs/debugging
3. Process context in chunks if needed (e.g., split by sections, paragraphs)
4. Before calling llm_query(), apply deterministic heuristics to each chunk such as regex matches, character counts, and sampling the first 10-15 words.
   - For example, run a regex to capture "$ digits" when searching for revenue or using parallel regex search for all types of "AI" questions.
   - Measure the chunk length, word counts, or presence of keywords to rank chunks.
   - Use these filtering steps to narrow down candidate chunks to a handful before any LM call.
5. Use llm_query() only after heuristics indicate a chunk is likely relevant; mention the heuristic evidence when deciding to call the LM.
6. Use regular Python for filtering, counting, and aggregation.
7. Be efficient—minimize LM calls (max 50 per execution) and avoid redundant prompts by caching or reusing results within the REPL.

Example pattern showing the heuristic-first approach:
```python
chunks = context.split('\\n---\\n')
candidates = []
pattern = re.compile(r'Q3 2022.*Revenue:\\s*\\$([0-9,.]+)M')

for chunk in chunks:
    header = chunk.strip().split('\\n')[0]
    snippet = ' '.join(chunk.split()[:15])
    if pattern.search(chunk) or 'Q3 2022' in header:
        candidates.append((chunk, snippet))

results = []
for chunk, snippet in candidates[:5]:
    print(f\"Heuristic candidates: {snippet}\")
    response = await llm_query(\"Does this chunk have the data we need?\", chunk)
    if \"yes\" in response.lower():
        results.append(chunk)

result = len(results)
print(f\"Processed {len(candidates)} heuristic candidates and found {result} relevant chunks\") 
```

Generate ONLY the Python code, no explanations."""

from nons.core.node import Node
from nons.core.types import ModelConfig, ModelProvider
from .rlm_operator import RLMOperator, RLMExecutionResult


class RLMIteration(BaseModel):
    """Single iteration of RLM loop"""

    iteration: int
    code: str
    execution_result: Dict[str, Any]
    had_error: bool
    fixed_code: Optional[str] = None
    confidence: float
    verifier_reasoning: str


class RLMResult(BaseModel):
    """Final result from RLM network"""

    success: bool
    final_output: Any
    iterations: List[RLMIteration]
    total_llm_calls: int
    total_iterations: int
    final_confidence: float
    stop_reason: str  # "confidence_threshold", "max_iterations", "error"


class RLMNetwork:
    """
    Network implementing Plan-Execute-Verify-Refine loop for RLM.

    Architecture:
    - Planner Node: Node('generate') with planning system prompt
    - Fixer Node: Node('generate') with fixing system prompt
    - Verifier Node: Node('generate') with verification system prompt
    - RLM Operator: Executes code with llm_query()

    All nodes use gpt-5.2 for cost efficiency.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        confidence_threshold: float = 0.8,
        max_llm_calls_per_execution: int = 50
    ):
        """
        Initialize RLM Network.

        Args:
            max_iterations: Maximum refinement iterations
            confidence_threshold: Stop if confidence exceeds this
            max_llm_calls_per_execution: Max LLM calls in single code execution
        """
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        # Create model config (gpt-4o-mini for all nodes)
        self.model_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000
        )

        # Initialize RLM operator
        self.rlm_operator = RLMOperator(
            model_config=self.model_config,
            max_llm_calls=max_llm_calls_per_execution
        )

        # Create nodes
        self.planner_node = self._create_planner_node()
        self.fixer_node = self._create_fixer_node()
        self.verifier_node = self._create_verifier_node()

        # Track iterations
        self.iterations: List[RLMIteration] = []

    def _create_planner_node(self) -> Node:
        """
        Create Planner node with system prompt.

        Planner generates Python code that:
        - Uses context variable (stored in REPL)
        - Calls llm_query(prompt, context_chunk) for semantic operations
        - Stores final result in 'result' variable
        - Uses print() for intermediate outputs
        """
        system_prompt = """You are a Python code planner for processing long documents.

Your task: Generate Python code that solves the given task using the available context.

Available in execution environment:
- context: The full document/data (can be very long, 10M+ tokens)
- llm_query(prompt, context_chunk): Async function to query LLM on chunks
  - Use this for hard semantic operations only after lighter heuristics
  - Pass small context_chunk (not entire context) to stay under token limits
  - Returns string response from LLM

Code requirements:
1. Store final answer in variable named 'result'
2. Use print() for intermediate outputs/debugging
3. Process context in chunks if needed (e.g., split by sections, paragraphs)
4. Before calling llm_query(), apply deterministic heuristics to each chunk such as regex matches, character counts, and sampling the first 10-15 words.
   - For example, run a regex to capture "$ digits" when searching for revenue or using parallel regex search for all types of "AI" questions.
   - Measure the chunk length, word counts, or presence of keywords to rank chunks.
   - Use these filtering steps to narrow down candidate chunks to a handful before any LM call.
5. Use llm_query() only after heuristics indicate a chunk is likely relevant; mention the heuristic evidence when deciding to call the LM.
6. Use regular Python for filtering, counting, and aggregation.
7. Be efficient—minimize LM calls (max 50 per execution) and avoid redundant prompts by caching or reusing results within the REPL.

Example pattern showing the heuristic-first approach:
```python
chunks = context.split('\\n---\\n')
candidates = []
pattern = re.compile(r'Q3 2022.*Revenue:\\s*\\$([0-9,.]+)M')

for chunk in chunks:
    header = chunk.strip().split('\\n')[0]
    snippet = ' '.join(chunk.split()[:15])
    if pattern.search(chunk) or 'Q3 2022' in header:
        candidates.append((chunk, snippet))

results = []
for chunk, snippet in candidates[:5]:
    print(f\"Heuristic candidates: {snippet}\")
    response = await llm_query(\"Does this chunk have the data we need?\", chunk)
    if \"yes\" in response.lower():
        results.append(chunk)

result = len(results)
print(f\"Processed {len(candidates)} heuristic candidates and found {result} relevant chunks\") 
```

Generate ONLY the Python code, no explanations."""

class RLMResult(BaseModel):
    """Final result from RLM network"""

    success: bool
    final_output: Any
    iterations: List[RLMIteration]
    total_llm_calls: int
    total_iterations: int
    final_confidence: float
    stop_reason: str  # "confidence_threshold", "max_iterations", "error"


class RLMNetwork:
    """
    Network implementing Plan-Execute-Verify-Refine loop for RLM.

    Architecture:
    - Planner Node: Node('generate') with planning system prompt
    - Fixer Node: Node('generate') with fixing system prompt
    - Verifier Node: Node('generate') with verification system prompt
    - RLM Operator: Executes code with llm_query()

    All nodes use gpt-5.2 for cost efficiency.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        confidence_threshold: float = 0.8,
        max_llm_calls_per_execution: int = 50
    ):
        """
        Initialize RLM Network.

        Args:
            max_iterations: Maximum refinement iterations
            confidence_threshold: Stop if confidence exceeds this
            max_llm_calls_per_execution: Max LLM calls in single code execution
        """
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        # Create model config (gpt-4o-mini for all nodes)
        self.model_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000
        )

        # Initialize RLM operator
        self.rlm_operator = RLMOperator(
            model_config=self.model_config,
            max_llm_calls=max_llm_calls_per_execution
        )

        # Create nodes
        self.planner_node = self._create_planner_node()
        self.fixer_node = self._create_fixer_node()
        self.verifier_node = self._create_verifier_node()

        # Track iterations
        self.iterations: List[RLMIteration] = []

    def _create_planner_node(self) -> Node:
        """
        Create Planner node with system prompt.

        Planner generates Python code that:
        - Uses context variable (stored in REPL)
        - Calls llm_query(prompt, context_chunk) for semantic operations
        - Stores final result in 'result' variable
        - Uses print() for intermediate outputs
        """
        system_prompt = """You are a Python code planner for processing long documents.

Your task: Generate Python code that solves the given task using the available context.

Available in execution environment:
- context: The full document/data (can be very long, 10M+ tokens)
- llm_query(prompt, context_chunk): Async function to query LLM on chunks
  - Use this for hard semantic operations only after lighter heuristics
  - Pass small context_chunk (not entire context) to stay under token limits
  - Returns string response from LLM

Code requirements:
1. Store final answer in variable named 'result'
2. Use print() for intermediate outputs/debugging
3. Process context in chunks if needed (e.g., split by sections, paragraphs)
4. Before calling llm_query(), apply deterministic heuristics to each chunk such as regex matches, character counts, and sampling the first 10-15 words.
   - For example, run a regex to capture "$ digits" when searching for revenue or using parallel regex search for all types of "AI" questions.
   - Measure the chunk length, word counts, or presence of keywords to rank chunks.
   - Use these filtering steps to narrow down candidate chunks to a handful before any LM call.
5. Use llm_query() only after heuristics indicate a chunk is likely relevant; mention the heuristic evidence when deciding to call the LM.
6. Use regular Python for filtering, counting, and aggregation.
7. Be efficient—minimize LM calls (max 50 per execution) and avoid redundant prompts by caching or reusing results within the REPL.

Example pattern showing the heuristic-first approach:
```python
chunks = context.split('\\n---\\n')
candidates = []
pattern = re.compile(r'Q3 2022.*Revenue:\\s*\\$([0-9,.]+)M')

for chunk in chunks:
    header = chunk.strip().split('\\n')[0]
    snippet = ' '.join(chunk.split()[:15])
    if pattern.search(chunk) or 'Q3 2022' in header:
        candidates.append((chunk, snippet))

results = []
for chunk, snippet in candidates[:5]:
    print(f\"Heuristic candidates: {snippet}\")
    response = await llm_query(\"Does this chunk have the data we need?\", chunk)
    if \"yes\" in response.lower():
        results.append(chunk)

result = len(results)
print(f\"Processed {len(candidates)} heuristic candidates and found {result} relevant chunks\") 
```

Generate ONLY the Python code, no explanations."""


class RLMNetwork:
    """
    Network implementing Plan-Execute-Verify-Refine loop for RLM.

    Architecture:
    - Planner Node: Node('generate') with planning system prompt
    - Fixer Node: Node('generate') with fixing system prompt
    - Verifier Node: Node('generate') with verification system prompt
    - RLM Operator: Executes code with llm_query()

    All nodes use gpt-5.2 for cost efficiency.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        confidence_threshold: float = 0.8,
        max_llm_calls_per_execution: int = 50
    ):
        """
        Initialize RLM Network.

        Args:
            max_iterations: Maximum refinement iterations
            confidence_threshold: Stop if confidence exceeds this
            max_llm_calls_per_execution: Max LLM calls in single code execution
        """
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        # Create model config (gpt-4o-mini for all nodes)
        self.model_config = ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000
        )

        # Initialize RLM operator
        self.rlm_operator = RLMOperator(
            model_config=self.model_config,
            max_llm_calls=max_llm_calls_per_execution
        )

        # Create nodes
        self.planner_node = self._create_planner_node()
        self.fixer_node = self._create_fixer_node()
        self.verifier_node = self._create_verifier_node()

        # Track iterations
        self.iterations: List[RLMIteration] = []

    def _create_planner_node(self) -> Node:
        """
        Create Planner node with system prompt.

        Planner generates Python code that:
        - Uses context variable (stored in REPL)
        - Calls llm_query(prompt, context_chunk) for semantic operations
        - Stores final result in 'result' variable
        - Uses print() for intermediate outputs
        """
        system_prompt = """You are a Python code planner for processing long documents.

Your task: Generate Python code that solves the given task using the available context.

Available in execution environment:
- context: The full document/data (can be very long, 10M+ tokens)
- llm_query(prompt, context_chunk): Async function to query LLM on chunks
  - Use this for hard semantic operations only after lighter heuristics
  - Pass small context_chunk (not entire context) to stay under token limits
  - Returns string response from LLM

Code requirements:
1. Store final answer in variable named 'result'
2. Use print() for intermediate outputs/debugging
3. Process context in chunks if needed (e.g., split by sections, paragraphs)
4. Before calling llm_query(), apply deterministic heuristics to each chunk such as regex matches, character counts, and sampling the first 10-15 words.
   - For example, run a regex to capture "$ digits" when searching for revenue or using parallel regex search for all types of "AI" questions.
   - Measure the chunk length, word counts, or presence of keywords to rank chunks.
   - Use these filtering steps to narrow down candidate chunks to a handful before any LM call.
5. Use llm_query() only after heuristics indicate a chunk is likely relevant; mention the heuristic evidence when deciding to call the LM.
6. Use regular Python for filtering, counting, and aggregation.
7. Be efficient—minimize LM calls (max 50 per execution) and avoid redundant prompts by caching or reusing results within the REPL.

Heuristic techniques to use BEFORE llm_query():

A. Regex patterns - Use when looking for structured data or specific formats:
   - Financial data: re.search(r'\\$[0-9,.]+[MB]', chunk) or re.search(r'revenue.*\\$([0-9,.]+)', chunk, re.IGNORECASE)
   - Dates: re.search(r'Q[1-4]\\s+20\\d{2}', chunk) or re.search(r'\\d{4}-\\d{2}-\\d{2}', chunk)
   - Email/URLs: re.search(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', chunk)
   - Why: Regex is 1000x faster than LLM calls and perfect for pattern matching

B. String length filtering - Use to eliminate trivial or oversized chunks:
   - Minimum content: len(chunk) > 200 to skip headers/footers
   - Maximum size: len(chunk) < 5000 to avoid token limit issues
   - Why: Prevents wasting LLM calls on empty sections or chunks too large to process

C. Word count analysis - Use to assess content density and relevance:
   - Substantial content: len(chunk.split()) > 50 for meaningful sections
   - Keyword density: chunk.lower().count('ai') / len(chunk.split()) > 0.01
   - Why: Word counts help identify information-rich sections worth LLM analysis

D. First N words/chars preview - Use for quick content identification:
   - Preview: ' '.join(chunk.split()[:15]) to see topic without reading all
   - Snippet: chunk[:100] for fast header/title extraction
   - Why: Lets you quickly scan what a chunk is about before expensive LLM call

E. Keyword presence - Use for topic filtering:
   - Simple match: 'revenue' in chunk.lower() or 'artificial intelligence' in chunk.lower()
   - Multiple keywords: any(kw in chunk.lower() for kw in ['ai', 'machine learning', 'neural'])
   - Why: Fast boolean checks eliminate irrelevant chunks immediately

F. Section structure - Use to understand document organization:
   - Headers: chunk.strip().split('\\n')[0] to extract section titles
   - Subsections: chunk.count('##') or chunk.count('\\n\\n') for structure
   - Why: Document structure helps identify where relevant info is likely located

Once you feel like you've exhausted the usage of the heuristics, move on over to using llm_query to do some comprehensive searches.

These searches can happen after you've narrowed down chunks or narrowed down the texts to a set of sentences as well to reduce the cognitive load of number of LLM calls.

Ensure you are only returning Python code to be executed in the Python REPL sandbox.
"""

        return Node(
            operator_name="generate",
            model_config=self.model_config,
            additional_prompt_context=system_prompt
        )

    def _create_fixer_node(self) -> Node:
        """
        Create Fixer node with system prompt.

        Fixer takes broken code and error message, returns fixed code.
        """
        system_prompt = """You are a Python code fixer for RLM executions.

Given broken code and an error message, fix the code to resolve the error.

Common issues:
1. Forgot to use 'await' with llm_query() (it's async)
2. Trying to process entire context at once (too large)
3. Not storing result in 'result' variable
4. Syntax errors or undefined variables
5. Exceeding max LLM calls (50)

Fix the code while preserving the original intent.

Generate ONLY the fixed Python code, no explanations."""

        return Node(
            operator_name="generate",
            model_config=self.model_config,
            additional_prompt_context=system_prompt
        )

    def _create_verifier_node(self) -> Node:
        """
        Create Verifier node with system prompt.

        Verifier analyzes execution result and returns confidence 0-1.
        """
        system_prompt = """You are a verification agent for RLM executions.

Analyze the execution result and determine confidence in the answer.

Consider:
1. Did the code execute successfully?
2. Is there a valid result?
3. Does the approach make sense for the task?
4. Are there any logical errors in the solution?
5. How does the entire distribution of plausible answers look? Calibrate your confidence by describing alternative outcomes and how the probability mass spreads across them (even if the mass concentrates on one best answer).

Return your response in this format:
CONFIDENCE: <float 0-1>
REASONING: <brief explanation>

Example:
CONFIDENCE: 0.9
REASONING: Code executed successfully, found 5 relevant sections using semantic filtering. Approach is sound."""

        return Node(
            operator_name="generate",
            model_config=self.model_config,
            additional_prompt_context=system_prompt
        )

    def _parse_verification_response(self, response: str) -> tuple[float, str]:
        """Parse verifier response to extract confidence and reasoning."""
        lines = response.strip().split('\n')
        confidence = 0.5
        reasoning = "Could not parse verification response"

        for line in lines:
            if line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return confidence, reasoning

    async def run(self, task: str, context: Any) -> RLMResult:
        """
        Execute RLM loop: Plan → Execute → Verify → Refine.

        Args:
            task: The task description
            context: The document/data to process

        Returns:
            RLMResult with final output and execution trace
        """
        self.iterations = []
        total_llm_calls = 0

        # Store context in operator
        self.rlm_operator.reset_environment()
        self.rlm_operator.execution_environment["context"] = context

        # Initial planning prompt
        current_prompt = f"Task: {task}\n\nGenerate Python code to solve this task."

        for iteration in range(self.max_iterations):
            print(f"\n=== Iteration {iteration + 1}/{self.max_iterations} ===")
            print("Python REPL: executing the generated plan inside the persistent environment, refining its state with each pass.")
            print("The REPL retains llm_query, the document context, and previous outputs so subsequent refinements can focus on the weakest links.")

            # Step 1: Plan (generate code)
            print("Planning...")
            code_response = await self.planner_node.execute(current_prompt)

            # Extract code from markdown blocks if present
            code = self._extract_code(code_response)
            print(f"Generated code ({len(code)} chars)")

            # Step 2: Execute
            print("Executing...")
            exec_result = await self.rlm_operator.execute(code)
            total_llm_calls += exec_result.llm_calls

            repl_state = self.rlm_operator.get_environment_info()
            print("Python REPL state summary:", repl_state)

            # Step 2.5: If error, fix once and retry
            fixed_code = None
            if exec_result.error:
                print("Error occurred:")
                print(exec_result.error)
                print("Calling fixer...")

                fix_prompt = f"""Original code:
```python
{code}
```

Error:
{exec_result.error}

Fix the code to resolve this error."""

                fixed_response = await self.fixer_node.execute(fix_prompt)
                fixed_code = self._extract_code(fixed_response)
                print(f"Fixed code ({len(fixed_code)} chars)")
                print("Retrying execution...")

                # Retry with fixed code
                exec_result = await self.rlm_operator.execute(fixed_code)
                total_llm_calls += exec_result.llm_calls

            # Step 3: Verify
            print("Verifying...")
            verify_prompt = f"""Task: {task}

Code executed:
```python
{fixed_code if fixed_code else code}
```

Execution output:
{exec_result.output}

Result: {exec_result.result}

Error: {exec_result.error if exec_result.error else "None"}

Verify the result and provide confidence. Calibrate the entire probability distribution of outcomes, noting any alternative answers you considered and how you weighed them relative to your final confidence."""

            verification_response = await self.verifier_node.execute(verify_prompt)
            confidence, reasoning = self._parse_verification_response(verification_response)

            print(f"Confidence: {confidence:.2f}")
            print("Reasoning:", reasoning)
            print("Refinement focus: adjusting the Python REPL plan based on verifier feedback and exploring the other plausible answer regions.")

            # Record iteration
            # had_error is True if we had to call the fixer (fixed_code is not None)
            # or if the final execution still has an error
            iteration_record = RLMIteration(
                iteration=iteration + 1,
                code=code,
                execution_result={
                    "output": exec_result.output,
                    "result": str(exec_result.result),
                    "error": exec_result.error,
                    "llm_calls": exec_result.llm_calls
                },
                had_error=fixed_code is not None or exec_result.error is not None,
                fixed_code=fixed_code,
                confidence=confidence,
                verifier_reasoning=reasoning
            )
            self.iterations.append(iteration_record)

            # Step 4: Check stopping criteria
            if exec_result.error:
                # If still erroring after fix, stop
                return RLMResult(
                    success=False,
                    final_output=None,
                    iterations=self.iterations,
                    total_llm_calls=total_llm_calls,
                    total_iterations=len(self.iterations),
                    final_confidence=confidence,
                    stop_reason="error"
                )

            if confidence >= self.confidence_threshold:
                # Success - confidence threshold met
                return RLMResult(
                    success=True,
                    final_output=exec_result.result,
                    iterations=self.iterations,
                    total_llm_calls=total_llm_calls,
                    total_iterations=len(self.iterations),
                    final_confidence=confidence,
                    stop_reason="confidence_threshold"
                )

            # Step 5: Refine for next iteration
            current_prompt = f"""Task: {task}

Previous attempt:
Code:
```python
{fixed_code if fixed_code else code}
```

Result: {exec_result.result}
Output: {exec_result.output}

Verification feedback (confidence {confidence:.2f}):
{reasoning}

Generate improved Python code that addresses the feedback."""

        # Max iterations reached
        final_result = self.iterations[-1].execution_result["result"] if self.iterations else None
        return RLMResult(
            success=True,
            final_output=final_result,
            iterations=self.iterations,
            total_llm_calls=total_llm_calls,
            total_iterations=len(self.iterations),
            final_confidence=confidence,
            stop_reason="max_iterations"
        )

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Python code from markdown blocks"""
        match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1)
        # Try without language specifier
        match = re.search(r'```\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1)
        # Return as-is if no code blocks
        return text.strip()
