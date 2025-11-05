"""
Base operator implementations for NoN (Network of Networks).

This module contains the 10 fundamental operators that serve as mathematical
primitives for compound AI systems. These operators are irreducible yet
infinitely composable building blocks.
"""

from typing import List, Optional, Union, Any
from ..core.types import (
    Content,
    StructuredOutput,
    TransformSpec,
    ExtractionCriteria,
    ClassificationSchema,
    GenerationSpec,
    ValidationCriteria,
    ExpansionTopic,
    SynthesisFocus,
    ComparisonDimensions,
    RoutingLogic,
    Classification,
    ValidationResult,
    ComparisonAnalysis,
    RouteDecision,
    ModelConfig,
)
from .registry import operator
from ..utils.providers import create_provider
from ..core.config import get_default_model_config


@operator(
    metadata={
        "name": "transform",
        "description": "The identity function of NoNs - reshapes input content according to transformation directive",
        "examples": [
            "transform(content, 'convert to JSON format')",
            "transform(content, 'translate to Spanish')",
            "transform(content, 'rewrite in formal tone')",
        ],
        "tags": ["core", "formatting", "conversion"],
    }
)
async def transform(
    content: Content,
    transformation_type: TransformSpec,
    model_config: Optional[ModelConfig] = None,
) -> Content:
    """
    Reshapes input content according to transformation directive.

    The identity function of NoNs that handles format conversion, style transfer,
    structural reorganization, and translation tasks.

    Args:
        content: The input content to transform
        transformation_type: Specification of how to transform the content
        model_config: Optional model configuration (uses default if not provided)

    Returns:
        Transformed content maintaining the same semantic meaning
    """
    if isinstance(content, dict):
        content_str = str(content)
    else:
        content_str = content

    prompt = f"""Transform the following content according to this specification: {transformation_type}

Content to transform:
{content_str}

Provide only the transformed content without additional explanation."""

    # Get model configuration and create provider
    if model_config is None:
        model_config = get_default_model_config()

    provider = create_provider(model_config)
    result, metrics = await provider.generate_completion(prompt)

    return result


@operator(
    metadata={
        "name": "synthesize",
        "description": "Interweaves two concepts into unified context while maintaining fidelity",
        "examples": [
            "synthesize(report1, report2, 'focus on common themes')",
            "synthesize(data, analysis, 'create unified narrative')",
        ],
        "tags": ["core", "fusion", "merging"],
    }
)
async def synthesize(
    original_content: Content,
    new_content: Content,
    synthesis_focus: Optional[SynthesisFocus] = None,
    model_config: Optional[ModelConfig] = None,
) -> Content:
    """
    Interweaves two concepts into unified context.

    Averages information density while maintaining fidelity. Used for merging
    perspectives, reconciling conflicts, and creating unified narratives.

    Args:
        original_content: The base content to synthesize with
        new_content: The additional content to weave in
        synthesis_focus: Optional focus for the synthesis process
        model_config: Optional model configuration (uses default if not provided)

    Returns:
        Synthesized content combining both inputs
    """
    focus_instruction = f" with focus on: {synthesis_focus}" if synthesis_focus else ""

    prompt = f"""Synthesize these two pieces of content into a unified narrative{focus_instruction}.
Maintain the key information from both while creating coherent flow.

Original content:
{original_content}

New content:
{new_content}

Provide the synthesized result without additional explanation."""

    if model_config is None:
        model_config = get_default_model_config()

    provider = create_provider(model_config)
    result, metrics = await provider.generate_completion(prompt)

    return result


@operator(
    metadata={
        "name": "expand",
        "description": "Multiplicative growth of information depth around a specific topic",
        "examples": [
            "expand(summary, 'technical details', 3)",
            "expand(outline, 'market analysis', 2)",
        ],
        "tags": ["core", "elaboration", "depth"],
    }
)
async def expand(
    content: Content,
    expansion_topic: ExpansionTopic,
    depth_level: int,
    model_config: Optional[ModelConfig] = None,
) -> Content:
    """
    Multiplicative growth of information depth around a specific topic.

    Used for deep dives, contextual enrichment, and technical elaboration.

    Args:
        content: The content to expand upon
        expansion_topic: The specific topic or angle to expand
        depth_level: Integer indicating depth of expansion (1-5)
        model_config: Optional model configuration (uses default if not provided)

    Returns:
        Expanded content with greater detail on the specified topic
    """
    depth_instruction = {
        1: "briefly elaborate",
        2: "provide moderate detail",
        3: "provide comprehensive detail",
        4: "provide extensive analysis",
        5: "provide exhaustive coverage",
    }.get(depth_level, "provide detail")

    prompt = f"""Expand the following content on the topic "{expansion_topic}".
{depth_instruction.capitalize()} while maintaining relevance to the original content.

Original content:
{content}

Expansion topic: {expansion_topic}
Depth level: {depth_level}

Provide the expanded content without additional explanation."""

    if model_config is None:
        model_config = get_default_model_config()

    provider = create_provider(model_config)
    result, metrics = await provider.generate_completion(prompt)

    return result


@operator(
    metadata={
        "name": "condense",
        "description": "Distills content to essential components with understanding, not truncation",
        "examples": [
            "condense(report, target_length=500)",
            "condense(analysis, preservation_priority='key findings')",
        ],
        "tags": ["core", "summarization", "compression"],
    }
)
async def condense(
    content: Content,
    target_length: Optional[int] = None,
    preservation_priority: Optional[str] = None,
    model_config: Optional[ModelConfig] = None,
) -> Content:
    """
    Distills content to essential components.

    Compression with understanding, not truncation. Used for summarization,
    executive briefs, and removing redundancy.

    Args:
        content: The content to condense
        target_length: Optional target length in characters/words
        preservation_priority: What aspects to prioritize preserving
        model_config: Optional model configuration (uses default if not provided)

    Returns:
        Condensed content maintaining essential information
    """
    length_instruction = (
        f" to approximately {target_length} characters" if target_length else ""
    )
    priority_instruction = (
        f" while prioritizing: {preservation_priority}" if preservation_priority else ""
    )

    prompt = f"""Condense the following content{length_instruction}{priority_instruction}.
Maintain all essential information and key insights while removing redundancy.

Content to condense:
{content}

Provide the condensed content without additional explanation."""

    if model_config is None:
        model_config = get_default_model_config()

    provider = create_provider(model_config)
    result, metrics = await provider.generate_completion(prompt)

    return result


@operator(
    metadata={
        "name": "extract",
        "description": "Surgical isolation of specific information matching extraction criteria",
        "examples": [
            "extract(document, 'all dates and times')",
            "extract(text, 'key financial metrics', 'JSON')",
        ],
        "tags": ["core", "data-mining", "isolation"],
    }
)
async def extract(
    content: Content,
    extraction_criteria: ExtractionCriteria,
    output_format: Optional[str] = None,
    model_config: Optional[ModelConfig] = None,
) -> StructuredOutput[Any]:
    """
    Surgical isolation of specific information matching extraction criteria.

    Used for data mining, answer extraction, pulling quotes, and finding facts.

    Args:
        content: The content to extract information from
        extraction_criteria: Specification of what to extract
        output_format: Optional format for the extracted data
        model_config: Optional model configuration (uses default if not provided)

    Returns:
        StructuredOutput containing the extracted information
    """
    format_instruction = f" in {output_format} format" if output_format else ""

    prompt = f"""Extract information from the following content based on these criteria: {extraction_criteria}
Present the extracted information{format_instruction}.

Content:
{content}

Extraction criteria: {extraction_criteria}

Provide only the extracted information without additional explanation."""

    if model_config is None:
        model_config = get_default_model_config()

    provider = create_provider(model_config)
    extracted_data, metrics = await provider.generate_completion(prompt)

    return StructuredOutput(
        data=extracted_data,
        metadata={"extraction_criteria": extraction_criteria, "format": output_format},
    )


@operator(
    metadata={
        "name": "classify",
        "description": "Categorical assignment into predefined or emergent classes",
        "examples": [
            "classify(text, 'sentiment: positive/negative/neutral')",
            "classify(document, 'document type schema', confidence_threshold=0.8)",
        ],
        "tags": ["core", "categorization", "decision-boundary"],
    }
)
async def classify(
    content: Content,
    classification_schema: ClassificationSchema,
    confidence_threshold: Optional[float] = None,
    model_config: Optional[ModelConfig] = None,
) -> Classification:
    """
    Categorical assignment into predefined or emergent classes.

    The decision boundary operator used for routing decisions, sentiment analysis,
    and topic categorization.

    Args:
        content: The content to classify
        classification_schema: Schema defining categories and rules
        confidence_threshold: Minimum confidence level required
        model_config: Optional model configuration (uses default if not provided)

    Returns:
        Classification result with category, confidence, and reasoning
    """
    threshold_note = (
        f" (minimum confidence: {confidence_threshold})" if confidence_threshold else ""
    )

    prompt = f"""Classify the following content according to this schema: {classification_schema}
Provide the category, confidence score (0-1), and brief reasoning{threshold_note}.

Content to classify:
{content}

Classification schema: {classification_schema}

Respond in the format: Category | Confidence | Reasoning"""

    if model_config is None:
        model_config = get_default_model_config()

    provider = create_provider(model_config)
    response, metrics = await provider.generate_completion(prompt)

    # Parse response (simple parsing, could be enhanced with structured output)
    parts = response.split("|")
    if len(parts) >= 3:
        category = parts[0].strip()
        try:
            confidence = float(parts[1].strip())
        except ValueError:
            confidence = 0.5
        reasoning = parts[2].strip()
    else:
        category = response.strip()
        confidence = 0.5
        reasoning = "Parsed from LLM response"

    return Classification(category=category, confidence=confidence, reasoning=reasoning)


@operator(
    metadata={
        "name": "compare",
        "description": "Differential analysis between content pieces along specified dimensions",
        "examples": [
            "compare(doc1, doc2, 'technical accuracy')",
            "compare(version1, version2, 'feature completeness', [baseline])",
        ],
        "tags": ["core", "analysis", "differential"],
    }
)
async def compare(
    content_a: Content,
    content_b: Content,
    comparison_dimensions: Optional[ComparisonDimensions] = None,
    additional_content: Optional[List[Content]] = None,
    model_config: Optional[ModelConfig] = None,
) -> ComparisonAnalysis:
    """
    Differential analysis between content pieces along specified dimensions.

    Used for fact-checking, version comparison, and competitive analysis.

    Args:
        content_a: First content piece to compare
        content_b: Second content piece to compare
        comparison_dimensions: Optional dimensions to focus comparison on
        additional_content: Optional additional content pieces for context
        model_config: Optional model configuration (uses default if not provided)

    Returns:
        ComparisonAnalysis with differences, similarities, and conclusion
    """
    dimensions_note = (
        f" focusing on: {comparison_dimensions}" if comparison_dimensions else ""
    )
    additional_note = (
        f" with {len(additional_content)} additional context pieces"
        if additional_content
        else ""
    )

    prompt = f"""Compare these two pieces of content{dimensions_note}{additional_note}.
Identify key differences, similarities, and provide an overall conclusion.

Content A:
{content_a}

Content B:
{content_b}

Provide structured comparison in this format:
DIFFERENCES:
- [list differences, one per line]

SIMILARITIES:
- [list similarities, one per line]

CONCLUSION:
[provide conclusion]"""

    if model_config is None:
        model_config = get_default_model_config()

    provider = create_provider(model_config)
    response, metrics = await provider.generate_completion(prompt)

    # Parse response (simple parsing)
    differences = []
    similarities = []
    conclusion = ""

    sections = response.split("SIMILARITIES:")
    if len(sections) >= 2:
        diff_section = sections[0].replace("DIFFERENCES:", "").strip()
        differences = [
            line.strip("- ").strip()
            for line in diff_section.split("\n")
            if line.strip().startswith("-")
        ]

        remaining = sections[1].split("CONCLUSION:")
        if len(remaining) >= 2:
            sim_section = remaining[0].strip()
            similarities = [
                line.strip("- ").strip()
                for line in sim_section.split("\n")
                if line.strip().startswith("-")
            ]
            conclusion = remaining[1].strip()
        else:
            similarities = [
                line.strip("- ").strip()
                for line in remaining[0].split("\n")
                if line.strip().startswith("-")
            ]

    if not conclusion:
        conclusion = "Comparison completed"

    return ComparisonAnalysis(
        differences=differences if differences else ["No differences identified"],
        similarities=similarities if similarities else ["No similarities identified"],
        conclusion=conclusion,
    )


@operator(
    metadata={
        "name": "generate",
        "description": "Pure creation of novel content from specifications without input dependency",
        "examples": [
            "generate('write a product roadmap', constraints='1 page')",
            "generate('create test data', style_guide='JSON format')",
        ],
        "tags": ["core", "creation", "generation"],
    }
)
async def generate(
    generation_specification: GenerationSpec,
    constraints: Optional[str] = None,
    style_guide: Optional[str] = None,
    model_config: Optional[ModelConfig] = None,
) -> Content:
    """
    Pure creation of novel content from specifications.

    Used for creative writing, code generation, and initial content creation
    without dependency on input content.

    Args:
        generation_specification: Specification of what to generate
        constraints: Optional constraints on the generation
        style_guide: Optional style guide to follow
        model_config: Optional model configuration (uses default if not provided)

    Returns:
        Generated content matching the specification
    """
    constraints_note = f"\nConstraints: {constraints}" if constraints else ""
    style_note = f"\nStyle guide: {style_guide}" if style_guide else ""

    prompt = f"""Generate content according to this specification: {generation_specification}{constraints_note}{style_note}

Provide only the generated content without additional explanation."""

    if model_config is None:
        model_config = get_default_model_config()

    provider = create_provider(model_config)
    result, metrics = await provider.generate_completion(prompt)

    return result


@operator(
    metadata={
        "name": "validate",
        "description": "Truth-testing against validation criteria or external knowledge",
        "examples": [
            "validate(claim, 'factual accuracy', 'strict')",
            "validate(code, 'syntax correctness', 'moderate')",
        ],
        "tags": ["core", "verification", "truth-testing"],
    }
)
async def validate(
    content: Content,
    validation_criteria: ValidationCriteria,
    strictness_level: str,
    model_config: Optional[ModelConfig] = None,
) -> ValidationResult:
    """
    Truth-testing against validation criteria or external knowledge.

    Used for fact verification, logical consistency, and constraint satisfaction.

    Args:
        content: The content to validate
        validation_criteria: Criteria to validate against
        strictness_level: Level of strictness (strict/moderate/lenient)
        model_config: Optional model configuration (uses default if not provided)

    Returns:
        ValidationResult with validity, reasoning, and confidence
    """
    prompt = f"""Validate the following content against these criteria: {validation_criteria}
Use {strictness_level} strictness level. Provide validation result, reasoning, and confidence.

Content to validate:
{content}

Validation criteria: {validation_criteria}
Strictness: {strictness_level}

Respond in the format: Valid/Invalid | Reasoning | Confidence (0-1)"""

    if model_config is None:
        model_config = get_default_model_config()

    provider = create_provider(model_config)
    response, metrics = await provider.generate_completion(prompt)

    # Parse response
    parts = response.split("|")
    if len(parts) >= 3:
        is_valid = parts[0].strip().lower() in ["valid", "true", "yes"]
        reasoning = parts[1].strip()
        try:
            confidence = float(parts[2].strip())
        except ValueError:
            confidence = 0.5
    else:
        is_valid = "valid" in response.lower()
        reasoning = response.strip()
        confidence = 0.5

    return ValidationResult(
        is_valid=is_valid, validation_reasoning=reasoning, confidence=confidence
    )


@operator(
    metadata={
        "name": "route",
        "description": "Dynamic decision-making on information flow and execution paths",
        "examples": [
            "route(content, 'complexity-based routing', ['simple', 'complex', 'expert'])",
            "route(query, 'model selection logic', ['gpt-4', 'claude', 'local'])",
        ],
        "tags": ["core", "control-flow", "decision-making"],
    }
)
async def route(
    content: Content,
    routing_logic: RoutingLogic,
    available_paths: List[str],
    model_config: Optional[ModelConfig] = None,
) -> RouteDecision:
    """
    Dynamic decision-making on information flow.

    Control flow operator determining subsequent execution paths. Used for
    adaptive workflows, complexity-based model selection, and conditional branching.

    Args:
        content: The content to base routing decision on
        routing_logic: Logic specification for routing decision
        available_paths: List of available paths to route to
        model_config: Optional model configuration (uses default if not provided)

    Returns:
        RouteDecision with selected path, confidence, and reasoning
    """
    prompt = f"""Analyze the content and determine the best routing path based on: {routing_logic}

Content:
{content}

Available paths: {', '.join(available_paths)}
Routing logic: {routing_logic}

Respond in the format: Selected_Path | Confidence (0-1) | Reasoning"""

    if model_config is None:
        model_config = get_default_model_config()

    provider = create_provider(model_config)
    response, metrics = await provider.generate_completion(prompt)

    # Parse response
    parts = response.split("|")
    if len(parts) >= 3:
        selected_path = parts[0].strip()
        try:
            routing_confidence = float(parts[1].strip())
        except ValueError:
            routing_confidence = 0.5
        reasoning = parts[2].strip()
    else:
        # Try to find matching path in response
        selected_path = available_paths[0] if available_paths else "default"
        for path in available_paths:
            if path.lower() in response.lower():
                selected_path = path
                break
        routing_confidence = 0.5
        reasoning = response.strip()

    return RouteDecision(
        selected_path=selected_path,
        routing_confidence=routing_confidence,
        reasoning=reasoning,
    )
