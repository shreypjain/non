"""
Deterministic operator implementation for NoN (Network of Networks).

This module contains pure-function deterministic operators with content hashing,
memoization, and no LLM calls. These operators provide reliable, debuggable
data transforms for ensemble patterns and structured data processing.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import json
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from nons.core.types import InputSchema, OutputSchema, OperatorMetadata
from .registry import operator


def _canon(obj: Any) -> bytes:
    """
    Canonicalize the object to bytes for consistent hashing.

    Args:
        obj: The object to canonicalize.

    Returns:
        The canonicalized object as bytes.
    """
    def make_serializable(obj):
        """Convert objects to JSON-serializable format."""
        if hasattr(obj, '__dict__'):
            # Handle dataclasses and custom objects
            if hasattr(obj, '__dataclass_fields__'):
                # This is a dataclass
                from dataclasses import asdict
                return asdict(obj)
            else:
                # Regular object with __dict__
                return obj.__dict__
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: make_serializable(value) for key, value in obj.items()}
        else:
            return obj

    serializable_obj = make_serializable(obj)
    return json.dumps(serializable_obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")


def _hash(obj: Any) -> str:
    """
    Hash the object to a string using SHA256.

    Args:
        obj: The object to hash.

    Returns:
        The canonicalized object hash as hex string.
    """
    return hashlib.sha256(_canon(obj)).hexdigest()


class DeterministicOp(ABC):
    """
    Base class for deterministic operators with content hashing and caching.

    These operators are pure functions that:
    - Take structured input and return structured output
    - Use content hashing for cache keys
    - Have no side effects (no LLM calls, no randomness)
    - Are fully reproducible
    """

    def __init__(self, name: str, enable_cache: bool = True):
        self.name = name
        self.enable_cache = enable_cache
        self._cache: Dict[str, Any] = {}

    @abstractmethod
    def transform(self, input_data: Any) -> Any:
        """
        Pure function transformation of input data.

        Args:
            input_data: The input to transform.

        Returns:
            The transformed output.
        """
        pass

    def __call__(self, input_data: Any) -> Any:
        """
        Execute the deterministic operation with caching.

        Args:
            input_data: The input to process.

        Returns:
            The processed output.
        """
        if self.enable_cache:
            cache_key = _hash(input_data)
            if cache_key in self._cache:
                return self._cache[cache_key]

            result = self.transform(input_data)
            self._cache[cache_key] = result
            return result
        else:
            return self.transform(input_data)

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()

    def cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_enabled": self.enable_cache
        }


@dataclass
class Candidate:
    """A candidate result with metadata."""
    id: str
    content: Any
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PackedCandidates:
    """A collection of candidates with shared metadata."""
    candidates: List[Candidate]
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_count: Optional[int] = None


class PackCandidates(DeterministicOp):
    """
    Pack individual results into a structured candidate format.

    Takes a list of results and converts them into Candidate objects
    with optional scoring and metadata.
    """

    def __init__(self, enable_cache: bool = True):
        super().__init__("pack_candidates", enable_cache)

    def transform(self, input_data: Any) -> PackedCandidates:
        """
        Transform input data into packed candidates.

        Args:
            input_data: Can be:
                - List of strings/objects
                - Dict with 'results' key
                - Dict with 'candidates' key already formatted

        Returns:
            PackedCandidates object.
        """
        if isinstance(input_data, dict):
            if "candidates" in input_data:
                # Already in candidate format
                candidates = [
                    Candidate(**c) if isinstance(c, dict) else Candidate(id=str(i), content=c)
                    for i, c in enumerate(input_data["candidates"])
                ]
                return PackedCandidates(
                    candidates=candidates,
                    metadata=input_data.get("metadata", {}),
                    total_count=len(candidates)
                )
            elif "results" in input_data:
                # Extract from results key
                results = input_data["results"]
            else:
                # Treat whole dict as single result
                results = [input_data]
        elif isinstance(input_data, list):
            results = input_data
        else:
            # Single item
            results = [input_data]

        candidates = [
            Candidate(
                id=f"candidate_{i}",
                content=result,
                score=result.get("score") if isinstance(result, dict) else None
            )
            for i, result in enumerate(results)
        ]

        return PackedCandidates(
            candidates=candidates,
            metadata={"packed_at": _hash(input_data)[:8]},
            total_count=len(candidates)
        )


class ExtractWinners(DeterministicOp):
    """
    Extract winning candidates based on scoring criteria.

    Supports multiple selection strategies:
    - top_k: Select top K candidates by score
    - threshold: Select candidates above threshold
    - percentile: Select top percentile of candidates
    """

    def __init__(self, strategy: str = "top_k", k: int = 1, threshold: float = 0.5, percentile: float = 0.9, enable_cache: bool = True):
        super().__init__("extract_winners", enable_cache)
        self.strategy = strategy
        self.k = k
        self.threshold = threshold
        self.percentile = percentile

    def transform(self, input_data: Any) -> List[Candidate]:
        """
        Extract winning candidates from packed candidates.

        Args:
            input_data: PackedCandidates or dict with candidates.

        Returns:
            List of winning candidates.
        """
        if isinstance(input_data, PackedCandidates):
            candidates = input_data.candidates
        elif isinstance(input_data, dict) and "candidates" in input_data:
            candidates = [
                Candidate(**c) if isinstance(c, dict) else Candidate(id=str(i), content=c)
                for i, c in enumerate(input_data["candidates"])
            ]
        else:
            raise ValueError("Input must be PackedCandidates or dict with 'candidates' key")

        # Ensure all candidates have scores
        scored_candidates = []
        for candidate in candidates:
            if candidate.score is None:
                # Default scoring based on content hash (deterministic)
                score = int(_hash(candidate.content)[:8], 16) / 0xffffffff
                candidate.score = score
            scored_candidates.append(candidate)

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x.score, reverse=True)

        if self.strategy == "top_k":
            return scored_candidates[:self.k]
        elif self.strategy == "threshold":
            return [c for c in scored_candidates if c.score >= self.threshold]
        elif self.strategy == "percentile":
            cutoff_idx = int(len(scored_candidates) * (1 - self.percentile))
            return scored_candidates[:max(1, len(scored_candidates) - cutoff_idx)]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class Majority(DeterministicOp):
    """
    Perform majority voting on candidate results.

    Supports different voting strategies:
    - simple: Most frequent result wins
    - weighted: Weighted by candidate scores
    - consensus: Require minimum agreement threshold
    """

    def __init__(self, strategy: str = "simple", min_consensus: float = 0.5, enable_cache: bool = True):
        super().__init__("majority", enable_cache)
        self.strategy = strategy
        self.min_consensus = min_consensus

    def transform(self, input_data: Any) -> Dict[str, Any]:
        """
        Perform majority voting on candidates.

        Args:
            input_data: List of candidates or PackedCandidates.

        Returns:
            Dict with majority result and voting metadata.
        """
        if isinstance(input_data, PackedCandidates):
            candidates = input_data.candidates
        elif isinstance(input_data, list):
            candidates = input_data
        else:
            raise ValueError("Input must be list of candidates or PackedCandidates")

        if not candidates:
            return {"result": None, "confidence": 0.0, "vote_counts": {}}

        # Extract content for voting
        if self.strategy == "simple":
            # Count content occurrences
            content_votes = [_hash(c.content if hasattr(c, 'content') else c) for c in candidates]
            vote_counts = Counter(content_votes)
            winner_hash = vote_counts.most_common(1)[0][0]

            # Find original content for winner
            winner_content = None
            for c in candidates:
                content = c.content if hasattr(c, 'content') else c
                if _hash(content) == winner_hash:
                    winner_content = content
                    break

            confidence = vote_counts[winner_hash] / len(candidates)

        elif self.strategy == "weighted":
            # Weight votes by candidate scores
            content_scores = {}
            content_mapping = {}

            for c in candidates:
                content = c.content if hasattr(c, 'content') else c
                score = getattr(c, 'score', 1.0) or 1.0
                content_hash = _hash(content)

                if content_hash not in content_scores:
                    content_scores[content_hash] = 0.0
                    content_mapping[content_hash] = content

                content_scores[content_hash] += score

            winner_hash = max(content_scores.keys(), key=lambda x: content_scores[x])
            winner_content = content_mapping[winner_hash]

            total_score = sum(content_scores.values())
            confidence = content_scores[winner_hash] / total_score if total_score > 0 else 0.0
            vote_counts = {h: s for h, s in content_scores.items()}

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Check consensus requirement
        if confidence < self.min_consensus:
            return {
                "result": None,
                "confidence": confidence,
                "vote_counts": dict(vote_counts),
                "consensus_met": False,
                "min_consensus": self.min_consensus
            }

        return {
            "result": winner_content,
            "confidence": confidence,
            "vote_counts": dict(vote_counts),
            "consensus_met": True,
            "strategy": self.strategy
        }


class SelectById(DeterministicOp):
    """
    Select specific candidates by their IDs.

    Useful for extracting specific results from a candidate pool
    or implementing custom selection logic.
    """

    def __init__(self, target_ids: Union[str, List[str]], enable_cache: bool = True):
        super().__init__("select_by_id", enable_cache)
        self.target_ids = [target_ids] if isinstance(target_ids, str) else target_ids

    def transform(self, input_data: Any) -> List[Candidate]:
        """
        Select candidates by their IDs.

        Args:
            input_data: PackedCandidates or list of candidates.

        Returns:
            List of selected candidates.
        """
        if isinstance(input_data, PackedCandidates):
            candidates = input_data.candidates
        elif isinstance(input_data, list):
            candidates = input_data
        else:
            raise ValueError("Input must be list of candidates or PackedCandidates")

        selected = []
        for candidate in candidates:
            candidate_id = getattr(candidate, 'id', None) or str(candidate)
            if candidate_id in self.target_ids:
                selected.append(candidate)

        return selected


# Register deterministic operators with the NoN operator registry

@operator(
    input_schema=InputSchema(
        required_params=["input_data"],
        optional_params=[],
        param_types={"input_data": "Any"}
    ),
    output_schema=OutputSchema(
        return_type="PackedCandidates",
        description="Structured candidate data with metadata"
    ),
    metadata=OperatorMetadata(
        name="pack_candidates",
        description="Pack individual results into structured candidate format",
        examples=["pack_candidates([result1, result2]) -> PackedCandidates"],
        tags=["deterministic", "ensemble", "structure"]
    )
)
async def pack_candidates_op(input_data: Any) -> PackedCandidates:
    """Pack individual results into structured candidate format."""
    op = PackCandidates()
    return op(input_data)


@operator(
    input_schema=InputSchema(
        required_params=["input_data"],
        optional_params=["strategy", "k", "threshold", "percentile"],
        param_types={
            "input_data": "Any",
            "strategy": "str",
            "k": "int",
            "threshold": "float",
            "percentile": "float"
        }
    ),
    output_schema=OutputSchema(
        return_type="List[Candidate]",
        description="List of winning candidates"
    ),
    metadata=OperatorMetadata(
        name="extract_winners",
        description="Extract winning candidates based on scoring criteria",
        examples=["extract_winners(candidates, strategy='top_k', k=3)"],
        tags=["deterministic", "ensemble", "selection"]
    )
)
async def extract_winners_op(input_data: Any, strategy: str = "top_k", k: int = 1, threshold: float = 0.5, percentile: float = 0.9) -> List[Candidate]:
    """Extract winning candidates based on scoring criteria."""
    op = ExtractWinners(strategy=strategy, k=k, threshold=threshold, percentile=percentile)
    return op(input_data)


@operator(
    input_schema=InputSchema(
        required_params=["input_data"],
        optional_params=["strategy", "min_consensus"],
        param_types={
            "input_data": "Any",
            "strategy": "str",
            "min_consensus": "float"
        }
    ),
    output_schema=OutputSchema(
        return_type="Dict[str, Any]",
        description="Majority voting result with confidence and metadata"
    ),
    metadata=OperatorMetadata(
        name="majority",
        description="Perform majority voting on candidate results",
        examples=["majority(candidates, strategy='weighted')"],
        tags=["deterministic", "ensemble", "voting"]
    )
)
async def majority_op(input_data: Any, strategy: str = "simple", min_consensus: float = 0.5) -> Dict[str, Any]:
    """Perform majority voting on candidate results."""
    op = Majority(strategy=strategy, min_consensus=min_consensus)
    return op(input_data)


@operator(
    input_schema=InputSchema(
        required_params=["input_data", "target_ids"],
        optional_params=[],
        param_types={
            "input_data": "Any",
            "target_ids": "Union[str, List[str]]"
        }
    ),
    output_schema=OutputSchema(
        return_type="List[Candidate]",
        description="List of selected candidates"
    ),
    metadata=OperatorMetadata(
        name="select_by_id",
        description="Select specific candidates by their IDs",
        examples=["select_by_id(candidates, ['id1', 'id2'])"],
        tags=["deterministic", "selection", "filter"]
    )
)
async def select_by_id_op(input_data: Any, target_ids: Union[str, List[str]]) -> List[Candidate]:
    """Select specific candidates by their IDs."""
    op = SelectById(target_ids=target_ids)
    return op(input_data)

