"""
Regression tests for three core/operator bug fixes.

Issue #7  - Agent tool execution passes null params (nons/core/agents/agent.py)
Issue #14 - Operator metadata crashes when registered with dict metadata
             (nons/operators/registry.py)
Issue #8  - ExtractWinners mutates input candidates, causing unbounded cache
             growth (nons/operators/deterministic.py)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nons.core.types import RouteDecision, OperatorMetadata
from nons.core.agents.agent import Agent
from nons.core.agents.registry import ToolRegistry
from nons.operators.registry import OperatorRegistry
from nons.operators.deterministic import (
    Candidate,
    PackedCandidates,
    ExtractWinners,
)


# ---------------------------------------------------------------------------
# Issue #7 – Agent._execute_tool passes None params instead of {}
# ---------------------------------------------------------------------------


class TestAgentNullParams:
    """Regression tests for Issue #7."""

    def _make_agent(self, registry: ToolRegistry) -> Agent:
        """Return an Agent wired with a stub NoN network."""
        mock_network = MagicMock()
        return Agent(network=mock_network, registry=registry)

    @pytest.mark.asyncio
    async def test_execute_tool_uses_empty_dict_when_params_is_none(self):
        """
        When RouteDecision.params is None, _execute_tool must pass {} to
        ToolRegistry.execute, not None.

        Before the fix, ``hasattr(decision, 'params')`` was always True for
        Pydantic models so None was forwarded verbatim.
        """
        registry = ToolRegistry()
        captured: list = []

        async def my_tool(params, **ctx):
            captured.append(params)
            return {"done": True}

        registry._register("my_tool", "A test tool", my_tool)

        agent = self._make_agent(registry)

        decision = RouteDecision(
            selected_path="my_tool",
            routing_confidence=0.9,
            reasoning="test",
            params=None,  # Explicitly None – the problematic case
        )

        result = await agent._execute_tool(decision, state={})

        # The tool must have been called with an empty dict, not None
        assert len(captured) == 1
        assert captured[0] == {}, (
            "Expected {} to be passed when params is None, "
            f"got {captured[0]!r} instead"
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_tool_forwards_params_when_provided(self):
        """
        When RouteDecision.params is a real dict, it is forwarded as-is.
        This ensures the fix does not break the happy path.
        """
        registry = ToolRegistry()
        captured: list = []

        async def my_tool(params, **ctx):
            captured.append(params)
            return {"done": True}

        registry._register("my_tool", "A test tool", my_tool)

        agent = self._make_agent(registry)

        decision = RouteDecision(
            selected_path="my_tool",
            routing_confidence=0.9,
            reasoning="test",
            params={"key": "value"},
        )

        await agent._execute_tool(decision, state={})

        assert captured[0] == {"key": "value"}


# ---------------------------------------------------------------------------
# Issue #14 – OperatorRegistry.register() crashes on dict metadata
# ---------------------------------------------------------------------------


class TestOperatorRegistryDictMetadata:
    """Regression tests for Issue #14."""

    def test_register_with_dict_metadata_does_not_raise(self):
        """
        Passing a plain dict as ``metadata`` must succeed and produce a
        RegisteredOperator whose .metadata is an OperatorMetadata instance.

        Before the fix, attribute access like .metadata.description would
        crash with AttributeError because the dict was stored verbatim.
        """
        registry = OperatorRegistry()

        async def my_op(text: str) -> str:
            """My operator docstring."""
            return text

        metadata_dict = {
            "name": "my_op",
            "description": "A dict-metadata operator",
            "examples": ["example()"],
            "tags": ["test"],
        }

        registered = registry.register(func=my_op, name="my_op", metadata=metadata_dict)

        # Attribute access must not raise
        assert registered.metadata.description == "A dict-metadata operator"
        assert isinstance(registered.metadata, OperatorMetadata)
        assert registered.metadata.tags == ["test"]

    def test_register_with_dict_metadata_name_is_used(self):
        """The 'name' key inside the dict is propagated to OperatorMetadata."""
        registry = OperatorRegistry()

        async def another_op(x: int) -> int:
            return x

        registered = registry.register(
            func=another_op,
            name="another_op",
            metadata={"name": "another_op", "description": "desc", "examples": [], "tags": []},
        )

        assert registered.metadata.name == "another_op"

    def test_register_with_operatormetadata_instance_still_works(self):
        """
        Passing an OperatorMetadata instance (the original interface) must
        continue to work after the dict-normalisation change.
        """
        registry = OperatorRegistry()

        async def typed_op(x: str) -> str:
            return x

        meta = OperatorMetadata(
            name="typed_op",
            description="Uses typed metadata",
            examples=[],
            tags=["typed"],
        )

        registered = registry.register(func=typed_op, name="typed_op", metadata=meta)

        assert registered.metadata.description == "Uses typed metadata"
        assert isinstance(registered.metadata, OperatorMetadata)

    def test_register_with_none_metadata_auto_generates(self):
        """
        Passing no metadata (None) still auto-generates an OperatorMetadata
        from the function name and docstring – existing behaviour preserved.
        """
        registry = OperatorRegistry()

        async def auto_op(x: str) -> str:
            """Auto-generated docstring."""
            return x

        registered = registry.register(func=auto_op, name="auto_op")

        assert isinstance(registered.metadata, OperatorMetadata)
        assert "Auto-generated docstring" in registered.metadata.description


# ---------------------------------------------------------------------------
# Issue #8 – ExtractWinners mutates input Candidate objects
# ---------------------------------------------------------------------------


class TestExtractWinnersNoMutation:
    """Regression tests for Issue #8."""

    def test_transform_does_not_mutate_original_candidates(self):
        """
        Candidates that enter ExtractWinners.transform() with score=None must
        still have score=None after the call; the assigned score must only
        appear on the returned copies.

        Before the fix, ``candidate.score = score`` modified the dataclass
        in-place, changing its hash and causing unbounded cache growth.
        """
        candidates_without_scores = [
            Candidate(id="a", content="alpha"),
            Candidate(id="b", content="beta"),
            Candidate(id="c", content="gamma"),
        ]
        packed = PackedCandidates(candidates=candidates_without_scores)

        op = ExtractWinners(strategy="top_k", k=2, enable_cache=False)
        winners = op(packed)

        # Returned winners must have scores assigned
        assert all(w.score is not None for w in winners)

        # Originals must be untouched
        for original in candidates_without_scores:
            assert original.score is None, (
                f"Candidate '{original.id}' was mutated: "
                f"score changed to {original.score!r}"
            )

    def test_cache_key_stable_across_repeated_calls(self):
        """
        Calling ExtractWinners twice on the same PackedCandidates must hit the
        cache on the second call (cache size stays at 1, not growing).

        Mutation of the input would change its hash between calls, producing a
        new cache entry each time.
        """
        candidates = [
            Candidate(id="x", content="xray"),
            Candidate(id="y", content="yankee"),
        ]
        packed = PackedCandidates(candidates=candidates)

        op = ExtractWinners(strategy="top_k", k=1, enable_cache=True)

        op(packed)
        assert op.cache_stats()["cache_size"] == 1

        op(packed)
        # Second call must hit cache – size stays at 1
        assert op.cache_stats()["cache_size"] == 1, (
            "Cache grew on the second call with identical input, "
            "indicating the input was mutated between calls"
        )

    def test_returned_winner_scores_are_deterministic(self):
        """
        The score assigned to a scoreless candidate must be the same on every
        call (derived from a content hash), even without caching.
        """
        packed = PackedCandidates(
            candidates=[Candidate(id="z", content="zulu")]
        )

        op = ExtractWinners(strategy="top_k", k=1, enable_cache=False)
        result1 = op(packed)
        result2 = op(packed)

        assert result1[0].score == result2[0].score
