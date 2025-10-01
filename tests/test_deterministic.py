"""
Tests for deterministic operators.

Tests the pure-function deterministic operators with content hashing,
memoization, and ensemble voting patterns.
"""

import pytest
from typing import List, Dict, Any

from nons.operators.deterministic import (
    _canon, _hash, DeterministicOp,
    PackCandidates, ExtractWinners, Majority, SelectById,
    Candidate, PackedCandidates,
    pack_candidates_op, extract_winners_op, majority_op, select_by_id_op
)


class TestContentHashing:
    """Test content hashing functions."""

    def test_canon_consistency(self):
        """Test that canonicalization is consistent."""
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}  # Different order

        canon1 = _canon(data1)
        canon2 = _canon(data2)

        assert canon1 == canon2, "Canonicalization should be order-independent"
        assert isinstance(canon1, bytes), "Canonicalization should return bytes"

    def test_hash_consistency(self):
        """Test that hashing is consistent and deterministic."""
        data = {"test": "value", "numbers": [1, 2, 3]}

        hash1 = _hash(data)
        hash2 = _hash(data)

        assert hash1 == hash2, "Hash should be deterministic"
        assert isinstance(hash1, str), "Hash should return string"
        assert len(hash1) == 64, "SHA256 hash should be 64 characters"

    def test_hash_different_inputs(self):
        """Test that different inputs produce different hashes."""
        data1 = {"a": 1}
        data2 = {"a": 2}

        hash1 = _hash(data1)
        hash2 = _hash(data2)

        assert hash1 != hash2, "Different inputs should produce different hashes"

    def test_hash_complex_types(self):
        """Test hashing with complex nested types."""
        complex_data = {
            "nested": {"deep": {"value": 42}},
            "list": [1, 2, {"inner": "test"}],
            "string": "test",
            "null": None,
            "bool": True
        }

        hash_result = _hash(complex_data)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64


class TestDeterministicOp:
    """Test the base DeterministicOp class."""

    class TestOp(DeterministicOp):
        """Test implementation of DeterministicOp."""

        def transform(self, input_data: Any) -> Any:
            # Simple transformation: add "processed_" prefix
            if isinstance(input_data, str):
                return f"processed_{input_data}"
            return {"processed": input_data}

    def test_op_without_cache(self):
        """Test operation without caching."""
        op = self.TestOp("test_op", enable_cache=False)

        result1 = op("test_input")
        result2 = op("test_input")

        assert result1 == "processed_test_input"
        assert result2 == "processed_test_input"
        assert op.cache_stats()["cache_size"] == 0

    def test_op_with_cache(self):
        """Test operation with caching."""
        op = self.TestOp("test_op", enable_cache=True)

        result1 = op("test_input")
        result2 = op("test_input")  # Should hit cache

        assert result1 == result2
        assert op.cache_stats()["cache_size"] == 1
        assert op.cache_stats()["cache_enabled"] == True

    def test_cache_clear(self):
        """Test cache clearing."""
        op = self.TestOp("test_op")

        op("input1")
        op("input2")
        assert op.cache_stats()["cache_size"] == 2

        op.clear_cache()
        assert op.cache_stats()["cache_size"] == 0

    def test_different_inputs_different_cache(self):
        """Test that different inputs create different cache entries."""
        op = self.TestOp("test_op")

        op("input1")
        op("input2")
        op("input1")  # Should hit cache

        assert op.cache_stats()["cache_size"] == 2


class TestPackCandidates:
    """Test the PackCandidates operator."""

    def test_pack_list_input(self):
        """Test packing a simple list."""
        op = PackCandidates()
        input_data = ["result1", "result2", "result3"]

        result = op(input_data)

        assert isinstance(result, PackedCandidates)
        assert len(result.candidates) == 3
        assert result.total_count == 3
        assert result.candidates[0].id == "candidate_0"
        assert result.candidates[0].content == "result1"

    def test_pack_dict_with_results(self):
        """Test packing dict with results key."""
        op = PackCandidates()
        input_data = {"results": ["A", "B", "C"]}

        result = op(input_data)

        assert len(result.candidates) == 3
        assert result.candidates[1].content == "B"

    def test_pack_dict_with_candidates(self):
        """Test packing dict that already has candidates."""
        op = PackCandidates()
        input_data = {
            "candidates": [
                {"id": "test1", "content": "content1", "score": 0.8},
                "raw_content2"
            ],
            "metadata": {"source": "test"}
        }

        result = op(input_data)

        assert len(result.candidates) == 2
        assert result.candidates[0].id == "test1"
        assert result.candidates[0].score == 0.8
        assert result.metadata["source"] == "test"

    def test_pack_single_item(self):
        """Test packing a single item."""
        op = PackCandidates()
        input_data = "single_item"

        result = op(input_data)

        assert len(result.candidates) == 1
        assert result.candidates[0].content == "single_item"

    def test_pack_with_scored_results(self):
        """Test packing results that already have scores."""
        op = PackCandidates()
        input_data = [
            {"content": "result1", "score": 0.9},
            {"content": "result2", "score": 0.7}
        ]

        result = op(input_data)

        assert len(result.candidates) == 2
        assert result.candidates[0].score == 0.9
        assert result.candidates[1].score == 0.7


class TestExtractWinners:
    """Test the ExtractWinners operator."""

    def setup_method(self):
        """Setup test candidates."""
        self.packed_candidates = PackedCandidates(
            candidates=[
                Candidate(id="c1", content="content1", score=0.9),
                Candidate(id="c2", content="content2", score=0.7),
                Candidate(id="c3", content="content3", score=0.5),
                Candidate(id="c4", content="content4", score=0.3)
            ]
        )

    def test_extract_top_k(self):
        """Test top-k extraction strategy."""
        op = ExtractWinners(strategy="top_k", k=2)

        result = op(self.packed_candidates)

        assert len(result) == 2
        assert result[0].score == 0.9  # Highest score first
        assert result[1].score == 0.7

    def test_extract_threshold(self):
        """Test threshold extraction strategy."""
        op = ExtractWinners(strategy="threshold", threshold=0.6)

        result = op(self.packed_candidates)

        assert len(result) == 2  # Only scores >= 0.6
        assert all(c.score >= 0.6 for c in result)

    def test_extract_percentile(self):
        """Test percentile extraction strategy."""
        op = ExtractWinners(strategy="percentile", percentile=0.5)  # Top 50%

        result = op(self.packed_candidates)

        assert len(result) == 2  # Top 50% of 4 candidates
        assert result[0].score >= result[1].score

    def test_extract_with_none_scores(self):
        """Test extraction when some candidates have no scores."""
        candidates_no_scores = PackedCandidates(
            candidates=[
                Candidate(id="c1", content="content1"),  # No score
                Candidate(id="c2", content="content2", score=0.8),
                Candidate(id="c3", content="content3")   # No score
            ]
        )

        op = ExtractWinners(strategy="top_k", k=2)
        result = op(candidates_no_scores)

        assert len(result) == 2
        assert all(c.score is not None for c in result)  # Scores should be assigned

    def test_extract_invalid_strategy(self):
        """Test extraction with invalid strategy."""
        op = ExtractWinners(strategy="invalid_strategy")

        with pytest.raises(ValueError, match="Unknown strategy"):
            op(self.packed_candidates)

    def test_extract_from_dict(self):
        """Test extraction from dict format."""
        op = ExtractWinners(strategy="top_k", k=1)
        input_dict = {
            "candidates": [
                {"id": "c1", "content": "content1", "score": 0.9},
                {"id": "c2", "content": "content2", "score": 0.7}
            ]
        }

        result = op(input_dict)

        assert len(result) == 1
        assert result[0].score == 0.9


class TestMajority:
    """Test the Majority operator."""

    def setup_method(self):
        """Setup test candidates."""
        self.candidates = [
            Candidate(id="c1", content="answer_A", score=0.8),
            Candidate(id="c2", content="answer_A", score=0.7),  # Same content
            Candidate(id="c3", content="answer_B", score=0.9),
            Candidate(id="c4", content="answer_C", score=0.6)
        ]

    def test_simple_majority(self):
        """Test simple majority voting."""
        op = Majority(strategy="simple", min_consensus=0.4)

        result = op(self.candidates)

        assert result["consensus_met"] == True
        assert result["result"] == "answer_A"  # Most frequent
        assert result["confidence"] == 0.5  # 2 out of 4
        assert result["strategy"] == "simple"

    def test_weighted_majority(self):
        """Test weighted majority voting."""
        op = Majority(strategy="weighted", min_consensus=0.3)

        result = op(self.candidates)

        assert result["consensus_met"] == True
        # answer_A has total score 1.5, answer_B has 0.9, answer_C has 0.6
        # answer_A should win but check based on total scores
        assert result["strategy"] == "weighted"
        assert isinstance(result["confidence"], float)

    def test_consensus_not_met(self):
        """Test when consensus threshold is not met."""
        op = Majority(strategy="simple", min_consensus=0.8)  # Very high threshold

        result = op(self.candidates)

        assert result["consensus_met"] == False
        assert result["result"] is None
        assert result["confidence"] < 0.8

    def test_empty_candidates(self):
        """Test majority with empty candidate list."""
        op = Majority()

        result = op([])

        assert result["result"] is None
        assert result["confidence"] == 0.0
        assert result["vote_counts"] == {}

    def test_packed_candidates_input(self):
        """Test majority with PackedCandidates input."""
        op = Majority(strategy="simple")
        packed = PackedCandidates(candidates=self.candidates)

        result = op(packed)

        assert result["result"] == "answer_A"

    def test_invalid_strategy(self):
        """Test majority with invalid strategy."""
        op = Majority(strategy="invalid_strategy")

        with pytest.raises(ValueError, match="Unknown strategy"):
            op(self.candidates)


class TestSelectById:
    """Test the SelectById operator."""

    def setup_method(self):
        """Setup test candidates."""
        self.candidates = [
            Candidate(id="c1", content="content1"),
            Candidate(id="c2", content="content2"),
            Candidate(id="c3", content="content3"),
            Candidate(id="c4", content="content4")
        ]

    def test_select_single_id(self):
        """Test selecting a single candidate by ID."""
        op = SelectById("c2")

        result = op(self.candidates)

        assert len(result) == 1
        assert result[0].id == "c2"
        assert result[0].content == "content2"

    def test_select_multiple_ids(self):
        """Test selecting multiple candidates by ID."""
        op = SelectById(["c1", "c3"])

        result = op(self.candidates)

        assert len(result) == 2
        ids = [c.id for c in result]
        assert "c1" in ids
        assert "c3" in ids

    def test_select_nonexistent_id(self):
        """Test selecting non-existent ID."""
        op = SelectById("nonexistent")

        result = op(self.candidates)

        assert len(result) == 0

    def test_select_from_packed_candidates(self):
        """Test selecting from PackedCandidates."""
        op = SelectById(["c2", "c4"])
        packed = PackedCandidates(candidates=self.candidates)

        result = op(packed)

        assert len(result) == 2
        ids = [c.id for c in result]
        assert "c2" in ids
        assert "c4" in ids

    def test_select_mixed_existing_nonexisting(self):
        """Test selecting mix of existing and non-existing IDs."""
        op = SelectById(["c1", "nonexistent", "c3"])

        result = op(self.candidates)

        assert len(result) == 2  # Only existing ones
        ids = [c.id for c in result]
        assert "c1" in ids
        assert "c3" in ids


class TestRegisteredOperators:
    """Test the registered NoN operators."""

    @pytest.mark.asyncio
    async def test_pack_candidates_op(self):
        """Test the registered pack_candidates operator."""
        input_data = ["item1", "item2", "item3"]

        result = await pack_candidates_op(input_data)

        assert isinstance(result, PackedCandidates)
        assert len(result.candidates) == 3

    @pytest.mark.asyncio
    async def test_extract_winners_op(self):
        """Test the registered extract_winners operator."""
        packed = PackedCandidates(
            candidates=[
                Candidate(id="c1", content="content1", score=0.9),
                Candidate(id="c2", content="content2", score=0.7)
            ]
        )

        result = await extract_winners_op(packed, strategy="top_k", k=1)

        assert len(result) == 1
        assert result[0].score == 0.9

    @pytest.mark.asyncio
    async def test_majority_op(self):
        """Test the registered majority operator."""
        candidates = [
            Candidate(id="c1", content="answer_A"),
            Candidate(id="c2", content="answer_A"),
            Candidate(id="c3", content="answer_B")
        ]

        result = await majority_op(candidates, strategy="simple")

        assert result["result"] == "answer_A"
        assert result["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_select_by_id_op(self):
        """Test the registered select_by_id operator."""
        candidates = [
            Candidate(id="c1", content="content1"),
            Candidate(id="c2", content="content2")
        ]

        result = await select_by_id_op(candidates, ["c1"])

        assert len(result) == 1
        assert result[0].id == "c1"


class TestIntegration:
    """Test integration scenarios and edge cases."""

    def test_full_ensemble_pipeline(self):
        """Test a complete ensemble processing pipeline."""
        # Raw input data
        raw_responses = [
            "Response A with high quality",
            "Response B with medium quality",
            "Response A with high quality",  # Duplicate
            "Response C with low quality"
        ]

        # Step 1: Pack candidates
        pack_op = PackCandidates()
        packed = pack_op(raw_responses)

        assert len(packed.candidates) == 4

        # Step 2: Extract top candidates
        extract_op = ExtractWinners(strategy="top_k", k=3)
        top_candidates = extract_op(packed)

        assert len(top_candidates) <= 3

        # Step 3: Majority vote
        majority_op = Majority(strategy="simple", min_consensus=0.4)
        majority_result = majority_op(top_candidates)

        assert "result" in majority_result
        assert "confidence" in majority_result

        # Step 4: Select specific candidates
        if top_candidates:
            select_op = SelectById([top_candidates[0].id])
            selected = select_op(top_candidates)
            assert len(selected) == 1

    def test_error_handling(self):
        """Test error handling in deterministic operators."""
        # Test ExtractWinners with invalid input
        extract_op = ExtractWinners()
        with pytest.raises(ValueError):
            extract_op("invalid_input")

        # Test Majority with invalid input
        majority_op = Majority()
        with pytest.raises(ValueError):
            majority_op("invalid_input")

        # Test SelectById with invalid input
        select_op = SelectById("test_id")
        with pytest.raises(ValueError):
            select_op("invalid_input")

    def test_deterministic_behavior(self):
        """Test that operations are truly deterministic."""
        input_data = ["test1", "test2", "test3"]

        # Run same operations multiple times
        pack_op = PackCandidates()
        result1 = pack_op(input_data)
        result2 = pack_op(input_data)

        # Results should be identical
        assert result1.total_count == result2.total_count
        assert len(result1.candidates) == len(result2.candidates)

        # Candidate hashes should be the same
        for i in range(len(result1.candidates)):
            assert _hash(result1.candidates[i].content) == _hash(result2.candidates[i].content)

    def test_large_candidate_set(self):
        """Test performance with larger candidate sets."""
        # Create large candidate set
        large_input = [f"candidate_{i}" for i in range(100)]

        pack_op = PackCandidates()
        packed = pack_op(large_input)

        assert len(packed.candidates) == 100

        # Test majority voting on large set
        majority_op = Majority(strategy="simple")
        result = majority_op(packed)

        assert "result" in result
        assert isinstance(result["confidence"], float)