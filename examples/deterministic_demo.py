#!/usr/bin/env python3
"""
Deterministic Operators Demo

This example demonstrates the deterministic operators for NoN, showing:
1. Pure-function operators with content hashing and caching
2. Ensemble voting patterns
3. Structured candidate processing
4. Integration with NoN networks

The deterministic operators provide reliable, debuggable data transforms
with no LLM calls and no randomness.
"""

import asyncio
import json
from typing import List, Dict, Any

from nons.core.network import NoN
from nons.core.node import Node
from nons.operators.deterministic import (
    PackCandidates, ExtractWinners, Majority, SelectById,
    Candidate, PackedCandidates, _hash, _canon
)

async def demo_basic_deterministic_ops():
    """Demonstrate basic usage of deterministic operators."""
    print("🎯 BASIC DETERMINISTIC OPERATORS DEMO")
    print("=" * 50)

    # 1. PackCandidates - Convert raw results to structured format
    print("\n1. PackCandidates - Structure raw results")
    pack_op = PackCandidates()

    raw_results = [
        "The capital of France is Paris",
        "Paris is the capital of France",
        "France's capital city is Paris"
    ]

    packed = pack_op(raw_results)
    print(f"Input: {raw_results}")
    print(f"Packed: {len(packed.candidates)} candidates")
    for i, candidate in enumerate(packed.candidates):
        print(f"  Candidate {i}: {candidate.id} -> {candidate.content}")

    # 2. ExtractWinners - Select top candidates
    print("\n2. ExtractWinners - Select top candidates")
    extract_op = ExtractWinners(strategy="top_k", k=2)

    winners = extract_op(packed)
    print(f"Top 2 winners:")
    for winner in winners:
        print(f"  {winner.id}: {winner.content} (score: {winner.score:.4f})")

    # 3. Majority voting - Find consensus
    print("\n3. Majority - Find consensus")
    majority_op = Majority(strategy="simple", min_consensus=0.4)

    majority_result = majority_op(packed)
    print(f"Majority result: {majority_result['result']}")
    print(f"Confidence: {majority_result['confidence']:.2f}")
    print(f"Consensus met: {majority_result['consensus_met']}")

    # 4. SelectById - Pick specific candidates
    print("\n4. SelectById - Select specific candidates")
    select_op = SelectById(target_ids=["candidate_0", "candidate_2"])

    selected = select_op(packed)
    print(f"Selected candidates:")
    for candidate in selected:
        print(f"  {candidate.id}: {candidate.content}")

    print("\n" + "=" * 50)

async def demo_ensemble_flow_1():
    """
    Ensemble Flow 1: Multi-generation with majority voting

    Pattern: Generate multiple candidates → Pack → Vote → Extract winner
    """
    print("\n🏆 ENSEMBLE FLOW 1: MULTI-GENERATION WITH VOTING")
    print("=" * 50)

    # Simulate multiple LLM responses (normally these would come from different models/prompts)
    simulated_responses = [
        {"response": "Renewable energy is crucial for environmental sustainability", "model": "claude"},
        {"response": "Clean energy sources are vital for our planet's future", "model": "gpt"},
        {"response": "Sustainable power generation protects the environment", "model": "gemini"},
        {"response": "Renewable energy is crucial for environmental sustainability", "model": "claude-2"},  # Duplicate
        {"response": "Green energy solutions are essential for climate action", "model": "gpt-4"}
    ]

    print(f"Generated {len(simulated_responses)} responses from different models")

    # Step 1: Pack candidates
    pack_op = PackCandidates()
    packed = pack_op(simulated_responses)
    print(f"\nPacked into {len(packed.candidates)} candidates")

    # Step 2: Majority voting with simple strategy (low threshold to demonstrate consensus)
    majority_op = Majority(strategy="simple", min_consensus=0.2)
    majority_result = majority_op(packed)

    print(f"\nMajority Voting Results:")
    if majority_result['consensus_met'] and majority_result['result']:
        result_content = majority_result['result']
        if isinstance(result_content, dict) and 'response' in result_content:
            print(f"  Winner: {result_content['response']}")
        else:
            print(f"  Winner: {result_content}")
    else:
        print(f"  Winner: No consensus reached (threshold not met)")
    print(f"  Confidence: {majority_result['confidence']:.2f}")
    print(f"  Consensus met: {majority_result['consensus_met']}")
    print(f"  Vote distribution: {len(majority_result['vote_counts'])} unique responses")

    # Step 3: Extract top candidates for backup options
    extract_op = ExtractWinners(strategy="top_k", k=3)
    top_candidates = extract_op(packed)

    print(f"\nTop 3 candidates (as backup options):")
    for i, candidate in enumerate(top_candidates):
        response_data = candidate.content
        model = response_data.get('model', 'unknown') if isinstance(response_data, dict) else 'unknown'
        response = response_data.get('response', str(response_data)) if isinstance(response_data, dict) else str(response_data)
        print(f"  {i+1}. [{model}] {response[:60]}... (score: {candidate.score:.4f})")

async def demo_ensemble_flow_2():
    """
    Ensemble Flow 2: Scored candidates with weighted voting

    Pattern: Generate scored candidates → Pack with scores → Weighted vote → Select by criteria
    """
    print("\n🎲 ENSEMBLE FLOW 2: SCORED CANDIDATES WITH WEIGHTED VOTING")
    print("=" * 50)

    # Simulate scored candidates (e.g., from different evaluation criteria)
    scored_candidates_data = {
        "candidates": [
            {"id": "response_a", "content": "Solar and wind power are the most promising renewable technologies", "score": 0.85},
            {"id": "response_b", "content": "Nuclear energy provides clean, reliable baseload power", "score": 0.72},
            {"id": "response_c", "content": "Hydroelectric dams offer sustainable energy generation", "score": 0.68},
            {"id": "response_d", "content": "Solar and wind power are the most promising renewable technologies", "score": 0.91},  # Duplicate with higher score
            {"id": "response_e", "content": "Geothermal energy taps into Earth's natural heat", "score": 0.55}
        ]
    }

    print(f"Processing {len(scored_candidates_data['candidates'])} scored candidates")

    # Step 1: Pack candidates (already structured, but standardizes format)
    pack_op = PackCandidates()
    packed = pack_op(scored_candidates_data)
    print(f"\nPacked candidates with scores:")
    for candidate in packed.candidates:
        print(f"  {candidate.id}: score={candidate.score:.2f}")

    # Step 2: Weighted majority voting
    weighted_majority_op = Majority(strategy="weighted", min_consensus=0.4)
    weighted_result = weighted_majority_op(packed)

    print(f"\nWeighted Majority Voting Results:")
    if weighted_result['consensus_met'] and weighted_result['result']:
        result_content = weighted_result['result']
        if isinstance(result_content, dict) and 'content' in result_content:
            print(f"  Winner: {result_content['content']}")
        else:
            print(f"  Winner: {result_content}")
    else:
        print(f"  Winner: No consensus reached")
    print(f"  Confidence: {weighted_result['confidence']:.2f}")
    print(f"  Strategy: {weighted_result['strategy']}")

    # Step 3: Extract high-scoring candidates above threshold
    threshold_extract_op = ExtractWinners(strategy="threshold", threshold=0.7)
    high_quality = threshold_extract_op(packed)

    print(f"\nHigh-quality candidates (score ≥ 0.7):")
    for candidate in high_quality:
        print(f"  {candidate.id}: {candidate.content[:50]}... (score: {candidate.score:.2f})")

    # Step 4: Select specific candidates by ID for further processing
    select_op = SelectById(target_ids=["response_a", "response_d"])
    selected_similar = select_op(packed)

    print(f"\nSelected similar responses for deduplication:")
    for candidate in selected_similar:
        print(f"  {candidate.id}: score={candidate.score:.2f}")
        print(f"    Content: {candidate.content}")

async def demo_caching_and_hashing():
    """Demonstrate content hashing and caching capabilities."""
    print("\n🔄 CACHING AND CONTENT HASHING DEMO")
    print("=" * 50)

    # Show content hashing consistency
    test_data = {"text": "Hello world", "numbers": [1, 2, 3]}
    hash1 = _hash(test_data)
    hash2 = _hash({"numbers": [1, 2, 3], "text": "Hello world"})  # Different order

    print(f"Original data: {test_data}")
    print(f"Hash 1: {hash1}")
    print(f"Hash 2 (reordered): {hash2}")
    print(f"Hashes match: {hash1 == hash2}")

    # Demonstrate caching
    print(f"\nCaching demonstration:")
    pack_op = PackCandidates(enable_cache=True)

    data = ["item1", "item2", "item3"]

    # First call - should compute
    import time
    start = time.time()
    result1 = pack_op(data)
    time1 = time.time() - start

    # Second call - should hit cache
    start = time.time()
    result2 = pack_op(data)
    time2 = time.time() - start

    print(f"First call time: {time1:.6f}s")
    print(f"Second call time: {time2:.6f}s")
    print(f"Results identical: {result1.total_count == result2.total_count}")
    print(f"Cache stats: {pack_op.cache_stats()}")

    # Clear cache and try again
    pack_op.clear_cache()
    print(f"After cache clear: {pack_op.cache_stats()}")

async def demo_non_integration():
    """Demonstrate integration with NoN networks."""
    print("\n🌐 INTEGRATION WITH NoN NETWORKS")
    print("=" * 50)

    # Create a network that uses deterministic operators
    try:
        # Note: These operators are registered with the NoN registry
        network = NoN.from_operators([
            'pack_candidates',    # Structure the data
            'majority',          # Find consensus
            'extract_winners'    # Select top results
        ])

        print("Created NoN network with deterministic operators:")
        print(f"Network layers: {len(network.layers)}")

        # Test data for the network
        test_input = {
            "input_data": [
                "Machine learning enables intelligent automation",
                "AI systems can automate complex decision making",
                "Automated intelligence through machine learning",
                "Machine learning enables intelligent automation"  # Duplicate
            ]
        }

        print(f"\nInput: {len(test_input['input_data'])} responses")

        # Execute the network
        result = await network.forward(test_input)
        print(f"\nNetwork output: {type(result)}")
        print(f"Final result: {result}")

    except Exception as e:
        print(f"Network integration demo skipped: {e}")
        print("(This is expected if running without full NoN environment)")

async def main():
    """Run all deterministic operator demonstrations."""
    print("🔧 DETERMINISTIC OPERATORS COMPREHENSIVE DEMO")
    print("=" * 60)
    print("Demonstrating pure-function operators with content hashing,")
    print("memoization, and ensemble voting patterns.")
    print("=" * 60)

    await demo_basic_deterministic_ops()
    await demo_ensemble_flow_1()
    await demo_ensemble_flow_2()
    await demo_caching_and_hashing()
    await demo_non_integration()

    print("\n✅ ALL DETERMINISTIC OPERATOR DEMOS COMPLETED")
    print("=" * 60)
    print("Key benefits demonstrated:")
    print("  • Content hashing for reproducible results")
    print("  • Caching for performance optimization")
    print("  • Ensemble voting for robust decision making")
    print("  • Pure functions with no side effects")
    print("  • Integration with NoN network architecture")

if __name__ == "__main__":
    asyncio.run(main())