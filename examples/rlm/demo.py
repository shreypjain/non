#!/usr/bin/env python3
"""
RLM Demo - OOLONG-style Benchmark Problems

This demo shows RLM processing long documents with 3 simple benchmark problems:
1. Find specific information in long document
2. Count/aggregate data across sections
3. Multi-hop reasoning (who said what about what)

Key Patterns Demonstrated:
1. Processing 10M+ token documents in chunks
2. Using llm_query() for semantic operations
3. Iterative refinement until high confidence
4. Full execution trace for debugging

Architecture:
- Synthetic long documents (repeated sections)
- RLM Network with Plan-Execute-Verify loop
- Automatic error fixing and refinement
"""

import asyncio
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from examples.rlm.rlm_network import RLMNetwork


def create_earnings_report(num_sections: int = 1000) -> str:
    """
    Create synthetic earnings report with repeated sections.

    This simulates a 10M+ token document by repeating sections.
    Each section has random but consistent data.
    """
    template = """
SECTION {i}: Q{quarter} {year} EARNINGS REPORT

Company: TechCorp Inc.
Quarter: Q{quarter} {year}
CEO: {ceo}

Financial Highlights:
- Revenue: ${revenue}M ({growth}% YoY)
- Operating Income: ${op_income}M
- Net Income: ${net_income}M
- EPS: ${eps}

Key Metrics:
- Active Users: {users}M
- Average Revenue Per User: ${arpu}
- Customer Satisfaction: {satisfaction}%

CEO Statement:
"{ceo_statement}"

Product Updates:
- {product_update}

Risk Factors:
- {risk_factor}

---
"""

    sections = []

    ceos = ["John Smith", "Sarah Johnson", "Michael Chen", "Emily Rodriguez"]
    quarters = ["Q1", "Q2", "Q3", "Q4"]

    for i in range(num_sections):
        year = 2020 + (i // 4)
        quarter = quarters[i % 4]
        ceo = random.choice(ceos)

        section = template.format(
            i=i+1,
            quarter=quarter,
            year=year,
            ceo=ceo,
            revenue=random.randint(100, 500),
            growth=random.randint(-10, 30),
            op_income=random.randint(20, 100),
            net_income=random.randint(15, 80),
            eps=round(random.uniform(0.5, 3.0), 2),
            users=random.randint(10, 100),
            arpu=round(random.uniform(5, 50), 2),
            satisfaction=random.randint(70, 95),
            ceo_statement=random.choice([
                "We delivered strong results this quarter driven by product innovation.",
                "Our focus on operational efficiency led to margin expansion.",
                "Despite headwinds, we maintained growth momentum.",
                "We are investing heavily in AI and cloud infrastructure."
            ]),
            product_update=random.choice([
                "Launched new AI-powered analytics dashboard",
                "Released mobile app version 5.0",
                "Expanded into 3 new international markets",
                "Introduced enterprise tier pricing"
            ]),
            risk_factor=random.choice([
                "Increased competition in core markets",
                "Regulatory changes in key jurisdictions",
                "Supply chain disruptions",
                "Macroeconomic uncertainty"
            ])
        )
        sections.append(section)

    return "\n".join(sections)


async def demo_1_find_specific_info():
    """
    Demo 1: Find specific information in long document.

    Task: What were the Q3 2022 revenues?
    Expected: RLM should chunk document, use llm_query to find Q3 2022 sections,
    extract revenue numbers.
    """
    print("=" * 80)
    print("DEMO 1: Find Specific Information")
    print("=" * 80)
    print()

    # Create document
    print("Creating synthetic earnings report (1000 sections)...")
    document = create_earnings_report(num_sections=1000)
    doc_size = len(document)
    print(f"Document size: {doc_size:,} characters (~{doc_size/4:,.0f} tokens)")
    print()

    # Create RLM network
    network = RLMNetwork(
        max_iterations=5,
        confidence_threshold=0.85,
        max_llm_calls_per_execution=50
    )

    # Run task
    task = "Find all Q3 2022 revenue figures and return the average revenue."
    print(f"Task: {task}")
    print()

    result = await network.run(task, document)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Success: {result.success}")
    print(f"Final Output: {result.final_output}")
    print(f"Total Iterations: {result.total_iterations}")
    print(f"Total LLM Calls: {result.total_llm_calls}")
    print(f"Final Confidence: {result.final_confidence:.2f}")
    print(f"Stop Reason: {result.stop_reason}")
    print()

    # Show iteration trace
    print("ITERATION TRACE:")
    for it in result.iterations:
        print(f"\nIteration {it.iteration}:")
        print(f"  Had Error: {it.had_error}")
        if it.fixed_code:
            print(f"  Used Fixer: Yes")
        print(f"  LLM Calls: {it.execution_result['llm_calls']}")
        print(f"  Confidence: {it.confidence:.2f}")
        print(f"  Reasoning: {it.verifier_reasoning}")
    print()


async def demo_2_count_aggregate():
    """
    Demo 2: Count/aggregate data across sections.

    Task: How many times is "AI" mentioned across all CEO statements?
    Expected: RLM should extract CEO statements, use llm_query for semantic matching,
    count occurrences.
    """
    print("=" * 80)
    print("DEMO 2: Count and Aggregate")
    print("=" * 80)
    print()

    # Create document
    print("Creating synthetic earnings report (500 sections)...")
    document = create_earnings_report(num_sections=500)
    doc_size = len(document)
    print(f"Document size: {doc_size:,} characters (~{doc_size/4:,.0f} tokens)")
    print()

    # Create RLM network
    network = RLMNetwork(
        max_iterations=5,
        confidence_threshold=0.85,
        max_llm_calls_per_execution=50
    )

    # Run task
    task = 'Count how many CEO statements mention "AI" or artificial intelligence topics.'
    print(f"Task: {task}")
    print()

    result = await network.run(task, document)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Success: {result.success}")
    print(f"Final Output: {result.final_output}")
    print(f"Total Iterations: {result.total_iterations}")
    print(f"Total LLM Calls: {result.total_llm_calls}")
    print(f"Final Confidence: {result.final_confidence:.2f}")
    print(f"Stop Reason: {result.stop_reason}")
    print()


async def demo_3_multi_hop():
    """
    Demo 3: Multi-hop reasoning.

    Task: Who was CEO in Q3 2022 and what did they say about product updates?
    Expected: RLM should find Q3 2022 sections, identify CEO, extract their
    statements about products.
    """
    print("=" * 80)
    print("DEMO 3: Multi-hop Reasoning")
    print("=" * 80)
    print()

    # Create document
    print("Creating synthetic earnings report (800 sections)...")
    document = create_earnings_report(num_sections=800)
    doc_size = len(document)
    print(f"Document size: {doc_size:,} characters (~{doc_size/4:,.0f} tokens)")
    print()

    # Create RLM network
    network = RLMNetwork(
        max_iterations=5,
        confidence_threshold=0.85,
        max_llm_calls_per_execution=50
    )

    # Run task
    task = "Who was the CEO in Q3 2022 and what product updates did they announce?"
    print(f"Task: {task}")
    print()

    result = await network.run(task, document)

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Success: {result.success}")
    print(f"Final Output: {result.final_output}")
    print(f"Total Iterations: {result.total_iterations}")
    print(f"Total LLM Calls: {result.total_llm_calls}")
    print(f"Final Confidence: {result.final_confidence:.2f}")
    print(f"Stop Reason: {result.stop_reason}")
    print()


async def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("RLM (Recursive Language Model) Demo")
    print("Processing Long Documents with LLM-Augmented Python REPL")
    print("=" * 80)
    print()

    print("Architecture:")
    print("1. Context stored in Python REPL (not in prompt)")
    print("2. Code uses llm_query() for semantic operations on chunks")
    print("3. Plan → Execute → Verify → Refine loop")
    print("4. Automatic error fixing (single retry)")
    print("5. Confidence-based stopping criteria")
    print()
    print("Each iteration runs inside the Python REPL so planners can reuse intermediate state and llm_query helpers.")
    print("Refine steps explicitly tune what the REPL is doing next (chunking, filtering, aggregation) based on verifier reasoning.")
    print()

    input("Press Enter to start Demo 1...")
    await demo_1_find_specific_info()

    input("\nPress Enter to start Demo 2...")
    await demo_2_count_aggregate()

    input("\nPress Enter to start Demo 3...")
    await demo_3_multi_hop()

    print("\n" + "=" * 80)
    print("All demos complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
