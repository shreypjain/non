#!/usr/bin/env python3
"""
Test script for measuring multi-agent planner latency.

Measures:
- Overall execution time
- Individual agent execution times
- Tool call latencies
- Agent delegation overhead
"""

import asyncio
import sys
import os
import time
from typing import Dict, Any, List
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples"))

from multi_agent_planner import (
    create_planner_agent,
    create_code_writer_agent,
    create_file_manager_agent,
    create_research_agent,
    sub_agents,
)


class LatencyMetrics:
    """Track latency metrics for multi-agent execution"""

    def __init__(self):
        self.overall_start = None
        self.overall_end = None
        self.step_times = []
        self.agent_times = defaultdict(list)
        self.tool_times = defaultdict(list)

    def start(self):
        """Start overall timing"""
        self.overall_start = time.time()

    def end(self):
        """End overall timing"""
        self.overall_end = time.time()

    def record_step(self, step_num: int, duration: float):
        """Record step execution time"""
        self.step_times.append((step_num, duration))

    def record_agent(self, agent_name: str, duration: float):
        """Record agent execution time"""
        self.agent_times[agent_name].append(duration)

    def record_tool(self, tool_name: str, duration: float):
        """Record tool execution time"""
        self.tool_times[tool_name].append(duration)

    def get_overall_time(self) -> float:
        """Get total execution time"""
        if self.overall_start and self.overall_end:
            return self.overall_end - self.overall_start
        return 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return {
            "overall_time_seconds": self.get_overall_time(),
            "total_steps": len(self.step_times),
            "avg_step_time": (
                sum(t for _, t in self.step_times) / len(self.step_times)
                if self.step_times
                else 0
            ),
            "agent_times": {
                name: {
                    "total_calls": len(times),
                    "avg_time": sum(times) / len(times) if times else 0,
                    "min_time": min(times) if times else 0,
                    "max_time": max(times) if times else 0,
                }
                for name, times in self.agent_times.items()
            },
            "tool_times": {
                name: {
                    "total_calls": len(times),
                    "avg_time": sum(times) / len(times) if times else 0,
                }
                for name, times in self.tool_times.items()
            },
        }

    def print_report(self):
        """Print a formatted latency report"""
        print("\n" + "=" * 60)
        print("LATENCY REPORT")
        print("=" * 60)

        summary = self.get_summary()

        print(f"\nOverall Execution Time: {summary['overall_time_seconds']:.3f}s")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Average Step Time: {summary['avg_step_time']:.3f}s")

        if summary["agent_times"]:
            print("\nAgent Execution Times:")
            for agent, stats in summary["agent_times"].items():
                print(f"  {agent}:")
                print(f"    Calls: {stats['total_calls']}")
                print(f"    Avg: {stats['avg_time']:.3f}s")
                print(f"    Min: {stats['min_time']:.3f}s")
                print(f"    Max: {stats['max_time']:.3f}s")

        if summary["tool_times"]:
            print("\nTool Execution Times:")
            for tool, stats in summary["tool_times"].items():
                print(f"  {tool}:")
                print(f"    Calls: {stats['total_calls']}")
                print(f"    Avg: {stats['avg_time']:.3f}s")

        print("\n" + "=" * 60)


async def test_multi_agent_with_latency():
    """Test the multi-agent system with latency measurements"""
    metrics = LatencyMetrics()

    print("=== Multi-Agent Planner Latency Test ===\n")

    # Initialize sub-agents
    print("Initializing sub-agents...")
    init_start = time.time()
    sub_agents["code_writer"] = create_code_writer_agent()
    sub_agents["file_manager"] = create_file_manager_agent()
    sub_agents["research"] = create_research_agent()
    init_time = time.time() - init_start
    print(f"Sub-agents initialized in {init_time:.3f}s\n")

    # Create planner
    print("Creating planner agent...")
    planner_start = time.time()
    planner = create_planner_agent()
    planner_time = time.time() - planner_start
    print(f"Planner created in {planner_time:.3f}s\n")

    # Run task with timing
    task = "Create a Python script that reads data from a file and generates a report"
    print(f"Task: {task}\n")
    print("Executing task...\n")

    metrics.start()
    step_count = 0

    try:
        async for result in planner.run(task):
            step_start = time.time()

            if "step" in result:
                step_count = result["step"]
                tool_name = result.get("tool", "N/A")

                print(f"Step {step_count}: {tool_name}")

                # Track tool timing if available
                if "result" in result:
                    step_duration = time.time() - step_start
                    metrics.record_step(step_count, step_duration)
                    metrics.record_tool(tool_name, step_duration)

                    # Check for sub-agent delegation
                    if "sub_agent" in result.get("result", {}):
                        sub_agent_name = result["result"]["sub_agent"]
                        metrics.record_agent(sub_agent_name, step_duration)
                        print(f"  -> Delegated to: {sub_agent_name}")

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback

        traceback.print_exc()

    metrics.end()

    print(f"\nCompleted {step_count} steps")

    # Print latency report
    metrics.print_report()

    return metrics


async def run_multiple_tests(num_runs: int = 3):
    """Run multiple tests and aggregate results"""
    print(f"\n{'=' * 60}")
    print(f"Running {num_runs} test iterations")
    print(f"{'=' * 60}\n")

    all_metrics = []

    for i in range(num_runs):
        print(f"\n--- Test Run {i + 1}/{num_runs} ---")
        metrics = await test_multi_agent_with_latency()
        all_metrics.append(metrics)

        # Clear sub_agents for next run
        sub_agents.clear()

        if i < num_runs - 1:
            print("\nWaiting 2s before next run...")
            await asyncio.sleep(2)

    # Aggregate results
    print(f"\n{'=' * 60}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 60}")

    overall_times = [m.get_overall_time() for m in all_metrics]
    print(f"\nOverall Execution Times:")
    print(f"  Average: {sum(overall_times) / len(overall_times):.3f}s")
    print(f"  Min: {min(overall_times):.3f}s")
    print(f"  Max: {max(overall_times):.3f}s")
    print(f"  Std Dev: {(sum((t - sum(overall_times) / len(overall_times)) ** 2 for t in overall_times) / len(overall_times)) ** 0.5:.3f}s")

    print(f"\n{'=' * 60}\n")


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Test multi-agent planner latency")
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of test runs (default: 1)"
    )
    args = parser.parse_args()

    if args.runs > 1:
        await run_multiple_tests(args.runs)
    else:
        await test_multi_agent_with_latency()


if __name__ == "__main__":
    asyncio.run(main())
