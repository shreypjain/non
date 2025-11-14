#!/usr/bin/env python3
"""
Test multi-agent system with real API calls (Claude/Gemini/OpenAI).
Measures actual latency with real LLM providers.
"""

import asyncio
import sys
import os
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from nons.core.types import ModelConfig, ModelProvider
from nons.core.node import Node
from nons.core.network import create_network
from nons.core.agents.agent import Agent
from nons.core.agents.registry import ToolRegistry
from nons.core.scheduler import start_scheduler, stop_scheduler
from pydantic import BaseModel, Field
import nons.operators.base


class LatencyMetrics:
    """Track latency for real API calls"""

    def __init__(self):
        self.llm_calls = []
        self.tool_calls = []
        self.total_start = None
        self.total_end = None

    def record_llm(self, duration_ms):
        self.llm_calls.append(duration_ms)

    def record_tool(self, duration_ms):
        self.tool_calls.append(duration_ms)

    def start(self):
        self.total_start = time.time()

    def end(self):
        self.total_end = time.time()

    def get_total_time(self):
        if self.total_start and self.total_end:
            return self.total_end - self.total_start
        return 0

    def print_summary(self, provider_name):
        print(f"\n{'=' * 60}")
        print(f"LATENCY SUMMARY - {provider_name}")
        print(f"{'=' * 60}")

        total_time = self.get_total_time()
        print(f"\nTotal Execution Time: {total_time:.3f}s")

        if self.llm_calls:
            print(f"\nLLM Calls: {len(self.llm_calls)}")
            print(f"  Average: {sum(self.llm_calls) / len(self.llm_calls):.1f}ms")
            print(f"  Min: {min(self.llm_calls):.1f}ms")
            print(f"  Max: {max(self.llm_calls):.1f}ms")
            print(f"  Total: {sum(self.llm_calls):.1f}ms")

        if self.tool_calls:
            print(f"\nTool Calls: {len(self.tool_calls)}")
            print(f"  Average: {sum(self.tool_calls) / len(self.tool_calls):.1f}ms")
            print(f"  Total: {sum(self.tool_calls):.1f}ms")

        print(f"\n{'=' * 60}\n")


async def test_simple_agent(provider: str, model: str):
    """Test simple agent with real API"""
    print(f"\n{'=' * 60}")
    print(f"Testing Simple Agent: {provider.upper()} ({model})")
    print(f"{'=' * 60}\n")

    metrics = LatencyMetrics()

    try:
        # Start scheduler
        await start_scheduler()

        # Create simple tool
        registry = ToolRegistry()

        class CalcParams(BaseModel):
            a: float = Field(description="First number")
            b: float = Field(description="Second number")

        class StopParams(BaseModel):
            answer: str = Field(description="Final answer")

        @registry.tool(name="add", description="Add two numbers", param_model=CalcParams)
        async def add_tool(params: CalcParams, **context):
            return {"result": params.a + params.b}

        @registry.tool(name="finish", description="Return final answer", param_model=StopParams, is_stop_tool=True)
        async def finish_tool(params: StopParams, **context):
            return {"answer": params.answer}

        # Create agent
        node = Node("generate")
        network = create_network(
            layers=[node],
            provider=provider,
            model=model,
            system_prompt="""You are a calculator agent. Given a math problem, use the add tool to calculate, then return the final answer with the finish tool.

Return JSON format:
{
    "selected_path": "tool_name",
    "routing_confidence": 0.95,
    "reasoning": "explanation",
    "params": {"a": 5, "b": 3}
}""",
        )

        agent = Agent(network=network, registry=registry, max_steps=5)

        # Run task
        task = "What is 15 + 27?"
        print(f"Task: {task}\n")

        metrics.start()
        step_count = 0

        async for result in agent.run(task):
            step_count += 1
            step_start = time.time()

            print(f"Step {step_count}:")
            print(f"  Tool: {result.get('tool', 'N/A')}")
            print(f"  Success: {result.get('success', False)}")

            step_duration = (time.time() - step_start) * 1000
            metrics.record_tool(step_duration)

            if result.get("is_stop"):
                print(f"  Final Answer: {result.get('result', {})}")
                break

            if step_count >= 5:
                print("  (Max steps reached)")
                break

        metrics.end()

        # Get node stats
        stats = node.get_execution_stats()
        print(f"\nNode Statistics:")
        print(f"  Executions: {stats['execution_count']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Total cost: ${stats['total_cost']:.6f}")

        # Extract LLM latencies from node metrics
        for exec_time in [stats.get('last_execution_time')]:
            if exec_time:
                metrics.record_llm(exec_time * 1000)

        metrics.print_summary(f"{provider}/{model}")

        return True, metrics

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None
    finally:
        await stop_scheduler()


async def test_multi_agent(provider: str, model: str):
    """Test multi-agent planner with real API"""
    print(f"\n{'=' * 60}")
    print(f"Testing Multi-Agent Planner: {provider.upper()} ({model})")
    print(f"{'=' * 60}\n")

    metrics = LatencyMetrics()

    try:
        # Start scheduler
        await start_scheduler()

        # Import multi-agent components
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples"))

        # Create registries
        from nons.core.agents.registry import ToolRegistry

        planner_registry = ToolRegistry()
        code_writer_registry = ToolRegistry()

        # Planner tools
        class PlanParams(BaseModel):
            steps: list = Field(description="Plan steps")

        class DelegateParams(BaseModel):
            agent_name: str = Field(description="Agent to delegate to")
            task: str = Field(description="Task description")

        class StopParams(BaseModel):
            final_result: str = Field(description="Final result")

        @planner_registry.tool(name="create_plan", description="Create a plan", param_model=PlanParams)
        async def create_plan(params: PlanParams, **context):
            return {"plan": params.steps, "total_steps": len(params.steps)}

        @planner_registry.tool(name="complete_task", description="Complete the task", param_model=StopParams, is_stop_tool=True)
        async def complete_task(params: StopParams, **context):
            return {"final_result": params.final_result}

        # Code writer tools
        class CodeParams(BaseModel):
            description: str = Field(description="Code description")

        @code_writer_registry.tool(name="write_code", description="Write code", param_model=CodeParams)
        async def write_code(params: CodeParams, **context):
            return {"code": f"# {params.description}\npass", "status": "generated"}

        @code_writer_registry.tool(name="finish_code", description="Finish coding", param_model=StopParams, is_stop_tool=True)
        async def finish_code(params: StopParams, **context):
            return {"final_result": params.final_result}

        # Create sub-agents
        sub_agents = {}

        code_node = Node("generate")
        code_network = create_network(
            layers=[code_node],
            provider=provider,
            model=model,
            system_prompt="""You are a code writer. Use write_code to generate code, then finish_code when done.

Return JSON: {"selected_path": "tool_name", "routing_confidence": 0.9, "reasoning": "...", "params": {...}}""",
        )
        sub_agents["code_writer"] = Agent(code_network, code_writer_registry, max_steps=3)

        # Planner delegation tool
        @planner_registry.tool(name="delegate_to_code_writer", description="Delegate to code writer", param_model=DelegateParams)
        async def delegate_to_code(params: DelegateParams, **context):
            results = []
            async for result in sub_agents["code_writer"].run(params.task):
                results.append(result)
            return {"sub_agent": "code_writer", "results": results[-1] if results else {}}

        # Create planner
        planner_node = Node("generate")
        planner_network = create_network(
            layers=[planner_node],
            provider=provider,
            model=model,
            system_prompt="""You are a planner. Break down tasks, delegate to code_writer, and complete when done.

Return JSON: {"selected_path": "tool_name", "routing_confidence": 0.9, "reasoning": "...", "params": {...}}""",
        )
        planner = Agent(planner_network, planner_registry, max_steps=5)

        # Run task
        task = "Create a simple Python hello world script"
        print(f"Task: {task}\n")

        metrics.start()
        step_count = 0

        async for result in planner.run(task):
            step_count += 1

            if "step" in result:
                print(f"Step {step_count}:")
                print(f"  Tool: {result.get('tool', 'N/A')}")

                if "sub_agent" in result.get("result", {}):
                    print(f"  -> Delegated to: {result['result']['sub_agent']}")

            if result.get("is_stop"):
                print(f"\n✓ Task completed")
                break

            if step_count >= 5:
                print("\n(Max steps reached)")
                break

        metrics.end()

        # Get node stats
        planner_stats = planner_node.get_execution_stats()
        code_stats = code_node.get_execution_stats()

        print(f"\nPlanner Node Statistics:")
        print(f"  Executions: {planner_stats['execution_count']}")
        print(f"  Total tokens: {planner_stats['total_tokens']}")
        print(f"  Total cost: ${planner_stats['total_cost']:.6f}")

        print(f"\nCode Writer Node Statistics:")
        print(f"  Executions: {code_stats['execution_count']}")
        print(f"  Total tokens: {code_stats['total_tokens']}")
        print(f"  Total cost: ${code_stats['total_cost']:.6f}")

        total_cost = planner_stats['total_cost'] + code_stats['total_cost']
        total_tokens = planner_stats['total_tokens'] + code_stats['total_tokens']

        print(f"\nCombined Statistics:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Total cost: ${total_cost:.6f}")

        metrics.print_summary(f"Multi-Agent {provider}/{model}")

        return True, metrics

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None
    finally:
        await stop_scheduler()


async def main():
    print("\n" + "=" * 60)
    print("REAL API LATENCY TESTING")
    print("=" * 60)

    # Test configurations
    configs = [
        ("anthropic", "claude-sonnet-4-5-20250929"),
        ("google", "gemini-2.0-flash"),
        ("openai", "gpt-4o-mini"),
    ]

    results = {}

    for provider, model in configs:
        print(f"\n{'#' * 60}")
        print(f"# TESTING: {provider.upper()} - {model}")
        print(f"{'#' * 60}")

        # Test simple agent
        success1, metrics1 = await test_simple_agent(provider, model)

        # Test multi-agent
        success2, metrics2 = await test_multi_agent(provider, model)

        results[f"{provider}/{model}"] = {
            "simple_agent": success1,
            "multi_agent": success2,
            "simple_metrics": metrics1,
            "multi_metrics": metrics2,
        }

        # Brief pause between providers
        await asyncio.sleep(2)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for config, result in results.items():
        status1 = "✓ PASS" if result["simple_agent"] else "✗ FAIL"
        status2 = "✓ PASS" if result["multi_agent"] else "✗ FAIL"
        print(f"\n{config}:")
        print(f"  Simple Agent: {status1}")
        print(f"  Multi-Agent: {status2}")

        if result["simple_metrics"]:
            m = result["simple_metrics"]
            print(f"  Simple Latency: {m.get_total_time():.3f}s")

        if result["multi_metrics"]:
            m = result["multi_metrics"]
            print(f"  Multi-Agent Latency: {m.get_total_time():.3f}s")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
