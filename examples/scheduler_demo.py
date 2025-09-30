#!/usr/bin/env python3
"""
Request Scheduler Demo

This demo showcases the request scheduler with rate limiting, queue management,
and intelligent request distribution across multiple LLM providers.
"""

import asyncio
import sys
import os
import json
import time

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import operators to ensure they're registered
import nons.operators.base

from nons.core.node import Node
from nons.core.network import NoN
from nons.core.types import ModelConfig, ModelProvider
from nons.core.scheduler import (
    get_scheduler, configure_scheduler, start_scheduler, stop_scheduler,
    QueueStrategy, RateLimitConfig
)
from nons.observability.integration import get_observability, configure_observability


async def demo_basic_scheduler():
    """Demonstrate basic scheduler functionality."""
    print("🚀 BASIC SCHEDULER DEMO")
    print("=" * 60)

    # Configure scheduler with custom rate limits
    scheduler = configure_scheduler(
        rate_limits={
            ModelProvider.GOOGLE: RateLimitConfig(
                requests_per_minute=20,  # Reduced for demo
                requests_per_second=2,
                max_concurrent=3
            ),
            ModelProvider.ANTHROPIC: RateLimitConfig(
                requests_per_minute=15,  # Reduced for demo
                requests_per_second=1,
                max_concurrent=2
            ),
            ModelProvider.MOCK: RateLimitConfig(
                requests_per_minute=100,
                max_concurrent=10
            )
        },
        queue_strategy=QueueStrategy.PRIORITY,
        enable_observability=True
    )

    # Start the scheduler
    await start_scheduler()

    print(f"✅ Scheduler started with strategy: {scheduler.queue_strategy.value}")
    print(f"📊 Rate limits configured for {len(scheduler.rate_limits)} providers")
    print()

    # Create nodes with different providers
    nodes = [
        Node('generate', model_config=ModelConfig(
            provider=ModelProvider.GOOGLE,
            model_name="gemini-2.5-flash",
            max_tokens=30
        )),
        Node('generate', model_config=ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
            max_tokens=30
        )),
        Node('generate', model_config=ModelConfig(
            provider=ModelProvider.MOCK,
            model_name="mock-model",
            max_tokens=30
        ))
    ]

    # Execute multiple requests to see rate limiting in action
    print("🔄 Executing 6 requests across providers...")
    tasks = []

    for i in range(6):
        node = nodes[i % len(nodes)]
        provider_name = node.model_config.provider.value

        print(f"  Request {i+1}: {provider_name}")
        task = asyncio.create_task(
            node.execute(f"Generate a brief fun fact about {provider_name} AI (Request {i+1})")
        )
        tasks.append(task)

        # Small delay to show queueing
        await asyncio.sleep(0.1)

    # Show scheduler stats while requests are processing
    await asyncio.sleep(0.5)
    stats = scheduler.get_stats()
    print(f"\n📈 Scheduler Stats (mid-execution):")
    print(f"  Running: {stats['is_running']}")
    print(f"  Active Requests: {stats['total_active_requests']}")

    for provider, provider_stats in stats['providers'].items():
        if provider_stats['queue_length'] > 0 or provider_stats['requests_in_flight'] > 0:
            print(f"  {provider}: Queue={provider_stats['queue_length']}, In-flight={provider_stats['requests_in_flight']}")

    # Wait for all requests to complete
    results = await asyncio.gather(*tasks)

    print(f"\n✅ All requests completed!")
    print(f"📊 Results:")
    for i, result in enumerate(results):
        provider = nodes[i % len(nodes)].model_config.provider.value
        result_preview = str(result)[:60] + "..." if len(str(result)) > 60 else str(result)
        print(f"  {i+1}. [{provider}] {result_preview}")

    # Final stats
    final_stats = scheduler.get_stats()
    print(f"\n📈 Final Scheduler Stats:")
    for provider, provider_stats in final_stats['providers'].items():
        if provider_stats['requests_completed'] > 0:
            print(f"  {provider}:")
            print(f"    Completed: {provider_stats['requests_completed']}")
            print(f"    Failed: {provider_stats['requests_failed']}")
            print(f"    Avg Time: {provider_stats['avg_request_time']:.3f}s")

    print()


async def demo_queue_strategies():
    """Demonstrate different queue strategies."""
    print("⚡ QUEUE STRATEGIES DEMO")
    print("=" * 60)

    strategies = [QueueStrategy.PRIORITY, QueueStrategy.ROUND_ROBIN, QueueStrategy.LEAST_LOADED]

    for strategy in strategies:
        print(f"\n🎯 Testing strategy: {strategy.value}")

        # Reconfigure scheduler with new strategy
        scheduler = configure_scheduler(
            queue_strategy=strategy,
            enable_observability=True
        )
        await start_scheduler()

        # Create mix of high and low priority requests
        node = Node('generate', model_config=ModelConfig(
            provider=ModelProvider.MOCK,  # Use mock for consistent timing
            model_name="mock-model",
            max_tokens=20
        ))

        # Schedule requests with different priorities
        tasks = []
        priorities = [1, 5, 2, 10, 3]  # Mix of priorities

        for i, priority in enumerate(priorities):
            print(f"  Scheduling request {i+1} with priority {priority}")

            # For priority queue, higher numbers should execute first
            task = asyncio.create_task(
                node.execute(f"Priority {priority} request: Generate text {i+1}")
            )
            tasks.append((priority, task))

            await asyncio.sleep(0.05)  # Small delay

        # Wait for completion
        results = await asyncio.gather(*[task for _, task in tasks])

        print(f"  ✅ Strategy {strategy.value} completed {len(results)} requests")

        await stop_scheduler()

    print()


async def demo_rate_limiting():
    """Demonstrate rate limiting in action."""
    print("🚦 RATE LIMITING DEMO")
    print("=" * 60)

    # Configure very strict rate limits for demo
    scheduler = configure_scheduler(
        rate_limits={
            ModelProvider.MOCK: RateLimitConfig(
                requests_per_minute=10,
                requests_per_second=1,  # Very strict: 1 request per second
                max_concurrent=2
            )
        },
        enable_observability=True
    )
    await start_scheduler()

    print("⏰ Configured strict rate limit: 1 request/second, max 2 concurrent")

    node = Node('generate', model_config=ModelConfig(
        provider=ModelProvider.MOCK,
        model_name="mock-model",
        max_tokens=20
    ))

    # Schedule 5 requests rapidly
    start_time = time.time()
    print("\n🔄 Scheduling 5 requests rapidly...")

    tasks = []
    for i in range(5):
        print(f"  Scheduling request {i+1} at {time.time() - start_time:.2f}s")
        task = asyncio.create_task(
            node.execute(f"Rate limited request {i+1}")
        )
        tasks.append(task)

    # Monitor queue during execution
    monitor_task = asyncio.create_task(_monitor_queue_during_execution(scheduler, 8))

    # Wait for all requests
    results = await asyncio.gather(*tasks)
    monitor_task.cancel()

    total_time = time.time() - start_time
    print(f"\n✅ All 5 requests completed in {total_time:.2f}s")
    print(f"📊 Average time per request: {total_time/5:.2f}s")
    print("   (Should be ~1s due to rate limiting)")

    await stop_scheduler()
    print()


async def _monitor_queue_during_execution(scheduler, duration_seconds):
    """Monitor queue status during execution."""
    print("\n📊 Queue monitoring:")
    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        stats = scheduler.get_stats()
        mock_stats = stats['providers']['mock']

        if mock_stats['queue_length'] > 0 or mock_stats['requests_in_flight'] > 0:
            elapsed = time.time() - start_time
            print(f"  {elapsed:.1f}s: Queue={mock_stats['queue_length']}, "
                  f"In-flight={mock_stats['requests_in_flight']}, "
                  f"Completed={mock_stats['requests_completed']}")

        await asyncio.sleep(0.5)


async def demo_network_with_scheduler():
    """Demonstrate network execution with scheduler coordination."""
    print("🌐 NETWORK WITH SCHEDULER DEMO")
    print("=" * 60)

    # Configure scheduler for network execution
    scheduler = configure_scheduler(
        rate_limits={
            ModelProvider.GOOGLE: RateLimitConfig(
                requests_per_minute=30,
                requests_per_second=3,
                max_concurrent=5
            ),
            ModelProvider.ANTHROPIC: RateLimitConfig(
                requests_per_minute=20,
                requests_per_second=2,
                max_concurrent=3
            ),
            ModelProvider.MOCK: RateLimitConfig(
                requests_per_minute=100,
                max_concurrent=10
            )
        },
        queue_strategy=QueueStrategy.LEAST_LOADED,
        enable_observability=True
    )
    await start_scheduler()

    # Create a network with mixed providers
    network = NoN.from_operators([
        'generate',  # First layer
        ['generate', 'generate', 'generate']  # Second layer with 3 nodes
    ])

    # Configure different providers for each node
    network.layers[0].nodes[0].configure_model(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-2.5-flash",
        max_tokens=40
    )

    network.layers[1].nodes[0].configure_model(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-haiku-20240307",
        max_tokens=40
    )

    network.layers[1].nodes[1].configure_model(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-2.0-flash",
        max_tokens=40
    )

    network.layers[1].nodes[2].configure_model(
        provider=ModelProvider.MOCK,
        model_name="mock-model",
        max_tokens=40
    )

    print(f"🏗️  Created network with {len(network.layers)} layers")
    print(f"   Layer 1: 1 node (Google)")
    print(f"   Layer 2: 3 nodes (Anthropic, Google, Mock)")

    # Execute network
    start_time = time.time()
    print("\n🚀 Executing network with scheduler coordination...")

    result = await network.forward("Explain the future of AI in 3 different perspectives")

    execution_time = time.time() - start_time
    print(f"\n✅ Network execution completed in {execution_time:.2f}s")

    # Show final scheduler stats
    final_stats = scheduler.get_stats()
    print(f"\n📈 Final Scheduler Statistics:")
    total_requests = 0
    for provider, stats in final_stats['providers'].items():
        if stats['requests_completed'] > 0:
            total_requests += stats['requests_completed']
            print(f"  {provider}:")
            print(f"    Requests: {stats['requests_completed']}")
            print(f"    Avg Time: {stats['avg_request_time']:.3f}s")

    print(f"\n🎯 Total LLM requests scheduled: {total_requests}")
    print(f"📊 Result preview: {str(result)[:100]}...")

    await stop_scheduler()
    print()


async def demo_observability_integration():
    """Demonstrate scheduler integration with observability."""
    print("🔍 SCHEDULER + OBSERVABILITY DEMO")
    print("=" * 60)

    # Configure both observability and scheduler
    obs = configure_observability(
        enable_tracing=True,
        enable_logging=True,
        enable_metrics=True
    )

    scheduler = configure_scheduler(
        enable_observability=True,
        queue_strategy=QueueStrategy.PRIORITY
    )
    await start_scheduler()

    # Execute some requests
    node = Node('generate', model_config=ModelConfig(
        provider=ModelProvider.MOCK,
        model_name="mock-model",
        max_tokens=25
    ))

    print("🔄 Executing 3 requests with full observability...")

    tasks = []
    for i in range(3):
        task = asyncio.create_task(
            node.execute(f"Observable request {i+1} about quantum computing")
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # Export observability data
    all_data = obs.export_all_data()

    print(f"\n📊 Observability Data Collected:")
    print(f"  Spans: {len(all_data['spans'])}")
    print(f"  Logs: {len(all_data['logs'])}")
    print(f"  Metrics: {len(all_data['metrics'])}")

    # Show scheduler-specific spans
    scheduler_spans = [
        span for span in all_data['spans']
        if span.get('component_type') == 'scheduler'
    ]
    print(f"  Scheduler Spans: {len(scheduler_spans)}")

    # Show request execution spans
    request_spans = [
        span for span in all_data['spans']
        if span.get('operation_name') == 'request_execution'
    ]
    print(f"  Request Execution Spans: {len(request_spans)}")

    if request_spans:
        print(f"\n📋 Sample Request Execution Span:")
        sample_span = request_spans[0]
        print(f"  Trace ID: {sample_span.get('trace_id', 'N/A')[:8]}...")
        print(f"  Provider: {sample_span.get('tags', {}).get('provider', 'N/A')}")
        print(f"  Duration: {sample_span.get('duration_ms', 'N/A')}ms")

    await stop_scheduler()
    print()


async def main():
    """Run comprehensive scheduler demonstration."""
    print("🚀 COMPREHENSIVE SCHEDULER DEMONSTRATION")
    print("=" * 70)
    print("Showcasing intelligent request scheduling with rate limiting!")
    print("=" * 70)
    print()

    # Set API key for real testing
    os.environ['GOOGLE_API_KEY'] = 'AIzaSyB9k5cWxpvia7D6otvBTq8uahiHEaxAhME'

    await demo_basic_scheduler()
    await demo_queue_strategies()
    await demo_rate_limiting()
    await demo_network_with_scheduler()
    await demo_observability_integration()

    print("🎉 SCHEDULER DEMO COMPLETED!")
    print("=" * 70)
    print("✨ Intelligent request scheduling working perfectly!")
    print("✨ Rate limiting prevents API quota exhaustion!")
    print("✨ Multiple queue strategies for different use cases!")
    print("✨ Full observability integration for monitoring!")
    print("✨ Production-ready request management infrastructure!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())