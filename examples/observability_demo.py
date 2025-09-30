#!/usr/bin/env python3
"""
Comprehensive Observability Demo

This demo showcases the full observability stack with tracing, logging,
and metrics collection across NoN execution with database-ready exports.
"""

import asyncio
import sys
import os
import json

# Add the nons package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nons.core.node import Node
from nons.core.network import NoN
from nons.core.types import ModelConfig, ModelProvider
from nons.observability.integration import (
    get_observability, configure_observability,
    trace_network_operation, trace_layer_operation, trace_node_operation
)
from nons.observability.tracing import SpanKind
import nons.operators.base


async def demo_basic_observability():
    """Demonstrate basic observability features."""
    print("🔍 BASIC OBSERVABILITY DEMO")
    print("=" * 60)

    # Configure observability
    obs = configure_observability(
        enable_tracing=True,
        enable_logging=True,
        enable_metrics=True
    )

    # Manual tracing example
    async with obs.trace_operation(
        operation_name="demo_basic_operation",
        kind=SpanKind.NODE,
        component_type="demo",
        component_id="basic_demo"
    ) as span:
        # Add some metadata
        span.add_tags({
            "demo_type": "basic",
            "version": "1.0"
        })

        # Record some metrics
        obs.metrics_collector.increment_counter(
            "demo.operations.started",
            component_type="demo",
            component_id="basic_demo"
        )

        # Simulate work with some logging
        logger = obs.log_manager.get_logger("demo")
        logger.info(
            "Performing basic operation",
            step="initialization",
            component_type="demo",
            component_id="basic_demo"
        )

        await asyncio.sleep(0.1)  # Simulate work

        logger.info(
            "Basic operation completed",
            step="completion",
            component_type="demo",
            component_id="basic_demo"
        )

        # Record completion metrics
        obs.metrics_collector.increment_counter(
            "demo.operations.completed",
            component_type="demo",
            component_id="basic_demo"
        )

    print("✅ Basic observability operation completed")
    print()


async def demo_network_observability():
    """Demonstrate observability with actual NoN execution."""
    print("🌐 NETWORK OBSERVABILITY DEMO")
    print("=" * 60)

    obs = get_observability()

    # Create a simple network
    async with obs.trace_operation(
        operation_name="network_creation",
        kind=SpanKind.NETWORK,
        component_type="network",
        component_id="demo_network"
    ) as span:
        network = NoN.from_operators([
            'generate',
            ['generate', 'condense']
        ])

        span.add_tags({
            "layers": len(network.layers),
            "total_nodes": sum(len(layer.nodes) for layer in network.layers)
        })

    # Execute the network with full observability
    async with obs.trace_operation(
        operation_name="network_execution",
        kind=SpanKind.NETWORK,
        component_type="network",
        component_id=network.network_id
    ) as network_span:
        # Set up node configurations with different providers for testing
        network.layers[0].nodes[0].configure_model(
            provider=ModelProvider.GOOGLE,
            model_name="gemini-2.5-flash",
            max_tokens=50
        )

        network.layers[1].nodes[0].configure_model(
            provider=ModelProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
            max_tokens=50
        )

        network.layers[1].nodes[1].configure_model(
            provider=ModelProvider.GOOGLE,
            model_name="gemini-2.0-flash",
            max_tokens=50
        )

        # Execute with automatic observability
        result = await network.forward("Explain the benefits of renewable energy")

        # Record network-level metrics
        total_cost = sum(node.get_total_cost() for layer in network.layers for node in layer.nodes)
        total_tokens = sum(node.get_total_tokens() for layer in network.layers for node in layer.nodes)

        obs.record_cost_and_tokens(
            network_span,
            token_count=total_tokens,
            cost_usd=total_cost,
            provider="mixed",
            model="multiple"
        )

        network_span.add_tags({
            "result_length": len(str(result)),
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens
        })

    print("✅ Network execution with observability completed")
    print()


async def demo_database_export():
    """Demonstrate database-ready export functionality."""
    print("💾 DATABASE EXPORT DEMO")
    print("=" * 60)

    obs = get_observability()

    # Get all observability data
    all_data = obs.export_all_data()

    print(f"📊 Exported Data Summary:")
    print(f"  Spans: {len(all_data['spans'])}")
    print(f"  Logs: {len(all_data['logs'])}")
    print(f"  Metrics: {len(all_data['metrics'])}")
    print()

    # Show sample span
    if all_data['spans']:
        print("📋 Sample Span (Database-Ready):")
        sample_span = all_data['spans'][0]
        print(json.dumps(sample_span, indent=2, default=str)[:500] + "...")
        print()

    # Show sample log
    if all_data['logs']:
        print("📋 Sample Log (Database-Ready):")
        sample_log = all_data['logs'][0]
        print(json.dumps(sample_log, indent=2, default=str)[:300] + "...")
        print()

    # Show sample metric
    if all_data['metrics']:
        print("📋 Sample Metric (Database-Ready):")
        sample_metric = all_data['metrics'][0]
        print(json.dumps(sample_metric, indent=2, default=str)[:300] + "...")
        print()

    # Get trace summaries
    trace_ids = list(set(span['trace_id'] for span in all_data['spans']))
    print(f"🔗 Found {len(trace_ids)} unique traces")

    for trace_id in trace_ids[:2]:  # Show first 2 traces
        summary = obs.get_trace_summary(trace_id)
        print(f"\n📈 Trace Summary: {trace_id[:8]}...")
        for key, value in summary.items():
            if key != 'trace_id':
                print(f"  {key}: {value}")

    print()


async def demo_performance_metrics():
    """Demonstrate performance and cost metrics collection."""
    print("⚡ PERFORMANCE METRICS DEMO")
    print("=" * 60)

    obs = get_observability()

    # Create multiple nodes with different configurations
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
        ))
    ]

    # Execute nodes multiple times to collect metrics
    for i, node in enumerate(nodes):
        for j in range(3):
            async with obs.trace_operation(
                operation_name="node_execution",
                kind=SpanKind.NODE,
                component_type="node",
                component_id=node.node_id
            ) as span:
                result = await node.execute(f"Test prompt {j+1}")

                # Get metrics from node
                last_metrics = node.get_last_metrics()
                if last_metrics:
                    obs.record_cost_and_tokens(
                        span,
                        token_count=last_metrics.token_usage.total_tokens,
                        cost_usd=last_metrics.cost_info.total_cost_usd,
                        provider=last_metrics.provider,
                        model=node.model_config.model_name
                    )

    # Show metrics summaries
    print("📊 Metrics Summaries:")

    # Get timing metrics
    timing_summary = obs.metrics_collector.get_metric_summary("node_execution.duration")
    if timing_summary:
        print(f"\n⏱️  Execution Times:")
        print(f"  Count: {timing_summary.count}")
        print(f"  Average: {timing_summary.avg_value:.1f}ms")
        print(f"  Min: {timing_summary.min_value:.1f}ms")
        print(f"  Max: {timing_summary.max_value:.1f}ms")

    # Get cost metrics
    cost_summary = obs.metrics_collector.get_metric_summary("llm.cost.total")
    if cost_summary:
        print(f"\n💰 Cost Metrics:")
        print(f"  Count: {cost_summary.count}")
        print(f"  Total: ${cost_summary.sum_value:.6f}")
        print(f"  Average: ${cost_summary.avg_value:.6f}")

    # Get token metrics
    token_summary = obs.metrics_collector.get_metric_summary("llm.tokens.total")
    if token_summary:
        print(f"\n🔢 Token Metrics:")
        print(f"  Count: {token_summary.count}")
        print(f"  Total: {int(token_summary.sum_value):,}")
        print(f"  Average: {int(token_summary.avg_value):,}")

    print()


async def main():
    """Run comprehensive observability demonstration."""
    print("🔍 COMPREHENSIVE OBSERVABILITY DEMONSTRATION")
    print("=" * 70)
    print("Demonstrating tracing, logging, and metrics with database exports!")
    print("=" * 70)
    print()

    await demo_basic_observability()
    await demo_network_observability()
    await demo_database_export()
    await demo_performance_metrics()

    # Final statistics
    obs = get_observability()
    stats = obs.get_stats()

    print("📈 FINAL OBSERVABILITY STATISTICS")
    print("=" * 70)
    print(f"Active Spans: {stats['tracing']['active_spans']}")
    print(f"Completed Spans: {stats['tracing']['completed_spans']}")
    print(f"Total Log Entries: {stats['logging']['total_entries']}")
    print(f"Total Metric Points: {stats['metrics']['total_points']}")
    print(f"Unique Metrics: {stats['metrics']['unique_metrics']}")
    print()

    print("🎉 OBSERVABILITY DEMO COMPLETED!")
    print("=" * 70)
    print("✨ Full tracing, logging, and metrics collection working!")
    print("✨ Database-ready exports for all observability data!")
    print("✨ Automatic correlation across traces, logs, and metrics!")
    print("✨ Production-ready observability infrastructure!")
    print("=" * 70)


if __name__ == "__main__":
    # Set Google API key for real testing
    os.environ['GOOGLE_API_KEY'] = 'AIzaSyB9k5cWxpvia7D6otvBTq8uahiHEaxAhME'
    asyncio.run(main())