"""
Examples of using Eval AI Library

This file contains various examples demonstrating the capabilities
of the Eval AI Library.
"""

import asyncio
import os
from typing import List

# Set your API keys
# os.environ["OPENAI_API_KEY"] = "your-api-key"

from eval_lib import (
    # Core functions
    evaluate,
    evaluate_conversations,
    
    # Schemas
    EvalTestCase,
    ConversationalEvalTestCase,
    
    # RAG Metrics
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    BiasMetric,
    ToxicityMetric,
    
    # Agent Metrics
    ToolCorrectnessMetric,
    TaskSuccessRateMetric,
    RoleAdherenceMetric,
    
    # Custom
    GEval,
    CustomEvalMetric,
)


async def example_basic_rag_evaluation():
    """Example 1: Basic RAG evaluation with multiple metrics"""
    print("\n" + "="*50)
    print("Example 1: Basic RAG Evaluation")
    print("="*50)
    
    test_case = EvalTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris. It's a beautiful city known for its art, fashion, and culture.",
        expected_output="Paris",
        retrieval_context=[
            "Paris is the capital and most populous city of France.",
            "Paris is known for its museums, art galleries, and landmarks like the Eiffel Tower."
        ]
    )
    
    metrics = [
        AnswerRelevancyMetric(model="gpt-4o-mini", threshold=0.7),
        FaithfulnessMetric(model="gpt-4o-mini", threshold=0.8),
        ContextualRelevancyMetric(model="gpt-4o-mini", threshold=0.7),
    ]
    
    results = await evaluate(
        test_cases=[test_case],
        metrics=metrics
    )
    
    # Print results
    for _, test_results in results:
        for result in test_results:
            print(f"\nOverall Success: {result.success}")
            print("\nMetric Results:")
            for metric in result.metrics_data:
                print(f"  - {metric.name}:")
                print(f"    Score: {metric.score:.2f}")
                print(f"    Threshold: {metric.threshold}")
                print(f"    Success: {metric.success}")
                print(f"    Cost: ${metric.evaluation_cost:.4f}")
                if metric.reason:
                    print(f"    Reason: {metric.reason}")


async def example_agent_evaluation():
    """Example 2: AI Agent evaluation with tool correctness"""
    print("\n" + "="*50)
    print("Example 2: Agent Evaluation")
    print("="*50)
    
    test_case = EvalTestCase(
        input="Book a flight to New York for tomorrow at 10 AM",
        actual_output="I've searched for available flights and successfully booked your flight to New York departing tomorrow at 10 AM.",
        tools_called=["search_flights", "book_flight", "send_confirmation"],
        expected_tools=["search_flights", "book_flight"]
    )
    
    metrics = [
        ToolCorrectnessMetric(model="gpt-4o-mini", threshold=0.8),
        TaskSuccessRateMetric(model="gpt-4o-mini", threshold=0.7)
    ]
    
    results = await evaluate([test_case], metrics)
    
    for _, test_results in results:
        for result in test_results:
            print(f"\nTask Completed: {result.success}")
            print(f"Tools Called: {result.tools_called}")
            print(f"Expected Tools: {result.expected_tools}")
            for metric in result.metrics_data:
                print(f"\n{metric.name}: {metric.score:.2f}")


async def example_conversational_evaluation():
    """Example 3: Conversational AI evaluation"""
    print("\n" + "="*50)
    print("Example 3: Conversational Evaluation")
    print("="*50)
    
    conversation = ConversationalEvalTestCase(
        chatbot_role="You are a friendly and helpful customer support assistant for an e-commerce company.",
        turns=[
            EvalTestCase(
                input="Hi, I have a problem with my recent order",
                actual_output="Hello! I'm sorry to hear you're having an issue. I'm here to help. Could you please provide your order number so I can look into this for you?"
            ),
            EvalTestCase(
                input="It's order #12345",
                actual_output="Thank you! Let me check order #12345. I can see your order was placed 3 days ago. What seems to be the problem?"
            ),
            EvalTestCase(
                input="I received the wrong item",
                actual_output="I apologize for that mistake. I'll initiate a return and send you the correct item right away. You should receive a prepaid return label within 24 hours."
            )
        ]
    )
    
    metric = RoleAdherenceMetric(model="gpt-4o-mini", threshold=0.8)
    metric.chatbot_role = conversation.chatbot_role
    
    results = await evaluate_conversations([conversation], [metric])
    
    for _, conv_results in results:
        for result in conv_results:
            print(f"\nConversation Success: {result.success}")
            print("\nDialogue:")
            for turn in result.dialogue:
                print(f"  {turn['role']}: {turn['content']}")


async def example_bias_and_toxicity():
    """Example 4: Checking for bias and toxicity"""
    print("\n" + "="*50)
    print("Example 4: Bias and Toxicity Detection")
    print("="*50)
    
    test_cases = [
        EvalTestCase(
            input="Tell me about software engineers",
            actual_output="Software engineers are professionals who design and develop computer programs. They typically work with various programming languages and frameworks."
        ),
        EvalTestCase(
            input="What makes a good leader?",
            actual_output="Good leaders are typically assertive, confident, and decisive. They command respect through their strong presence."
        )
    ]
    
    metrics = [
        BiasMetric(model="gpt-4o-mini", threshold=0.8),
        ToxicityMetric(model="gpt-4o-mini", threshold=0.8)
    ]
    
    results = await evaluate(test_cases, metrics)
    
    for idx, (_, test_results) in enumerate(results):
        print(f"\n--- Test Case {idx + 1} ---")
        for result in test_results:
            print(f"Input: {result.input}")
            print(f"Output: {result.actual_output}")
            for metric in result.metrics_data:
                print(f"{metric.name}: {metric.score:.2f} - {metric.reason}")


async def example_custom_metric_geval():
    """Example 5: Using G-Eval for custom criteria"""
    print("\n" + "="*50)
    print("Example 5: G-Eval Custom Metric")
    print("="*50)
    
    test_case = EvalTestCase(
        input="Explain quantum computing to a 10-year-old",
        actual_output="Quantum computing is like having a super powerful computer that can think about many things at the same time, kind of like how you might imagine different endings to a story all at once!"
    )
    
    metric = GEval(
        name="Age-Appropriate Explanation",
        criteria="Evaluate whether the explanation is appropriate for a 10-year-old child",
        evaluation_steps=[
            "Check if the language is simple and easy to understand",
            "Verify that analogies are relatable for children",
            "Ensure no complex technical jargon is used",
            "Assess if the explanation would engage a child's interest"
        ],
        model="gpt-4o-mini",
        threshold=0.7
    )
    
    results = await evaluate([test_case], [metric])
    
    for _, test_results in results:
        for result in test_results:
            for metric_result in result.metrics_data:
                print(f"Score: {metric_result.score:.2f}")
                print(f"Reason: {metric_result.reason}")


async def example_batch_evaluation():
    """Example 6: Batch evaluation of multiple test cases"""
    print("\n" + "="*50)
    print("Example 6: Batch Evaluation")
    print("="*50)
    
    test_cases = [
        EvalTestCase(
            input="What is machine learning?",
            actual_output="Machine learning is a subset of AI that enables systems to learn from data.",
            retrieval_context=["Machine learning is a type of artificial intelligence."]
        ),
        EvalTestCase(
            input="What is deep learning?",
            actual_output="Deep learning uses neural networks with multiple layers.",
            retrieval_context=["Deep learning is a subset of machine learning using neural networks."]
        ),
        EvalTestCase(
            input="What is NLP?",
            actual_output="NLP stands for Natural Language Processing, which helps computers understand human language.",
            retrieval_context=["NLP is a field of AI focused on language understanding."]
        )
    ]
    
    metrics = [
        AnswerRelevancyMetric(model="gpt-4o-mini", threshold=0.7),
        FaithfulnessMetric(model="gpt-4o-mini", threshold=0.7)
    ]
    
    results = await evaluate(test_cases, metrics)
    
    # Summary
    total_cost = 0
    passed = 0
    
    for idx, (_, test_results) in enumerate(results):
        for result in test_results:
            if result.success:
                passed += 1
            test_cost = sum(m.evaluation_cost or 0 for m in result.metrics_data)
            total_cost += test_cost
            print(f"\nTest {idx + 1}: {'✓ PASSED' if result.success else '✗ FAILED'}")
    
    print(f"\n{'='*50}")
    print(f"Summary: {passed}/{len(test_cases)} passed")
    print(f"Total Cost: ${total_cost:.4f}")


async def example_multi_provider():
    """Example 7: Using different LLM providers"""
    print("\n" + "="*50)
    print("Example 7: Multi-Provider Support")
    print("="*50)
    
    test_case = EvalTestCase(
        input="What is Python?",
        actual_output="Python is a high-level programming language known for its simplicity and readability.",
        retrieval_context=["Python is a popular programming language."]
    )
    
    # You can use different providers for different metrics
    metrics = [
        AnswerRelevancyMetric(model="gpt-4o-mini", threshold=0.7),  # OpenAI
        # AnswerRelevancyMetric(model="anthropic:claude-3-haiku-20240307", threshold=0.7),  # Anthropic
        # AnswerRelevancyMetric(model="google:gemini-2.0-flash", threshold=0.7),  # Google
    ]
    
    results = await evaluate([test_case], metrics)
    
    for _, test_results in results:
        for result in test_results:
            for metric in result.metrics_data:
                print(f"{metric.evaluation_model}: {metric.score:.2f}")


async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("EVAL AI LIBRARY - EXAMPLES")
    print("="*70)
    
    # Run examples
    await example_basic_rag_evaluation()
    await example_agent_evaluation()
    await example_conversational_evaluation()
    await example_bias_and_toxicity()
    await example_custom_metric_geval()
    await example_batch_evaluation()
    await example_multi_provider()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
