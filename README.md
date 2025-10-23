# Eval AI Library

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive AI Model Evaluation Framework with advanced techniques including **Probability-Weighted Scoring** and **Auto Chain-of-Thought**. Support for multiple LLM providers and 15+ evaluation metrics for RAG systems and AI agents.

## Features

- 🎯 **15+ Evaluation Metrics**: RAG metrics and agent-specific evaluations
- 🧠 **G-Eval Implementation**: State-of-the-art evaluation with probability-weighted scoring
- 🔗 **Chain-of-Thought**: Automatic generation of evaluation steps from criteria
- 🤖 **Multi-Provider Support**: OpenAI, Azure OpenAI, Google Gemini, Anthropic Claude, Ollama
- 📊 **RAG Metrics**: Answer relevancy, faithfulness, contextual precision/recall, and more
- 🔧 **Agent Metrics**: Tool correctness, task success rate, role adherence, knowledge retention
- 🎨 **Custom Metrics**: Advanced custom evaluation with CoT and probability weighting
- 📦 **Data Generation**: Built-in test case generator from documents
- ⚡ **Async Support**: Full async/await support for efficient evaluation
- 💰 **Cost Tracking**: Automatic cost calculation for LLM API calls
- 📝 **Detailed Logging**: Comprehensive evaluation logs for transparency

## Installation
```bash
pip install eval-ai-library
```

### Development Installation
```bash
git clone https://github.com/yourusername/eval-ai-library.git
cd eval-ai-library
pip install -e ".[dev]"
```

## Quick Start

### Basic RAG Evaluation
```python
import asyncio
from eval_lib import (
    evaluate,
    EvalTestCase,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)

async def main():
    # Create test case
    test_case = EvalTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris, a beautiful city known for its art and culture.",
        expected_output="Paris",
        retrieval_context=["Paris is the capital and largest city of France."]
    )
    
    # Define metrics
    metrics = [
        AnswerRelevancyMetric(
            model="gpt-4o-mini",
            threshold=0.7,
            temperature=0.5  # Softmax temperature for score aggregation
        ),
        FaithfulnessMetric(
            model="gpt-4o-mini",
            threshold=0.8,
            temperature=0.5
        )
    ]
    
    # Evaluate
    results = await evaluate(
        test_cases=[test_case],
        metrics=metrics
    )
    
    # Print results with detailed logs
    for _, test_results in results:
        for result in test_results:
            print(f"Success: {result.success}")
            for metric in result.metrics_data:
                print(f"{metric.name}: {metric.score:.2f}")
                print(f"Reason: {metric.reason}")
                print(f"Cost: ${metric.evaluation_cost:.6f}")
                # Access detailed evaluation log
                if hasattr(metric, 'evaluation_log'):
                    print(f"Log: {metric.evaluation_log}")

asyncio.run(main())
```

### G-Eval with Probability-Weighted Scoring

G-Eval implements the state-of-the-art evaluation method from the paper ["G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"](https://arxiv.org/abs/2303.16634). It uses **probability-weighted scoring** (score = Σ p(si) × si) for fine-grained, continuous evaluation scores.
```python
from eval_lib import GEval, EvalTestCase

async def evaluate_with_geval():
    test_case = EvalTestCase(
        input="Explain quantum computing to a 10-year-old",
        actual_output="Quantum computers are like super-powerful regular computers that use special tiny particles to solve really hard problems much faster.",
        expected_output="A simple explanation using analogies suitable for children"
    )
    
    # G-Eval with auto chain-of-thought
    metric = GEval(
        model="gpt-4o",  # Works best with GPT-4
        threshold=0.7,  # Score range: 0-100
        name="Clarity & Simplicity",
        criteria="Evaluate how clear and age-appropriate the explanation is for a 10-year-old child",
        # evaluation_steps is auto-generated from criteria if not provided
        n_samples=20,  # Number of samples for probability estimation (default: 20)
        sampling_temperature=2.0  # High temperature for diverse sampling (default: 2.0)
    )
    
    result = await metric.evaluate(test_case)
    
    print(f"Score: {result['score']:.2f}/100")  # Fine-grained score like 73.45
    print(f"Success: {result['success']}")
    print(f"Reason: {result['reason']}")
    print(f"Sampled scores: {result['metadata']['sampled_scores']}")  # See all 20 samples
    print(f"Score distribution: {result['evaluation_log']['score_distribution']}")

asyncio.run(evaluate_with_geval())
```

### Custom Evaluation with Advanced Features

The CustomEvalMetric now includes **Chain-of-Thought** and **Probability-Weighted Scoring** from G-Eval for maximum accuracy:
```python
from eval_lib import CustomEvalMetric

async def custom_evaluation():
    test_case = EvalTestCase(
        input="How do I reset my password?",
        actual_output="To reset your password, click 'Forgot Password' on the login page, enter your email, and follow the link sent to your inbox.",
        expected_output="Clear step-by-step instructions"
    )
    
    metric = CustomEvalMetric(
        model="gpt-4o",
        threshold=0.7,
        name="HelpfulnessScore",
        criteria="Evaluate if the response provides clear, actionable steps that directly answer the user's question"
        # Auto-generates evaluation steps using CoT
        # Auto-applies probability-weighted scoring (20 samples)
    )
    
    result = await metric.evaluate(test_case)
    
    # Access detailed evaluation log
    log = result['evaluation_log']
    print(f"Auto-generated steps: {log['evaluation_steps']}")
    print(f"Sampled scores: {log['sampled_scores']}")
    print(f"Score distribution: {log['score_distribution']}")
    print(f"Final score: {log['final_score']:.2f}")

asyncio.run(custom_evaluation())
```

### Agent Evaluation
```python
from eval_lib import (
    evaluate,
    EvalTestCase,
    ToolCorrectnessMetric,
    TaskSuccessRateMetric
)

async def evaluate_agent():
    test_case = EvalTestCase(
        input="Book a flight to New York for tomorrow",
        actual_output="I've found available flights and booked your trip to New York for tomorrow.",
        tools_called=["search_flights", "book_flight"],
        expected_tools=["search_flights", "book_flight"]
    )
    
    metrics = [
        ToolCorrectnessMetric(model="gpt-4o-mini", threshold=0.8),
        TaskSuccessRateMetric(
            model="gpt-4o-mini",
            threshold=0.7,
            temperature=1.1  # Controls score aggregation strictness
        )
    ]
    
    results = await evaluate([test_case], metrics)
    return results

asyncio.run(evaluate_agent())
```

### Conversational Evaluation
```python
from eval_lib import (
    evaluate_conversations,
    ConversationalEvalTestCase,
    EvalTestCase,
    RoleAdherenceMetric,
    KnowledgeRetentionMetric
)

async def evaluate_conversation():
    conversation = ConversationalEvalTestCase(
        chatbot_role="You are a helpful customer support assistant. Be professional and empathetic.",
        turns=[
            EvalTestCase(
                input="I need help with my order",
                actual_output="I'd be happy to help you with your order. Could you please provide your order number?"
            ),
            EvalTestCase(
                input="It's #12345",
                actual_output="Thank you! Let me look up order #12345 for you."
            )
        ]
    )
    
    metrics = [
        RoleAdherenceMetric(
            model="gpt-4o-mini",
            threshold=0.8,
            temperature=0.5  # Softmax temperature for verdict aggregation
        ),
        KnowledgeRetentionMetric(
            model="gpt-4o-mini",
            threshold=0.7,
            temperature=0.5
        )
    ]
    
    # Set chatbot role for role adherence
    metrics[0].chatbot_role = conversation.chatbot_role
    
    results = await evaluate_conversations([conversation], metrics)
    
    # Access detailed logs
    for result in results:
        print(f"Dialogue: {result.evaluation_log['dialogue']}")
        print(f"Verdicts: {result.evaluation_log['verdicts']}")
        print(f"Score: {result.score}")
    
    return results

asyncio.run(evaluate_conversation())
```

## Available Metrics

### RAG Metrics

#### AnswerRelevancyMetric
Measures how relevant the answer is to the question using multi-step evaluation:
1. Infers user intent
2. Extracts atomic statements from answer
3. Generates verdicts (fully/mostly/partial/minor/none) for each statement
4. Aggregates using softmax
```python
metric = AnswerRelevancyMetric(
    model="gpt-4o-mini",
    threshold=0.7,
    temperature=0.5  # Controls aggregation strictness
)
```

#### FaithfulnessMetric
Checks if the answer is faithful to the provided context:
1. Extracts factual claims from answer
2. Verifies each claim against context (fully/mostly/partial/minor/none)
3. Aggregates faithfulness score
```python
metric = FaithfulnessMetric(
    model="gpt-4o-mini",
    threshold=0.8,
    temperature=0.5
)
```

#### ContextualRelevancyMetric
Evaluates relevance of retrieved context to the question.
```python
metric = ContextualRelevancyMetric(
    model="gpt-4o-mini",
    threshold=0.7,
    temperature=0.5
)
```

#### ContextualPrecisionMetric
Measures precision of context retrieval - are the retrieved chunks relevant?
```python
metric = ContextualPrecisionMetric(
    model="gpt-4o-mini",
    threshold=0.7
)
```

#### ContextualRecallMetric
Measures recall of relevant context - was all relevant information retrieved?
```python
metric = ContextualRecallMetric(
    model="gpt-4o-mini",
    threshold=0.7
)
```

#### BiasMetric
Detects bias and prejudice in AI-generated output. Score range: 0 (strong bias) to 100 (no bias).
```python
metric = BiasMetric(
    model="gpt-4o-mini",
    threshold=0.7  # Score range: 0-100
)
```

#### ToxicityMetric
Identifies toxic content in responses. Score range: 0 (highly toxic) to 100 (no toxicity).
```python
metric = ToxicityMetric(
    model="gpt-4o-mini",
    threshold=0.7  # Score range: 0-100
)
```

#### RestrictedRefusalMetric
Checks if the AI appropriately refuses harmful or out-of-scope requests.
```python
metric = RestrictedRefusalMetric(
    model="gpt-4o-mini",
    threshold=0.7
)
```

### Agent Metrics

#### ToolCorrectnessMetric
Validates that the agent calls the correct tools in the right sequence.
```python
metric = ToolCorrectnessMetric(
    model="gpt-4o-mini",
    threshold=0.8
)
```

#### TaskSuccessRateMetric
Measures task completion success across conversation:
1. Infers user's goal
2. Generates success criteria
3. Evaluates each criterion (fully/mostly/partial/minor/none)
4. Aggregates into final score
```python
metric = TaskSuccessRateMetric(
    model="gpt-4o-mini",
    threshold=0.7,
    temperature=1.1  # Higher = more lenient aggregation
)
```

#### RoleAdherenceMetric
Evaluates how well the agent maintains its assigned role:
1. Compares each response against role description
2. Generates adherence verdicts (fully/mostly/partial/minor/none)
3. Aggregates across all turns
```python
metric = RoleAdherenceMetric(
    model="gpt-4o-mini",
    threshold=0.8,
    temperature=0.5
)
# Don't forget to set: metric.chatbot_role = "Your role description"
```

#### KnowledgeRetentionMetric
Checks if the agent remembers and recalls information from earlier in the conversation:
1. Analyzes conversation for retention quality
2. Generates retention verdicts (fully/mostly/partial/minor/none)
3. Aggregates into retention score
```python
metric = KnowledgeRetentionMetric(
    model="gpt-4o-mini",
    threshold=0.7,
    temperature=0.5
)
```

### Custom & Advanced Metrics

#### GEval
State-of-the-art evaluation using probability-weighted scoring from the [G-Eval paper](https://arxiv.org/abs/2303.16634):
- **Auto Chain-of-Thought**: Automatically generates evaluation steps from criteria
- **Probability-Weighted Scoring**: score = Σ p(si) × si using 20 samples
- **Fine-Grained Scores**: Continuous scores (e.g., 73.45) instead of integers
```python
metric = GEval(
    model="gpt-4o",  # Best with GPT-4 for probability estimation
    threshold=0.7,
    name="Coherence",
    criteria="Evaluate logical flow and structure of the response",
    evaluation_steps=None,  # Auto-generated if not provided
    n_samples=20,  # Number of samples for probability estimation
    sampling_temperature=2.0  # High temperature for diverse sampling
)
```

#### CustomEvalMetric
Enhanced custom evaluation with CoT and probability-weighted scoring:
```python
metric = CustomEvalMetric(
    model="gpt-4o",
    threshold=0.7,
    name="QualityScore",
    criteria="Your custom evaluation criteria"
    # Automatically uses:
    # - Chain-of-Thought (generates evaluation steps)
    # - Probability-Weighted Scoring (20 samples, temp=2.0)
)
```

## Understanding Evaluation Results

### Score Ranges

- **RAG Metrics** (Answer Relevancy, Faithfulness, etc.): 0.0 - 1.0
- **Safety Metrics** (Bias, Toxicity): 0.0 - 1.0
- **G-Eval & Custom Metrics**: 0.0 - 1.0
- **Agent Metrics** (Task Success, Role Adherence, etc.): 0.0 - 1.0

## Temperature Parameter

Many metrics use a **temperature** parameter for score aggregation (via temperature-weighted scoring):

- **Lower (0.1-0.3)**: **STRICT** - All scores matter equally, low scores heavily penalize the final result. Best for critical applications where even one bad verdict should fail the metric.
- **Medium (0.4-0.8)**: **BALANCED** - Moderate weighting between high and low scores. Default behavior for most use cases (default: 0.5).
- **Higher (1.0-2.0)**: **LENIENT** - High scores (fully/mostly) dominate, effectively ignoring partial/minor/none verdicts. Best for exploratory evaluation or when you want to focus on positive signals.

**How it works:** Temperature controls exponential weighting of scores. Higher temperature exponentially boosts high scores (1.0, 0.9), making low scores (0.7, 0.3, 0.0) matter less. Lower temperature treats all scores more equally.

**Example:**
```python
# Verdicts: [fully, mostly, partial, minor, none] = [1.0, 0.9, 0.7, 0.3, 0.0]

# STRICT: All verdicts count
metric = FaithfulnessMetric(temperature=0.1)  
# Result: ~0.52 (heavily penalized by "minor" and "none")

# BALANCED: Moderate weighting
metric = AnswerRelevancyMetric(temperature=0.5)  
# Result: ~0.73 (balanced consideration)

# LENIENT: Only "fully" and "mostly" matter
metric = TaskSuccessRateMetric(temperature=2.0)  
# Result: ~0.95 (ignores "partial", "minor", "none")
```

## LLM Provider Configuration

### OpenAI
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"

from eval_lib import chat_complete

response, cost = await chat_complete(
    "gpt-4o-mini",  # or "openai:gpt-4o-mini"
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Azure OpenAI
```python
os.environ["AZURE_OPENAI_API_KEY"] = "your-api-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-endpoint.openai.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "your-deployment-name"

response, cost = await chat_complete(
    "azure:gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Google Gemini
```python
os.environ["GOOGLE_API_KEY"] = "your-api-key"

response, cost = await chat_complete(
    "google:gemini-2.0-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Anthropic Claude
```python
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

response, cost = await chat_complete(
    "anthropic:claude-sonnet-4-0",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Ollama (Local)
```python
os.environ["OLLAMA_API_KEY"] = "ollama"  # Can be any value
os.environ["OLLAMA_API_BASE_URL"] = "http://localhost:11434/v1"

response, cost = await chat_complete(
    "ollama:llama2",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Test Data Generation

The library includes a powerful test data generator that can create realistic test cases either from scratch or based on your documents.

### Supported Document Formats

- **Documents**: PDF, DOCX, DOC, TXT, RTF, ODT
- **Structured Data**: CSV, TSV, XLSX, JSON, YAML, XML
- **Web**: HTML, Markdown
- **Presentations**: PPTX
- **Images**: PNG, JPG, JPEG (with OCR support)

### Generate from Scratch
```python
from eval_lib.datagenerator.datagenerator import DatasetGenerator

generator = DatasetGenerator(
    model="gpt-4o-mini",
    agent_description="A customer support chatbot",
    input_format="User question or request",
    expected_output_format="Helpful response",
    test_types=["functionality", "edge_cases"],
    max_rows=20,
    question_length="mixed",  # "short", "long", or "mixed"
    question_openness="mixed",  # "open", "closed", or "mixed"
    trap_density=0.1,  # 10% trap questions
    language="en"
)

dataset = await generator.generate_from_scratch()
```

### Generate from Documents
```python
generator = DatasetGenerator(
    model="gpt-4o-mini",
    agent_description="Technical support agent",
    input_format="Technical question",
    expected_output_format="Detailed answer with references",
    test_types=["retrieval", "accuracy"],
    max_rows=50,
    chunk_size=1024,
    chunk_overlap=100,
    max_chunks=30
)

file_paths = ["docs/user_guide.pdf", "docs/faq.md"]
dataset = await generator.generate_from_documents(file_paths)

# Convert to test cases
from eval_lib import EvalTestCase
test_cases = [
    EvalTestCase(
        input=item["input"],
        expected_output=item["expected_output"],
        retrieval_context=[item.get("context", "")]
    )
    for item in dataset
]
```

## Best Practices

### 1. Choose the Right Model

- **G-Eval**: Use GPT-4 for best results with probability-weighted scoring
- **Other Metrics**: GPT-4o-mini is cost-effective and sufficient
- **Custom Eval**: Use GPT-4 for complex criteria, GPT-4o-mini for simple ones

### 2. Set Appropriate Thresholds
```python
# Safety metrics - high bar
BiasMetric(threshold=80.0)
ToxicityMetric(threshold=85.0)

# Quality metrics - moderate bar
AnswerRelevancyMetric(threshold=0.7)
FaithfulnessMetric(threshold=0.75)

# Agent metrics - context-dependent
TaskSuccessRateMetric(threshold=0.7)  # Most tasks
RoleAdherenceMetric(threshold=0.9)  # Strict role requirements
```

### 3. Use Temperature Wisely
```python
# STRICT evaluation - critical applications where all verdicts matter
# Use when: You need high accuracy and can't tolerate bad verdicts
metric = FaithfulnessMetric(temperature=0.1)

# BALANCED - general use (default)
# Use when: Standard evaluation with moderate requirements
metric = AnswerRelevancyMetric(temperature=0.5)

# LENIENT - exploratory evaluation or focusing on positive signals
# Use when: You want to reward good answers and ignore occasional mistakes
metric = TaskSuccessRateMetric(temperature=2.0)
```

**Real-world examples:**
```python
# Production RAG system - must be accurate
faithfulness = FaithfulnessMetric(
    model="gpt-4o-mini",
    threshold=0.8,
    temperature=0.2  # STRICT: One "none" verdict significantly impacts score
)

# Customer support chatbot - moderate standards
role_adherence = RoleAdherenceMetric(
    model="gpt-4o-mini",
    threshold=0.7,
    temperature=0.5  # BALANCED: Standard evaluation
)

# Experimental feature testing - focus on successes
task_success = TaskSuccessRateMetric(
    model="gpt-4o-mini",
    threshold=0.6,
    temperature=1.5  # LENIENT: Focuses on "fully" and "mostly" completions
)
```

### 4. Leverage Evaluation Logs
```python
result = await metric.evaluate(test_case)

# Always check the log for insights
log = result['evaluation_log']

# For debugging failures:
if not result['success']:
    print(f"Failed because: {log['final_reason']}")
    print(f"Verdicts: {log.get('verdicts', [])}")
    print(f"Steps taken: {log.get('evaluation_steps', [])}")
```

### 5. Batch Evaluation for Efficiency
```python
# Evaluate multiple test cases at once
results = await evaluate(
    test_cases=[test_case1, test_case2, test_case3],
    metrics=[metric1, metric2, metric3]
)

# Calculate aggregate statistics
total_cost = sum(
    metric.evaluation_cost or 0
    for _, test_results in results
    for result in test_results
    for metric in result.metrics_data
)

success_rate = sum(
    1 for _, test_results in results
    for result in test_results
    if result.success
) / len(results)

print(f"Total cost: ${total_cost:.4f}")
print(f"Success rate: {success_rate:.2%}")
```

## Cost Tracking

All evaluations automatically track API costs:
```python
results = await evaluate(test_cases, metrics)

for _, test_results in results:
    for result in test_results:
        for metric in result.metrics_data:
            print(f"{metric.name}: ${metric.evaluation_cost:.6f}")
```

**Cost Estimates** (as of 2025):
- **G-Eval with GPT-4**: ~$0.10-0.15 per evaluation (20 samples)
- **Custom Eval with GPT-4**: ~$0.10-0.15 per evaluation (20 samples + CoT)
- **Standard metrics with GPT-4o-mini**: ~$0.001-0.005 per evaluation
- **Faithfulness/Answer Relevancy**: ~$0.003-0.010 per evaluation (multiple LLM calls)

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | For Azure |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | For Azure |
| `AZURE_OPENAI_DEPLOYMENT` | Azure deployment name | For Azure |
| `GOOGLE_API_KEY` | Google API key | For Google |
| `ANTHROPIC_API_KEY` | Anthropic API key | For Anthropic |
| `OLLAMA_API_KEY` | Ollama API key | For Ollama |
| `OLLAMA_API_BASE_URL` | Ollama base URL | For Ollama |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:
```bibtex
@software{eval_ai_library,
  author = {Meshkov, Aleksandr},
  title = {Eval AI Library: Comprehensive AI Model Evaluation Framework},
  year = {2025},
  url = {https://github.com/meshkovQA/Eval-ai-library.git}
}
```

### References

This library implements techniques from:
```bibtex
@inproceedings{liu2023geval,
  title={G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment},
  author={Liu, Yang and Iter, Dan and Xu, Yichong and Wang, Shuohang and Xu, Ruochen and Zhu, Chenguang},
  booktitle={Proceedings of EMNLP},
  year={2023}
}
```

## Support

- 📧 Email: alekslynx90@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/meshkovQA/Eval-ai-library.git/issues)
- 📖 Documentation: [Full Documentation](https://github.com/meshkovQA/Eval-ai-library.git#readme)

## Acknowledgments

This library was developed to provide a comprehensive solution for evaluating AI models across different use cases and providers, with state-of-the-art techniques including G-Eval's probability-weighted scoring and automatic chain-of-thought generation.