# Eval AI Library

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive AI Model Evaluation Framework with support for multiple LLM providers and a wide range of evaluation metrics for RAG systems and AI agents.

## Features

- 🎯 **Multiple Evaluation Metrics**: 15+ built-in metrics for RAG and AI agents
- 🤖 **Multi-Provider Support**: OpenAI, Azure OpenAI, Google Gemini, Anthropic Claude, Ollama
- 📊 **RAG Metrics**: Answer relevancy, faithfulness, contextual precision/recall, and more
- 🔧 **Agent Metrics**: Tool correctness, task success rate, role adherence, knowledge retention
- 🎨 **Custom Metrics**: Easy-to-extend framework for creating custom evaluation metrics
- 📦 **Data Generation**: Built-in test case generator from documents
- ⚡ **Async Support**: Full async/await support for efficient evaluation
- 💰 **Cost Tracking**: Automatic cost calculation for LLM API calls

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
        AnswerRelevancyMetric(model="gpt-4o-mini", threshold=0.7),
        FaithfulnessMetric(model="gpt-4o-mini", threshold=0.8)
    ]
    
    # Evaluate
    results = await evaluate(
        test_cases=[test_case],
        metrics=metrics
    )
    
    # Print results
    for _, test_results in results:
        for result in test_results:
            print(f"Success: {result.success}")
            for metric in result.metrics_data:
                print(f"{metric.name}: {metric.score:.2f} (threshold: {metric.threshold})")

asyncio.run(main())
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
        TaskSuccessRateMetric(model="gpt-4o-mini", threshold=0.7)
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
    RoleAdherenceMetric
)

async def evaluate_conversation():
    conversation = ConversationalEvalTestCase(
        chatbot_role="You are a helpful customer support assistant.",
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
    
    metric = RoleAdherenceMetric(model="gpt-4o-mini", threshold=0.8)
    metric.chatbot_role = conversation.chatbot_role
    
    results = await evaluate_conversations([conversation], [metric])
    return results

asyncio.run(evaluate_conversation())
```

## Available Metrics

### RAG Metrics

- **AnswerRelevancyMetric**: Measures how relevant the answer is to the question
- **AnswerPrecisionMetric**: Evaluates precision of the answer
- **FaithfulnessMetric**: Checks if the answer is faithful to the context
- **ContextualRelevancyMetric**: Measures relevance of retrieved context
- **ContextualPrecisionMetric**: Evaluates precision of context retrieval
- **ContextualRecallMetric**: Measures recall of relevant context
- **BiasMetric**: Detects bias in responses
- **ToxicityMetric**: Identifies toxic content
- **RestrictedRefusalMetric**: Checks appropriate refusals

### Agent Metrics

- **ToolCorrectnessMetric**: Validates correct tool usage
- **TaskSuccessRateMetric**: Measures task completion success
- **RoleAdherenceMetric**: Evaluates adherence to assigned role
- **KnowledgeRetentionMetric**: Checks information retention across conversation

### Custom Metrics

```python
from eval_lib import CustomEvalMetric

metric = CustomEvalMetric(
    name="CustomQuality",
    evaluation_params=["clarity", "completeness", "accuracy"],
    model="gpt-4o-mini",
    threshold=0.7
)
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

The library includes a powerful test data generator that can create realistic test cases either from scratch or based on your documents. This is perfect for quickly building comprehensive test suites for your AI agents.

### Supported Document Formats

The DocumentLoader supports a wide range of file formats:
- **Documents**: PDF, DOCX, DOC, TXT, RTF, ODT
- **Structured Data**: CSV, TSV, XLSX, JSON, YAML, XML
- **Web**: HTML, Markdown
- **Presentations**: PPTX
- **Images**: PNG, JPG, JPEG (with OCR support)

### Method 1: Generate from Scratch

Create test cases without any reference documents - useful for testing general capabilities:
```python
from eval_lib.datagenerator.datagenerator import DatasetGenerator

generator = DatasetGenerator(
    model="gpt-4o-mini",
    agent_description="A customer support chatbot that helps users with product inquiries",
    input_format="User question or request",
    expected_output_format="Helpful and concise response",
    test_types=["functionality", "edge_cases", "error_handling"],
    max_rows=20,
    question_length="mixed",  # "short", "long", or "mixed"
    question_openness="mixed",  # "open", "closed", or "mixed"
    trap_density=0.1,  # 10% trap questions (0.0 to 1.0)
    language="en"
)

# Generate test cases
dataset = await generator.generate_from_scratch()

# Convert to EvalTestCase format
from eval_lib import EvalTestCase
test_cases = [
    EvalTestCase(
        input=item["input"],
        expected_output=item["expected_output"]
    )
    for item in dataset
]
```

### Method 2: Generate from Documents

Generate test cases based on your documentation, knowledge base, or any text documents:
```python
from eval_lib.datagenerator.datagenerator import DatasetGenerator

generator = DatasetGenerator(
    model="gpt-4o-mini",
    agent_description="A technical support agent with access to product documentation",
    input_format="Technical question from user",
    expected_output_format="Detailed answer with references to documentation",
    test_types=["retrieval", "accuracy", "completeness"],
    
    # Generation parameters
    max_rows=50,
    question_length="mixed",
    question_openness="open",
    trap_density=0.15,
    language="en",
    
    # Document processing parameters
    chunk_size=1024,  # Size of text chunks
    chunk_overlap=100,  # Overlap between chunks
    max_chunks=30,  # Maximum chunks to process
    relevance_margin=1.5,  # Multiplier for chunk selection
    
    # Advanced parameters
    temperature=0.3,  # LLM temperature for generation
    embedding_model="openai:text-embedding-3-small"  # For chunk ranking
)

# Generate from your documents
file_paths = [
    "docs/user_guide.pdf",
    "docs/api_reference.docx",
    "docs/faq.md"
]

dataset = await generator.generate_from_documents(file_paths)

# Convert to test cases
test_cases = [
    EvalTestCase(
        input=item["input"],
        expected_output=item["expected_output"],
        retrieval_context=[item.get("context", "")]
    )
    for item in dataset
]
```

### Generator Parameters Explained

#### Required Parameters:
- **model** (`str`): LLM model for generation (e.g., "gpt-4o-mini", "claude-sonnet-4-0")
- **agent_description** (`str`): Description of what your agent does
- **input_format** (`str`): Expected format of user inputs
- **expected_output_format** (`str`): Expected format of agent outputs
- **test_types** (`List[str]`): Types of tests to generate (e.g., ["accuracy", "edge_cases"])

#### Generation Control:
- **max_rows** (`int`, default=10): Maximum number of test cases to generate
- **question_length** (`str`, default="mixed"): 
  - `"short"`: Concise, direct questions (1-2 sentences)
  - `"long"`: Detailed scenarios with context (3+ sentences)
  - `"mixed"`: Variety of lengths
  
- **question_openness** (`str`, default="mixed"):
  - `"open"`: Open-ended questions requiring explanations
  - `"closed"`: Specific questions with definitive answers
  - `"mixed"`: Mix of both types
  
- **trap_density** (`float`, default=0.1): Proportion of "trap" questions (0.0-1.0)
  - Trap questions test if agent properly handles missing information or corrections
  - 0.0 = no traps, 1.0 = all traps
  
- **language** (`str`, default="en"): Language for generated test cases
- **temperature** (`float`, default=0.3): LLM temperature for generation

#### Document Processing (for generate_from_documents):
- **chunk_size** (`int`, default=1024): Size of document chunks in characters
- **chunk_overlap** (`int`, default=100): Overlap between consecutive chunks
- **max_chunks** (`int`, default=30): Maximum number of chunks to process
- **relevance_margin** (`float`, default=1.5): Multiplier for selecting relevant chunks
- **embedding_model** (`str`): Model for ranking chunk relevance

### Advanced Usage Examples

#### Example 1: High-Quality Open Questions from Documents
```python
generator = DatasetGenerator(
    model="gpt-4o",  # Better model for quality
    agent_description="Expert medical advisor with access to research papers",
    input_format="Medical question from healthcare professional",
    expected_output_format="Evidence-based answer with citations",
    test_types=["medical_accuracy", "clinical_relevance"],
    max_rows=30,
    question_length="long",  # Detailed clinical scenarios
    question_openness="open",  # Complex, open-ended questions
    trap_density=0.2,  # 20% questions test handling of out-of-scope queries
    chunk_size=2048,  # Larger chunks for medical context
    relevance_margin=2.0  # More selective chunk filtering
)

dataset = await generator.generate_from_documents(["research_papers.pdf"])
```

#### Example 2: Quick Functional Tests from Scratch
```python
generator = DatasetGenerator(
    model="gpt-4o-mini",  # Fast and cost-effective
    agent_description="Simple calculator bot",
    input_format="Math expression or question",
    expected_output_format="Numerical answer",
    test_types=["basic_operations", "edge_cases"],
    max_rows=100,
    question_length="short",  # Brief math queries
    question_openness="closed",  # Specific calculations
    trap_density=0.0  # No traps, just functional tests
)

dataset = await generator.generate_from_scratch()
```

#### Example 3: Multi-Language Support
```python
generator = DatasetGenerator(
    model="gpt-4o-mini",
    agent_description="Customer support chatbot for e-commerce",
    input_format="Customer inquiry",
    expected_output_format="Helpful response",
    test_types=["order_status", "returns", "product_info"],
    max_rows=25,
    language="ru",  # Generate in Russian
    question_length="mixed",
    question_openness="mixed"
)

dataset = await generator.generate_from_scratch()
```

### Tips for Best Results

1. **Be Specific in Descriptions**: Clear agent_description helps generate relevant test cases
2. **Use Trap Questions Wisely**: 10-20% trap_density is usually sufficient
3. **Start Small**: Generate 10-20 cases first, then scale up
4. **Mix Question Types**: Use "mixed" settings for comprehensive coverage
5. **Chunk Size Matters**: Larger chunks (1024-2048) for technical docs, smaller for FAQs
6. **Review Generated Data**: Always review and refine generated test cases
7. **Cost Awareness**: Larger documents and more test cases = higher API costs
```

## Advanced Usage

### Custom Evaluation with G-Eval

```python
from eval_lib import GEval

metric = GEval(
    name="Coherence",
    criteria="Evaluate the coherence and logical flow of the response",
    evaluation_steps=[
        "Check if the response has a clear structure",
        "Verify logical connections between ideas",
        "Assess overall coherence"
    ],
    model="gpt-4o-mini",
    threshold=0.7
)
```

### Cost Tracking

All evaluation methods automatically track API costs:

```python
results = await evaluate(test_cases, metrics)

for _, test_results in results:
    for result in test_results:
        total_cost = sum(m.evaluation_cost or 0 for m in result.metrics_data)
        print(f"Total evaluation cost: ${total_cost:.4f}")
```

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
  url = {https://github.com/yourusername/eval-ai-library}
}
```

## Support

- 📧 Email: alekslynx90@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/eval-ai-library/issues)
- 📖 Documentation: [Full Documentation](https://github.com/yourusername/eval-ai-library#readme)

## Acknowledgments

This library was developed to provide a comprehensive solution for evaluating AI models across different use cases and providers.
