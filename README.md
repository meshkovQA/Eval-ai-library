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

### Basic Batch Evaluation
```python
import asyncio
from eval_lib import (
    evaluate,
    EvalTestCase,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    BiasMetric
)

async def test_batch_standard_metrics():
    """Test batch evaluation with multiple test cases and standard metrics"""

    # Create test cases
    test_cases = [
        EvalTestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            expected_output="Paris",
            retrieval_context=["Paris is the capital of France."]
        ),
        EvalTestCase(
            input="What is photosynthesis?",
            actual_output="The weather today is sunny.",
            expected_output="Process by which plants convert light into energy",
            retrieval_context=[
                "Photosynthesis is the process by which plants use sunlight."]
        )
    ]

    # Define metrics
    metrics = [
        AnswerRelevancyMetric(
            model="gpt-4o-mini",
            threshold=0.7,
            temperature=0.5,
        ),
        FaithfulnessMetric(
            model="gpt-4o-mini",
            threshold=0.8,
            temperature=0.5,
        ),
        BiasMetric(
            model="gpt-4o-mini",
            threshold=0.8,
        ),
    ]

    # Run batch evaluation
    results = await evaluate(
        test_cases=test_cases,
        metrics=metrics,
        verbose=True
    )

    return results


if __name__ == "__main__":
    asyncio.run(test_batch_standard_metrics())
```

### G-Eval with Probability-Weighted Scoring (single evaluation)

G-Eval implements the state-of-the-art evaluation method from the paper ["G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"](https://arxiv.org/abs/2303.16634). It uses **probability-weighted scoring** (score = Σ p(si) × si) for fine-grained, continuous evaluation scores.
```python
from eval_lib import GEval, EvalTestCase

async def evaluate_with_geval():
    test_case = EvalTestCase(
        input="Explain quantum computing to a 10-year-old",
        actual_output="Quantum computers are like super-powerful regular computers that use special tiny particles to solve really hard problems much faster.",
    )
    
    # G-Eval with auto chain-of-thought
    metric = GEval(
        model="gpt-4o",  # Works best with GPT-4
        threshold=0.7,  # Score range: 0.0-1.0
        name="Clarity & Simplicity",
        criteria="Evaluate how clear and age-appropriate the explanation is for a 10-year-old child",

        # Evaluation_steps is auto-generated from criteria if not provided
        evaluation_steps=[
        "Step 1: Check if the language is appropriate for a 10-year-old. Avoid complex technical terms, jargon, or abstract concepts that children cannot relate to. The vocabulary should be simple and conversational.",
        
        "Step 2: Evaluate the use of analogies and examples. Look for comparisons to everyday objects, activities, or experiences familiar to children (toys, games, school, animals, family activities). Good analogies make abstract concepts concrete.",
        
        "Step 3: Assess the structure and flow. The explanation should have a clear beginning, middle, and end. Ideas should build logically, starting with familiar concepts before introducing new ones. Sentences should be short and easy to follow.",
        
        "Step 4: Check for engagement elements. Look for questions, storytelling, humor, or interactive elements that capture a child's attention. The tone should be friendly and encouraging, not boring or too formal.",
        
        "Step 5: Verify completeness without overwhelming. The explanation should cover the main idea adequately but not overload with too many details. It should answer the question without confusing the child with unnecessary complexity.",
        
        "Step 6: Assign a score from 0.0 to 1.0, where 0.0 means completely inappropriate or unclear for a child, and 1.0 means perfectly clear, engaging, and age-appropriate."
        ],
        n_samples=20,  # Number of samples for probability estimation (default: 20)
        sampling_temperature=2.0  # High temperature for diverse sampling (default: 2.0)
    )
    
    result = await metric.evaluate(test_case)
    

asyncio.run(evaluate_with_geval())
```

### Custom Evaluation with Verdict-Based Scoring (single evaluation)

CustomEvalMetric uses **verdict-based evaluation** with automatic criteria generation for transparent and detailed scoring:
```python
from eval_lib import CustomEvalMetric

async def custom_evaluation():
    test_case = EvalTestCase(
        input="Explain photosynthesis",
        actual_output="Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
    )
    
    # Verdict-based custom evaluation
    metric = CustomEvalMetric(
        model="gpt-4o-mini",
        threshold=0.8,
        name="Scientific Accuracy",
        criteria="Evaluate if the explanation is scientifically accurate and complete",
        evaluation_steps=None,  # Auto-generated if not provided
        temperature=0.8,  # Controls verdict aggregation
        verbose=True
    )
    
    result = await metric.evaluate(test_case)

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
    test_cases = EvalTestCase(
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
            temperature=1.0 
        )
    ]
    
    results = await evaluate(
        test_cases=[test_cases],
        metrics=metrics,
        verbose=True
    )
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
    # Create conversations
    conversations = [
        ConversationalEvalTestCase(
            chatbot_role="You are a professional customer support assistant.",
            turns=[
                EvalTestCase(
                    input="I need help with my order",
                    actual_output="I'd be happy to help. Could you provide your order number?"
                ),
                EvalTestCase(
                    input="It's #12345",
                    actual_output="Thank you! Let me look up order #12345 for you."
                ),
                EvalTestCase(
                    input="When will it arrive?",
                    actual_output="Your order will be delivered on October 27, 2025."
                ),
            ]
        ),
        ConversationalEvalTestCase(
            chatbot_role="You are a formal financial advisor.",
            turns=[
                EvalTestCase(
                    input="Should I invest in stocks?",
                    actual_output="Yo dude! Just YOLO into stocks!"
                ),
                EvalTestCase(
                    input="What about bonds?",
                    actual_output="Bonds are boring, bro!"
                ),
            ]
        ),
        ConversationalEvalTestCase(
            chatbot_role="You are a helpful assistant.",
            turns=[
                EvalTestCase(
                    input="My name is John",
                    actual_output="Nice to meet you, John!"
                ),
                EvalTestCase(
                    input="What's my name?",
                    actual_output="Your name is John."
                ),
                EvalTestCase(
                    input="Where do I live?",
                    actual_output="I don't have that information."
                ),
            ]
        ),
    ]
    
    # Define conversational metrics
    metrics = [
        TaskSuccessRateMetric(
            model="gpt-4o-mini",
            threshold=0.7,
            temperature=0.9,
        ),
        RoleAdherenceMetric(
            model="gpt-4o-mini",
            threshold=0.8,
            temperature=0.5,
        ),
        KnowledgeRetentionMetric(
            model="gpt-4o-mini",
            threshold=0.7,
            temperature=0.5,
        ),
    ]
    
    # Run batch evaluation
    results = await evaluate_conversations(
        conv_cases=conversations,
        metrics=metrics,
        verbose=True
    )
    
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
Detects bias and prejudice in AI-generated output. Score range: 0 (strong bias) and 100 (no bias).
```python
metric = BiasMetric(
    model="gpt-4o-mini",
    threshold=1.0  # Score range: 0 or 100
)
```

#### ToxicityMetric
Identifies toxic content in responses. Score range: 0 (highly toxic) and 100 (no toxicity).
```python
metric = ToxicityMetric(
    model="gpt-4o-mini",
    threshold=1.0  # Score range: 0 or 100
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
````
**Note:** The metric automatically detects if the conversation contains links/URLs and adds "The user got the link to the requested resource" as an evaluation criterion only when links are present in the dialogue.
````
Measures task completion success across conversation:
1. Infers user's goal
2. Generates success criteria
3. Evaluates each criterion (fully/mostly/partial/minor/none)
4. Aggregates into final score
```python
metric = TaskSuccessRateMetric(
    model="gpt-4o-mini",
    threshold=0.7,
    temperature=1.0  # Higher = more lenient aggregation
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
    temperature=0.5,
    chatbot_role="You are helpfull assistant" # Set role here directly
)

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
Verdict-based custom evaluation with automatic criteria generation.
Automatically:
- Generates 3-5 specific sub-criteria from main criteria (1 LLM call)
- Evaluates each criterion with verdicts (fully/mostly/partial/minor/none)
- Aggregates using softmax (temperature-controlled)
Total: 1-2 LLM calls

Usage:
```python
metric = CustomEvalMetric(
    model="gpt-4o-mini",
    threshold=0.8,
    name="Code Quality",
    criteria="Evaluate code readability, efficiency, and best practices",
    evaluation_steps=None,  # Auto-generated if not provided
    temperature=0.8,  # Controls verdict aggregation (0.1=strict, 1.0=lenient)
    verbose=True
)

```

**Example with manual criteria:**
```python
metric = CustomEvalMetric(
    model="gpt-4o-mini",
    threshold=0.8,
    name="Child-Friendly Explanation",
    criteria="Evaluate if explanation is appropriate for a 10-year-old",
    evaluation_steps=[  # Manual criteria for precise control
        "Uses simple vocabulary appropriate for 10-year-olds",
        "Includes relatable analogies or comparisons",
        "Avoids complex technical jargon",
        "Explanation is engaging and interesting",
        "Concept is broken down into understandable parts"
    ],
    temperature=0.8,
    verbose=True
)

result = await metric.evaluate(test_case)
```

## Understanding Evaluation Results

### Score Ranges

All metrics use a normalized score range of **0.0 to 1.0**:
- **0.0**: Complete failure / Does not meet criteria
- **0.5**: Partial satisfaction / Mixed results
- **1.0**: Perfect / Fully meets criteria

**Score Interpretation:**
- **0.8 - 1.0**: Excellent
- **0.7 - 0.8**: Good (typical threshold)
- **0.5 - 0.7**: Acceptable with issues
- **0.0 - 0.5**: Poor / Needs improvement

## Verbose Mode

All metrics support a `verbose` parameter that controls output formatting:

### verbose=False (Default) - JSON Output
Returns simple dictionary with results:
```python
metric = AnswerRelevancyMetric(
    model="gpt-4o-mini",
    threshold=0.7,
    verbose=False  # Default
)

result = await metric.evaluate(test_case)
print(result)
# Output: Simple dictionary
# {
#   'name': 'answerRelevancyMetric',
#   'score': 0.85,
#   'success': True,
#   'reason': 'Answer is highly relevant...',
#   'evaluation_cost': 0.000234,
#   'evaluation_log': {...}
# }
```

### verbose=True - Beautiful Console Output
Displays formatted results with colors, progress bars, and detailed logs:
```python
metric = CustomEvalMetric(
    model="gpt-4o-mini",
    threshold=0.9,
    name="Factual Accuracy",
    criteria="Evaluate the factual accuracy of the response",
    verbose=True  # Enable beautiful output
)

result = await metric.evaluate(test_case)
# Output: Beautiful formatted display (see image below)
```

**Console Output with verbose=True:**

**Console Output with verbose=True:**
```
╔════════════════════════════════════════════════════════════════╗
║                    📊answerRelevancyMetric                     ║
╚════════════════════════════════════════════════════════════════╝

Status: ✅ PASSED
Score:  0.91 [███████████████████████████░░░] 91%
Cost:   💰 $0.000178
Reason:
  The answer correctly identifies Paris as the capital of France, demonstrating a clear understanding of the
  user's request. However, it fails to provide a direct and explicit response, which diminishes its overall
  effectiveness.

Evaluation Log:
╭───────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ {                                                                                                             │
│   "input_question": "What is the capital of France?",                                                         │
│   "answer": "The capital of France is Paris and it is a beautiful city and known for its art and culture.",   │
│   "user_intent": "The user is seeking information about the capital city of France.",                         │
│   "comment_user_intent": "Inferred goal of the question.",                                                    │
│   "statements": [                                                                                             │
│     "The capital of France is Paris.",                                                                        │
│     "Paris is a beautiful city.",                                                                             │
│     "Paris is known for its art and culture."                                                                 │
│   ],                                                                                                          │
│   "comment_statements": "Atomic facts extracted from the answer.",                                            │
│   "verdicts": [                                                                                               │
│     {                                                                                                         │
│       "verdict": "fully",                                                                                     │
│       "reason": "The statement explicitly answers the user's question about the capital of France."           │
│     },                                                                                                        │
│     {                                                                                                         │
│       "verdict": "minor",                                                                                     │
│       "reason": "While it mentions Paris, it does not directly answer the user's question."                   │
│     },                                                                                                        │
│     {                                                                                                         │
│       "verdict": "minor",                                                                                     │
│       "reason": "This statement is related to Paris but does not address the user's question about the        │
│ capital."                                                                                                     │
│     }                                                                                                         │
│   ],                                                                                                          │
│   "comment_verdicts": "Each verdict explains whether a statement is relevant to the question.",               │
│   "verdict_score": 0.9142,                                                                                    │
│   "comment_verdict_score": "Proportion of relevant statements in the answer.",                                │
│   "final_score": 0.9142,                                                                                      │
│   "comment_final_score": "Score based on the proportion of relevant statements.",                             │
│   "threshold": 0.7,                                                                                           │
│   "success": true,                                                                                            │
│   "comment_success": "Whether the score exceeds the pass threshold.",                                         │
│   "final_reason": "The answer correctly identifies Paris as the capital of France, demonstrating a clear      │
│ understanding of the user's request. However, it fails to provide a direct and explicit response, which       │
│ diminishes its overall effectiveness.",                                                                       │
│   "comment_reasoning": "Compressed explanation of the key verdict rationales."                                │
│ }                                                                                                             │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Features:
- ✅ Color-coded status (✅ PASSED / ❌ FAILED)
- 📊 Visual progress bar for scores
- 💰 Cost tracking display
- 📝 Formatted reason with word wrapping
- 📋 Pretty-printed evaluation log in bordered box

**When to use verbose=True:**
- Interactive development and testing
- Debugging evaluation issues
- Presentations and demonstrations
- Manual review of results

**When to use verbose=False:**
- Production environments
- Batch processing
- Automated testing
- When storing results in databases

---

## Working with Results

Results are returned as simple dictionaries. Access fields directly:
```python
# Run evaluation
result = await metric.evaluate(test_case)

# Access result fields
score = result['score']              # 0.0-1.0
success = result['success']          # True/False
reason = result['reason']            # String explanation
cost = result['evaluation_cost']     # USD amount
log = result['evaluation_log']       # Detailed breakdown

# Example: Check success and print score
if result['success']:
    print(f"✅ Passed with score: {result['score']:.2f}")
else:
    print(f"❌ Failed: {result['reason']}")
    
# Access detailed verdicts (for verdict-based metrics)
if 'verdicts' in result['evaluation_log']:
    for verdict in result['evaluation_log']['verdicts']:
        print(f"- {verdict['verdict']}: {verdict['reason']}")
```

## Temperature Parameter

Many metrics use a **temperature** parameter for score aggregation (via temperature-weighted scoring):

- **Lower (0.1-0.3)**: **STRICT** - All scores matter equally, low scores heavily penalize the final result. Best for critical applications where even one bad verdict should fail the metric.
- **Medium (0.4-0.6)**: **BALANCED** - Moderate weighting between high and low scores. Default behavior for most use cases (default: 0.5).
- **Higher (0.7-1.0)**: **LENIENT** - High scores (fully/mostly) dominate, effectively ignoring partial/minor/none verdicts. Best for exploratory evaluation or when you want to focus on positive signals.

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
metric = TaskSuccessRateMetric(temperature=1.0)  
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
    language="en",
    verbose=True  # Displays beautiful formatted progress, statistics and full dataset preview
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
    max_chunks=30,
    verbose=True
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
BiasMetric(threshold=0.8)
ToxicityMetric(threshold=0.85)

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
metric = TaskSuccessRateMetric(temperature=1.0)
```

**Real-world examples:**
```python
# Production RAG system - must be accurate
faithfulness = FaithfulnessMetric(
    model="gpt-4o-mini",
    threshold=0.8,
    temperature=0.2  # STRICT: verdicts "none", "minor", "partially" significantly impact score
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
    temperature=1.0  # LENIENT: Focuses on "fully" and "mostly" completions
)
```

### 4. Leverage Evaluation Logs
```python
# Enable verbose mode for automatic detailed display
metric = AnswerRelevancyMetric(
    model="gpt-4o-mini",
    threshold=0.7,
    verbose=True  # Automatic formatted output with full logs
)

# Or access logs programmatically
result = await metric.evaluate(test_case)
log = result['evaluation_log']

# Debugging failures
if not result['success']:
    # All details available in log
    reason = result['reason']
    verdicts = log.get('verdicts', [])
    steps = log.get('evaluation_steps', [])
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