def get_question_style_guidance(question_openness: str, question_length: str) -> str:
    """Generate guidance based on question style parameters."""
    guidance = []

    if question_openness == "open":
        guidance.append(
            "- Favor open-ended questions that allow for detailed, explanatory responses")
        guidance.append(
            "- Include 'how', 'why', 'explain', 'describe' type questions")
    elif question_openness == "closed":
        guidance.append(
            "- Focus on specific, factual questions with definitive answers")
        guidance.append(
            "- Include yes/no questions, specific data requests, and factual lookups")
    else:  # mixed
        guidance.append("- Mix both open-ended and closed questions")
        guidance.append(
            "- Balance exploratory questions with specific factual queries")

    if question_length == "short":
        guidance.append("- Keep inputs concise and direct (1-2 sentences)")
    elif question_length == "long":
        guidance.append(
            "- Include detailed context and background in inputs (3+ sentences)")
        guidance.append(
            "- Provide scenarios with multiple parts or complex requirements")
    else:  # mixed
        guidance.append(
            "- Vary input length from brief queries to detailed scenarios")

    return "\n".join(guidance) if guidance else ""


def get_trap_guidance(trap_density: float) -> str:
    """Generate guidance for trap questions based on density."""
    if trap_density == 0:
        return "**No Trap Questions:** All questions should be answerable using the provided information."

    trap_percentage = int(trap_density * 100)

    return f"""**Trap Questions ({trap_percentage}% of total):**
Create realistic scenarios where:
- The user asks about information NOT present in the reference material
- Questions contain subtle factual errors or misconceptions
- Requests involve information that would require knowledge beyond the provided context
- Make traps subtle and realistic - they should feel like genuine user mistakes or knowledge gaps
- The agent should be able to politely indicate the limitation or correct the misconception"""


def dataset_generation_prompt(
    chunk: str,
    rows_per_chunk: int,
    agent_description: str,
    input_format: str,
    expected_output_format: str,
    test_types: list[str],
    question_length: str,
    question_openness: str,
    trap_density: float,
    language: str
) -> str:
    """
    Create a more realistic and detailed prompt for dataset generation.
    """

    question_style_guidance = get_question_style_guidance(
        question_openness, question_length)

    trap_guidance = get_trap_guidance(trap_density)

    prompt = f"""You are an expert test case designer creating realistic evaluation scenarios for an AI agent.

**Agent Context:**
{agent_description}

**Input Format:** {input_format}
**Expected Output Format:** {expected_output_format}

**Reference Information:**
\"\"\"{chunk}\"\"\"

**Your Task:**
Create {rows_per_chunk} realistic test cases that would naturally occur when evaluating this agent. Each test case should:

1. **Reflect Real User Interactions:** Generate inputs that actual users would provide in real scenarios
2. **Be Contextually Relevant:** Use the reference information naturally, as if a user discovered it and wants to interact with the agent about it
3. **Vary in Complexity:** Include both straightforward and nuanced scenarios
4. **Test Different Capabilities:** Cover various aspects of the agent's expected functionality

**Test Types Available:** {', '.join(test_types)}

**Question Characteristics:**
- **Length:** {question_length}
- **Style:** {question_openness}
{question_style_guidance}

**Quality Requirements:**
- Use natural, conversational language that real users would employ
- Avoid academic or artificial phrasing
- Include context and background when users would naturally provide it
- Make questions specific enough to have clear, verifiable answers
- Ensure variety in question types and complexity levels

{trap_guidance}

**Output Format:**
Generate exactly {rows_per_chunk} JSON objects in an array. Each object must have:
- "input": The user's natural input/question
- "expected_output": The ideal agent response
- "test_type": One of {test_types}
- "is_trap": boolean indicating if this is a trap question

**Language:** {language}

Return ONLY the JSON array, no additional text or formatting."""

    return prompt


def dataset_generation_from_scratch_prompt(
    max_rows: int,
    agent_description: str,
    input_format: str,
    expected_output_format: str,
    test_types: list[str],
    question_length: str,
    question_openness: str,
    trap_density: float,
    language: str
) -> str:
    """
    Create a realistic prompt for generating dataset from scratch without reference material.
    """

    question_style_guidance = get_question_style_guidance(
        question_openness, question_length)
    trap_guidance = get_trap_guidance(trap_density)

    prompt = f"""You are an expert test case designer creating comprehensive evaluation scenarios for an AI agent.

**Agent Context:**
{agent_description}

**Input Format:** {input_format}
**Expected Output Format:** {expected_output_format}

**Your Task:**
Create {max_rows} diverse, realistic test cases that thoroughly evaluate this agent's capabilities. Design scenarios that would naturally occur in real-world usage.

**Test Design Principles:**

1. **Real-World Scenarios:** Create inputs that actual users would provide in genuine situations
2. **Comprehensive Coverage:** Test different aspects of the agent's functionality and knowledge domains
3. **Varied Complexity:** Include simple queries, moderate challenges, and complex multi-step scenarios
4. **Edge Cases:** Include boundary conditions and unusual but valid requests
5. **Common Use Cases:** Focus on frequent user interaction patterns

**Test Types Available:** {', '.join(test_types)}

**Question Characteristics:**
- **Length:** {question_length}
- **Style:** {question_openness}
{question_style_guidance}

**Quality Requirements:**
- Use natural, conversational language that real users would employ
- Avoid repetitive patterns or artificial academic phrasing
- Include appropriate context when users would naturally provide it
- Create specific, testable scenarios with clear success criteria
- Ensure broad coverage of the agent's expected capabilities
- Make each test case unique and valuable for evaluation

{trap_guidance}

**Diversity Guidelines:**
- Vary question topics and domains relevant to the agent
- Include different user personas and interaction styles
- Test both common workflows and edge cases
- Balance straightforward requests with more complex scenarios

**Output Format:**
Generate exactly {max_rows} JSON objects in an array. Each object must have:
- "input": The user's natural input/question
- "expected_output": The ideal agent response
- "test_type": One of {test_types}
- "is_trap": boolean indicating if this is a trap question

**Language:** {language}

Return ONLY the JSON array, no additional text or formatting."""

    return prompt
