# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX

### Added
- Initial release of Eval AI Library
- Support for multiple LLM providers (OpenAI, Azure, Google Gemini, Anthropic Claude, Ollama)
- RAG evaluation metrics:
  - Answer Relevancy Metric
  - Answer Precision Metric
  - Faithfulness Metric
  - Contextual Relevancy Metric
  - Contextual Precision Metric
  - Contextual Recall Metric
  - Bias Metric
  - Toxicity Metric
  - Restricted Refusal Metric
- Agent evaluation metrics:
  - Tool Correctness Metric
  - Task Success Rate Metric
  - Role Adherence Metric
  - Knowledge Retention Metric
- Custom evaluation capabilities:
  - G-Eval framework
  - Custom Eval Metric
- Data generation tools:
  - Document Loader
  - Test Case Generator
- Conversational evaluation support
- Automatic cost tracking for API calls
- Comprehensive documentation and examples
- Type hints support (PEP 561)

### Features
- Async/await support for efficient evaluation
- Flexible metric configuration with thresholds
- Detailed evaluation logs and reasoning
- Batch evaluation capabilities
- Multi-turn conversation evaluation

## [Unreleased]

### Planned
- Additional metrics for specific use cases
- Integration tests
- CI/CD pipeline

---

## Version History

### Version Numbering
This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backward compatible manner
- **PATCH** version for backward compatible bug fixes

### How to Update This File
When making changes:
1. Add new entries under `[Unreleased]`
2. Organize by type: Added, Changed, Deprecated, Removed, Fixed, Security
3. When releasing, move unreleased items to a new version section
4. Keep descriptions clear and concise
