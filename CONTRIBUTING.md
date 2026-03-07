# Contributing to Eval AI Library

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/meshkovQA/Eval-ai-library.git
cd Eval-ai-library
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

pip install -e ".[full]"
pip install pytest
```

## How to Contribute

### Reporting Bugs

Open an issue on [GitHub Issues](https://github.com/meshkovQA/Eval-ai-library/issues) with:
- A clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

### Suggesting Features

Open an issue with the "feature request" label describing the use case and proposed solution.

### Submitting Changes

1. Fork the repository
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Make your changes
4. Run the test suite to ensure nothing is broken:
   ```bash
   pytest tests/
   ```
5. Commit with a clear message:
   ```bash
   git commit -m "Add: brief description of change"
   ```
6. Push and open a Pull Request against `main`

## Code Guidelines

- Follow existing code style and project conventions
- Add tests for new functionality
- Keep changes focused — one feature or fix per PR
- Update pricing in `eval_lib/price.py` if adding new models
- New LLM providers should follow the OpenAI-compatible pattern in `eval_lib/llm_client.py`

## Project Structure

```
eval_lib/
  __init__.py          # Public API
  llm_client.py        # LLM providers and client logic
  price.py             # Model pricing data
  metrics/             # Evaluation metrics
  connector/           # API Connector (web UI backend)
  static/              # CSS/JS for web dashboard
tests/                 # Test suite
```

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
