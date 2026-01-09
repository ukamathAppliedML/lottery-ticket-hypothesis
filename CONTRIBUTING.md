# Contributing to Lottery Ticket Hypothesis

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/lottery-ticket-hypothesis.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install development dependencies: `pip install -e ".[dev]"`

## Development Workflow

### Code Style

We use the following tools for code quality:

```bash
# Format code
black src/ tests/ examples/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pruning.py -v
```

### Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Areas for Contribution

### High Priority

- [ ] Additional model architectures (Transformers, ViT)
- [ ] LLM pruning examples (BERT, GPT-2)
- [ ] TensorRT integration guide
- [ ] Performance benchmarks on various GPUs

### Medium Priority

- [ ] Visualization tools for pruning analysis
- [ ] Gradual pruning schedules
- [ ] Knowledge distillation integration
- [ ] ONNX export improvements

### Documentation

- [ ] More detailed API documentation
- [ ] Tutorial notebooks
- [ ] Deployment guides for different platforms

## Code of Conduct

Be respectful and constructive in all interactions. We're all here to learn and build cool things together.

## Questions?

Open an issue or reach out to the maintainers. We're happy to help!
