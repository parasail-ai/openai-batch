# AGENTS.md

This file contains conventions and commands for agentic coding assistants working in this repository.

## Build, Lint, and Test Commands

### Format Code
```bash
black openai_batch/ tests/
```

### Run Tests
```bash
# Run all tests (excludes slow and live tests by default)
pytest

# Run a single test
pytest tests/test_basic.py::test_version

# Run tests in a specific file
pytest tests/test_basic.py

# Run slow tests (may make API calls)
pytest -m slow

# Run live tests (requires actual API keys)
pytest -m live

# Run all tests including slow and live
pytest -m 'slow or live'
```

## Code Style Guidelines

### Formatting
- Use Black formatter with line length 100
- Target Python 3.8+

### Imports
- Organize imports: stdlib → third-party → local (separated by blank lines)
- Import specific classes/functions rather than modules
- Use `from typing import ...` for type hints

### Type Hints
- Always include type hints for function signatures
- Use `Optional[T]` for nullable parameters
- Use `Union` for multiple possible types
- Use `Tuple` for multiple return values
- Import from `typing` module

### Naming Conventions
- **Classes**: PascalCase (Batch, Provider, BatchType)
- **Functions/variables**: snake_case (add_to_batch, submission_input_file)
- **Constants**: CAPS_SNAKE_CASE (FINISHED_STATES)
- **Private methods**: prefix with underscore (_ensure_submission_file)
- **Enums**: PascalCase (BatchType) with lowercase values

### File and Path Handling
- Use `pathlib.Path` instead of string paths for file operations
- Use encoding="utf-8" when opening files
- Use context managers (`with` statements) for file I/O

### Error Handling
- Use `ValueError` for user-facing validation errors
- Include descriptive error messages
- Use assert for programming errors that should never occur
- Use try/except for fallback logic (e.g., optional dependencies)

### Classes
- Use `@dataclass` for simple data containers (Provider)
- Implement `__enter__`/`__exit__` for context manager support
- Initialize attributes in `__init__` even if None

### Docstrings
- Use `"""` triple quotes for module and function docstrings
- Include description and parameter/return details
- Keep docstrings concise

### Testing
- Use pytest fixtures (e.g., tmp_path)
- Use parametrization for similar test cases
- Use `pytest.raises(ValueError, match="pattern")` for expected exceptions
- Mark slow tests with `@pytest.mark.slow`
- Mark live tests with `@pytest.mark.live`

### Code Organization
- Separate utility functions into `_utils.py` module
- Use `__init__.py` to expose public API
- Keep backward compatibility in mind; deprecate old APIs with warnings

### API Interaction
- Use `dry_run` parameter for testing without API calls
- Return mock objects/values when dry_run=True
- Pass through OpenAI client parameters (extra_headers, extra_query, extra_body, timeout)

### Constants
- Group related constants as Enums (BatchType)
- Use module-level constants for magic strings/numbers

### Validation
- Validate arguments early in functions
- Use isinstance() for type checking
- Provide specific validation error messages

### Resources
- Black line length: 100
- Test framework: pytest
- Main dependency: openai>=1.60
