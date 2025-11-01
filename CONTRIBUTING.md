# Contributing to RAG Förderkatalog

Vielen Dank für dein Interesse, zu diesem Projekt beizutragen! 🎉

## 🚀 Quick Start

1. **Fork & Clone**
   ```bash
   git clone https://github.com/dgaida/rag_foerderkatalog.git
   cd rag_foerderkatalog
   ```

2. **Setup Development Environment**
   ```bash
   # Mit Conda
   conda env create -f environment.yml
   conda activate rag_foerderkatalog

   # Oder mit pip
   make install-dev
   ```

3. **Run Tests**
   ```bash
   make test
   ```

## 📋 Development Workflow

### Branch Strategy

- `main` — Stable production code
- `develop` — Development branch
- `feature/*` — New features
- `bugfix/*` — Bug fixes
- `hotfix/*` — Critical production fixes

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation

3. **Run quality checks**
   ```bash
   make check-all
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

### Commit Message Convention

Wir verwenden [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation changes
- `style:` — Code style changes (formatting)
- `refactor:` — Code refactoring
- `test:` — Adding tests
- `chore:` — Maintenance tasks

**Beispiele:**
```bash
feat: add keyword search functionality
fix: resolve dimension mismatch in embeddings
docs: update README with new examples
test: add unit tests for llm_wrapper
```

## 🧪 Testing

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Fast tests only (without slow markers)
make test-fast

# Specific test file
pytest tests/test_engine.py -v
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test names: `test_extract_year_valid_date()`
- Use fixtures for common setup
- Mock external dependencies (Ollama, LLMClient)

**Example:**
```python
def test_search_returns_dataframe(mock_engine):
    """Test that search returns a valid DataFrame."""
    result = mock_engine.search("test query", k=5)
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= 5
```

## 🎨 Code Style

### Formatting

We use **Black** and **isort** for consistent code formatting:

```bash
# Format code
make format

# Check formatting
make format-check
```

### Linting

```bash
# Run all linters
make lint

# Type checking
make type-check
```

### Style Guidelines

- **Line length**: 100 characters
- **Docstrings**: Google style
- **Type hints**: Use everywhere
- **Imports**: Organized with isort

**Example:**
```python
def process_data(
    data: pd.DataFrame,
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """Process dataframe and return filtered results.

    Args:
        data: Input dataframe with project data.
        threshold: Minimum score threshold. Defaults to 0.5.

    Returns:
        List of dictionaries containing filtered results.

    Raises:
        ValueError: If data is empty.
    """
    if data.empty:
        raise ValueError("Data cannot be empty")
    return data.to_dict('records')
```

## 📚 Documentation

### Docstrings

All functions, classes, and modules must have docstrings:

- Use **Google style** docstrings
- Include **Args**, **Returns**, **Raises** sections
- Add **Examples** for complex functions

### README Updates

When adding features:
1. Update the main README.md
2. Add examples if applicable
3. Update feature list

## 🐛 Bug Reports

### Before Submitting

- Check existing issues
- Verify it's reproducible
- Gather system information

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
1. Step 1
2. Step 2
3. See error

**Expected Behavior**
What should happen

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.11.5]
- Package Version: [e.g., 1.0.0]

**Additional Context**
Any other relevant information
```

## ✨ Feature Requests

### Before Submitting

- Check if feature already exists
- Search existing feature requests
- Consider if it fits project scope

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches you've thought about
```

## 🔍 Code Review Process

### What We Look For

- ✅ Tests pass and coverage is maintained
- ✅ Code follows style guidelines
- ✅ Documentation is complete
- ✅ Commit messages are clear
- ✅ No unnecessary changes

### Review Timeline

- First review: Within 48 hours
- Follow-up: Within 24 hours
- Merge: After approval + CI passing

## 📦 Pull Request Process

1. **Update your fork**
   ```bash
   git checkout main
   git pull upstream main
   git checkout feature/your-feature
   git rebase main
   ```

2. **Push your changes**
   ```bash
   git push origin feature/your-feature
   ```

3. **Create Pull Request**
   - Use descriptive title
   - Reference related issues
   - Describe changes made
   - Add screenshots if UI changes

4. **Address Review Comments**
   - Make requested changes
   - Push updates
   - Re-request review

### PR Template

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123

## Changes Made
- Added feature X
- Fixed bug Y
- Updated documentation

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] Changelog updated
```

## 🎯 Development Setup

### Useful Commands

```bash
# Development
make install-dev      # Install dev dependencies
make run-debug        # Run with debug logging
make clean            # Clean build artifacts

# Testing
make test             # Run all tests
make test-cov         # Tests with coverage
make test-watch       # Watch mode for TDD

# Quality
make format           # Format code
make lint             # Run linters
make type-check       # Type checking
make check-all        # All quality checks

# CI Simulation
make ci               # Run all CI checks locally
```

### Git Hooks

Setup pre-commit hook:

```bash
cp .git/hooks/pre-commit.sample .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
make git-pre-commit
```

## 🤝 Community

### Getting Help

- 📖 Read the [README](README.md)
- 🐛 Check [existing issues](https://github.com/dgaida/rag-foerderkatalog/issues)
- 💬 Ask in discussions

### Code of Conduct

- Be respectful and inclusive
- Constructive feedback only
- Help others learn
- Focus on the issue, not the person

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing!** 🙏

Your efforts help make this project better for everyone.
