# Contributing to AI-FastAPI-MLOps

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

- Check if the issue already exists
- Use the issue template
- Include detailed information:
  - Python version
  - Operating system
  - Steps to reproduce
  - Expected vs actual behavior

### Suggesting Enhancements

- Search existing issues first
- Clearly describe the enhancement
- Explain why it would be useful
- Include examples if possible

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow the code style
   - Add tests for new features
   - Update documentation
4. **Run tests locally**
   ```bash
   pytest tests/ -v
   black api/
   flake8 api/
   ```
5. **Commit your changes**
   ```bash
   git commit -m "feat: Add your feature"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ai-fastapi-mlops.git
cd ai-fastapi-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
pytest tests/ -v
```

## Code Style

- Follow PEP 8
- Use Black for formatting
- Use type hints
- Write docstrings for functions
- Keep functions focused and small

## Testing

- Write tests for new features
- Maintain >80% code coverage
- Test edge cases
- Use descriptive test names

## Commit Messages

Follow Conventional Commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
- `style:` Formatting
- `chore:` Maintenance

## Questions?

Feel free to open an issue or reach out:
- Email: priyanshj1304@gmail.com
- GitHub: @Priyanshjain10

Thank you for contributing! 🎉
