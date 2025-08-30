# Development Scripts

## Quick Setup
```powershell
# Install dependencies
poetry install

# Set up pre-commit hooks
poetry run pre-commit install

# Run tests to verify setup
poetry run pytest tests/ -v
```

## Common Development Commands

### Testing
```powershell
# Run all tests
poetry run pytest tests/ -v

# Run tests with coverage
poetry run pytest tests/ -v --cov=src/virality_analyzer --cov-report=html

# Run specific test file
poetry run pytest tests/test_features.py -v

# Run tests in watch mode (requires pytest-watch)
poetry run ptw tests/
```

### Code Quality
```powershell
# Format code
poetry run black src/ tests/ examples/

# Sort imports
poetry run isort src/ tests/ examples/

# Lint code
poetry run flake8 src/ tests/

# Type checking
poetry run mypy src/

# Security check
poetry run safety check
poetry run bandit -r src/
```

### Running the Application
```powershell
# Start FastAPI server
poetry run uvicorn virality_analyzer.api.endpoints:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit app
poetry run streamlit run src/virality_analyzer/api/app.py

# Run basic example
poetry run python examples/basic_usage.py

# Run advanced example
poetry run python examples/advanced_analysis.py
```

### Docker Development
```powershell
# Build Docker image
docker build -t virality-analyzer .

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### VS Code Integration

The project includes comprehensive VS Code configuration:

- **Tasks** (`Ctrl+Shift+P` → "Tasks: Run Task"):
  - Install Dependencies
  - Run Tests / Run Tests with Coverage
  - Lint Code / Format Code / Type Check
  - Run FastAPI Server / Run Streamlit App
  - Run Examples
  - Docker Build / Docker Compose Up/Down

- **Debug Configurations** (`F5` or Debug panel):
  - Debug Basic/Advanced Examples
  - Debug FastAPI Server
  - Debug Streamlit App
  - Debug Tests
  - Debug Current File

- **Recommended Extensions**:
  - Python extension pack
  - Black formatter
  - Pylance language server
  - Jupyter notebooks
  - GitHub Copilot

### Environment Variables

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### Debugging Tips

1. **Breakpoints**: Set breakpoints in VS Code and use the debug configurations
2. **Interactive Debugging**: Use `breakpoint()` in your code for interactive debugging
3. **Logging**: The project uses structured logging - check logs for detailed information
4. **Test Debugging**: Use "Debug Specific Test File" configuration to debug failing tests

### Performance Profiling

```powershell
# Profile with cProfile
poetry run python -m cProfile -o profile.stats examples/basic_usage.py

# Analyze with snakeviz
poetry run snakeviz profile.stats
```

### Documentation

```powershell
# Generate API documentation
poetry run sphinx-build -b html docs/ docs/_build/

# Serve documentation locally
cd docs/_build && python -m http.server 8080
```
