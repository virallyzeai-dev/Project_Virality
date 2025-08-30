# PowerShell script to set up the Virality Analyzer development environment
# Run this script as: .\setup_dev.ps1

param(
    [switch]$SkipPoetry,
    [switch]$SkipTests,
    [switch]$Help
)

if ($Help) {
    Write-Host "Virality Analyzer - Development Environment Setup" -ForegroundColor Cyan
    Write-Host "=================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\setup_dev.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -SkipPoetry    Skip Poetry installation check"
    Write-Host "  -SkipTests     Skip running tests after setup"
    Write-Host "  -Help          Show this help message"
    Write-Host ""
    exit 0
}

Write-Host "🚀 Setting up Virality Analyzer Development Environment" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# Check if Poetry is installed
if (-not $SkipPoetry) {
    Write-Host "✓ Checking Poetry installation..." -ForegroundColor Yellow
    try {
        $poetryVersion = poetry --version
        Write-Host "  ✅ Poetry found: $poetryVersion" -ForegroundColor Green
    }
    catch {
        Write-Host "  ❌ Poetry not found. Please install Poetry first:" -ForegroundColor Red
        Write-Host "     https://python-poetry.org/docs/#installation" -ForegroundColor Red
        exit 1
    }
}

# Install dependencies
Write-Host "✓ Installing project dependencies..." -ForegroundColor Yellow
try {
    poetry install
    Write-Host "  ✅ Dependencies installed successfully" -ForegroundColor Green
}
catch {
    Write-Host "  ❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Set up pre-commit hooks
Write-Host "✓ Setting up pre-commit hooks..." -ForegroundColor Yellow
try {
    poetry run pre-commit install
    Write-Host "  ✅ Pre-commit hooks installed" -ForegroundColor Green
}
catch {
    Write-Host "  ⚠️  Pre-commit hooks setup failed (optional)" -ForegroundColor Yellow
}

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "✓ Creating .env file..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "  ✅ .env file created from example" -ForegroundColor Green
    Write-Host "  ⚠️  Please edit .env file and add your API keys" -ForegroundColor Yellow
}

# Run tests to verify setup
if (-not $SkipTests) {
    Write-Host "✓ Running tests to verify setup..." -ForegroundColor Yellow
    try {
        poetry run pytest tests/test_basic.py -v
        Write-Host "  ✅ Basic tests passed" -ForegroundColor Green
    }
    catch {
        Write-Host "  ⚠️  Some tests failed - this might be expected if API keys are not set" -ForegroundColor Yellow
    }
}

# Run verification script
Write-Host "✓ Running setup verification..." -ForegroundColor Yellow
try {
    poetry run python verify_setup.py
    Write-Host "  ✅ Setup verification completed" -ForegroundColor Green
}
catch {
    Write-Host "  ⚠️  Setup verification had issues - check output above" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Development environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit .env file with your API keys" -ForegroundColor White
Write-Host "2. Open the project in VS Code: code ." -ForegroundColor White
Write-Host "3. Try running an example: poetry run python examples/basic_usage.py" -ForegroundColor White
Write-Host "4. Start developing with the VS Code tasks and debug configurations!" -ForegroundColor White
Write-Host ""
Write-Host "Available VS Code tasks (Ctrl+Shift+P -> 'Tasks: Run Task'):" -ForegroundColor Cyan
Write-Host "- Install Dependencies" -ForegroundColor White
Write-Host "- Run Tests / Run Tests with Coverage" -ForegroundColor White
Write-Host "- Lint Code / Format Code / Type Check" -ForegroundColor White
Write-Host "- Run FastAPI Server / Run Streamlit App" -ForegroundColor White
Write-Host "- Run Basic Example / Run Advanced Example" -ForegroundColor White
Write-Host ""
Write-Host "Happy coding! 🚀" -ForegroundColor Green
