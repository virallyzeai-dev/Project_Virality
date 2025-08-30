#!/usr/bin/env python3
"""
Quick setup verification script for the Virality Analyzer project.
Run this script to verify that the development environment is properly configured.
"""

import sys
import subprocess
import importlib
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"✓ {description}...")
    try:
        result = subprocess.run(
            command.split(), 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        if result.returncode == 0:
            print(f"  ✅ Success")
            return True
        else:
            print(f"  ❌ Failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def check_import(module_name: str, description: str) -> bool:
    """Check if a module can be imported."""
    print(f"✓ {description}...")
    try:
        importlib.import_module(module_name)
        print(f"  ✅ Success")
        return True
    except ImportError as e:
        print(f"  ❌ Failed: {e}")
        return False


def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists."""
    print(f"✓ {description}...")
    if Path(file_path).exists():
        print(f"  ✅ Success")
        return True
    else:
        print(f"  ❌ Failed: File not found")
        return False


def main():
    """Run all verification checks."""
    print("🚀 Virality Analyzer - Development Environment Verification")
    print("=" * 60)
    
    checks = []
    
    # Check Python version
    print(f"✓ Python version: {sys.version}")
    if sys.version_info >= (3, 9):
        print("  ✅ Python version is compatible")
        checks.append(True)
    else:
        print("  ❌ Python 3.9+ required")
        checks.append(False)
    
    # Check key files exist
    checks.append(check_file_exists("pyproject.toml", "Poetry configuration"))
    checks.append(check_file_exists("src/virality_analyzer/__init__.py", "Main package"))
    checks.append(check_file_exists(".vscode/tasks.json", "VS Code tasks"))
    checks.append(check_file_exists(".vscode/launch.json", "VS Code debug config"))
    
    # Check Poetry
    checks.append(run_command("poetry --version", "Poetry installation"))
    
    # Check key dependencies can be imported
    sys.path.insert(0, "src")
    checks.append(check_import("virality_analyzer", "Main package import"))
    checks.append(check_import("pandas", "Pandas"))
    checks.append(check_import("numpy", "NumPy"))
    checks.append(check_import("scikit-learn", "Scikit-learn"))
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"🎉 All checks passed! ({passed}/{total})")
        print("\n✨ Your development environment is ready!")
        print("\nNext steps:")
        print("1. Run 'poetry run pytest tests/' to run tests")
        print("2. Try 'poetry run python examples/basic_usage.py'")
        print("3. Start coding with VS Code debugging and tasks!")
    else:
        print(f"⚠️  Some checks failed ({passed}/{total})")
        print("\n🔧 Please resolve the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
