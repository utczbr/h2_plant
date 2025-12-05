"""
Test runner for the h2_plant testing framework.

This script provides a convenient way to run different test suites
with different options based on the testing specification.
"""

import sys
import subprocess
from pathlib import Path


def run_tests(suite=None, verbose=False, coverage=False, benchmark=False, markers=None):
    """
    Run tests with specified options.

    Args:
        suite (str): Specific test suite to run ('unit', 'integration', 'e2e', 'performance')
        verbose (bool): Enable verbose output
        coverage (bool): Generate coverage report
        benchmark (bool): Run performance benchmarks only
        markers (str): Pytest markers to select specific tests
    """
    cmd = ["pytest"]

    # Add output options
    if verbose:
        cmd.append("-v")
    
    # Handle different test types
    if benchmark:
        cmd.extend(["tests/performance/", "--benchmark-only"])
    elif suite == "unit":
        cmd.extend(["tests/core/", "tests/optimization/", "tests/components/"])
    elif suite == "integration":
        cmd.append("tests/integration/")
    elif suite == "e2e":
        cmd.append("tests/e2e/")
    elif suite == "performance":
        cmd.append("tests/performance/")
    else:
        # All tests
        cmd.append("tests/")
    
    # Coverage options
    if coverage:
        cmd.extend(["--cov=h2_plant", "--cov-report=html", "--cov-report=term"])
    
    # Markers
    if markers:
        cmd.extend(["-m", markers])
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    return result.returncode


def main():
    """Main entry point for test runner."""
    if len(sys.argv) < 2:
        print("Usage: python -m tests.runner <suite> [options]")
        print("Suites: unit, integration, e2e, performance, all")
        print("Options: --verbose, --coverage, --benchmark, --markers=<marker>")
        return 1
    
    suite = sys.argv[1] if sys.argv[1] not in ['--verbose', '--coverage', '--benchmark'] else 'all'
    
    verbose = '--verbose' in sys.argv
    coverage = '--coverage' in sys.argv
    benchmark = '--benchmark' in sys.argv
    
    # Extract markers if provided
    markers = None
    for arg in sys.argv:
        if arg.startswith('--markers='):
            markers = arg.split('=', 1)[1]
            break
    
    return run_tests(
        suite=suite,
        verbose=verbose,
        coverage=coverage,
        benchmark=benchmark,
        markers=markers
    )


if __name__ == "__main__":
    sys.exit(main())