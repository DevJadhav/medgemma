"""
Enable running the pipeline CLI as a module.

Usage:
    uv run python -m medai_compass.pipelines run
    uv run python -m medai_compass.pipelines train model=medgemma_27b
    uv run python -m medai_compass.pipelines tune --scheduler asha
    uv run python -m medai_compass.pipelines evaluate --checkpoint /path/to/model
    uv run python -m medai_compass.pipelines serve --port 8000
    uv run python -m medai_compass.pipelines data --source ./data --output ./processed
    uv run python -m medai_compass.pipelines config
    uv run python -m medai_compass.pipelines verify
"""

import sys

from medai_compass.pipelines.cli import main

if __name__ == "__main__":
    sys.exit(main())
