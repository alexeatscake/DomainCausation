set -e
uv sync
uv run python scripts/uncorrelated.py
uv run python scripts/separate.py
uv run python scripts/correlated.py
uv run python scripts/separate_and_correlated.py
uv run python scripts/domain_difference.py