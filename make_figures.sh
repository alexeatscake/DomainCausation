set -e
uv sync
uv run python scripts/uncorrelated.py
uv run python scripts/separate.py
uv run python scripts/correlated.py
uv run python scripts/separate_and_correlated.py
uv run python scripts/domain_difference.py
uv run python scripts/control_simple.py
uv run python scripts/control_confounding.py
uv run python scripts/control_moredomain.py
uv run python scripts/control_noise.py
uv run python scripts/control_shift.py
