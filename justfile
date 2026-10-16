# Development entry point. `just` comes from PyPI via the `just` dependency group.
# Full list: `just` or `just --list`.

# list recipes (default)
default:
    @just --list

# ruff: lint
lint:
    uv run ruff check src tests examples
    uv run ruff format --check src tests examples

# ruff: autofix + format
fmt:
    uv run ruff check --fix src tests examples
    uv run ruff format src tests examples

# ty: type check
typecheck:
    uv run ty check

# pytest; extra args pass through (`just test -k judge`)
test *args:
    uv run pytest {{ args }}

# everything CI runs
ci: lint typecheck test

# simulhausen-harbor (integrations/harbor, its own uv project): lint, format check, ty, pytest
harbor-ci:
    uv sync --directory integrations/harbor --locked
    uv run --directory integrations/harbor ruff check .
    uv run --directory integrations/harbor ruff format --check .
    uv run --directory integrations/harbor ty check
    uv run --directory integrations/harbor pytest

# simulhausen-harbor: ruff autofix + format
harbor-fmt:
    uv run --directory integrations/harbor ruff check --fix .
    uv run --directory integrations/harbor ruff format .

# build sdist + wheel into dist/
build:
    uv build

# run every pre-commit hook (prek)
precommit:
    uv run prek run --all-files
