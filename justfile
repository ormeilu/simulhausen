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

# build sdist + wheel into dist/
build:
    uv build

# run every pre-commit hook (prek)
precommit:
    uv run prek run --all-files

# frameworks the contract tests cover; `just contract <name>` runs one of them
contract_frameworks := "ag2 agent_framework autogen camel claude_agent crewai deepagents dspy google_adk haystack langchain llamaindex openai_agents pydantic_ai smolagents strands"

# contract tests of one framework adapter against the real library, installed into a throwaway overlay
contract framework *args:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{ framework }}" in
        ag2) with=(ag2) ;;
        agent_framework) with=(agent-framework-core agent-framework-orchestrations) ;;
        autogen) with=(autogen-agentchat) ;;
        camel) with=(camel-ai 'mcp<2') ;;
        claude_agent) with=(claude-agent-sdk) ;;
        crewai) with=(crewai) ;;
        deepagents) with=(deepagents) ;;
        dspy) with=(dspy) ;;
        google_adk) with=(google-adk) ;;
        haystack) with=(haystack-ai) ;;
        langchain) with=(langchain langgraph langgraph-supervisor langchain-classic) ;;
        llamaindex) with=(llama-index-core) ;;
        openai_agents) with=(openai-agents) ;;
        pydantic_ai) with=(pydantic-ai-slim) ;;
        smolagents) with=(smolagents) ;;
        strands) with=(strands-agents) ;;
        *) echo "unknown framework '{{ framework }}'; one of: {{ contract_frameworks }}" >&2; exit 1 ;;
    esac
    flags=()
    for requirement in "${with[@]}"; do flags+=(--with "$requirement"); done
    # -p no:simulhausen comes from addopts; --no-cov: one adapter is not the package.
    uv run --no-group harbor "${flags[@]}" pytest "tests/contract/test_{{ framework }}_contract.py" --no-cov -q {{ args }}

# contract tests of every framework adapter, one overlay after another
contract-all:
    #!/usr/bin/env bash
    set -uo pipefail
    failed=()
    for framework in {{ contract_frameworks }}; do
        echo "== $framework"
        just contract "$framework" || failed+=("$framework")
    done
    if [ "${#failed[@]}" -gt 0 ]; then echo "failed: ${failed[*]}" >&2; exit 1; fi
