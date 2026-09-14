# Contributing

`main` is protected: changes land through a pull request, and CI must be green
before one can be merged.

## Getting set up

Requires Python 3.14+, or [uv](https://docs.astral.sh/uv/), which fetches an
interpreter for you.

```sh
git clone https://github.com/shubhambakshi374/WAI.git
cd WAI
uv sync --all-groups
```

## Before you open a PR

Run what CI runs. All four, in this order:

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```

Tests that hit real provider APIs are deselected by default and cost money to
run; opt in with `-m live` only if you mean it.

## What the review looks for

- **Tests that would fail without the change.** A test asserting that code was
  written is not worth its maintenance.
- **Layering.** `core`, `providers`, `config`, `storage`, `tools` and `cloud`
  must stay importable without Textual — `tests/test_layering.py` enforces it.
  If that test fails, move the code into `wai.tui` rather than deleting the test.
- **Anything reaching a cluster or a cloud account is gated.** Reads may run
  freely; mutations dry-run first and then ask. Operations that run code, mint
  credentials, rewrite authorization or take capacity out of service are
  `PRIVILEGED` and demand a typed confirmation — see `wai/cloud/kube.py`.
- **Tool output is redacted before it reaches a model provider.** Results are
  transmitted verbatim to whichever LLM is configured, so a gap in
  `wai/cloud/redact.py` is a credential leaving someone's machine.

## Commit messages

Say what changed and why it was worth changing. The body is the place for the
reasoning that would otherwise be lost — a constraint discovered, a subtlety
that makes the obvious approach wrong. Skip anything a reader can get from the
diff.
