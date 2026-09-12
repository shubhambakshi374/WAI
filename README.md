# WAI

A terminal coding and DevOps harness with bring-your-own-key support for eight
LLM providers — and, ahead of it, a workflow designer that turns the harness
into a software factory.

> **Status: Phase 1.** Streaming chat across all eight providers, with the
> session, config and provider layers that Phase 2 is built on. No tool
> execution or agent loop yet.

## Install

Requires Python 3.11+.

```bash
uv tool install wai          # or: pipx install wai
```

From a checkout:

```bash
uv sync --all-groups
uv run wai
```

## Quick start

```bash
wai config set-key anthropic     # stored in the OS keyring, never on disk
wai config doctor                # which providers can authenticate
wai                              # launch the TUI
```

Headless, for scripts and CI:

```bash
wai chat --once "explain this failing rollout" --model claude-sonnet-5
```

## Providers

| Provider | Credential | Notes |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | extended thinking |
| `openai` | `OPENAI_API_KEY` | |
| `openrouter` | `OPENROUTER_API_KEY` | routes to many upstream models |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-reasoner` streams reasoning |
| `azure_foundry` | `AZURE_OPENAI_API_KEY` | needs `base_url`; model = deployment name |
| `gemini` | `GEMINI_API_KEY` | |
| `mistral` | `MISTRAL_API_KEY` | |
| `bedrock` | **AWS credential chain** | `AWS_PROFILE`, instance/IRSA roles — no API key |

Credentials resolve in this order: `WAI_<PROVIDER>_API_KEY`, then the native
environment variable above, then the OS keyring. Environment wins so CI and
headless runs work without a keyring backend.

Bedrock is deliberately different: it authenticates through the standard boto3
chain, because requiring an API key would break the way a DevOps tool is
normally deployed.

## Configuration

`wai config path` prints the location (`~/.config/wai/config.toml` on Linux,
`~/Library/Application Support/wai/config.toml` on macOS). It never contains
secrets.

```toml
default_profile = "sonnet"

[profiles.sonnet]
provider = "anthropic"
model = "claude-sonnet-5"
max_tokens = 8192

[profiles.ops]
provider = "bedrock"
model = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
system = "You are a careful SRE. Explain before you act."

[providers.bedrock]
region = "eu-west-1"

[providers.azure_foundry]
base_url = "https://my-resource.openai.azure.com"
api_version = "2024-10-21"
```

Sessions are appended as JSONL under the platform data directory as each turn
completes, so an interrupted run never loses history.

## Keys

| | |
|---|---|
| `Enter` | send |
| `Ctrl+J` | newline |
| `Ctrl+P` | switch model |
| `Ctrl+N` | new session |
| `Ctrl+C` | cancel the stream |
| `Ctrl+Q` | quit |

## Architecture

```
wai/core        normalized types, the streaming event union, retries
wai/providers   one adapter per provider, all folding onto that union
wai/config      configuration and credential resolution
wai/storage     JSONL session persistence
wai/runner      headless turn execution
wai/tui         the Textual front end
```

**`wai.core` and `wai.providers` must never import `textual`.** The Phase 2
workflow engine drives providers headlessly; if the provider layer were
entangled with the UI, Phase 2 would start with a rewrite. `tests/test_layering.py`
enforces this — if it fails, move the offending code into `wai.tui` rather than
deleting the test.

Every adapter normalizes its provider's stream onto one event union
(`wai/core/events.py`), which already defines tool-call and reasoning events
even though Phase 1 emits only text. That is deliberate: it keeps the agent
loop from being a breaking change.

## Development

```bash
uv sync --all-groups
uv run pytest                       # unit tests, no network
uv run pytest -m live               # hits real APIs; costs money; needs keys
uv run ruff check . && uv run mypy
uv run textual run --dev wai.tui.app:WaiApp   # with `uv run textual console`
```

Snapshot tests render the TUI to SVG and will churn when Textual is upgraded:
`uv run pytest tests/test_snapshots.py --snapshot-update`.

## Roadmap

- **Phase 1 — skeleton and chat.** ✅ Eight providers, streaming TUI, sessions, BYOK config.
- **Phase 2 — the workflow designer.** Compose and run multi-step workflows; the reason the layering above is enforced.
- **Phase 3+ —** tool execution and the agent loop, then the software factory built on the workflow engine.

## License

Dual licensed under either of

- MIT ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option. Unless you state otherwise, any contribution you intentionally
submit for inclusion in this work shall be dual licensed as above, without any
additional terms or conditions.
