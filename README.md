# WAI

A terminal coding and DevOps harness with bring-your-own-key support for eight
LLM providers — and, ahead of it, a workflow designer that turns the harness
into a software factory.

> **Status: Phase 1.5.** Streaming chat across all eight providers, plus
> read-only filesystem tools driven by an agent loop. No writes and no shell
> yet.

## Install

Requires Python 3.11+ on the machine; you do not need to manage it yourself if
you use [uv](https://docs.astral.sh/uv/), which fetches an interpreter for you.

**Today, from git** --- installs a `wai` command on your PATH:

```bash
uv tool install git+https://github.com/shubhambakshi374/WAI
```

**Once released to PyPI:**

```bash
uv tool install wai          # or: pipx install wai
```

**From a clone**, for development or to run an unreleased change:

```bash
uv tool install .            # a real `wai` command, installed from the checkout
uvx --from . wai --version   # or run it once, installing nothing
```

Upgrade with `uv tool upgrade wai`, remove with `uv tool uninstall wai`.

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

## The workspace

Every session has a **workspace**: a rooted filesystem context the model can
read. It defaults to the current directory.

```bash
wai tools list                                   # what the model can call, and where
wai chat --once "what does the retry logic do?"  # the model reads the repo to answer
wai --allow-path /etc/nginx                      # add a root
wai --no-tools                                   # plain chat
```

The workspace is also the Phase 2 primitive: a flow will construct one and
hand the same instance to every step, which is why it lives in
`wai/workspace.py` rather than inside the tools.

| Tool | What it does |
|---|---|
| `read_file` | Line-numbered read with `offset`/`limit`. Refuses binaries. |
| `list_dir` | Directory contents, directories first. |
| `glob` | Find files by pattern, newest first. |
| `grep` | Regex content search. Uses `rg` when installed, else pure Python. |

`.gitignore` is honoured and `.git` is always skipped, so the model sees your
source rather than `node_modules`.

### What it will not read

Two rules, both enforced in `Workspace.resolve` before anything touches disk:

1. **Nothing outside the workspace.** Symlinks are resolved *before* the
   containment check, so a link inside the root pointing at `~/.ssh` does not
   escape it.
2. **No credential files, even inside the workspace** --- `.env*`, `*.pem`,
   `*.key`, `id_rsa*`, `.netrc`, `.npmrc`, `.aws/credentials` and similar.

The second rule exists because WAI ships file contents to external model
providers by design. "Model reads `.env`, quotes it back, key lands in a
provider's logs" is the most plausible way this tool leaks a credential.
Refusals are explicit, so the model reports them instead of retrying. Set
`deny_secrets = false` under `[workspace]` if you genuinely need it off.

```toml
[workspace]
extra_roots = ["/etc/nginx"]

[tools]
enabled = true
max_iterations = 25
max_file_bytes = 262144
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
wai/core        normalized types, the event unions, retries
wai/providers   one adapter per provider, all folding onto that union
wai/config      configuration and credential resolution
wai/storage     JSONL session persistence
wai/workspace   the rooted filesystem context, and its containment rules
wai/tools       read-only filesystem tools
wai/runner      one inference call
wai/agent       the loop: inference, tool execution, repeat
wai/tui         the Textual front end
```

**`wai.core`, `wai.providers`, `wai.workspace`, `wai.tools` and `wai.agent`
must never import `textual`.** The Phase 2
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
- **Phase 1.5 — the workspace and read-only tools.** ✅ Agent loop, `read_file`/`list_dir`/`glob`/`grep`, sandboxed.
- **Phase 2 — the workflow designer.** Compose and run multi-step workflows over a shared workspace; the reason the layering above is enforced.
- **Phase 3+ —** writes and shell behind an approval gate, then the software factory built on the workflow engine.

## License

Dual licensed under either of

- MIT ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option. Unless you state otherwise, any contribution you intentionally
submit for inclusion in this work shall be dual licensed as above, without any
additional terms or conditions.
