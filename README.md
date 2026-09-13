# WAI

A terminal coding and DevOps harness with bring-your-own-key support for eight
LLM providers — and, ahead of it, a workflow designer that turns the harness
into a software factory.

> **Status: Phase 2b.** Streaming chat across all eight providers; filesystem
> tools behind a diff-first approval gate; and **Kubernetes reads that draw
> you a picture** --- usage charts, storage, and a topology view. AWS, Azure
> and GCP tools land in 2c–2e.

## Install

Requires Python 3.11+, or [uv](https://docs.astral.sh/uv/), which fetches an
interpreter for you.

**WAI is not on PyPI**, so `uv tool install wai` will not work. Clone it:

```bash
git clone https://github.com/shubhambakshi374/WAI
cd WAI
uv sync
uv run wai              # the TUI
uv run wai tools list   # anything else
```

To get a `wai` command on your PATH instead of typing `uv run`:

```bash
uv tool install .                 # from a clone
uv tool install --editable .      # ...or track your edits live
```

That installs a snapshot, so after changing the code either re-run it with
`--force` or use `--editable` from the start. `uv tool uninstall wai` removes
it. Installing straight from the remote works too, without cloning:

```bash
uv tool install git+https://github.com/shubhambakshi374/WAI
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

| Tool | What it does | |
|---|---|---|
| `read_file` | Line-numbered read with `offset`/`limit`. Refuses binaries. | read-only |
| `list_dir` | Directory contents, directories first. | read-only |
| `glob` | Find files by pattern, newest first. | read-only |
| `grep` | Regex content search. Uses `rg` when installed, else pure Python. | read-only |
| `write_file` | Create a file, or replace one wholesale. | **asks first** |
| `edit_file` | Replace an exact string. Refuses an ambiguous match. | **asks first** |
| `delete_path` | Delete a file, or a directory with `recursive`. | **asks first** |

`.gitignore` is honoured and `.git` is always skipped, so the model sees your
source rather than `node_modules`.

### Changes always ask first

Nothing is written before you say yes. Each change opens a prompt showing the
**actual unified diff** --- approving a change you cannot see is not consent ---
plus whether git could get the file back:

```
EDIT — approval required
src/wai/core/retry.py
tracked by git and unmodified — recoverable with git checkout

  @@ -12,7 +12,7 @@
  -    base: float = 0.5,
  +    base: float = 1.0,

              [ Reject (n) ]  [ Always allow edit_file (a) ]  [ Approve (y) ]
```

Reject is focused by default, so Enter takes the safe option. "Always allow"
is scoped to **one tool**, lives in memory for **one session**, is never
written to disk, and is shown in the status bar the whole time it is active.

`edit_file` requires `old_string` to match exactly once, so an ambiguous edit
is refused rather than guessed at. `write_file` is for new files and full
rewrites; the model is told to prefer `edit_file`, which keeps diffs small and
reviewable.

Headless runs cannot prompt, so `wai chat --once` **refuses changes** unless
you pass `--yes`:

```bash
wai chat --once "bump the version" --yes
```

Writes additionally refuse anything inside `.git`, and the same secret
denylist applies --- so the model cannot create a `.env` either. Files are
written atomically (temp file, then rename), so an interrupted write leaves
the original intact.

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

## Slash commands

Anything starting with `/` is a command, not a prompt.

| | |
|---|---|
| `/help` | List commands |
| `/provider` · `/provider use <name>` | LLM provider status, or switch |
| `/key <provider>` · `/key rm <provider>` | Store a key (masked, straight to the OS keyring) |
| `/model` · `/model <id>` · `/models` | Pick or set a model |
| `/login` · `/login <cloud>` | Cloud auth status, or sign in |
| `/kube` · `/kube use <ctx>` · `/kube add <path>` | Kubernetes contexts |
| `/tools` | Tools, installed integrations, standing approvals |
| `/new` | Start a fresh session |

`wai login` and `wai kube list|use|add` do the same from the shell.

## Kubernetes

Read-only in this release. Ask a question, get a chart:

```
> what is running in the qam namespace and how is it sized?

◆ Deployment/api  3/3 ready
└─ ◇ ReplicaSet/api-6f4
   ├─ ● Pod/api-6f4-2xk  Running 1/1
   │  ├─ · → mounts PersistentVolumeClaim/data
   │  └─ · ← selects by Service/api
   └─ ● Pod/api-6f4-9dm  CrashLoopBackOff
◈ Service/api  ClusterIP
├─ · → selects 2 Pods
└─ · ← routes-to by Ingress/public
```

`k8s_topology` follows `ownerReferences` for the tree and infers the rest ---
Service selectors, Ingress backends, volume mounts, ConfigMap references, HPA
targets --- so it is a graph, not just a listing. `k8s_usage` charts requests
against limits against live usage; `k8s_top` and `k8s_storage` do the same for
node/pod metrics and volumes. Press Enter on any chart to expand it full-screen.

**The model never sees the chart.** Tools return a compact text summary for the
LLM and the visual separately, so a whole-cluster topology costs almost no
context. That is what makes this affordable rather than a novelty.

### What it will not pretend to know

- **No metrics-server, no live usage.** `k8s_top` names the missing component
  and how to install it. Requests, limits and counts still work.
- **Volume fill level is not available.** metrics-server exposes no volume
  statistics --- that needs Prometheus. `k8s_storage` charts provisioned size
  relative to the largest claim and says so. A bound PVC has requested ==
  capacity, so charting one against the other would show every volume at 100%
  and read as "full".

## Clouds

Optional extras, so you only carry what you use:

```bash
uv tool install 'wai[k8s]'          # or aws, azure, gcp, all
```

Uninstalled integrations show up in `/tools` with the command to add them,
rather than silently not being there.

`/login azure` uses a **device-code flow through `azure-identity`** and works
with no `az` installed. `/login gcp` needs either `gcloud` or a service-account
key in `GOOGLE_APPLICATION_CREDENTIALS` — Google has no device-code flow that
works without a registered client, so there is no way around that.

### Two things it does not do

**It does not modify `~/.kube/config` by default.** `/kube use` records the
context in WAI's own config, because changing your global context as a side
effect of a chat message would silently retarget every other terminal you have
open. If you want kubectl-like behaviour:

```toml
[cloud]
kube_context_scope = "global"   # default "wai"
```

or per invocation: `/kube use <ctx> --global` (and `--local` to override the
other way). The global write sets exactly one key and preserves the rest of the
file, atomically.

**It redacts secrets before the model sees them.** Tool results are transmitted
to whichever LLM provider is active, so an unredacted Kubernetes Secret would
put base64 credentials in a third party's logs with no undo. Secret payloads,
credential-shaped keys, and `{name: API_TOKEN, value: …}` pairs are replaced
with a visible marker. `[cloud] secret_redaction = false` opts out.

### Protected environments

```toml
[cloud.protected]
patterns = ["*prod*", "*production*"]   # matched case-insensitively
accounts = ["123456789012"]
mode = "confirm"                        # or "deny"
```

Matching is case-insensitive on purpose: real clusters are as likely to be
called `AKS_EU_PROD` as `prod-eu`, and a rule that misses on case is worse
than no rule. `confirm` will require typing the target's name rather than
pressing a key.

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
| `y` / `n` / `a` | at an approval prompt: approve, reject, always allow that tool |
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
- **Phase 1.75 — writes behind an approval gate.** ✅ `write_file`/`edit_file`/`delete_path`, diff-first prompts.
- **Phase 2a — DevOps foundations.** ✅ Cloud auth, contexts, redaction, protected environments, slash commands.
- **Phase 2b — Kubernetes, visualised.** ✅ Topology, usage, storage, metrics. Reads only.
- **Phase 2c–2e —** AWS, Azure/GCP, and the justified CLI fallback.
- **Phase 3 — the workflow designer.** Compose and run multi-step workflows over a shared workspace; the reason the layering above is enforced.
- **Phase 3+ —** shell execution, then the software factory built on the workflow engine.

## License

Dual licensed under either of

- MIT ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option. Unless you state otherwise, any contribution you intentionally
submit for inclusion in this work shall be dual licensed as above, without any
additional terms or conditions.
