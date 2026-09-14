# Branch rulesets

`main.json` is the ruleset protecting the default branch. GitHub does not read
rulesets from the repository — this file is the source of truth for what the
ruleset *should* be, so a change made in the UI can be diffed against it.

## Applying it

Either import `main.json` at **Settings → Rules → Rulesets → New ruleset →
Import a ruleset**, or:

```sh
gh api --method POST repos/:owner/:repo/rulesets --input .github/rulesets/main.json
```

To update an existing one, find its id with `gh api repos/:owner/:repo/rulesets`
and `PUT` to `repos/:owner/:repo/rulesets/{id}`.

## What it does

| Rule | Effect |
|---|---|
| `pull_request` | Direct pushes to `main` are refused. One approval required, stale approvals dismissed on push, review threads must be resolved. |
| `required_status_checks` | `check (3.14)` must pass, and the branch must be up to date with `main` first. |
| `non_fast_forward` | No force-pushing `main`. |
| `deletion` | `main` cannot be deleted. |

## The bypass actor

`actor_id: 5` is the built-in **admin** repository role, so the repository
owner can still push directly and force-push when they need to. **Verify this
id after importing** — the numeric ids for built-in roles are not part of
GitHub's documented API surface, and the UI is authoritative. Open the ruleset
and confirm the bypass list reads "Repository admin".

Two consequences worth being deliberate about:

- With `bypass_mode: always`, the ruleset stops other people pushing to `main`
  but does not stop *you* doing it by accident. Set it to `pull_request` if you
  would rather route your own work through PRs too and keep only an escape
  hatch for merges.
- On a personal public repository, people who are not collaborators **already**
  cannot push — they must fork and open a pull request. The ruleset's real work
  is protecting against your own accidents, enforcing green CI before a merge,
  and covering the case where you add a collaborator later.
