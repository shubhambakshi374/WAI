# Branch rulesets

`main.json` is the ruleset protecting the default branch. GitHub does not read
rulesets from the repository — this file is the source of truth for what the
ruleset *should* be, so a change made in the UI can be diffed against it.

It is written in the shape the API accepts, so it can be applied directly
rather than transcribed.

## Applying it

The ruleset already exists. Update it in place, keeping its id:

```sh
id=$(gh api repos/:owner/:repo/rulesets --jq '.[] | select(.name=="protect main") | .id')
gh api --method PUT repos/:owner/:repo/rulesets/$id --input .github/rulesets/main.json
```

To create it somewhere fresh, `POST` to `repos/:owner/:repo/rulesets` instead,
or import the file at **Settings → Rules → Rulesets → New ruleset → Import**.

Check what is actually live, which is the point of keeping this file:

```sh
gh api repos/:owner/:repo/rulesets/$id --jq '{name,enforcement,bypass_actors,rules}'
```

## What it does

| Rule | Effect |
|---|---|
| `pull_request` | Direct pushes to `main` are refused. **No approval is required** — see below. Stale approvals are dismissed on push and review threads must be resolved. |
| `required_status_checks` | `check (3.14)` must pass, and the branch must be up to date with `main` first. |
| `code_quality` | Code-scanning findings at `errors` severity block the merge. |
| `non_fast_forward` | No force-pushing `main`. |
| `deletion` | `main` cannot be deleted. |

## Why zero required approvals

It was one, briefly, and it blocked the only person who could satisfy it:
**GitHub does not let you approve your own pull request.** On a repository with
a single maintainer that turns every PR into `gh pr merge --admin`, which is
worse than no rule at all — it trains you to reach for the override, and the
override skips the status checks too.

Zero approvals keeps everything that actually holds:

- changes still go through a pull request, so `main` always has a diff and a CI
  run behind it;
- CI still has to be green, which is the rule doing the real work here;
- an outside contributor still cannot merge their own PR, because they have no
  write access at all — the approval count was never what stopped them.

Set it back to `1` the day a second person has write access. That is the day it
starts protecting something.

## The bypass actor

`actor_id: 5` is the built-in **admin** repository role, so the owner can push
directly and force-push when they need to. The numeric ids for built-in roles
are not part of GitHub's documented API surface, so confirm in the UI that the
bypass list reads "Repository admin" rather than trusting the number.

Note that a bypass is not automatic: `gh pr merge` refuses with *"the base
branch policy prohibits the merge"* until you pass `--admin` explicitly. That
is a good default — the override should be a deliberate act.

## Things GitHub adds on its own

Importing this file produced a ruleset with three settings that were never in
it: the `code_quality` rule, `require_extra_approval_for_unattributed_changes`,
and an empty `required_reviewers`. They are folded in above so the file matches
what is live. Re-read the live ruleset after any UI change rather than assuming
this file is still accurate.
