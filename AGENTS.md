# Agent instructions

## Temporary workflow: repository-history epic

These instructions apply to work on
[EPIC-01: Temporal repository-history example](https://github.com/mylonasc/gestaltdb-examples/issues/7)
and its RH-indexed implementation issues. The specification is
[`repo-history/EPIC_01_REPO_HISTORY_EXAMPLE.md`](repo-history/EPIC_01_REPO_HISTORY_EXAMPLE.md).

- Use `dev-rh` as the epic's integration branch.
- Create implementation branches from the latest `origin/dev-rh`, preferably in
  separate worktrees. Keep changes scoped to the relevant RH issue.
- Open all epic implementation pull requests against **`dev-rh`**, not `main`.
  Set the base explicitly, for example:
  `gh pr create --base dev-rh --head <implementation-branch>`.
- Reference the relevant RH issue and include its acceptance-check results in
  each PR. Integrate the epic's implementation changes through these PRs rather
  than committing them directly to `main`.

### Required cleanup when the epic is complete

Remove this temporary workflow section once the parent epic **#7** and all
relevant implementation issues are completed and closed. This includes
**RH-01 through RH-12 (#8 through #19)** and any additional issues brought into
the epic. Verify both their acceptance criteria and their GitHub closed state;
PRs merged into a non-default branch may not automatically close linked issues.

Do not remove these instructions while any of that work remains incomplete or
open. Once the completion condition is met, remove the section in an epic cleanup
PR targeting `dev-rh`; do not carry the temporary PR-base requirement into the
post-epic workflow. If this file still contains only these temporary instructions,
delete `AGENTS.md` entirely. Preserve any unrelated guidance added in the meantime.
