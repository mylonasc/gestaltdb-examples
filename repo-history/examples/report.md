# Repository-history inspection report

Run `runbook-live` · horizon 2026-11-25T12:18:29.598826+00:00 · repository `mylonasc/tf_gnns` @ `9bfe6c19`.

Coverage: {'commits': 18, 'changes': 145, 'snapshots': 2388, 'work_items': 3, 'sections': 146, 'links': 13, 'assignments': 147} · 4 gaps (content-unsupported, github-historical-revisions-unavailable).

Factual links (provider, mentions, structural) are solid blue in the figure; inferred topical links are dashed amber.

## Q1 — What changed across the commit range?

Range `(3eaddf86, 9bfe6c19]`: 16 commits, 124 file changes (truncated)

- `7cd81bd4` changes in skill ([source](https://github.com/mylonasc/tf_gnns/commit/7cd81bd48367c652b6a43d6dddc817bd73397321))
- `0f05ec00` added an index creation step before running the tests (tests also check for agen ([source](https://github.com/mylonasc/tf_gnns/commit/0f05ec009fcc7e244b044273103ccc74546ee883))
- `f787d50f` Merge pull request #10 from mylonasc/agentic-doc-metaskill ([source](https://github.com/mylonasc/tf_gnns/commit/f787d50f5bb8fc27b2ed0a0d3a0aa0a729e91eeb))
- `92d041fb` updated version ([source](https://github.com/mylonasc/tf_gnns/commit/92d041fb199838cb712462cadb368da4a4186340))
- `3f7ea987` Merge pull request #11 from mylonasc/version_bump ([source](https://github.com/mylonasc/tf_gnns/commit/3f7ea9879666c5b04fb5dfec8bab926a26079e42))
- `7464e12d` updated uv locl ([source](https://github.com/mylonasc/tf_gnns/commit/7464e12d5751bf6b6bdb9258b05d18163f2369a4))
- `6324b89c` Merge pull request #12 from mylonasc/version_bump ([source](https://github.com/mylonasc/tf_gnns/commit/6324b89cd52dceb9963e44cb5c3e1716dd5a1698))
- `9bfe6c19` Discourage probe-script experiments in the benchmark agent prompt; trust fetched ([source](https://github.com/mylonasc/tf_gnns/commit/9bfe6c19c02723dbbf101b9e311de684a4728bb1))

## Q2 — Which work items explain those changes?

- PR/issue #11 (pull_request) via **MENTIONS**: updated version — Text references work item #11.
- PR/issue #11 (pull_request) via **CLOSED_BY**: updated version — PR #11 merged at 2026-09-14 20:46:37+00:00; merge style is non-head over 1 PR commit(s).
- PR/issue #12 (pull_request) via **MENTIONS**: updated uv locl — Text references work item #12.
- PR/issue #12 (pull_request) via **CLOSED_BY**: updated uv locl — PR #12 merged at 2026-09-14 20:51:10+00:00; merge style is non-head over 1 PR commit(s).
- PR/issue #10 (pull_request) via **MENTIONS**: Agentic doc metaskill — Text references work item #10.
- PR/issue #10 (pull_request) via **CLOSED_BY**: Agentic doc metaskill — PR #10 merged at 2026-09-14 20:34:20+00:00; merge style is non-head over 10 PR commit(s).

## Q3 — Which observed branches contain these commits?

Anchor reachable from tips: none: all observed tips postdate the anchor, and the bounded selection cannot see past its boundary parents.
Merged tip `0f05ec00` reachable from: origin/agentic-doc-metaskill.

## Q4 — How did the top topic emerge or recur?

Top topic `topic:811fe7ae972c801a5726` (17 assignments):
- 2026-11-25T12:18:29.598826+00:00 · `076d562d13dd113153aad093` · reuse conf=0.97
- 2026-11-25T12:18:29.598826+00:00 · `1e06bc4661d53aec7e01c512` · reuse conf=0.60
- 2026-11-25T12:18:29.598826+00:00 · `235668a1321f49da0f9fe42c` · reuse conf=0.66
- 2026-11-25T12:18:29.598826+00:00 · `38ead6ffc36ba7d238b01d42` · reuse conf=0.65
- 2026-11-25T12:18:29.598826+00:00 · `3d119ba42aac22251eca1962` · reuse conf=0.67
- 2026-11-25T12:18:29.598826+00:00 · `3f57ca40c54869f97b0b60ef` · reuse conf=0.60
- 2026-11-25T12:18:29.598826+00:00 · `4948f5751eb163c1fb94c5d5` · reuse conf=0.60
- 2026-11-25T12:18:29.598826+00:00 · `49e5ae1c4aea85d8c8da6490` · reuse conf=0.67
- 2026-11-25T12:18:29.598826+00:00 · `6087bcca5ce644bb7897be25` · reuse conf=0.63
- 2026-11-25T12:18:29.598826+00:00 · `634dc0b91098d3a3d3af1c4e` · reuse conf=0.63

## Q5 — Which files relate to work item #11 now?

- `.python-version` via RELATED_TO conf=1.00 · sections: no headings

## Q6 — Why was an entity labeled or connected?

Explanations for `snap:c68fc2c51be54865b55d4d35`:
- **label:derived:reuse** pages-yml — Reusing topic 'pages-yml': cosine 0.944 over matched terms ['docs', 'github', 'pages', 'workflows', 'yaml', 'yml']. (method `deterministic-tfidf-labeler`)
Explanations for work item #11:
- **link:CLOSED_BY** commit:f1418e795d48cd60f01206ed — PR #11 merged at 2026-09-14 20:46:37+00:00; merge style is non-head over 1 PR commit(s). (method `link-provider`)
- **link:MENTIONS** workitem:ee20175f246b45fecb56232a — Text references work item #11. (method `link-resolver`)
- **link:RELATED_TO** branch:55c2c8b15d18a4e6c738a15a — PR #11 targets head branch 'version_bump' (branch is deleted; tip not resolved). (method `link-provider`)
- **link:RELATED_TO** branch:95e9091d20e4af779c24a9f4 — PR #11 targets base branch 'main'. (method `link-provider`)
