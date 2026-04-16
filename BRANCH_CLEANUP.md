# Branch cleanup plan

This repository has multiple divergent feature/scratch branches. The goal is to preserve useful work without dumping everything into `main`.

## Keep

- `main`
- `integration/review-2026-04`
- `gptq-pro-cuda-kernel`
- `copilot/analyze-gptq-enhancements-v2` (only if this work is still wanted)

## Delete after preserving anything unique

- `copilot/sub-pr-2`
- `copilot/analyze-gptq-enhancements`

## Do not merge directly into `main`

- `fix/gemma4-ampere-main`

That branch is a large integration/refactor line and should be rebased/reviewed separately or cherry-picked selectively.

## Current staging

- PR #5 stages `gptq-pro-cuda-kernel` into `integration/review-2026-04`
- PR #6 stages `copilot/analyze-gptq-enhancements-v2` into `integration/review-2026-04`

## Why this structure exists

- Keeps `main` clean
- Preserves the focused CUDA-kernel work
- Preserves the only surviving Copilot GPTQ-Pro enhancement branch worth reviewing
- Avoids blindly merging duplicate or superseded scratch branches
- Leaves the large Gemma/refactor branch quarantined until intentionally handled

## Remaining manual cleanup

The current automation path can create branches and PRs but cannot directly delete remote branches. Once the staged work is reviewed, delete the superseded branches in GitHub UI or via git:

```bash
git push origin --delete copilot/sub-pr-2
git push origin --delete copilot/analyze-gptq-enhancements
```
