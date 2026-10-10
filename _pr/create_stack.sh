#!/bin/bash
# Push main and the notebook branches, then open one PR per notebook, each based on the one below.
# Run from the repo root on the Mac:  bash _pr/create_stack.sh
# Creates STEM-Link-AI-Engineer-3-0-Workspace/level4-lesson3 (private) if it does not exist. Override with ORG=...
# Safe to re-run: existing PRs are left alone. macOS bash 3.2 compatible.
set -e
BRANCHES="notebook/00-bench notebook/01-metrics notebook/02-hybrid notebook/03-rerank notebook/04-tuning notebook/05-failures notebook/06-your-corpus notebook/07-breakout-help-desk"
ORG="${ORG:-STEM-Link-AI-Engineer-3-0-Workspace}"
if ! git remote get-url origin >/dev/null 2>&1; then
  if gh repo view "$ORG/level4-lesson3" >/dev/null 2>&1; then
    git remote add origin "https://github.com/$ORG/level4-lesson3.git"
  else
    gh repo create "$ORG/level4-lesson3" --private --source . --remote origin
  fi
fi
git push -u origin main
base=main
for br in $BRANCHES; do
  git push -u origin "$br"
  title=$(git log -1 --format=%s "$br")
  if gh pr view "$br" >/dev/null 2>&1; then
    echo "PR exists for $br"
  else
    gh pr create --base "$base" --head "$br" --title "$title" \
      --body "$(git log -1 --format=%b "$br" | sed '/^Co-Authored-By/,$d')

Merge bottom-up with a merge commit (not squash), then delete the branch so the next PR retargets to main.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_0152Nq4q2oHDQ65GQCZoJ5ZH"
  fi
  base=$br
done
