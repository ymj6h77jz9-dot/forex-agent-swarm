#!/usr/bin/env bash
# install-cicd.sh — KRATOS v2
# Run this once from the repo root to activate the GitHub Actions CI/CD gate.
# Requires: git, and a token with `workflow` scope.
#
set -euo pipefail

echo "[KRATOS CI] Installing GitHub Actions workflow..."
mkdir -p .github/workflows
cp ci/kratos-ci.yml .github/workflows/kratos-ci.yml
git add .github/workflows/kratos-ci.yml
git commit -m "ci: activate KRATOS v2 pre-merge gate"
git push
echo "[KRATOS CI] Done — workflow active on next PR/push."
