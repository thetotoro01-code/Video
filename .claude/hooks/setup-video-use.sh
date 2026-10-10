#!/bin/bash
# Reinstall the video-use skill in fresh cloud containers (per its install.md).
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

REPO="$HOME/Developer/video-use"

if [ -d "$REPO/.git" ]; then
  git -C "$REPO" pull --ff-only --quiet || true
else
  mkdir -p "$HOME/Developer"
  git clone --quiet https://github.com/browser-use/video-use "$REPO"
fi

(cd "$REPO" && uv sync --quiet)

mkdir -p "$HOME/.claude/skills"
ln -sfn "$REPO" "$HOME/.claude/skills/video-use"

# Helpers need the repo venv (requests, numpy, PIL, librosa).
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$REPO/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi
