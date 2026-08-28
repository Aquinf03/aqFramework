#!/usr/bin/env bash
# Install aq (CLI + bundled kernel).
set -euo pipefail

INSTALL_DIR="${AQUIN_INSTALL_DIR:-$HOME/.local/share/aquin-framework}"
BRANCH="${AQUIN_BRANCH:-main}"
DEFAULT_RELEASE_URL="https://aq.aquin.app/framework/releases/latest.tar.gz"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v node >/dev/null || { echo "Node.js required (>=18)"; exit 1; }
command -v npm >/dev/null || { echo "npm required"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required"; exit 1; }

install_from_dir() {
  local root="$1"
  if [ ! -f "$root/aq/package.json" ] || [ ! -f "$root/aq/kernel/run.py" ]; then
    echo "Not a framework tree (need aq/package.json and aq/kernel/run.py): $root" >&2
    exit 1
  fi
  cd "$root/aq"
  npm install
  npm link
  echo ""
  echo "Installed. Run: aq help"
  echo "Source: $root/aq (kernel in aq/kernel/)"
  if ! command -v aq >/dev/null; then
    echo ""
    echo "If aq is not on PATH, add npm's global bin dir:"
    echo "  export PATH=\"\$(npm config get prefix)/bin:\$PATH\""
  fi
}

install_from_release() {
  local url="$1"
  command -v curl >/dev/null || { echo "curl required for release install"; exit 1; }
  command -v tar >/dev/null || { echo "tar required for release install"; exit 1; }

  mkdir -p "$INSTALL_DIR"
  local archive="$INSTALL_DIR/.release.tar.gz"
  echo "Downloading $url ..."
  curl -fsSL "$url" -o "$archive"
  tar xzf "$archive" -C "$INSTALL_DIR" --strip-components=0
  rm -f "$archive"

  install_from_dir "$INSTALL_DIR"
}

# 1) Run from your checkout: ./install.sh
if [ -f "$SCRIPT_DIR/aq/package.json" ] && [ -f "$SCRIPT_DIR/aq/kernel/run.py" ]; then
  install_from_dir "$SCRIPT_DIR"
  exit 0
fi

# 2) Explicit local path: AQUIN_SOURCE=/path/to/aqfw ./install.sh
if [ -n "${AQUIN_SOURCE:-}" ]; then
  install_from_dir "$AQUIN_SOURCE"
  exit 0
fi

# 3) Private git remote: AQUIN_REPO=git@github.com:you/aqfw.git ./install.sh
if [ -n "${AQUIN_REPO:-}" ]; then
  command -v git >/dev/null || { echo "git required when using AQUIN_REPO"; exit 1; }

  if [ -d "$INSTALL_DIR/.git" ]; then
    git -C "$INSTALL_DIR" fetch origin "$BRANCH"
    git -C "$INSTALL_DIR" checkout "$BRANCH"
    git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH" || true
  else
    git clone --depth 1 --branch "$BRANCH" "$AQUIN_REPO" "$INSTALL_DIR"
  fi

  install_from_dir "$INSTALL_DIR"
  exit 0
fi

# 4) R2 release via aq.aquin.app (default for curl | bash)
if [ "${AQUIN_NO_RELEASE:-}" != "1" ]; then
  install_from_release "${AQUIN_RELEASE_URL:-$DEFAULT_RELEASE_URL}"
  exit 0
fi

echo "Install options:" >&2
echo "  ./install.sh              (from a checkout that contains aq/)" >&2
echo "  AQUIN_SOURCE=/path/to/aqfw ./install.sh" >&2
echo "  AQUIN_REPO=<git-url> ./install.sh" >&2
echo "  curl -fsSL https://aq.aquin.app/framework/install.sh | bash" >&2
exit 1

if ! command -v aq >/dev/null; then
  echo ""
  echo "If aq is not on PATH, add npm's global bin dir:"
  echo "  export PATH=\"\$(npm config get prefix)/bin:\$PATH\""
fi
