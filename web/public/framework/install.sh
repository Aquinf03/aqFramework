#!/usr/bin/env bash
# Install aq (CLI + bundled kernel). Use bash: curl … | bash
# Never needs sudo — links into ~/.local (not /usr/local).
set -euo pipefail

INSTALL_DIR="${AQUIN_INSTALL_DIR:-$HOME/.local/share/aquin-framework}"
BRANCH="${AQUIN_BRANCH:-main}"
DEFAULT_RELEASE_URL="https://aq.aquin.app/releases/aq-latestv.tar.gz"
# User-writable npm global prefix (avoids EACCES on /usr/local/lib/node_modules)
NPM_PREFIX="${AQUIN_NPM_PREFIX:-$HOME/.local}"

# BASH_SOURCE is unset when the script is piped: curl … | bash
script_dir() {
  local src="${BASH_SOURCE[0]:-}"
  case "$src" in
    "" | bash | /bin/bash | /usr/bin/bash | sh | /bin/sh) return 0 ;;
  esac
  if [ ! -f "$src" ]; then
    return 0
  fi
  cd "$(dirname "$src")" && pwd
}

SCRIPT_DIR="$(script_dir || true)"

command -v node >/dev/null || { echo "Node.js required (>=18)"; exit 1; }
command -v npm >/dev/null || { echo "npm required"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required"; exit 1; }

ensure_path_hint() {
  local bin="$NPM_PREFIX/bin"
  mkdir -p "$bin"
  if ! echo ":$PATH:" | grep -q ":$bin:"; then
    echo ""
    echo "Add this to your shell profile (~/.bashrc or ~/.zshrc) if 'aq' is not found:"
    echo "  export PATH=\"$bin:\$PATH\""
    echo "Then: source ~/.bashrc   # or open a new terminal"
  fi
}

link_aq() {
  local root="$1"
  mkdir -p "$NPM_PREFIX/bin" "$NPM_PREFIX/lib/node_modules"
  # Prefer npm link into user prefix (no sudo). Fall back to a plain symlink.
  if npm link --prefix "$NPM_PREFIX" >/dev/null 2>&1; then
    return 0
  fi
  # Some npm versions dislike link --prefix; install the package globally under prefix.
  if npm install -g --prefix "$NPM_PREFIX" "$root/aq" >/dev/null 2>&1; then
    return 0
  fi
  ln -sfn "$root/aq/bin/aq" "$NPM_PREFIX/bin/aq"
  chmod +x "$root/aq/bin/aq" 2>/dev/null || true
}

install_from_dir() {
  local root="$1"
  if [ ! -f "$root/aq/package.json" ] || [ ! -f "$root/aq/kernel/run.py" ]; then
    echo "Not a framework tree (need aq/package.json and aq/kernel/run.py): $root" >&2
    exit 1
  fi
  cd "$root/aq"
  npm install
  link_aq "$root"
  if [ -f "$root/aq/kernel/requirements.txt" ]; then
    echo "Installing kernel Python deps (venv)…"
    python3 -m venv "$root/aq/kernel/.venv"
    # shellcheck disable=SC1091
    "$root/aq/kernel/.venv/bin/pip" install -U pip
    "$root/aq/kernel/.venv/bin/pip" install -r "$root/aq/kernel/requirements.txt"
  fi
  echo "Installed under $root (npm prefix: $NPM_PREFIX)"
  echo "Run: aq help"
  echo "LLM/LoRA needs recipe.model (hub id). QLoRA needs CUDA + bitsandbytes."
  ensure_path_hint
  export PATH="$NPM_PREFIX/bin:$PATH"
  if command -v aq >/dev/null; then
    aq help >/dev/null 2>&1 && echo "OK: aq is on PATH for this shell." || true
  fi
  if [ -e /usr/local/lib/node_modules/aq ]; then
    echo ""
    echo "Note: an old global install exists at /usr/local/lib/node_modules/aq."
    echo "If 'aq' still fails or points at a stale build, remove it once:"
    echo "  sudo npm unlink -g aq    # or: sudo rm -rf /usr/local/lib/node_modules/aq"
  fi
}

extract_release() {
  local archive="$1"
  local dest="$2"
  # Ignore macOS libarchive xattr noise when unpacking on Linux.
  if tar --help 2>&1 | grep -q -- '--warning='; then
    tar xzf "$archive" -C "$dest" --warning=no-unknown-keyword
  else
    tar xzf "$archive" -C "$dest" 2>/dev/null || tar xzf "$archive" -C "$dest"
  fi
}

install_from_release() {
  local url="$1"
  command -v curl >/dev/null || { echo "curl required." >&2; exit 1; }
  command -v tar >/dev/null || { echo "tar required." >&2; exit 1; }

  mkdir -p "$INSTALL_DIR"
  local archive="$INSTALL_DIR/.release.tar.gz"
  echo "Installing Aquin..."
  if ! curl -fsSL "$url" -o "$archive" 2>/dev/null; then
    echo "Install unavailable. Try again later." >&2
    exit 1
  fi
  extract_release "$archive" "$INSTALL_DIR"
  rm -f "$archive"

  install_from_dir "$INSTALL_DIR"
}

# 1) Run from your checkout: ./install.sh
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/aq/package.json" ] && [ -f "$SCRIPT_DIR/aq/kernel/run.py" ]; then
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
