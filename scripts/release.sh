#!/usr/bin/env bash
# Build a framework tarball and upload to Cloudflare R2 (aqfw-releases bucket).
# Requires: wrangler logged in, bucket created (see scripts/cloudflare/releases-worker/README.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-latest}"
BUCKET="${AQUIN_R2_BUCKET:-aqfw-releases}"

command -v wrangler >/dev/null || {
  echo "wrangler CLI required: npm i -g wrangler && wrangler login" >&2
  exit 1
}

TMP="$(mktemp -d)"
ARCHIVE="$TMP/aqfw-${VERSION}.tar.gz"
trap 'rm -rf "$TMP"' EXIT

echo "Building aq (TypeScript) ..."
(cd "$ROOT/aq" && npm run build)

echo "Packing tarball ..."
tar czf "$ARCHIVE" -C "$ROOT" \
  --exclude='aq/node_modules' \
  --exclude='.git' \
  aq install.sh README.md

REMOTE="${VERSION}.tar.gz"
echo "Uploading r2://${BUCKET}/${REMOTE} ..."
wrangler r2 object put "${BUCKET}/${REMOTE}" --file="$ARCHIVE" --content-type application/gzip

if [ "$VERSION" != "latest" ]; then
  echo "Updating latest.tar.gz alias ..."
  wrangler r2 object put "${BUCKET}/latest.tar.gz" --file="$ARCHIVE" --content-type application/gzip
fi

echo ""
echo "Published:"
echo "  R2:  ${BUCKET}/${REMOTE}"
echo "  URL: https://aq.aquin.app/framework/releases/${VERSION}.tar.gz"
if [ "$VERSION" != "latest" ]; then
  echo "  URL: https://aq.aquin.app/framework/releases/latest.tar.gz"
fi
