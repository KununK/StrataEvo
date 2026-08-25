#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${BFCL_ROOT:-${ROOT}/eval/vendor/bfcl}"
REVISION="${BFCL_REVISION:-6ea57973c7a6097fd7c5915698c54c17c5b1b6c8}"

if [[ ! -d "${DESTINATION}/.git" ]]; then
  mkdir -p "${DESTINATION}"
  git -C "${DESTINATION}" init
  git -C "${DESTINATION}" remote add origin https://github.com/ShishirPatil/gorilla.git
fi

git -C "${DESTINATION}" fetch --depth 1 origin "${REVISION}"
git -C "${DESTINATION}" checkout --detach FETCH_HEAD
printf 'BFCL source ready at %s (%s)\n' "${DESTINATION}" "${REVISION}"
