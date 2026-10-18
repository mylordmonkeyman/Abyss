#!/usr/bin/env bash
set -euo pipefail
cd /actions-runner
# A named volume retains the complete installation, registration and auto-updates.
# Never overwrite a configured runner during a restart or image rebuild.
if [[ ! -e ./config.sh ]]; then
  cp -a --no-preserve=ownership /opt/actions-runner-template/. .
fi
mkdir -p _tmp
if [[ "$#" -eq 0 ]]; then
  set -- ./run.sh
fi
if [[ "$1" == './run.sh' && ! -f .runner ]]; then
  echo 'Runner is not registered. Run the one-time config command in README.md first.' >&2
  exit 1
fi
exec "$@"
