#!/usr/bin/env bash
# Copy cached VideoUnmaskSwap replays from the RunPod volume to this Mac (run ON THE MAC).
# Needs a running pod (no GPU work). Get HOST and PORT from the pod's "Connect -> SSH over exposed TCP".
#   HOST=1.2.3.4 PORT=12345 bash scripts/copy_replays.sh
# Paths follow scene-graph-robot/notebooks/pilot (CACHE_DIR = notebooks/pilot/cache) unpacked at /workspace/sg.
set -euo pipefail
: "${HOST:?set HOST}"; : "${PORT:?set PORT}"
N_TRAIN=${N_TRAIN:-20}
REMOTE=/workspace/sg/notebooks/pilot/cache
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/replays"
SSH="ssh -p $PORT root@$HOST"
echo "== sizes on the pod (nothing copied yet)"
$SSH "ls $REMOTE/VideoUnmaskSwap/ep*_replay.npz 2>/dev/null | head -n $N_TRAIN | xargs -r du -ch | tail -1; \
      ls $REMOTE/VideoUnmaskSwap_test/ 2>/dev/null | head; du -sh $REMOTE/VideoUnmaskSwap_test 2>/dev/null || true"
read -r -p "Copy these into $DEST ? [y/N] " ok
[[ "$ok" == "y" ]] || exit 0
mkdir -p "$DEST/train" "$DEST/test"
FILES=$($SSH "ls $REMOTE/VideoUnmaskSwap/ep*_replay.npz | head -n $N_TRAIN")
for f in $FILES; do rsync -avz -e "ssh -p $PORT" "root@$HOST:$f" "$DEST/train/"; done
rsync -avz -e "ssh -p $PORT" "root@$HOST:$REMOTE/VideoUnmaskSwap_test/" "$DEST/test/" \
  --include='ep*_replay.npz' --include='motion.json' --exclude='*'
echo "done: $(ls "$DEST/train" | wc -l) train, $(ls "$DEST/test" | grep -c replay || true) test replays"
