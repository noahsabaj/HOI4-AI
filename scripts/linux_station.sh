#!/bin/sh
# record-ai on a Linux fleet node's own HOI4, through the X11 worker (hoi4_arena.xworker),
# on the node's station display. Arguments are record-ai's, output folder first:
#
#   fleet run --on <linux-node> --name hoi4-ai-linux -- sh scripts/linux_station.sh \
#       artifacts/record-NAME --minutes 240 --player scripted \
#       --mod artifacts/mods/arena-12x8-v4 --speeds 5
#
# The pushed folder needs the arena and the screen data beside the code:
# artifacts/mods/<arena>, artifacts/screens-1080p and artifacts/calibration-1080p.
# The game keeps its settings, logs and saves in artifacts/hoi4-user (-userdir).
set -eu
export DISPLAY="${DISPLAY:-:0}"
exec uv run --frozen python -m hoi4_arena record-ai "$@"
