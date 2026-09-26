#!/bin/sh
# The Python test suite on a CPU-only Linux computer: the fleet's Linux laptops, so that a
# run takes none of this PC's CPU. From a git worktree (tracked files only: never the main
# checkout, whose ignored data folders would be copied too):
#
#   fleet submit --mirror --wait --name hoi4-ai-tests --cpus 8 --ram-gb 12 -- sh scripts/test_on_linux.sh -n 8
#
# Arguments go to pytest; without any it runs the whole suite on every core. Keep -n equal
# to --cpus: each xdist worker runs one thread.
#
# The environment is uv.lock's but for torch. The lock pins torch's CUDA 12.8 build, which
# on Linux brings about 4 GB of CUDA libraries a computer without a GPU never loads; this
# installs the same torch and torchvision versions' CPU builds (about 200 MB; torchvision's
# operators only load beside the torch they were built for) and every other package as
# locked, into .venv, which fleet leaves in place between runs. It is rebuilt when uv.lock
# changes, from uv's cache on the node, in seconds.
set -eu
cd "$(dirname "$0")/.."

lock=$(sha256sum uv.lock | cut -c1-64)
if [ "$(cat .venv/built-from-lock 2>/dev/null)" != "$lock" ]; then
    rm -rf .venv
    uv venv --quiet --python 3.12 .venv
    uv export --frozen --extra dev --no-hashes --no-emit-project --no-annotate --no-header \
        > .venv/lock.txt
    cpu=
    for name in torch torchvision; do
        version=$(grep "^$name==" .venv/lock.txt | cut -d= -f3 | cut -d+ -f1 | cut -d' ' -f1)
        cpu="$cpu $name==$version+cpu"
    done
    grep -Ev '^(torch==|torchvision==|triton|nvidia-|cuda-)' .venv/lock.txt > .venv/cpu.txt
    uv pip sync --quiet --python .venv .venv/cpu.txt
    # shellcheck disable=SC2086 # one argument per package
    uv pip install --quiet --python .venv --no-deps \
        --index-url https://download.pytorch.org/whl/cpu $cpu
    echo "$lock" > .venv/built-from-lock
fi

# Temporary files stay in the job's folder, as fleet asks, and go when the run ends.
scratch="$PWD/.pytest-tmp"
rm -rf "$scratch"
mkdir -p "$scratch/tmp"
trap 'rm -rf "$scratch"' EXIT
export TMPDIR="$scratch/tmp"
# The package is not installed in the environment: Python started by a test imports it
# from here, as pytest itself does (pyproject's pythonpath).
export PYTHONPATH="$PWD/src"
# One thread per xdist worker. Torch would otherwise start one per core in every worker,
# and fleet's CPU limit is a quota, not a set of cores, so they would fight over it.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
if [ $# -eq 0 ]; then
    set -- -n auto
fi
status=0
.venv/bin/python -m pytest -q -p no:cacheprovider --dist worksteal --basetemp "$scratch/pytest" \
    "$@" || status=$?
# The job's peak memory (page cache included), to size --ram-gb by.
peak="/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.peak"
if [ -r "$peak" ]; then
    echo "peak memory: $(($(cat "$peak") / 1048576)) MB"
fi
exit $status
