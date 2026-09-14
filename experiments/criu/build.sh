#!/usr/bin/env bash
# Isolated Ubuntu 22.04 x86_64 build; downloads packages without installing them.
set -euo pipefail
umask 077
[[ $# == 1 ]] || { echo 'Usage: bash experiments/criu/build.sh NEW_TOOLS_DIRECTORY' >&2; exit 2; }
tools_dir="$(realpath -m "$1")"
mkdir -p "$(dirname "$tools_dir")"
mkdir "$tools_dir"

fetch_source() {
  git init -q "$tools_dir/$1"
  git -C "$tools_dir/$1" fetch -q --depth 1 "$2" "$3"
  git -C "$tools_dir/$1" checkout -q --detach FETCH_HEAD
}
fetch_source criu-4.2.1 https://github.com/checkpoint-restore/criu.git 9539417f3e3cfa4eb84c319cd71f4d52f1f08645
fetch_source cuda-checkpoint https://github.com/NVIDIA/cuda-checkpoint.git 00d5cce84c628088d6caa203fc4af40c1538b6f7
cd "$tools_dir"
apt-get download libprotobuf-c-dev protobuf-c-compiler libprotobuf-c1 libprotobuf-dev \
  protobuf-compiler libprotoc23 libprotobuf23 libprotobuf-lite23 libnl-3-dev \
  libnl-route-3-dev libnl-3-200 libnl-route-3-200 libnet1-dev libnet1 libbsd-dev libmd-dev libcap-dev libcap2
for package in ./*.deb; do
  dpkg-deb -x "$package" "$tools_dir/criu-deps"
done
python3 - "$tools_dir/criu-deps" <<'PY'
from pathlib import Path
import sys
deps = Path(sys.argv[1])
for path in deps.rglob('*.pc'):
    path.write_text(path.read_text().replace('prefix=/usr', 'prefix=' + str(deps / 'usr')))
PY
export PATH="$tools_dir/criu-deps/usr/bin:$PATH"
export PKG_CONFIG_PATH="$tools_dir/criu-deps/usr/lib/x86_64-linux-gnu/pkgconfig"
export CPATH="$tools_dir/criu-deps/usr/include"
export LIBRARY_PATH="$tools_dir/criu-deps/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$LIBRARY_PATH"
make -C "$tools_dir/criu-4.2.1" -j4 criu cuda_plugin > "$tools_dir/build.log" 2>&1
"$tools_dir/criu-4.2.1/criu/criu" --version
sha256sum ./*.deb > package-hashes.txt
