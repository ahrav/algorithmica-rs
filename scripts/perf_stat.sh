#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin_path="$root_dir/target/release/perf_harness"
repeat="${PERF_REPEAT:-5}"
build=1

usage() {
  cat <<'EOF'
Usage:
  scripts/perf_stat.sh [--repeat N] [--no-build] [--] <perf_harness args>

Examples:
  scripts/perf_stat.sh --bench gcd_scalar --len 100000 --iters 100
  PERF_EVENTS=cycles,instructions scripts/perf_stat.sh --bench prefix_sum_scalar --len 1000000

Notes:
  - PERF_EVENTS is a comma-separated list of perf events.
  - Adds --report unless you pass --no-report.
EOF
}

args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repeat)
      repeat="${2:-}"
      shift 2
      ;;
    --no-build)
      build=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      args+=("$@")
      break
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

if [[ $build -eq 1 ]]; then
  cargo build --release --bin perf_harness --manifest-path "$root_dir/Cargo.toml"
elif [[ ! -x "$bin_path" ]]; then
  echo "error: missing $bin_path; rerun without --no-build" >&2
  exit 1
fi

if [[ -n "${PERF_EVENTS:-}" ]]; then
  IFS=',' read -r -a events <<< "$PERF_EVENTS"
else
  events=(
    cycles instructions branches branch-misses
    cache-references cache-misses
    L1-dcache-loads L1-dcache-load-misses
    dTLB-loads dTLB-load-misses
    stalled-cycles-frontend stalled-cycles-backend
  )
fi

event_args=()
for ev in "${events[@]}"; do
  event_args+=("-e" "$ev")
done

report_set=0
for arg in "${args[@]}"; do
  if [[ "$arg" == "--report" || "$arg" == "--no-report" ]]; then
    report_set=1
    break
  fi
done
if [[ $report_set -eq 0 ]]; then
  args+=("--report")
fi

perf stat -r "$repeat" "${event_args[@]}" -- "$bin_path" "${args[@]}"
