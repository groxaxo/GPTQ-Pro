#!/usr/bin/env bash
# End-to-end GPTQ-Pro Ampere validation.
#
# Full RTX 3090/A100 run:
#   bash scripts/validate_gptq_pro_ampere.sh
#
# Short plumbing run:
#   bash scripts/validate_gptq_pro_ampere.sh --quick
#
# Select a specific physical GPU:
#   bash scripts/validate_gptq_pro_ampere.sh --gpu 2
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

GPU_SELECTOR="auto"
PYTHON_BIN="${PYTHON:-python3}"
OUTPUT_DIR=""
M_VALUES="1,2,3,4,5,6,8,12,16,24,32,64,128,256"
N_SIZE=4096
K_SIZE=4096
GROUP_SIZE=128
WARMUP=20
ITERATIONS=100
CHECK_COLUMNS=256
MIN_COSINE="0.999"
MAX_MEAN_ERROR="0.05"
MAX_ERROR="1.0"
QUICK=0
FULL_TESTS=0
SKIP_SANITIZER=0
RACECHECK=0
REQUIRE_SPEEDUP=0
ALLOW_BUSY_GPU=0
NATIVE_ARCH_ONLY=0

usage() {
  cat <<'EOF'
Usage: scripts/validate_gptq_pro_ampere.sh [options]

  --gpu INDEX|auto        GPU to use; default selects the least-busy GPU
  --python PATH           Python interpreter; default: python3
  --output-dir PATH       Artifact directory
  --quick                 Smaller matrices and fewer benchmark iterations
  --m-values LIST         Default: 1,2,3,4,5,6,8,12,16,24,32,64,128,256
  --n INT                 Default: 4096
  --k INT                 Default: 4096
  --group-size INT        Default: 128
  --warmup INT            Default: 20
  --iterations INT        Default: 100
  --check-columns INT     Default: 256
  --min-cosine FLOAT      Default: 0.999
  --max-mean-error FLOAT  Default: 0.05
  --max-error FLOAT       Default: 1.0
  --full-tests            Run the broader GPTQ-Pro CPU regression subset
  --skip-sanitizer        Skip compute-sanitizer memcheck
  --racecheck             Also run compute-sanitizer racecheck
  --require-speedup       Fail if AUTO is over 5% slower than legacy
  --allow-busy-gpu        Permit benchmarking on a busy GPU
  --native-arch-only      Compile only for the selected GPU architecture
  -h, --help              Show this message

The script writes a machine-readable validation-report.json and creates
VALIDATION_PASSED only when every enabled correctness gate succeeds.
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "${2:-}" ]] || die "missing value for $1"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) need_value "$@"; GPU_SELECTOR="$2"; shift 2 ;;
    --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --output-dir) need_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --m-values) need_value "$@"; M_VALUES="$2"; shift 2 ;;
    --n) need_value "$@"; N_SIZE="$2"; shift 2 ;;
    --k) need_value "$@"; K_SIZE="$2"; shift 2 ;;
    --group-size) need_value "$@"; GROUP_SIZE="$2"; shift 2 ;;
    --warmup) need_value "$@"; WARMUP="$2"; shift 2 ;;
    --iterations) need_value "$@"; ITERATIONS="$2"; shift 2 ;;
    --check-columns) need_value "$@"; CHECK_COLUMNS="$2"; shift 2 ;;
    --min-cosine) need_value "$@"; MIN_COSINE="$2"; shift 2 ;;
    --max-mean-error) need_value "$@"; MAX_MEAN_ERROR="$2"; shift 2 ;;
    --max-error) need_value "$@"; MAX_ERROR="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --full-tests) FULL_TESTS=1; shift ;;
    --skip-sanitizer) SKIP_SANITIZER=1; shift ;;
    --racecheck) RACECHECK=1; shift ;;
    --require-speedup) REQUIRE_SPEEDUP=1; shift ;;
    --allow-busy-gpu) ALLOW_BUSY_GPU=1; shift ;;
    --native-arch-only) NATIVE_ARCH_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

if [[ "$QUICK" -eq 1 ]]; then
  M_VALUES="1,2,4,5,8,16,64"
  N_SIZE=2048
  K_SIZE=2048
  GROUP_SIZE=128
  WARMUP=5
  ITERATIONS=20
  CHECK_COLUMNS=128
fi

[[ -f "${ROOT_DIR}/pyproject.toml" ]] || die "repository root not found"
[[ -f "${ROOT_DIR}/gptqmodel_ext/gptq_pro/gptq_pro_kernel_v3.cu" ]] || die "V3 CUDA source missing"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
command -v nvcc >/dev/null || die "nvcc is required"
command -v "$PYTHON_BIN" >/dev/null || die "Python not found: $PYTHON_BIN"

select_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F',' '{gsub(/ /,"",$1);gsub(/ /,"",$2);gsub(/ /,"",$3);s=$2*1000+$3;if(NR==1||s<b){b=s;i=$1}}END{print i}'
}

if [[ "$GPU_SELECTOR" == "auto" ]]; then
  GPU_INDEX="$(select_gpu)"
else
  [[ "$GPU_SELECTOR" =~ ^[0-9]+$ ]] || die "--gpu must be an index or auto"
  GPU_INDEX="$GPU_SELECTOR"
fi

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
(( GPU_INDEX < GPU_COUNT )) || die "GPU ${GPU_INDEX} does not exist"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="${ROOT_DIR}/validation-results/gptq-pro-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

exec > >(tee -a "${OUTPUT_DIR}/validation.log") 2>&1
START_EPOCH="$(date +%s)"
START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PASSED=0
trap 'rc=$?; if [[ "$PASSED" -ne 1 ]]; then echo FAILED > "${OUTPUT_DIR}/VALIDATION_FAILED"; echo "Validation failed with exit ${rc}. Artifacts: ${OUTPUT_DIR}"; fi' EXIT

cd "$ROOT_DIR"
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git branch --show-current 2>/dev/null || echo detached)"
{
  echo "sha=${GIT_SHA}"
  echo "branch=${GIT_BRANCH}"
  git status --short 2>/dev/null || true
} > "${OUTPUT_DIR}/git-state.txt"

nvidia-smi > "${OUTPUT_DIR}/nvidia-smi.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,memory.used,memory.total,utilization.gpu --format=csv,noheader > "${OUTPUT_DIR}/gpu-summary.csv"

GPU_STATE="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F',' -v g="$GPU_INDEX" '{gsub(/ /,"",$1);gsub(/ /,"",$2);gsub(/ /,"",$3);if($1==g)print $2","$3}')"
GPU_MEM="${GPU_STATE%%,*}"
GPU_UTIL="${GPU_STATE##*,}"
if [[ "$ALLOW_BUSY_GPU" -ne 1 ]] && (( GPU_MEM > 2048 || GPU_UTIL > 20 )); then
  die "GPU ${GPU_INDEX} is busy (${GPU_MEM} MiB, ${GPU_UTIL}% util); choose another GPU or use --allow-busy-gpu"
fi

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export GPTQMODEL_EXT_BUILD="${OUTPUT_DIR}/extension-cache"
export GPTQMODEL_EXT_VERBOSE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON_BIN" - <<'PY' > "${OUTPUT_DIR}/environment.json"
import json, os, platform, subprocess, sys
import torch
if not torch.cuda.is_available(): raise SystemExit("torch.cuda.is_available() is false")
major, minor = torch.cuda.get_device_capability(0)
if major < 8: raise SystemExit(f"compute capability 8.0+ required; got {major}.{minor}")
p = torch.cuda.get_device_properties(0)
info = {
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "device_name": p.name,
    "compute_capability": f"{major}.{minor}",
    "total_memory_bytes": p.total_memory,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
info["nvcc"] = subprocess.check_output(["nvcc", "--version"], text=True).strip()
print(json.dumps(info, indent=2))
PY
"$PYTHON_BIN" -m pip freeze > "${OUTPUT_DIR}/python-packages.txt"

printf '\n[1/7] Python and CPU source contracts\n'
"$PYTHON_BIN" -m py_compile gptqmodel/utils/gptq_pro.py gptqmodel/nn_modules/qlinear/gptq_pro.py scripts/benchmark_gptq_pro_kernel.py tests/kernels/test_gptq_pro_ampere_pipeline.py
"$PYTHON_BIN" -m pytest -q --confcutdir=tests/kernels tests/kernels/test_gptq_pro_ampere_pipeline.py | tee "${OUTPUT_DIR}/kernel-contract-tests.log"

if [[ "$FULL_TESTS" -eq 1 ]]; then
  "$PYTHON_BIN" -m pytest -q tests/qcfg/test_gptq_pro.py tests/kernels/test_selection.py tests/test_extension_registry.py tests/test_repository_consistency.py | tee "${OUTPUT_DIR}/broader-cpu-tests.log"
fi

printf '\n[2/7] Standalone CUDA build\n'
EXT_DIR="${ROOT_DIR}/gptqmodel_ext/gptq_pro"
VALIDATOR="${OUTPUT_DIR}/gptq_pro_validate"
GPU_CC="$("$PYTHON_BIN" -c 'import torch; a,b=torch.cuda.get_device_capability(0); print(f"{a}{b}")')"
if [[ "$NATIVE_ARCH_ONLY" -eq 1 ]]; then
  ARCH_FLAGS=("-arch=sm_${GPU_CC}")
else
  ARCH_FLAGS=(
    "-gencode" "arch=compute_80,code=sm_80"
    "-gencode" "arch=compute_86,code=sm_86"
    "-gencode" "arch=compute_87,code=sm_87"
    "-gencode" "arch=compute_87,code=compute_87"
  )
fi
(
  set -x
  nvcc -O3 -std=c++17 -lineinfo -Xptxas=-v,-warn-spills \
    -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
    "${ARCH_FLAGS[@]}" \
    "${EXT_DIR}/gptq_pro_validate.cu" "${EXT_DIR}/gptq_pro_kernel_v3.cu" \
    -o "$VALIDATOR"
) 2>&1 | tee "${OUTPUT_DIR}/standalone-build.log"

printf '\n[3/7] Standalone numerical validator\n'
"$VALIDATOR" | tee "${OUTPUT_DIR}/standalone-validator.log"

printf '\n[4/7] CUDA memory safety\n'
if [[ "$SKIP_SANITIZER" -eq 1 ]]; then
  echo "compute-sanitizer skipped"
elif command -v compute-sanitizer >/dev/null; then
  compute-sanitizer --tool memcheck --error-exitcode 86 "$VALIDATOR" 2>&1 | tee "${OUTPUT_DIR}/compute-sanitizer-memcheck.log"
  if [[ "$RACECHECK" -eq 1 ]]; then
    compute-sanitizer --tool racecheck --error-exitcode 87 "$VALIDATOR" 2>&1 | tee "${OUTPUT_DIR}/compute-sanitizer-racecheck.log"
  fi
else
  echo "WARNING: compute-sanitizer unavailable" | tee "${OUTPUT_DIR}/compute-sanitizer-memcheck.log"
fi

printf '\n[5/7] Real extension loader build/import\n'
"$PYTHON_BIN" - <<'PY' 2>&1 | tee "${OUTPUT_DIR}/runtime-loader.log"
from gptqmodel.utils.gptq_pro import ensure_gptq_pro_loaded
m = ensure_gptq_pro_loaded(verbose=True)
assert hasattr(m, "gptq_pro_gemm")
print(f"Loaded: {m.__file__}")
print(m.gptq_pro_gemm.__doc__)
PY

printf '\n[6/7] Raw numerical/performance benchmark\n'
"$PYTHON_BIN" scripts/benchmark_gptq_pro_kernel.py \
  --m-values "$M_VALUES" --n "$N_SIZE" --k "$K_SIZE" \
  --group-size "$GROUP_SIZE" --warmup "$WARMUP" \
  --iterations "$ITERATIONS" --check-columns "$CHECK_COLUMNS" \
  --output "${OUTPUT_DIR}/kernel-benchmark.json" \
  | tee "${OUTPUT_DIR}/kernel-benchmark.log"

printf '\n[7/7] Numerical gates and report\n'
export V_OUTPUT="$OUTPUT_DIR" V_MIN_COS="$MIN_COSINE" V_MAX_MEAN="$MAX_MEAN_ERROR" V_MAX_ERR="$MAX_ERROR" V_REQUIRE_SPEEDUP="$REQUIRE_SPEEDUP"
export V_START_EPOCH="$START_EPOCH" V_START_ISO="$START_ISO" V_GIT_SHA="$GIT_SHA" V_GIT_BRANCH="$GIT_BRANCH" V_GPU_INDEX="$GPU_INDEX"
"$PYTHON_BIN" - <<'PY'
import hashlib, json, os, time
from collections import defaultdict
from pathlib import Path
out = Path(os.environ["V_OUTPUT"])
data = json.loads((out / "kernel-benchmark.json").read_text())
min_cos = float(os.environ["V_MIN_COS"])
max_mean = float(os.environ["V_MAX_MEAN"])
max_err = float(os.environ["V_MAX_ERR"])
require_speedup = os.environ["V_REQUIRE_SPEEDUP"] == "1"
failures = []
by_m = defaultdict(dict)
for r in data["results"]:
    m, mode, n = int(r["m"]), str(r["mode"]), r["numerical"]
    by_m[m][mode] = r
    if float(n["cosine_similarity"]) < min_cos:
        failures.append(f"M={m} {mode}: cosine below {min_cos}")
    if float(n["mean_abs_error"]) > max_mean:
        failures.append(f"M={m} {mode}: mean error above {max_mean}")
    if float(n["max_abs_error"]) > max_err:
        failures.append(f"M={m} {mode}: max error above {max_err}")
speedups = []
for m, modes in sorted(by_m.items()):
    if "auto" in modes and "legacy" in modes:
        a = float(modes["auto"]["median_ms"])
        l = float(modes["legacy"]["median_ms"])
        s = l / a
        speedups.append({"m": m, "auto_ms": a, "legacy_ms": l, "speedup": s})
        print(f"M={m:4d}: AUTO={a:.5f} ms legacy={l:.5f} ms speedup={s:.3f}x")
        if require_speedup and a > l * 1.05:
            failures.append(f"M={m}: AUTO is over 5% slower than legacy")
gates = {
    "passed": not failures,
    "thresholds": {"min_cosine": min_cos, "max_mean_error": max_mean, "max_error": max_err, "require_speedup": require_speedup},
    "speedups": speedups,
    "failures": failures,
}
(out / "benchmark-gates.json").write_text(json.dumps(gates, indent=2))
hashes = {}
for p in sorted(out.iterdir()):
    if p.is_file() and p.name not in {"validation-report.json", "VALIDATION_PASSED", "VALIDATION_FAILED"}:
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
report = {
    "status": "PASS" if not failures else "FAIL",
    "started_at": os.environ["V_START_ISO"],
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "duration_seconds": int(time.time()) - int(os.environ["V_START_EPOCH"]),
    "git": {"sha": os.environ["V_GIT_SHA"], "branch": os.environ["V_GIT_BRANCH"]},
    "physical_gpu_index": int(os.environ["V_GPU_INDEX"]),
    "environment": json.loads((out / "environment.json").read_text()),
    "benchmark_shape": data["shape"],
    "gates": gates,
    "artifact_sha256": hashes,
}
(out / "validation-report.json").write_text(json.dumps(report, indent=2))
if failures:
    print("Validation failures:")
    for failure in failures: print(f"  - {failure}")
    raise SystemExit(1)
print("All numerical gates passed")
PY

echo PASSED > "${OUTPUT_DIR}/VALIDATION_PASSED"
rm -f "${OUTPUT_DIR}/VALIDATION_FAILED"
PASSED=1
printf '\n============================================================\n'
echo "GPTQ-Pro validation PASSED"
echo "Commit:    ${GIT_SHA}"
echo "GPU:       physical index ${GPU_INDEX}"
echo "Artifacts: ${OUTPUT_DIR}"
echo "Report:    ${OUTPUT_DIR}/validation-report.json"
printf '============================================================\n'
