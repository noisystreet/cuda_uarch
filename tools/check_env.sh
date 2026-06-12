#!/usr/bin/env bash
# check_env.sh — GPU reverse-engineering tool chain verification
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass_cnt=0
fail_cnt=0
warn_cnt=0

pass() { echo -e "  ${GREEN}[PASS]${NC} $1"; ((pass_cnt++)); }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; ((fail_cnt++)); }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; ((warn_cnt++)); }

echo "==================================="
echo " GPU Reverse-Engineering Tool Check"
echo "==================================="
echo ""

# ─── 1. nvidia-smi ───────────────────────────────────────────────────
echo "--- nvidia-smi ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null | head -1
    pass "nvidia-smi"
else
    fail "nvidia-smi not found"
fi

# ─── 2. CUDA Toolkit ─────────────────────────────────────────────────
echo ""
echo "--- CUDA Toolkit ---"
if command -v nvcc &>/dev/null; then
    nvcc --version | tail -4
    pass "nvcc"
else
    fail "nvcc not found"
fi

# ─── 3. cuobjdump ────────────────────────────────────────────────────
if command -v cuobjdump &>/dev/null; then
    pass "cuobjdump"
else
    fail "cuobjdump not found"
fi

# ─── 4. nvdisasm ─────────────────────────────────────────────────────
if command -v nvdisasm &>/dev/null; then
    pass "nvdisasm"
else
    fail "nvdisasm not found"
fi

# ─── 5. Nsight Compute ───────────────────────────────────────────────
echo ""
echo "--- Profiling Tools ---"
# ncu is the Nsight Compute CLI
if command -v ncu &>/dev/null; then
    ncu --version 2>/dev/null || ncu --help 2>/dev/null | head -1
    pass "ncu (Nsight Compute)"
else
    warn "ncu not found (optional, for profiling)"
fi

if command -v nsys &>/dev/null; then
    pass "nsys (Nsight Systems)"
else
    warn "nsys not found (optional)"
fi

# ─── 6. Python data stack ────────────────────────────────────────────
echo ""
echo "--- Python Data Stack ---"
PY_OK=true
for pkg in numpy pandas matplotlib; do
    if python3 -c "import $pkg" 2>/dev/null; then
        pass "python3 $pkg"
    else
        warn "python3 $pkg not installed (optional, for plotting)"
    fi
done

# ─── 7. Build system ─────────────────────────────────────────────────
echo ""
echo "--- Build System ---"
if command -v cmake &>/dev/null; then
    cmake --version | head -1
    pass "cmake"
else
    fail "cmake not found"
fi

if command -v make &>/dev/null; then
    pass "make"
else
    fail "make not found"
fi

# ─── 8. GPU count ────────────────────────────────────────────────────
echo ""
echo "--- GPU Count ---"
NUM_GPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 0)
if [ "$NUM_GPU" -gt 0 ]; then
    pass "GPU count: $NUM_GPU"
else
    fail "No GPU detected"
fi

# ─── Summary ─────────────────────────────────────────────────────────
echo ""
echo "==================================="
echo -e " Results: ${GREEN}$pass_cnt passed${NC}, ${RED}$fail_cnt failed${NC}, ${YELLOW}$warn_cnt warnings${NC}"
echo "==================================="

if [ "$fail_cnt" -gt 0 ]; then
    echo ""
    echo "Missing critical tools. Install via:"
    echo "  sudo apt install nvidia-cuda-toolkit  # Ubuntu/Debian"
    echo "  or download from https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

exit 0
