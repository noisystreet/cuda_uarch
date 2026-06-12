#!/usr/bin/env bash
# install-precommit.sh — Install git pre-commit hook for code formatting
set -euo pipefail

HOOK_DIR="$(git rev-parse --git-dir)/hooks"
HOOK_PATH="${HOOK_DIR}/pre-commit"

cat > "$HOOK_PATH" << 'HOOK'
#!/usr/bin/env bash
# pre-commit hook — auto-format staged .cu, .cuh, .h, .hpp, .cpp, .cc files
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

if ! command -v clang-format &>/dev/null; then
    echo -e "${RED}[ERROR] clang-format not found. Install it or skip with --no-verify.${NC}"
    exit 1
fi

# Check .clang-format exists
if [ ! -f ".clang-format" ]; then
    echo -e "${RED}[ERROR] .clang-format not found in project root.${NC}"
    exit 1
fi

# Get staged files matching our extensions
FILES=$(git diff --cached --name-only --diff-filter=ACMR \
    | grep -E '\.(cu|cuh|h|hpp|cpp|cc)$' \
    || true)

if [ -z "$FILES" ]; then
    exit 0
fi

echo -e "${GREEN}[INFO] Formatting staged files...${NC}"
FAILED=0
for FILE in $FILES; do
    if [ ! -f "$FILE" ]; then
        continue
    fi
    clang-format -i --style=file "$FILE"
    git add "$FILE"
    echo -e "  ${GREEN}[OK]${NC} $FILE"
done

if [ "$FAILED" -ne 0 ]; then
    echo -e "${RED}[ERROR] Some files failed to format.${NC}"
    exit 1
fi

echo -e "${GREEN}[INFO] All files formatted successfully.${NC}"
HOOK

chmod +x "$HOOK_PATH"
echo -e "\033[0;32m[OK]\033[0m Pre-commit hook installed at $HOOK_PATH"
