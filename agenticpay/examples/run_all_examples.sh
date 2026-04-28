#!/bin/bash

# claude-opus-4-5-20251101

# Change to script directory
cd "$(dirname "$0")"

# ============================================
# Run all run_all_tasks.sh scripts in subdirectories
# ============================================

# Get the base directory
BASE_DIR="$(pwd)"

# Directories to skip (names must match subdir entries, usually with trailing slash: "foo/")
SKIP_DIRS=(
    "single_buyer_product_seller/"
)

should_skip_dir() {
    local candidate="$1"
    local s
    for s in "${SKIP_DIRS[@]}"; do
        [[ -z "$s" ]] && continue
        [[ "$candidate" == "$s" ]] && return 0
    done
    return 1
}

# Find all subdirectories containing run_all_tasks.sh
echo "=========================================="
echo "Searching for run_all_tasks.sh in subdirectories..."
echo "=========================================="
echo ""

# Array to store directories with run_all_tasks.sh
DIRS=()

# Subdirectories: reverse lexicographic order (opposite of */ expansion)
shopt -s nullglob
ALL_SUBDIRS=(*/)
shopt -u nullglob

# Find all directories containing run_all_tasks.sh
if [ ${#ALL_SUBDIRS[@]} -gt 0 ]; then
    mapfile -t SORTED < <(printf '%s\n' "${ALL_SUBDIRS[@]}" | LC_ALL=C sort -r)
    for dir in "${SORTED[@]}"; do
        if [ -f "${dir}run_all_tasks.sh" ]; then
            if should_skip_dir "$dir"; then
                echo "Skip (SKIP_DIRS): ${dir}run_all_tasks.sh"
                continue
            fi
            DIRS+=("$dir")
            echo "Found: ${dir}run_all_tasks.sh"
        fi
    done
fi

echo ""
echo "Found ${#DIRS[@]} directories with run_all_tasks.sh"
echo ""

if [ ${#DIRS[@]} -eq 0 ]; then
    echo "No run_all_tasks.sh scripts found in subdirectories!"
    exit 1
fi

# Execute each run_all_tasks.sh
for dir in "${DIRS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Executing: ${dir}run_all_tasks.sh"
    echo "=========================================="
    echo ""
    
    # Change to the subdirectory and execute the script
    cd "${BASE_DIR}/${dir}"
    bash run_all_tasks.sh
    
    # Check if the script executed successfully
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to execute ${dir}run_all_tasks.sh"
        echo "Continuing with next directory..."
    fi
    
    # Return to base directory
    cd "$BASE_DIR"
    
    echo ""
    echo "Completed: ${dir}run_all_tasks.sh"
    echo ""
done

echo "=========================================="
echo "All run_all_tasks.sh scripts completed!"
echo "=========================================="
