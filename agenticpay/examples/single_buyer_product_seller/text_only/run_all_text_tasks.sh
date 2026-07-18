#!/bin/bash

# Batch runner for the TEXT-ONLY single-buyer/single-seller negotiation tasks.
# These are LLM-only variants (no product image) of Task4..Task28, meant for users
# running text models (CustomLLM / VLLMLLM / etc.) without VLM capability.

# Change to script directory
cd "$(dirname "$0")"

# Get project root (5 levels up: this script lives one dir deeper, under text_only/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RESULTS_BASE="$PROJECT_ROOT/agenticpay/results/single_buyer_product_seller_text"

# Function to find the latest result directory for a model
find_latest_result_dir() {
    local model_name="$1"
    local model_name_safe=$(echo "$model_name" | sed 's/[\/\\:]/_/g')
    local model_dir="$RESULTS_BASE/$model_name_safe"

    if [ -d "$model_dir" ]; then
        # Find the most recently created batch_evaluation_* directory (newest first)
        ls -td "$model_dir"/batch_evaluation_* 2>/dev/null | head -1
    fi
}

# Function to save run history to the latest result directory
save_run_history() {
    local log_file="$1"
    local model_name="$2"
    local task_name="$3"

    if [ ! -f "$log_file" ]; then
        return
    fi

    local latest_dir=$(find_latest_result_dir "$model_name")

    if [ -n "$latest_dir" ] && [ -d "$latest_dir" ]; then
        local history_file="$latest_dir/${task_name}_run_history.txt"
        cp "$log_file" "$history_file"
        echo "  Run history saved to: $history_file"
    fi
}

# ============================================
# Configuration: Model List
# ============================================
# Text-only models (passed to CustomLLM via --model). If the list is empty, each
# script uses its own default model (gemini-3-pro-all via CustomLLM).
# Example:
# MODELS=("gpt-3.5-turbo" "DeepSeek-R1" "claude-sonnet-4-5" "gemini-3-pro-all")
MODELS=()


# ============================================
# Configuration: Task List
# ============================================
# Configure which tasks to run. Use short names from TASK_SCRIPTS (e.g. "Task9") or
# full script stems (e.g. "Task24_s21_rent_house_1") if the .py file exists.
# If the list is empty, all available Task*.py scripts in this directory are run
# (excluding *example* files).
TASKS=()

# ============================================
# Execute Tasks
# ============================================
# Optional mapping from short names (TaskN) to script stem (no .py).
# Used when TASKS is non-empty; when TASKS is empty, all Task*.py are auto-discovered.
declare -A TASK_SCRIPTS
TASK_SCRIPTS["Task4"]="Task4_s1_beauty_product_negotiation"
TASK_SCRIPTS["Task5"]="Task5_s2_toothpaste_negotiation"
TASK_SCRIPTS["Task6"]="Task6_s3_riflescope_negotiation"
TASK_SCRIPTS["Task7"]="Task7_s4_headphones_negotiation"
TASK_SCRIPTS["Task8"]="Task8_s5_wall_lantern_negotiation"
TASK_SCRIPTS["Task9"]="Task9_s6_bookshelf_negotiation"
TASK_SCRIPTS["Task10"]="Task10_s7_sandals_negotiation"
TASK_SCRIPTS["Task11"]="Task11_s8_jeans_negotiation"
TASK_SCRIPTS["Task12"]="Task12_s9_beverage_negotiation"
TASK_SCRIPTS["Task13"]="Task13_s10_food_color_negotiation"
TASK_SCRIPTS["Task14"]="Task14_s11_taxi_1"
TASK_SCRIPTS["Task15"]="Task15_s12_taxi_2"
TASK_SCRIPTS["Task16"]="Task16_s13_taxi_3"
TASK_SCRIPTS["Task17"]="Task17_s14_taxi_4"
TASK_SCRIPTS["Task18"]="Task18_s15_taxi_5"
TASK_SCRIPTS["Task19"]="Task19_s16_food_delivery_1"
TASK_SCRIPTS["Task20"]="Task20_s17_food_delivery_2"
TASK_SCRIPTS["Task21"]="Task21_s18_food_delivery_3"
TASK_SCRIPTS["Task22"]="Task22_s19_food_delivery_4"
TASK_SCRIPTS["Task23"]="Task23_s20_food_delivery_5"
TASK_SCRIPTS["Task24"]="Task24_s21_rent_house_1"
TASK_SCRIPTS["Task25"]="Task25_s22_rent_house_2"
TASK_SCRIPTS["Task26"]="Task26_s23_rent_house_3"
TASK_SCRIPTS["Task27"]="Task27_s24_rent_house_4"
TASK_SCRIPTS["Task28"]="Task28_s25_rent_house_5"

# Discover all runnable Task*.py scripts in this directory (excludes *example* variants)
discover_all_task_scripts() {
    local -a found=()
    local f base
    local -a task_files=()
    shopt -s nullglob
    task_files=("$SCRIPT_DIR"/Task*.py)
    shopt -u nullglob
    for f in "${task_files[@]}"; do
        [ -f "$f" ] || continue
        base=$(basename "$f" .py)
        [[ "$base" == *example* ]] && continue
        found+=("$base")
    done
    if [ ${#found[@]} -eq 0 ]; then
        return
    fi
    # Sort by Task number (Task2 before Task10, etc.)
    mapfile -t _DISCOVERED_TASKS < <(printf '%s\n' "${found[@]}" | sort -V)
}

# Build TASKS_TO_RUN: list of script stems (no .py)
if [ ${#TASKS[@]} -eq 0 ]; then
    echo "TASKS list is empty. Auto-discovering all text-only Task scripts in this directory..."
    discover_all_task_scripts
    TASKS_TO_RUN=("${_DISCOVERED_TASKS[@]}")
    unset _DISCOVERED_TASKS
    if [ ${#TASKS_TO_RUN[@]} -eq 0 ]; then
        echo "Error: no Task*.py scripts found in $SCRIPT_DIR" >&2
        exit 1
    fi
else
    TASKS_TO_RUN=()
    for task_key in "${TASKS[@]}"; do
        script_name="${TASK_SCRIPTS[$task_key]}"
        if [ -n "$script_name" ]; then
            TASKS_TO_RUN+=("$script_name")
        elif [ -f "$SCRIPT_DIR/${task_key}.py" ]; then
            TASKS_TO_RUN+=("$task_key")
        else
            echo "Warning: Unknown task '$task_key' (not in TASK_SCRIPTS and no ${task_key}.py), skipping..." >&2
        fi
    done
    if [ ${#TASKS_TO_RUN[@]} -eq 0 ]; then
        echo "Error: no valid tasks to run after resolving TASKS." >&2
        exit 1
    fi
fi

echo "Text-only tasks to run: ${#TASKS_TO_RUN[@]}"
for script_name in "${TASKS_TO_RUN[@]}"; do
    echo "  - $script_name"
done
echo ""

if [ ${#MODELS[@]} -eq 0 ]; then
    # Model list is empty: each script uses its own default text model
    echo "Running selected text-only tasks (using default models)..."

    TEMP_LOG=$(mktemp)

    for script_name in "${TASKS_TO_RUN[@]}"; do
        if [ -f "${script_name}.py" ]; then
            echo ""
            echo "Running ${script_name}..."
            python "${script_name}.py" 2>&1 | tee "$TEMP_LOG"
        else
            echo "Warning: ${script_name}.py not found, skipping..."
        fi
    done

    rm -f "$TEMP_LOG"
else
    # Model list is provided: loop through each model and run selected tasks
    echo "Running selected text-only tasks with the following model list: ${MODELS[*]}"
    for model in "${MODELS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Using model: $model"
        echo "=========================================="
        echo ""

        TEMP_LOG=$(mktemp)

        for script_name in "${TASKS_TO_RUN[@]}"; do
            if [ -f "${script_name}.py" ]; then
                echo ""
                echo "Running ${script_name} (model: $model)..."
                python "${script_name}.py" --model "$model" 2>&1 | tee "$TEMP_LOG"
                save_run_history "$TEMP_LOG" "$model" "$script_name"
            else
                echo "Warning: ${script_name}.py not found, skipping..."
            fi
        done

        rm -f "$TEMP_LOG"

        echo ""
        echo "Completed all text-only tasks for model: $model"
        echo ""
    done
    echo "All text-only tasks for all models completed!"
fi
