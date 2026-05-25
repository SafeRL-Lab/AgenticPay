#!/bin/bash

# Change to script directory
cd "$(dirname "$0")"

# Get project root (4 levels up from script directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RESULTS_BASE="$PROJECT_ROOT/agenticpay/results/multi_buyer_multi_products_multi_seller"

# Function to find the latest result directory for a model
find_latest_result_dir() {
    local model_name="$1"
    local model_name_safe=$(echo "$model_name" | sed 's/[\/\\:]/_/g')
    local model_dir="$RESULTS_BASE/$model_name_safe"
    
    if [ -d "$model_dir" ]; then
        # Find the most recently created batch_evaluation_* directory
        # Use ls -t to sort by modification time (newest first)
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
    
    # Find the latest result directory
    local latest_dir=$(find_latest_result_dir "$model_name")
    
    if [ -n "$latest_dir" ] && [ -d "$latest_dir" ]; then
        # Save run history with task name prefix
        local history_file="$latest_dir/${task_name}_run_history.txt"
        cp "$log_file" "$history_file"
        echo "  Run history saved to: $history_file"
    fi
}

# ============================================
# Configuration: Model List
# ============================================
# Configure the model list to use here
# If the list is empty, each script will use its default model
# Example:
# MODELS=("gpt-5.4" "gemini-3.1-pro-preview" "claude-opus-4-7" "Qwen/Qwen2.5-VL-72B-Instruct" "Qwen/Qwen3-VL-32B-Instruct" "internvl3-38b")
MODELS=("gpt-5.4" "gemini-3.1-pro-preview-medium" "Qwen/Qwen3-VL-32B-Instruct" "internvl3-38b")


# ============================================
# Configuration: Task List
# ============================================
# Configure which tasks to run. Use short names from TASK_SCRIPTS (e.g. "Task9") or
# full script stems (e.g. "Task24_s21_rent_house_1") if the .py file exists.
# If the list is empty, all available Task*.py scripts in this directory are run
# (excluding *example* files such as Task1_*_example.py).
#
# 下面为「最后 20 个」任务（Task9–Task28），可按需增删或改序：
TASKS=(
    "Task29"
)

# TASKS=(
#     "Task5"
#     "Task6"
#     "Task7"
#     "Task8"
#     "Task9"
#     "Task15"
#     "Task16"
#     "Task17"
#     "Task18"
#     "Task19"
#     "Task20"
#     "Task21"
#     "Task22"
#     "Task23"
#     "Task24"
#     "Task25"
#     "Task26"
#     "Task27"
#     "Task28"
#     "Task29"
# )
# TASKS=()   # 取消上面数组并置空，则运行本目录下全部任务

# ============================================
# Execute Tasks
# ============================================
# Optional mapping from short names (TaskN) to script stem (no .py)
# Used when TASKS is non-empty; when TASKS is empty, all Task*.py are auto-discovered
declare -A TASK_SCRIPTS
TASK_SCRIPTS["Task1"]="Task1_parallel_two_buyer_two_seller_two_product_negotiation"
TASK_SCRIPTS["Task2"]="Task2_parallel_three_buyer_three_seller_two_product_negotiation"
TASK_SCRIPTS["Task3"]="Task3_sequential_two_buyer_two_seller_two_product_negotiation"
TASK_SCRIPTS["Task4"]="Task4_sequential_three_buyer_three_seller_three_product_negotiation"
TASK_SCRIPTS["Task5"]="Task5_s1_beauty_product_bundle_negotiation"
TASK_SCRIPTS["Task6"]="Task6_s2_beauty_product_bundle_negotiation"
TASK_SCRIPTS["Task7"]="Task7_s3_riflescope_epson_bundle_negotiation"
TASK_SCRIPTS["Task8"]="Task8_s4_headphones_speaker_bundle_negotiation"
TASK_SCRIPTS["Task9"]="Task9_s5_bed_wall_lantern_package_negotiation"
TASK_SCRIPTS["Task10"]="Task10_s6_bookshelf_sconce_negotiation"
TASK_SCRIPTS["Task11"]="Task11_s7_flipflops_tshirt_negotiation"
TASK_SCRIPTS["Task12"]="Task12_s8_jeans_shirt_package_negotiation"
TASK_SCRIPTS["Task13"]="Task13_s9_beverage_air_plants_package_negotiation"
TASK_SCRIPTS["Task14"]="Task14_s10_smokehouse_food_package_negotiation"
TASK_SCRIPTS["Task15"]="Task15_s11_taxi_1"
TASK_SCRIPTS["Task16"]="Task16_s12_taxi_2"
TASK_SCRIPTS["Task17"]="Task17_s13_taxi_3"
TASK_SCRIPTS["Task18"]="Task18_s14_taxi_4"
TASK_SCRIPTS["Task19"]="Task19_s15_taxi_5"
TASK_SCRIPTS["Task20"]="Task20_s16_food_delivery_1"
TASK_SCRIPTS["Task21"]="Task21_s17_food_delivery_2"
TASK_SCRIPTS["Task22"]="Task22_s18_food_delivery_3"
TASK_SCRIPTS["Task23"]="Task23_s19_food_delivery_4"
TASK_SCRIPTS["Task24"]="Task24_s20_food_delivery_5"
TASK_SCRIPTS["Task25"]="Task25_s21_rent_house_1"
TASK_SCRIPTS["Task26"]="Task26_s22_rent_house_2"
TASK_SCRIPTS["Task27"]="Task27_s23_rent_house_3"
TASK_SCRIPTS["Task28"]="Task28_s24_rent_house_4"
TASK_SCRIPTS["Task29"]="Task29_s25_rent_house_5"

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
        # Skip example entrypoints and any other *example* scripts
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
    echo "TASKS list is empty. Auto-discovering all Task scripts in this directory..."
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
            # Allow passing full script stem, e.g. Task15_s11_taxi_1
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

echo "Tasks to run: ${#TASKS_TO_RUN[@]}"
for script_name in "${TASKS_TO_RUN[@]}"; do
    echo "  - $script_name"
done
echo ""

if [ ${#MODELS[@]} -eq 0 ]; then
    # Model list is empty: use default behavior (each script uses its own default model)
    echo "Running selected tasks (using default models)..."
    
    # Create temporary log file
    TEMP_LOG=$(mktemp)
    
    # Run each task with tee to capture output
    for script_name in "${TASKS_TO_RUN[@]}"; do
        if [ -f "${script_name}.py" ]; then
            echo ""
            echo "Running ${script_name}..."
            python "${script_name}.py" 2>&1 | tee "$TEMP_LOG"
        else
            echo "Warning: ${script_name}.py not found, skipping..."
        fi
    done
    
    # Clean up
    rm -f "$TEMP_LOG"
else
    # Model list is provided: loop through each model and run selected tasks for each model
    echo "Running selected tasks with the following model list: ${MODELS[*]}"
    for model in "${MODELS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Using model: $model"
        echo "=========================================="
        echo ""
        
        # Create temporary log file for this model's tasks
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
        
        # Clean up temporary log file
        rm -f "$TEMP_LOG"
        
        echo ""
        echo "Completed all tasks for model: $model"
        echo ""
    done
    echo "All tasks for all models completed!"
fi
