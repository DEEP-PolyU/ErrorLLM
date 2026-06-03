set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

BENCHMARK="bird"
NUM_CPUS=12
TIMEOUT=30.0

BASE_MODEL="${PROJECT_ROOT}/model/codes-7b"
PLM_PATH="${PROJECT_ROOT}/model/electra-large-discriminator"
TRAIN_CONFIG="${PROJECT_ROOT}/src/training/detect.yaml"
INFER_CONFIG="${PROJECT_ROOT}/src/training/infer.yaml"

log() { echo -e "\n$(date '+%H:%M:%S') [Step $1] $2"; }

# ==============================================================================
#  Phase 1  Training Data Preparation  (train split)
# ==============================================================================

TRAIN_MODE="train"
TRAIN_DB_ROOT="${PROJECT_ROOT}/database/${BENCHMARK}/${TRAIN_MODE}/${TRAIN_MODE}_databases/"
TRAIN_DATASET="${PROJECT_ROOT}/database/${BENCHMARK}/${TRAIN_MODE}/${TRAIN_MODE}.json"
TRAIN_TABLES="${PROJECT_ROOT}/database/${BENCHMARK}/${TRAIN_MODE}/${TRAIN_MODE}_tables.json"
TRAIN_GT_SQL="${PROJECT_ROOT}/database/${BENCHMARK}/${TRAIN_MODE}/${TRAIN_MODE}_gold.sql"
TRAIN_GOLD_PATH="${PROJECT_ROOT}/data/gold_sqls/${BENCHMARK}/${TRAIN_MODE}.json"
TRAIN_PRED_SQL="your train split baseline sql predictions"

TRAIN_AST_DIR="${PROJECT_ROOT}/data/finetune/ast/${BENCHMARK}"
TRAIN_QSS="${PROJECT_ROOT}/data/finetune/qss/${BENCHMARK}/qss_${TRAIN_MODE}.json"
TRAIN_DETECT_INPUT="${PROJECT_ROOT}/data/finetune/detect/${BENCHMARK}/detect_${TRAIN_MODE}.json"
TRAIN_RULE_DETECT="${PROJECT_ROOT}/output/detect/${BENCHMARK}/rule_detection_${TRAIN_MODE}.json"
SYNTH_OUTPUT="${PROJECT_ROOT}/data/finetune/synthesized/${BENCHMARK}/synthesized_${TRAIN_MODE}.json"
FT_DETECT="${PROJECT_ROOT}/data/finetune/detect/${BENCHMARK}/ft_detect.json"

log 1 "AST construction (train)"
mkdir -p "$TRAIN_AST_DIR"
python3 "${PROJECT_ROOT}/src/ast_construction.py" \
    --input_path "$TRAIN_PRED_SQL" \
    --output_dir "$TRAIN_AST_DIR" \
    --dialect sqlite

log 2 "QSS construction (train)"
mkdir -p "$(dirname "$TRAIN_QSS")"
python3 "${PROJECT_ROOT}/src/qss_construction/build_qss.py" \
    --mode scratch \
    --dataset_path "$TRAIN_DATASET" \
    --tables_path "$TRAIN_TABLES" \
    --db_dir "$TRAIN_DB_ROOT" \
    --plm_path "$PLM_PATH" \
    --output_path "$TRAIN_QSS"

log 3 "Detect input construction (train)"
mkdir -p "$(dirname "$TRAIN_DETECT_INPUT")"
python3 "${PROJECT_ROOT}/src/utils/detect_input_construction.py" \
    --qss_path "$TRAIN_QSS" \
    --ast_path "${TRAIN_AST_DIR}/ast_${TRAIN_MODE}.json" \
    --dev_path "$TRAIN_DATASET" \
    --db_root_path "$TRAIN_DB_ROOT" \
    --output_path "$TRAIN_DETECT_INPUT" \
    --num_cpus "$NUM_CPUS" \
    --meta_time_out "$TIMEOUT"

log 4 "Rule-based detection (train)"
mkdir -p "$(dirname "$TRAIN_RULE_DETECT")"
python3 "${PROJECT_ROOT}/src/rules/detect.py" \
    --input "$TRAIN_DETECT_INPUT" \
    --output "$TRAIN_RULE_DETECT" \
    --gt_sql "$TRAIN_GT_SQL" \
    --db_root "$TRAIN_DB_ROOT" \
    --timeout "$TIMEOUT"

log 5 "Data synthesis (train)"
mkdir -p "$(dirname "$SYNTH_OUTPUT")"
python3 -m src.data_synthesize.synthesize_workflow \
    --db_root "$TRAIN_DB_ROOT" \
    --gold_path "$TRAIN_GOLD_PATH" \
    --pred_path "$TRAIN_PRED_SQL" \
    --output "$SYNTH_OUTPUT" \
    --seed 42

log 6 "Construct finetuning data"
mkdir -p "$(dirname "$FT_DETECT")"
python3 "${PROJECT_ROOT}/src/construct_finetuning_data.py" \
    --synth_file "$SYNTH_OUTPUT" \
    --detect_input "$TRAIN_DETECT_INPUT" \
    --rule_detect "$TRAIN_RULE_DETECT" \
    --gold_path "$TRAIN_GOLD_PATH" \
    --db_dir "$TRAIN_DB_ROOT" \
    --output "$FT_DETECT" \
    --timeout "$TIMEOUT"

log 7 "Training"
echo "  Training config: $TRAIN_CONFIG"
echo "  Run with LlamaFactory:"
echo "    cd LlamaFactory && CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train $TRAIN_CONFIG"
echo "  (Requires 4x A100 GPUs, ~3 epochs)"

# ==============================================================================
#  Phase 2  Inference & Evaluation  (dev split)
# ==============================================================================

DEV_MODE="dev"
DEV_DB_ROOT="${PROJECT_ROOT}/database/${BENCHMARK}/${DEV_MODE}/${DEV_MODE}_databases/"
DEV_DATASET="${PROJECT_ROOT}/database/${BENCHMARK}/${DEV_MODE}/${DEV_MODE}.json"
DEV_TABLES="${PROJECT_ROOT}/database/${BENCHMARK}/${DEV_MODE}/${DEV_MODE}_tables.json"
DEV_GT_SQL="${PROJECT_ROOT}/database/${BENCHMARK}/${DEV_MODE}/${DEV_MODE}_gold.sql"
DEV_PRED_SQL="your target sql path to be refined"

DEV_AST_DIR="${PROJECT_ROOT}/data/finetune/ast/${BENCHMARK}"
DEV_QSS="${PROJECT_ROOT}/data/finetune/qss/${BENCHMARK}/qss_${DEV_MODE}.json"
DEV_DETECT_INPUT="${PROJECT_ROOT}/data/finetune/detect/${BENCHMARK}/detect_${DEV_MODE}.json"
DEV_RULE_DETECT="${PROJECT_ROOT}/output/detect/${BENCHMARK}/rule_detection_${DEV_MODE}.json"
REFINE_INPUT="${PROJECT_ROOT}/output/ast_error_detector_inference/generated_predictions.jsonl"
REFINE_OUTPUT="${PROJECT_ROOT}/output/refine/${BENCHMARK}/refined_${DEV_MODE}.json"

log 8 "AST construction (dev)"
mkdir -p "$DEV_AST_DIR"
python3 "${PROJECT_ROOT}/src/ast_construction.py" \
    --input_path "$DEV_PRED_SQL" \
    --output_dir "$DEV_AST_DIR" \
    --dialect sqlite

log 9 "QSS construction (dev)"
mkdir -p "$(dirname "$DEV_QSS")"
python3 "${PROJECT_ROOT}/src/qss_construction/build_qss.py" \
    --mode scratch \
    --dataset_path "$DEV_DATASET" \
    --tables_path "$DEV_TABLES" \
    --db_dir "$DEV_DB_ROOT" \
    --plm_path "$PLM_PATH" \
    --output_path "$DEV_QSS"

log 10 "Detect input construction (dev)"
mkdir -p "$(dirname "$DEV_DETECT_INPUT")"
python3 "${PROJECT_ROOT}/src/utils/detect_input_construction.py" \
    --qss_path "$DEV_QSS" \
    --ast_path "${DEV_AST_DIR}/ast_${DEV_MODE}.json" \
    --dev_path "$DEV_DATASET" \
    --db_root_path "$DEV_DB_ROOT" \
    --output_path "$DEV_DETECT_INPUT" \
    --num_cpus "$NUM_CPUS" \
    --meta_time_out "$TIMEOUT"

log 11 "Rule-based detection (dev)"
mkdir -p "$(dirname "$DEV_RULE_DETECT")"
python3 "${PROJECT_ROOT}/src/rules/detect.py" \
    --input "$DEV_DETECT_INPUT" \
    --output "$DEV_RULE_DETECT" \
    --gt_sql "$DEV_GT_SQL" \
    --db_root "$DEV_DB_ROOT" \
    --timeout "$TIMEOUT"

log 12 "Inference"
echo "  Inference config: $INFER_CONFIG"
echo "    cd LlamaFactory && llamafactory-cli train $INFER_CONFIG"

log 13 "Refinement (dev)"
mkdir -p "$(dirname "$REFINE_OUTPUT")"
if [ -f "$REFINE_INPUT" ]; then
    python3 -m src.refinement.pipeline \
        --input "$REFINE_INPUT" \
        --output "$REFINE_OUTPUT" \
        --model gpt-4o \
        --verbose
else
    echo "  Skipping: inference output not found at $REFINE_INPUT"
fi

log 14 "Evaluation"
if [ -f "$REFINE_OUTPUT" ]; then
    python3 "${PROJECT_ROOT}/src/evaluation.py" \
        --predicted_sql_path "$REFINE_OUTPUT" \
        --ground_truth_path "$(dirname "$DEV_GT_SQL")/" \
        --data_mode "$DEV_MODE" \
        --db_root_path "$DEV_DB_ROOT" \
        --num_cpus "$NUM_CPUS" \
        --meta_time_out "$TIMEOUT" \
        --diff_json_path "$DEV_DATASET" \
        --benchmark "$BENCHMARK"
else
    echo "  Skipping: refinement output not found at $REFINE_OUTPUT"
fi

echo ""
echo "=============================================="
echo "ErrorLLM Pipeline Complete"
echo "=============================================="
