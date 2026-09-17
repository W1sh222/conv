cd eval/RULER/
# bash setup.sh
cd scripts
export RULER_NUM_SAMPLES="${RULER_NUM_SAMPLES:-100}"

# ./run.sh llama3.1-8b-chat synthetic  --stride 16  --metric conv
RULER_RUN_TAG="${RULER_RUN_TAG:-conv_kernel_7x7_initial_vertical_diag}" ./run32_2.sh llama3.1-8b-chat synthetic  --stride 8  --metric conv --block_topk_ratio "${BLOCK_TOPK_RATIO:-0.65}" --conv_weight_path initial_vertical_diag
# ./run.sh llama3.1-8b-chat synthetic  --stride 4  --metric conv
