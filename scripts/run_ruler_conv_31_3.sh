cd eval/RULER/
# bash setup.sh
cd scripts

# ./run.sh llama3.1-8b-chat synthetic  --stride 16  --metric conv
RULER_RUN_TAG="${RULER_RUN_TAG:-conv_kernel_7x7_ruler_mix_sparse_guarded_long_t07_24k32k_bf16_step18000}" ./run32_3.sh llama3.1-8b-chat synthetic  --stride 8  --metric conv
# ./run.sh llama3.1-8b-chat synthetic  --stride 4  --metric conv
