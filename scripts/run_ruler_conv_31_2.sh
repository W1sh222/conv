cd eval/RULER/
# bash setup.sh
cd scripts

# ./run.sh llama3.1-8b-chat synthetic  --stride 16  --metric conv
RULER_RUN_TAG="${RULER_RUN_TAG:-conv_llama_t065_stage3_shortmix_32k48k_bf16_ema_step2300}" ./run32_2.sh llama3.1-8b-chat synthetic  --stride 8  --metric conv
# ./run.sh llama3.1-8b-chat synthetic  --stride 4  --metric conv
