#!/bin/bash
set -x   # print commands as they run

SCRIPT="python -m p2_efficiency.scripts.run_nsys_profiler"
MODE="fwd_bcwd"
BATCH_SIZE=64
USE_AMP=true

if [ "$USE_AMP" = true ]; then
  COMMON_ARGS="--is-amp --autocast-dtype bfloat16 --mode ${MODE} --batch-size ${BATCH_SIZE}"
  IS_AMP="_amp"
else
  COMMON_ARGS="--no-is-amp --autocast-dtype bfloat16 --mode ${MODE} --batch-size ${BATCH_SIZE}"
  IS_AMP=""
fi

MAIN_DIR="p2_efficiency/outputs/nsys_profiler/${MODE}"

# context lengths
CONTEXTS=(128 256 512 1024)

# attention type
ATTN_TYPES=("naive" "compiled")

# model configs: name layers d_model d_ff heads
MODELS=(
  "12 768 3072 12"
  "24 1024 4096 16"
  "36 1280 5120 20"
  "48 1600 6400 25"
  "32 2560 10240 32"
)

for ATTN_TYPE in "${ATTN_TYPES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    read LAYERS D_MODEL D_FF HEADS <<< "$MODEL"

    for SEQ_LEN in "${CONTEXTS[@]}"; do
      OUT_DIR="${MAIN_DIR}/b_${BATCH_SIZE}_s_${SEQ_LEN}_l_${LAYERS}_d_${D_MODEL}_dff_${D_FF}_h_${HEADS}"
      mkdir -p "$OUT_DIR"
      OUT="${OUT_DIR}/${ATTN_TYPE}${IS_AMP}"
      ARGS=(--attn-type "$ATTN_TYPE" --num-layers "$LAYERS" --d-model "$D_MODEL" --d-ff "$D_FF" --num-heads "$HEADS" --context-length "$SEQ_LEN")
      uv run nsys profile --force-overwrite true -t cuda,nvtx,osrt -o "$OUT" $SCRIPT "${ARGS[@]}" $COMMON_ARGS
    done
  done
done


# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/fwd_bcwd/model_small_amp_cl_128 python -m p2_efficiency.scripts.run_profile_nsys --num-layers 12 --d-model 768 --d-ff 3072 --num-heads 12 --context-length 128 --is-amp True --autocast-dtype bfloat16
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_small_amp_cl_256 python p2_efficiency/run_profile.py --num-layers 12 --d-model 768 --d-ff 3072 --num-heads 12 --context-length 256 --is-amp True --autocast-dtype bfloat16
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_small_amp_cl_512 python p2_efficiency/run_profile.py --num-layers 12 --d-model 768 --d-ff 3072 --num-heads 12 --context-length 512 --is-amp True --autocast-dtype bfloat16
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_small_amp_cl_1024 python p2_efficiency/run_profile.py --num-layers 12 --d-model 768 --d-ff 3072 --num-heads 12 --context-length 1024 --is-amp True --autocast-dtype bfloat16
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_medium_amp_cl_128 python p2_efficiency/run_profile.py --num-layers 24 --d-model 1024 --d-ff 4096 --num-heads 16 --context-length 128 --is-amp True --autocast-dtype bfloat16
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_medium_amp_cl_256 python p2_efficiency/run_profile.py --num-layers 24 --d-model 1024 --d-ff 4096 --num-heads 16 --context-length 256 --is-amp True --autocast-dtype bfloat16
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_medium_amp_cl_512 python p2_efficiency/run_profile.py --num-layers 24 --d-model 1024 --d-ff 4096 --num-heads 16 --context-length 512 --is-amp True --autocast-dtype bfloat16
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_medium_amp_cl_1024 python p2_efficiency/run_profile.py --num-layers 24 --d-model 1024 --d-ff 4096 --num-heads 16 --context-length 1024 --is-amp True --autocast-dtype bfloat16
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_large_amp_cl_128 python p2_efficiency/run_profile.py --num-layers 36 --d-model 1280 --d-ff 5120 --num-heads 20 --context-length 128 --is-amp True --autocast-dtype bfloat16
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_large_amp_cl_256 python p2_efficiency/run_profile.py --num-layers 36 --d-model 1280 --d-ff 5120 --num-heads 20 --context-length 256 --is-amp True --autocast-dtype bfloat16
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_large_amp_cl_512 python p2_efficiency/run_profile.py --num-layers 36 --d-model 1280 --d-ff 5120 --num-heads 20 --context-length 512 --is-amp True --autocast-dtype bfloat16
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_large_amp_cl_1024 python p2_efficiency/run_profile.py --num-layers 36 --d-model 1280 --d-ff 5120 --num-heads 20 --context-length 1024 --is-amp True --autocast-dtype bfloat16 # fails on RTX5090
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_xl_amp_cl_128 python p2_efficiency/run_profile.py --num-layers 48 --d-model 1600 --d-ff 6400 --num-heads 25 --context-length 128 --is-amp True --autocast-dtype bfloat16
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_xl_amp_cl_256 python p2_efficiency/run_profile.py --num-layers 48 --d-model 1600 --d-ff 6400 --num-heads 25 --context-length 256 --is-amp True --autocast-dtype bfloat16
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_xl_amp_cl_512 python p2_efficiency/run_profile.py --num-layers 48 --d-model 1600 --d-ff 6400 --num-heads 25 --context-length 512 --is-amp True --autocast-dtype bfloat16 # fails on RTX5090
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_xl_amp_cl_1024 python p2_efficiency/run_profile.py --num-layers 48 --d-model 1600 --d-ff 6400 --num-heads 25 --context-length 1024 --is-amp True --autocast-dtype bfloat16 # fails on RTX5090
# uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/outputs/nsys/model_xxl_amp_cl_128 python p2_efficiency/run_profile.py --num-layers 32 --d-model 2560 --d-ff 10240 --num-heads 32 --context-length 128 --is-amp True --autocast-dtype bfloat16  # fails on RTX5090