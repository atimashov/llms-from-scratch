#!/bin/bash
set -x   # print commands as they run

# forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_small_cl_128 python p2_efficiency/run_profile.py --num-layers 12 --d-model 768 --d-ff 3072 --num-heads 12 --context-length 128 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_small_cl_256 python p2_efficiency/run_profile.py --num-layers 12 --d-model 768 --d-ff 3072 --num-heads 12 --context-length 256 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_small_cl_512 python p2_efficiency/run_profile.py --num-layers 12 --d-model 768 --d-ff 3072 --num-heads 12 --context-length 512 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_small_cl_1024 python p2_efficiency/run_profile.py --num-layers 12 --d-model 768 --d-ff 3072 --num-heads 12 --context-length 1024 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_medium_cl_128 python p2_efficiency/run_profile.py --num-layers 24 --d-model 1024 --d-ff 4096 --num-heads 16 --context-length 128 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_medium_cl_256 python p2_efficiency/run_profile.py --num-layers 24 --d-model 1024 --d-ff 4096 --num-heads 16 --context-length 256 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_medium_cl_512 python p2_efficiency/run_profile.py --num-layers 24 --d-model 1024 --d-ff 4096 --num-heads 16 --context-length 512 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_medium_cl_1024 python p2_efficiency/run_profile.py --num-layers 24 --d-model 1024 --d-ff 4096 --num-heads 16 --context-length 1024 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_large_cl_128 python p2_efficiency/run_profile.py --num-layers 36 --d-model 1280 --d-ff 5120 --num-heads 20 --context-length 128 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_large_cl_256 python p2_efficiency/run_profile.py --num-layers 36 --d-model 1280 --d-ff 5120 --num-heads 20 --context-length 256 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_large_cl_512 python p2_efficiency/run_profile.py --num-layers 36 --d-model 1280 --d-ff 5120 --num-heads 20 --context-length 512 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_large_cl_1024 python p2_efficiency/run_profile.py --num-layers 36 --d-model 1280 --d-ff 5120 --num-heads 20 --context-length 1024 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_xl_cl_128 python p2_efficiency/run_profile.py --num-layers 48 --d-model 1600 --d-ff 6400 --num-heads 25 --context-length 128 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_xl_cl_256 python p2_efficiency/run_profile.py --num-layers 48 --d-model 1600 --d-ff 6400 --num-heads 25 --context-length 256 --is-amp False --mode forward 
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_xl_cl_512 python p2_efficiency/run_profile.py --num-layers 48 --d-model 1600 --d-ff 6400 --num-heads 25 --context-length 512 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_xl_cl_1024 python p2_efficiency/run_profile.py --num-layers 48 --d-model 1600 --d-ff 6400 --num-heads 25 --context-length 1024 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_xxl_cl_128 python p2_efficiency/run_profile.py --num-layers 32 --d-model 2560 --d-ff 10240 --num-heads 32 --context-length 128 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_xxl_cl_256 python p2_efficiency/run_profile.py --num-layers 32 --d-model 2560 --d-ff 10240 --num-heads 32 --context-length 256 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_xxl_cl_512 python p2_efficiency/run_profile.py --num-layers 32 --d-model 2560 --d-ff 10240 --num-heads 32 --context-length 512 --is-amp False --mode forward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_forward_xxl_cl_1024 python p2_efficiency/run_profile.py --num-layers 32 --d-model 2560 --d-ff 10240 --num-heads 32 --context-length 1024 --is-amp False --mode forward

# forward+backward
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_small_cl_128 python p2_efficiency/run_profile.py --num-layers 12 --d-model 768 --d-ff 3072 --num-heads 12 --context-length 128 --is-amp False
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_small_cl_256 python p2_efficiency/run_profile.py --num-layers 12 --d-model 768 --d-ff 3072 --num-heads 12 --context-length 256 --is-amp False
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_small_cl_512 python p2_efficiency/run_profile.py --num-layers 12 --d-model 768 --d-ff 3072 --num-heads 12 --context-length 512 --is-amp False
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_small_cl_1024 python p2_efficiency/run_profile.py --num-layers 12 --d-model 768 --d-ff 3072 --num-heads 12 --context-length 1024 --is-amp False
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_medium_cl_128 python p2_efficiency/run_profile.py --num-layers 24 --d-model 1024 --d-ff 4096 --num-heads 16 --context-length 128 --is-amp False
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_medium_cl_256 python p2_efficiency/run_profile.py --num-layers 24 --d-model 1024 --d-ff 4096 --num-heads 16 --context-length 256 --is-amp False
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_medium_cl_512 python p2_efficiency/run_profile.py --num-layers 24 --d-model 1024 --d-ff 4096 --num-heads 16 --context-length 512 --is-amp False
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_medium_cl_1024 python p2_efficiency/run_profile.py --num-layers 24 --d-model 1024 --d-ff 4096 --num-heads 16 --context-length 1024 --is-amp False
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_large_cl_128 python p2_efficiency/run_profile.py --num-layers 36 --d-model 1280 --d-ff 5120 --num-heads 20 --context-length 128 --is-amp False
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_large_cl_256 python p2_efficiency/run_profile.py --num-layers 36 --d-model 1280 --d-ff 5120 --num-heads 20 --context-length 256 --is-amp False
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_large_cl_512 python p2_efficiency/run_profile.py --num-layers 36 --d-model 1280 --d-ff 5120 --num-heads 20 --context-length 512 --is-amp False
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_large_cl_1024 python p2_efficiency/run_profile.py --num-layers 36 --d-model 1280 --d-ff 5120 --num-heads 20 --context-length 1024 --is-amp False # fails on RTX5090
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_xl_cl_128 python p2_efficiency/run_profile.py --num-layers 48 --d-model 1600 --d-ff 6400 --num-heads 25 --context-length 128 --is-amp False
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_xl_cl_256 python p2_efficiency/run_profile.py --num-layers 48 --d-model 1600 --d-ff 6400 --num-heads 25 --context-length 256 --is-amp False
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_xl_cl_512 python p2_efficiency/run_profile.py --num-layers 48 --d-model 1600 --d-ff 6400 --num-heads 25 --context-length 512 --is-amp False # fails on RTX5090
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_xl_cl_1024 python p2_efficiency/run_profile.py --num-layers 48 --d-model 1600 --d-ff 6400 --num-heads 25 --context-length 1024 --is-amp False # fails on RTX5090
uv run nsys profile -t cuda,nvtx,osrt -o p2_efficiency/nsight/model_xxl_cl_128 python p2_efficiency/run_profile.py --num-layers 32 --d-model 2560 --d-ff 10240 --num-heads 32 --context-length 128 --is-amp False # fails on RTX5090
