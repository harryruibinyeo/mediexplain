#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# MediExplain SG — Inference Service entrypoint
#
# Starts a vLLM server that exposes an OpenAI-compatible API.
# The API service and LangChain agent call this at:
#   POST http://inference:8000/v1/chat/completions
#
# All values are configurable via environment variables so the same
# image works in local Podman, kind, and OpenShift without rebuilding.
# ---------------------------------------------------------------------------

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
QUANTIZATION="${QUANTIZATION:-awq}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-half}"

echo "========================================="
echo " MediExplain SG — Inference Service"
echo "========================================="
echo " Model:               $MODEL_NAME"
echo " Quantization:        $QUANTIZATION"
echo " Max context length:  $MAX_MODEL_LEN tokens"
echo " GPU memory util:     $GPU_MEMORY_UTILIZATION"
echo " Data type:           $DTYPE"
echo " Port:                $PORT"
echo "========================================="
echo ""
echo "Model will be downloaded from HuggingFace on first run (~4.5GB)."
echo "Subsequent starts reuse the cached model."
echo ""

exec python -m vllm.entrypoints.openai.api_server \
    --model                   "$MODEL_NAME" \
    --quantization            "$QUANTIZATION" \
    --dtype                   "$DTYPE" \
    --max-model-len           "$MAX_MODEL_LEN" \
    --gpu-memory-utilization  "$GPU_MEMORY_UTILIZATION" \
    --host                    0.0.0.0 \
    --port                    "$PORT" \
    --served-model-name       "$MODEL_NAME"

# Prometheus /metrics endpoint is enabled by default in vLLM 0.21+
# GET http://inference:8000/metrics
