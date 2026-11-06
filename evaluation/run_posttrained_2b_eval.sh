#!/bin/bash
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare

set -eo pipefail

# PBS executes a spool copy, so BASH_SOURCE cannot locate the project.
export EVAL_DIR=/lus/flare/projects/datascience/seonghapark/LongRoPE/evaluation
export EVAL_VENV=/lus/flare/projects/datascience/seonghapark/llm_evaluation/.venv-post2b
export EVAL_FRAMEWORKS="${EVAL_FRAMEWORKS:-frameworks/2026.1.0}"
export HARNESS_DIR=/lus/flare/projects/datascience/seonghapark/llm_evaluation/lm-evaluation-harness
module load "${EVAL_FRAMEWORKS}"
if [[ ! -f "${EVAL_VENV}/bin/activate" ]]; then
    echo "[ERROR] Missing ${EVAL_VENV}; follow the post-trained 2B setup in README.md." >&2
    exit 1
fi
source "${EVAL_VENV}/bin/activate"

ASSETS="${ASSETS:-/lus/flare/projects/datascience/seonghapark/assets}"
export PYTHONPATH="${EVAL_DIR}/../runtime:${HARNESS_DIR}"
export HF_HOME="${EVAL_DIR}/../experiments/prolong128k/cache/huggingface"
export HF_DATASETS_CACHE="${EVAL_DIR}/../experiments/prolong128k/cache/datasets"
export VLLM_CACHE_ROOT="${EVAL_DIR}/../experiments/prolong128k/cache/vllm_eval"
export TORCHINDUCTOR_CACHE_DIR="${EVAL_DIR}/../experiments/prolong128k/cache/inductor_eval"
export NLTK_DATA="${NLTK_DATA:-${ASSETS}/datasets/nltk_data}"
export LM_EVAL_URL_CACHE="${LM_EVAL_URL_CACHE:-${ASSETS}/datasets/ruler_url_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=hydra
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-${HTTP_PROXY}}"
export http_proxy="${HTTP_PROXY}" https_proxy="${HTTPS_PROXY}"
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"

cd "${EVAL_DIR}"
if [[ "${AUTO_CHAIN:-0}" == "1" ]]; then
    exec python -u "${EVAL_DIR}/chain_posttrained_2b.py" "$@"
fi
python -u "${EVAL_DIR}/eval_posttrained_2b.py" "$@"
