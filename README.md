# Spurious Prompts

Official implementation of [Spurious Prompts: Can Irrelevant Prompts Steer Large Language Models?](https://arxiv.org/abs/2605.29678).

This repository studies whether system prompts that are semantically unrelated to a task can still steer large language model behavior. This source release contains the code needed to search for such prompts with a black-box evolutionary loop and evaluate discovered candidates on held-out examples.

## Method

The search loop is intentionally simple:

1. Ask a generator model for candidate spurious prompts.
2. Filter candidates that mention task-relevant concepts too directly.
3. Score candidates on a fresh training partition.
4. Mutate the best candidates over several rounds.
5. Select on a held-out validation split.
6. Report final performance on the test split.

The default generator is `Qwen/Qwen3.5-27B`. Target models are configurable and can be evaluated either with local Transformers or through managed vLLM servers.

## Benchmarks

| Benchmark | Search script | Dataset |
| --- | --- | --- |
| GSM8K | `spurious_search.py` | `openai/gsm8k` |
| MedQA | `spurious_search_medqa.py` | `GBaker/MedQA-USMLE-4-options` |
| MATH-500 | `spurious_search_math500.py` | `HuggingFaceH4/MATH-500` |

## Repository Guide

| Path | Purpose |
| --- | --- |
| `spurious_search.py` | Spurious prompt search for GSM8K. |
| `spurious_search_medqa.py` | Spurious prompt search for MedQA. |
| `spurious_search_math500.py` | Spurious prompt search for MATH-500. |
| `vllm_target_runtime.py` | Shared helpers for managed vLLM servers and batched requests. |

## Setup

Use Python 3.11 or newer on a CUDA-capable machine.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Authenticate with Hugging Face if the selected models require access:

```bash
huggingface-cli login
```

## Running Search

By default, the search scripts expect an OpenAI-compatible vLLM generator endpoint at `http://127.0.0.1:8000`. You can either start that endpoint yourself, or pass `--manage-generator-server` so the script starts and stops it on demand.

Run the default GSM8K search:

```bash
python spurious_search.py
```

Run the default MedQA search:

```bash
python spurious_search_medqa.py
```

Run the default MATH-500 search:

```bash
python spurious_search_math500.py
```

A managed-server GSM8K run:

```bash
python spurious_search.py \
  --manage-generator-server \
  --use-target-vllm \
  --manage-target-server
```
