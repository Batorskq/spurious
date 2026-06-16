#!/usr/bin/env python3
"""Search for spurious system prompts that improve MATH-500 performance.

This script mirrors spurious_search.py but targets HuggingFaceH4/MATH-500.

Dataset split (fixed, seed-controlled):
  - 400 examples  →  "training" pool for the spurious search
  - 100 examples  →  held-out test set (evaluated once at the very end)

Within the 400 training examples, the same replay-buffer partitioning as
spurious_search.py applies (controlled by --mutation-rounds, default 3):
  - (mutation_rounds + 2) equal partitions
  - One fresh chunk per training round + one held-out validation chunk.
  - Every training example is fresh data in exactly one round,
    and may appear as 10% replay in later rounds.

Grading handles:
  - Pure integers and decimals
  - LaTeX fractions  (\\frac{a}{b} → a/b)
  - Symbolic expressions (normalized string comparison)
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
import gc
import json
import multiprocessing
import os
import random
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_GENERATOR_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-7B-Instruct"
GENERATOR_OUTPUT_FORMAT = "Final answer: <answer>"
REPLAY_FRACTION = 0.1

# Fixed random seed for the 400/100 train-test split of MATH-500.
# Never change this after the first run so the split is reproducible.
MATH500_SPLIT_SEED = 42
MATH500_TRAIN_SIZE = 400
MATH500_TEST_SIZE = 100

FORBIDDEN_MATH_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bmath(?:ematics|ematical)?\b", re.IGNORECASE),
    re.compile(r"\barithmetic\b", re.IGNORECASE),
    re.compile(r"\balgebra\b", re.IGNORECASE),
    re.compile(r"\bgeometry\b", re.IGNORECASE),
    re.compile(r"\bcalculus\b", re.IGNORECASE),
    re.compile(r"\bequation(?:s)?\b", re.IGNORECASE),
    re.compile(r"\btheorem(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bproof(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bcompute\b", re.IGNORECASE),
    re.compile(r"\bcalculation(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bcalculate\b", re.IGNORECASE),
    re.compile(r"\bnumber(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bnumeric(?:al)?\b", re.IGNORECASE),
    re.compile(r"\bcount(?:ing)?\b", re.IGNORECASE),
    re.compile(r"\bquantity\b", re.IGNORECASE),
    re.compile(r"\bsum\b", re.IGNORECASE),
    re.compile(r"\bsubtract\b", re.IGNORECASE),
    re.compile(r"\bmultiply\b", re.IGNORECASE),
    re.compile(r"\bdivide\b", re.IGNORECASE),
    re.compile(r"\bfraction(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bpercentage(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bratio(?:s)?\b", re.IGNORECASE),
    re.compile(r"\binteger(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bdecimal(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bword problem(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bscratchpad\b", re.IGNORECASE),
    re.compile(r"\bcalculator\b", re.IGNORECASE),
)


@dataclass
class GPUMemorySnapshot:
    device: str
    total_bytes: int
    used_bytes: int
    free_bytes: int


@dataclass
class CandidatePrompt:
    candidate_id: str
    prompt: str
    source_attempt: int
    source_index: int
    validation_notes: List[str]


@dataclass
class ExamplePrediction:
    split: str
    example_index: int
    question: str
    gold_answer: str
    extracted_prediction: Optional[str]
    correct: bool
    raw_prediction: str


class HTTPJSONRequestError(RuntimeError):
    def __init__(self, *, code: int, url: str, body: str):
        self.code = code
        self.url = url
        self.body = body
        super().__init__(f"HTTP {code} from {url}: {body}")


_PARALLEL_TARGET_WORKER_RUNNER: Optional["TargetModelRunner"] = None
_PARALLEL_TARGET_WORKER_DEVICE: Optional[str] = None


def parse_generator_host_port(base_url: str) -> Tuple[str, int]:
    parsed = urlparse(normalize_base_url(base_url))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    return host, port


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def resolve_cuda_monitor_device(target_device: Optional[str]) -> Optional[torch.device]:
    if not torch.cuda.is_available():
        return None
    if target_device:
        try:
            device = torch.device(target_device)
            if device.type == "cuda":
                return torch.device(f"cuda:{device.index or 0}")
        except (TypeError, RuntimeError):
            pass
    return torch.device("cuda:0")


def get_gpu_memory_snapshot(device: Optional[torch.device]) -> Optional[GPUMemorySnapshot]:
    if device is None or not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        return None
    return GPUMemorySnapshot(
        device=str(device),
        total_bytes=int(total_bytes),
        used_bytes=int(total_bytes - free_bytes),
        free_bytes=int(free_bytes),
    )


def capture_settled_gpu_memory(
    device: Optional[torch.device],
    *,
    max_wait_seconds: float = 15.0,
    poll_interval_seconds: float = 0.5,
    stable_tolerance_bytes: int = 64 * 1024 * 1024,
) -> Optional[GPUMemorySnapshot]:
    snapshot = get_gpu_memory_snapshot(device)
    if snapshot is None:
        return None
    deadline = time.monotonic() + max_wait_seconds
    previous = snapshot
    stable_count = 0
    while time.monotonic() < deadline:
        time.sleep(poll_interval_seconds)
        current = get_gpu_memory_snapshot(device)
        if current is None:
            return previous
        if abs(current.used_bytes - previous.used_bytes) <= stable_tolerance_bytes:
            stable_count += 1
            if stable_count >= 2:
                return current
        else:
            stable_count = 0
        previous = current
    return previous


def summarize_gpu_memory(snapshot: Optional[GPUMemorySnapshot]) -> str:
    if snapshot is None:
        return "unavailable"
    return (
        f"used={format_gib(snapshot.used_bytes)}, "
        f"free={format_gib(snapshot.free_bytes)}, "
        f"total={format_gib(snapshot.total_bytes)} on {snapshot.device}"
    )


def tail_text_lines(path: Path, limit: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:])


class GeneratorServerManager:
    def __init__(self, *, args: argparse.Namespace, run_dir: Path, monitor_device: Optional[torch.device]) -> None:
        self.args = args
        self.base_url = normalize_base_url(args.generator_base_url)
        self.host, self.port = parse_generator_host_port(args.generator_base_url)
        self.model_name = args.generator_model
        self.api_key = args.generator_api_key
        self.monitor_device = monitor_device
        self._language_model_only_compat = False
        self.process: Optional[subprocess.Popen] = None
        self.log_handle: Optional[Any] = None
        self.current_log_path: Optional[Path] = None
        self.logs_dir = run_dir / "generator_server_logs"
        ensure_dir(self.logs_dir)
        self.memory_events: List[Dict[str, Any]] = []

    def _build_command(self) -> List[str]:
        command = [
            "vllm", "serve", self.model_name,
            "--host", self.host, "--port", str(self.port),
            "--tensor-parallel-size", str(self.args.generator_tensor_parallel_size),
            "--pipeline-parallel-size", str(self.args.generator_pipeline_parallel_size),
            "--gpu-memory-utilization", str(self.args.generator_gpu_memory_utilization),
            "--max-model-len", str(self.args.generator_max_model_len),
        ]
        if self.args.generator_max_num_seqs is not None:
            command.extend(["--max-num-seqs", str(self.args.generator_max_num_seqs)])
        if self._language_model_only_compat:
            command.insert(3, "--language-model-only")
        elif self.args.generator_task:
            command[5:5] = ["--task", self.args.generator_task]
        return command

    def _is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _check_ready(self) -> bool:
        try:
            list_vllm_models(self.base_url, self.api_key, timeout=min(5.0, self.args.generator_timeout))
            return True
        except Exception:
            return False

    def start(self, phase_name: str) -> None:
        if self._is_running():
            return
        log_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", phase_name.strip()) or "generator"
        self.current_log_path = self.logs_dir / f"{log_name}.log"
        self.log_handle = self.current_log_path.open("w", encoding="utf-8")
        while True:
            print(f"Launching generator server for {self.model_name} ({phase_name}) on http://{self.host}:{self.port}")
            self.process = subprocess.Popen(
                self._build_command(), stdout=self.log_handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
            retry_without_task = False
            for _ in range(self.args.generator_ready_retries):
                if self._check_ready():
                    print(f"Generator server is ready for {phase_name}.")
                    return
                if not self._is_running():
                    log_tail = tail_text_lines(self.current_log_path) if self.current_log_path else ""
                    if self.args.generator_task and "unrecognized arguments: --task" in log_tail:
                        print("Managed vLLM CLI rejected --task; retrying with --language-model-only.")
                        if self.log_handle is not None:
                            self.log_handle.write(
                                "\n[compat] Retrying with --language-model-only after CLI rejected --task.\n"
                            )
                            self.log_handle.flush()
                        self._language_model_only_compat = True
                        retry_without_task = True
                        break
                    tp_error = re.search(r"AssertionError:\s+(\d+)\s+is not divisible by\s+(\d+)", log_tail)
                    if tp_error:
                        numerator = int(tp_error.group(1))
                        denominator = int(tp_error.group(2))
                        valid_sizes = [str(size) for size in range(1, numerator + 1) if numerator % size == 0]
                        raise RuntimeError(
                            "Generator server exited before becoming ready "
                            f"({phase_name}). The current tensor parallel size "
                            f"({denominator}) is incompatible with a model component that has "
                            f"{numerator} attention heads. Choose a tensor parallel size that "
                            f"divides {numerator}, for example: {', '.join(valid_sizes)}.\n{log_tail}"
                        )
                    oom_error = re.search(
                        r"torch\.OutOfMemoryError: CUDA out of memory\. Tried to allocate\s+(.+?)\.\s+GPU\s+(\d+)\s+has a total capacity of\s+(.+?)\s+of which\s+(.+?)\s+is free\.",
                        log_tail,
                    )
                    if oom_error:
                        allocation = oom_error.group(1)
                        gpu_index = oom_error.group(2)
                        total_capacity = oom_error.group(3)
                        free_capacity = oom_error.group(4)
                        raise RuntimeError(
                            "Generator server exited before becoming ready "
                            f"({phase_name}) because it ran out of GPU memory while initializing "
                            f"the vLLM KV cache. It tried to allocate {allocation} on GPU {gpu_index}, "
                            f"which had only {free_capacity} free out of {total_capacity}. "
                            f"Current settings: model={self.model_name}, "
                            f"tensor_parallel_size={self.args.generator_tensor_parallel_size}, "
                            f"max_model_len={self.args.generator_max_model_len}. "
                            "Try one of: expose more GPUs and increase tensor parallel size, switch "
                            "to a smaller --generator-model, or lower --generator-max-model-len.\n"
                            f"{log_tail}"
                        )
                    max_num_seqs_error = re.search(
                        r"max_num_seqs \((\d+)\) exceeds available Mamba cache blocks \((\d+)\)",
                        log_tail,
                    )
                    if max_num_seqs_error:
                        current_max_num_seqs = int(max_num_seqs_error.group(1))
                        available_mamba_blocks = int(max_num_seqs_error.group(2))
                        suggested_limit = min(available_mamba_blocks, max(32, available_mamba_blocks - 8))
                        raise RuntimeError(
                            "Generator server exited before becoming ready "
                            f"({phase_name}) because vLLM's max_num_seqs setting "
                            f"({current_max_num_seqs}) exceeds the available Mamba cache blocks "
                            f"({available_mamba_blocks}) for this layout. "
                            "Lower --generator-max-num-seqs or increase "
                            "--generator-gpu-memory-utilization. "
                            f"A safe next try is --generator-max-num-seqs {suggested_limit}.\n"
                            f"{log_tail}"
                        )
                    raise RuntimeError(f"Generator server exited before becoming ready ({phase_name}).\n{log_tail}")
                time.sleep(self.args.generator_ready_sleep_seconds)
            if retry_without_task:
                continue
            log_tail = tail_text_lines(self.current_log_path) if self.current_log_path else ""
            self.stop(phase_name, emit_memory_log=False)
            raise RuntimeError(f"Timed out waiting for generator server ({phase_name}).\n{log_tail}")

    def stop(self, phase_name: str, *, emit_memory_log: bool = True) -> None:
        if self.process is None:
            return
        before = capture_settled_gpu_memory(self.monitor_device, max_wait_seconds=2.0)
        if emit_memory_log:
            print(f"GPU memory before unloading {self.model_name} ({phase_name}): {summarize_gpu_memory(before)}")
        if self._is_running():
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=self.args.generator_shutdown_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10.0)
        if self.log_handle is not None:
            self.log_handle.close()
        after = capture_settled_gpu_memory(self.monitor_device)
        if emit_memory_log:
            print(f"GPU memory after unloading {self.model_name} ({phase_name}): {summarize_gpu_memory(after)}")
            if before is not None and after is not None:
                freed = before.used_bytes - after.used_bytes
                print(f"Freed by removing {self.model_name}: {format_gib(freed)}")
                self.memory_events.append({
                    "phase": phase_name, "model_name": self.model_name,
                    "before": asdict(before), "after": asdict(after), "freed_bytes": freed,
                })
        self.process = None
        self.log_handle = None
        self.current_log_path = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for spurious MATH-500 prompts with a generator + target model.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for run artifacts.")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--generator-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--generator-api-key", default="EMPTY")
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--generator-timeout", type=float, default=180.0)
    parser.add_argument("--generator-temperature", type=float, default=0.9)
    parser.add_argument("--generator-top-p", type=float, default=0.95)
    parser.add_argument("--generator-max-tokens", type=int, default=2200)
    parser.add_argument("--generator-seed", type=int, default=7)
    parser.add_argument("--num-candidates", type=int, default=24)
    parser.add_argument("--prompts-per-call", type=int, default=6)
    parser.add_argument("--max-generation-attempts", type=int, default=12)
    parser.add_argument(
        "--manage-generator-server",
        action="store_true",
        help=(
            "Launch and stop the vLLM generator server on demand so the large generator model "
            "is only resident in GPU memory during prompt generation and mutation."
        ),
    )
    parser.add_argument("--generator-ready-retries", type=int, default=180)
    parser.add_argument("--generator-ready-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--generator-gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--generator-max-model-len", type=int, default=9000)
    parser.add_argument(
        "--generator-max-num-seqs",
        type=int,
        default=None,
        help="Optional vLLM --max-num-seqs cap for the managed generator server.",
    )
    parser.add_argument("--generator-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--generator-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--generator-task", default="generate",
                        help="vLLM --task value for the generator server. 'generate' skips the vision encoder on VL models.")
    parser.add_argument("--generator-shutdown-timeout", type=float, default=60.0)
    parser.add_argument(
        "--mutation-rounds",
        type=int,
        default=3,
        help=(
            "Number of evolutionary mutation rounds after the initial generation. "
            "The 400 training examples are split into (mutation_rounds + 2) equal "
            "partitions: one fresh chunk per training round + one held-out validation chunk."
        ),
    )

    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--target-device", default=None)
    parser.add_argument("--target-attn-implementation", default=None)
    parser.add_argument(
        "--target-worker-devices",
        default=None,
        help=(
            "Comma-separated devices for replicated target-model evaluation, e.g. "
            "'cuda:0,cuda:1,cuda:2', or 'auto' for all visible CUDA devices. When set, "
            "the script loads one full target-model replica per device and scores "
            "different candidates in parallel."
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument(
        "--eval-thinking-tokens",
        type=int,
        default=400,
        help="Tokens for chain-of-thought phase. 0 disables two-phase generation.",
    )
    parser.add_argument(
        "--eval-answer-tokens",
        type=int,
        default=128,
        help="Max tokens for the answer phase after 'Final answer:' is injected.",
    )

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument(
        "--save-subset-predictions",
        action="store_true",
        help="Save per-example predictions for the train subset stage as JSONL.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_target_worker_devices(spec: Optional[str]) -> List[str]:
    if spec is None:
        return []
    text = str(spec).strip()
    if not text:
        return []
    if text.lower() == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("--target-worker-devices auto requested, but CUDA is not available.")
        count = torch.cuda.device_count()
        if count < 1:
            raise RuntimeError("--target-worker-devices auto requested, but no CUDA devices are visible.")
        return [f"cuda:{idx}" for idx in range(count)]
    devices = [part.strip() for part in text.split(",") if part.strip()]
    if not devices:
        raise RuntimeError("--target-worker-devices was provided, but no devices were parsed from it.")
    if len(set(devices)) != len(devices):
        raise RuntimeError(f"--target-worker-devices contains duplicates: {text}")
    return devices


def normalize_base_url(base_url: str) -> str:
    base = str(base_url).strip().rstrip("/")
    if not base:
        raise ValueError("generator_base_url must be non-empty")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def vllm_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def http_json_request(
    *,
    url: str,
    method: str,
    timeout: float,
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise HTTPJSONRequestError(code=exc.code, url=url, body=err_body) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Server returned non-JSON body from {url}: {body[:500]}") from exc


def parse_vllm_context_limit_error(err_text: str) -> Optional[Tuple[int, int, int]]:
    match = re.search(
        r"maximum context length is (\d+) tokens.*?requested (\d+) output tokens.*?"
        r"prompt contains at least (\d+) input tokens",
        err_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    max_context = int(match.group(1))
    requested_output_tokens = int(match.group(2))
    input_tokens = int(match.group(3))
    return max_context, requested_output_tokens, input_tokens


def list_vllm_models(base_url: str, api_key: Optional[str], timeout: float) -> List[str]:
    response = http_json_request(
        url=f"{base_url}/models", method="GET", timeout=timeout, headers=vllm_headers(api_key),
    )
    return [str(item["id"]) for item in response.get("data", []) if isinstance(item, dict) and "id" in item]


def resolve_generator_model_name(
    requested_model: str, *, base_url: str, api_key: Optional[str], timeout: float,
) -> str:
    available = list_vllm_models(base_url, api_key, timeout)
    if not available:
        raise RuntimeError(f"No models were listed at {base_url}/models.")
    if requested_model in available:
        return requested_model
    if len(available) == 1:
        return available[0]
    raise RuntimeError(
        f"Requested generator model {requested_model!r} not found. Available: {', '.join(available)}"
    )


def request_vllm_completion(
    *,
    base_url: str,
    api_key: Optional[str],
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    seed: Optional[int] = None,
) -> str:
    current_max_tokens = int(max_tokens)
    completion_url = f"{base_url}/completions"
    headers = vllm_headers(api_key)
    context_retry_safety_margin = 8

    for attempt in range(2):
        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": current_max_tokens,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "n": 1,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        try:
            response = http_json_request(
                url=completion_url, method="POST", timeout=timeout,
                headers=headers, payload=payload,
            )
            break
        except HTTPJSONRequestError as exc:
            context_error = parse_vllm_context_limit_error(exc.body)
            if exc.code != 400 or context_error is None or attempt > 0:
                raise
            max_context, requested_output_tokens, input_tokens = context_error
            if requested_output_tokens != current_max_tokens:
                raise RuntimeError(
                    "vLLM rejected the completion request for exceeding context limits, "
                    f"but the reported output budget ({requested_output_tokens}) did not match "
                    f"the request ({current_max_tokens}). Original error: {exc.body}"
                ) from exc
            adjusted_max_tokens = max_context - input_tokens - context_retry_safety_margin
            if adjusted_max_tokens < 1:
                raise RuntimeError(
                    "Generator prompt already fills the model context window and cannot leave "
                    f"room for any output. max_context={max_context}, input_tokens={input_tokens}. "
                    "Reduce prompt length or increase --generator-max-model-len."
                ) from exc
            if adjusted_max_tokens >= current_max_tokens:
                raise
            print(
                "Generator request exceeded the context budget; "
                f"retrying with max_tokens={adjusted_max_tokens} "
                f"(was {current_max_tokens}, input_tokens={input_tokens}, max_context={max_context})."
            )
            current_max_tokens = adjusted_max_tokens
    else:
        raise RuntimeError("Completion retry loop exited unexpectedly.")

    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"No choices in response: {json.dumps(response)[:1000]}")
    choice = choices[0]
    if not isinstance(choice, dict) or "text" not in choice:
        raise RuntimeError(f"Choice missing text: {json.dumps(response)[:1000]}")
    return str(choice["text"])


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

def build_generator_prompt(batch_size: int, existing_prompts: Sequence[str]) -> str:
    variety_hints = [
        "a Latin scribe",
        "a monastic archivist",
        "a travel diarist",
        "a stage director",
        "a botanist writing field notes",
        "a museum curator",
        "a quiet storyteller",
        "a court historian",
        "a librarian of lost cities",
        "a translator of old letters",
    ]
    return f"""
You are designing system prompts for a separate assistant.

Goal:
- Produce {batch_size} candidate system prompts that are intentionally spurious.
- The assistant will later receive MATH competition problems, but your prompt text itself must have nothing to do with mathematics.

Hard constraints for every candidate prompt:
- Do not mention mathematics or closely related ideas.
- Do not use words about arithmetic, calculation, equations, numbers, counting, algebra, geometry, proofs, or similar topics.
- Do not mention code, Python, calculators, scratchpads, or verifiers.
- The prompt must still be coherent and usable as a real system prompt.
- The prompt may use a non-mathematical persona, story, ritual, style guide, or language flavor.
- Include a final formatting rule telling the assistant to end with the exact final line: {GENERATOR_OUTPUT_FORMAT}
- Keep each prompt between about 60 and 140 words.
- Make the prompts meaningfully different from each other.

Preferred vibes:
- {", ".join(variety_hints)}

Already accepted prompts to avoid duplicating too closely:
{json.dumps(list(existing_prompts[-5:]), ensure_ascii=False)}

Return only valid JSON with this schema:
{{"prompts": ["prompt 1", "prompt 2"]}}

Do not use markdown fences.
Do not include commentary before or after the JSON.
""".strip()


def build_mutation_prompt(
    batch_size: int,
    seed_prompts: Sequence[str],
    existing_prompts: Sequence[str],
) -> str:
    return f"""
You are designing system prompts for a separate assistant.

Goal:
- Produce {batch_size} new candidate system prompts by mutating the seed prompts below.
- Each new prompt should vary the persona, style, or narrative of a seed while preserving its non-mathematical character.
- The assistant will later receive MATH competition problems, but your prompt text must have nothing to do with mathematics.

Seed prompts (top-performing — mutate their style or persona):
{json.dumps(list(seed_prompts), ensure_ascii=False)}

Hard constraints for every new candidate prompt:
- Do not mention mathematics or closely related ideas.
- Do not use words about arithmetic, calculation, equations, numbers, counting, algebra, geometry, proofs, or similar topics.
- Do not mention code, Python, calculators, scratchpads, or verifiers.
- The prompt must still be coherent and usable as a real system prompt.
- The prompt may use a non-mathematical persona, story, ritual, style guide, or language flavor.
- Include a final formatting rule telling the assistant to end with the exact final line: {GENERATOR_OUTPUT_FORMAT}
- Keep each prompt between about 60 and 140 words.
- Make the new prompts meaningfully different from the seeds and from each other.

Already accepted prompts to avoid duplicating too closely:
{json.dumps(list(existing_prompts[-10:]), ensure_ascii=False)}

Return only valid JSON with this schema:
{{"prompts": ["prompt 1", "prompt 2"]}}

Do not use markdown fences.
Do not include commentary before or after the JSON.
""".strip()


def extract_first_json_container(text: str) -> Optional[str]:
    for start_index, opening in enumerate(text):
        if opening not in "[{":
            continue
        closing = "}" if opening == "{" else "]"
        depth, in_string, escape = 0, False, False
        for index in range(start_index, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return text[start_index:index + 1]
    return None


def strip_thinking_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def parse_generator_response(text: str) -> List[str]:
    cleaned_text = strip_thinking_tags(text)
    container = extract_first_json_container(cleaned_text)
    if not container:
        return []
    try:
        payload = json.loads(container)
    except json.JSONDecodeError:
        return []
    prompts = payload.get("prompts", []) if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
    cleaned: List[str] = []
    for item in prompts:
        if isinstance(item, str):
            cleaned.append(clean_prompt_text(item))
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            cleaned.append(clean_prompt_text(item["prompt"]))
    return cleaned


def clean_prompt_text(text: str) -> str:
    lines = [line.rstrip() for line in str(text).strip().splitlines()]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines).strip())
    return cleaned


def validate_spurious_prompt(prompt: str) -> List[str]:
    notes: List[str] = []
    word_count = len(prompt.split())
    if word_count < 40:
        notes.append("too_short")
    if word_count > 220:
        notes.append("too_long")
    if "Final answer:" not in prompt:
        notes.append("missing_final_answer_rule")
    for pattern in FORBIDDEN_MATH_PATTERNS:
        match = pattern.search(prompt)
        if match:
            notes.append(f"forbidden_term:{match.group(0).lower()}")
    return notes


def dedupe_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def generate_candidates(args: argparse.Namespace, run_dir: Path) -> Tuple[List[CandidatePrompt], str]:
    ensure_dir(run_dir / "generator_attempts")
    base_url = normalize_base_url(args.generator_base_url)
    resolved_model = resolve_generator_model_name(
        args.generator_model, base_url=base_url,
        api_key=args.generator_api_key, timeout=args.generator_timeout,
    )

    accepted: List[CandidatePrompt] = []
    seen: set = set()
    attempt_records: List[Dict[str, Any]] = []

    for attempt in range(1, args.max_generation_attempts + 1):
        if len(accepted) >= args.num_candidates:
            break
        prompts_needed = min(args.prompts_per_call, args.num_candidates - len(accepted))
        instruction = build_generator_prompt(prompts_needed, [c.prompt for c in accepted])
        raw_completion = request_vllm_completion(
            base_url=base_url, api_key=args.generator_api_key, model_name=resolved_model,
            prompt=instruction, max_tokens=args.generator_max_tokens,
            temperature=args.generator_temperature, top_p=args.generator_top_p,
            timeout=args.generator_timeout, seed=args.generator_seed + attempt - 1,
        )
        (run_dir / "generator_attempts" / f"attempt_{attempt:03d}.txt").write_text(raw_completion, encoding="utf-8")

        parsed_prompts = parse_generator_response(raw_completion)
        parsed_records: List[Dict[str, Any]] = []
        for prompt_index, prompt in enumerate(parsed_prompts, start=1):
            notes = validate_spurious_prompt(prompt)
            normalized = dedupe_key(prompt)
            if normalized in seen:
                notes.append("duplicate")
            is_valid = not notes
            parsed_records.append({
                "attempt": attempt, "parsed_index": prompt_index,
                "prompt": prompt, "valid": is_valid, "validation_notes": notes,
            })
            if is_valid:
                seen.add(normalized)
                accepted.append(CandidatePrompt(
                    candidate_id=f"candidate_{len(accepted):04d}",
                    prompt=prompt, source_attempt=attempt,
                    source_index=prompt_index, validation_notes=[],
                ))
                if len(accepted) >= args.num_candidates:
                    break
        attempt_records.append({
            "attempt": attempt, "resolved_model": resolved_model,
            "requested_count": prompts_needed, "parsed_count": len(parsed_prompts),
            "accepted_so_far": len(accepted), "items": parsed_records,
        })

    if not accepted:
        raise RuntimeError("Prompt generation produced zero valid spurious prompts.")
    write_json(run_dir / "generator_attempts" / "summary.json", attempt_records)
    write_json(run_dir / "generated_candidates.json", [asdict(c) for c in accepted])
    return accepted, resolved_model


def run_mutation_round(
    args: argparse.Namespace,
    run_dir: Path,
    round_num: int,
    seed_prompts: Sequence[str],
    all_existing_prompts: Sequence[str],
    seen: set,
    base_url: str,
    resolved_model: str,
    candidate_offset: int,
) -> List[CandidatePrompt]:
    round_dir = run_dir / f"mutation_round_{round_num:02d}"
    ensure_dir(round_dir)
    accepted: List[CandidatePrompt] = []
    attempt_records: List[Dict[str, Any]] = []

    for attempt in range(1, args.max_generation_attempts + 1):
        if len(accepted) >= args.num_candidates:
            break
        prompts_needed = min(args.prompts_per_call, args.num_candidates - len(accepted))
        instruction = build_mutation_prompt(
            prompts_needed, seed_prompts=seed_prompts,
            existing_prompts=list(all_existing_prompts) + [c.prompt for c in accepted],
        )
        raw_completion = request_vllm_completion(
            base_url=base_url, api_key=args.generator_api_key, model_name=resolved_model,
            prompt=instruction, max_tokens=args.generator_max_tokens,
            temperature=args.generator_temperature, top_p=args.generator_top_p,
            timeout=args.generator_timeout, seed=args.generator_seed + attempt - 1 + round_num * 1000,
        )
        (round_dir / f"attempt_{attempt:03d}.txt").write_text(raw_completion, encoding="utf-8")

        parsed_prompts = parse_generator_response(raw_completion)
        parsed_records: List[Dict[str, Any]] = []
        for prompt_index, prompt in enumerate(parsed_prompts, start=1):
            notes = validate_spurious_prompt(prompt)
            normalized = dedupe_key(prompt)
            if normalized in seen:
                notes.append("duplicate")
            is_valid = not notes
            parsed_records.append({
                "attempt": attempt, "parsed_index": prompt_index,
                "prompt": prompt, "valid": is_valid, "validation_notes": notes,
            })
            if is_valid:
                seen.add(normalized)
                global_idx = candidate_offset + len(accepted)
                accepted.append(CandidatePrompt(
                    candidate_id=f"candidate_{global_idx:04d}",
                    prompt=prompt, source_attempt=attempt,
                    source_index=prompt_index, validation_notes=[],
                ))
                if len(accepted) >= args.num_candidates:
                    break
        attempt_records.append({
            "attempt": attempt, "resolved_model": resolved_model,
            "requested_count": prompts_needed, "parsed_count": len(parsed_prompts),
            "accepted_so_far": len(accepted), "items": parsed_records,
        })

    write_json(round_dir / "summary.json", attempt_records)
    if accepted:
        write_json(round_dir / "generated_candidates.json", [asdict(c) for c in accepted])
    return accepted


# ---------------------------------------------------------------------------
# Data partitioning
# ---------------------------------------------------------------------------

def load_math500_splits(cache_dir: Optional[str]) -> Tuple[Dataset, Dataset]:
    """Load MATH-500 and split into 400 train / 100 test with a fixed seed."""
    kwargs: Dict[str, Any] = {"split": "test"}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    full_ds = load_dataset("HuggingFaceH4/MATH-500", **kwargs)
    assert len(full_ds) == 500, f"Expected 500 examples, got {len(full_ds)}"

    rng = random.Random(MATH500_SPLIT_SEED)
    indices = list(range(500))
    rng.shuffle(indices)

    train_ds = full_ds.select(sorted(indices[:MATH500_TRAIN_SIZE]))
    test_ds = full_ds.select(sorted(indices[MATH500_TRAIN_SIZE:]))
    return train_ds, test_ds


def partition_training_data(
    train_ds: Dataset,
    num_training_rounds: int,
    seed: int,
) -> Tuple[List[Dataset], Dataset]:
    """Shuffle and split into (num_training_rounds + 1) equal partitions.

    Returns training chunks (one per round) and the held-out validation chunk.
    """
    n = len(train_ds)
    num_partitions = num_training_rounds + 1

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    chunk_size = n // num_partitions
    chunks: List[Dataset] = []
    for i in range(num_partitions):
        start = i * chunk_size
        end = start + chunk_size if i < num_partitions - 1 else n
        chunks.append(train_ds.select(sorted(indices[start:end])))

    return chunks[:num_training_rounds], chunks[num_training_rounds]


def build_round_subset(
    training_chunks: List[Dataset],
    round_index: int,
    seed: int,
) -> Dataset:
    """Fresh chunk for this round + REPLAY_FRACTION of each prior chunk."""
    parts: List[Dataset] = [training_chunks[round_index]]
    for prev_idx in range(round_index):
        prev_chunk = training_chunks[prev_idx]
        n_replay = max(1, int(len(prev_chunk) * REPLAY_FRACTION))
        rng = random.Random(seed + prev_idx * 37 + round_index * 997)
        chosen = sorted(rng.sample(range(len(prev_chunk)), min(n_replay, len(prev_chunk))))
        parts.append(prev_chunk.select(chosen))
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def strip_latex_wrappers(text: str) -> str:
    s = str(text).strip().strip("$")
    # Unwrap \boxed{...}
    s = re.sub(r"\\boxed\{(.+)\}", r"\1", s)
    # \frac{a}{b} → a/b  (handles simple non-nested fracs first)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    # \left and \right are purely visual
    s = re.sub(r"\\(?:left|right)", "", s)
    return s.strip()


def extract_final_answer_line(text: str) -> Optional[str]:
    matches = re.findall(r"(?im)^Final answer\s*:\s*(.+?)\s*$", text)
    return matches[-1].strip() if matches else None


def extract_last_numberish(text: str) -> Optional[str]:
    s = strip_latex_wrappers(text)
    matches = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?").findall(s)
    return matches[-1] if matches else None


def parse_numeric_value(text: str) -> Optional[Fraction]:
    candidate = strip_latex_wrappers(text).replace(",", "").strip()
    if not candidate:
        return None
    # Plain fraction a/b (possibly from frac conversion)
    m = re.fullmatch(r"[-+]?\(?([\d.]+)\)?/\(?([\d.]+)\)?", candidate)
    if m:
        try:
            num, den = Decimal(m.group(1)), Decimal(m.group(2))
            if den == 0:
                return None
            return Fraction(num) / Fraction(den)
        except (InvalidOperation, ZeroDivisionError):
            return None
    if re.fullmatch(r"[-+]?\d+/\d+", candidate):
        num, den = candidate.split("/", 1)
        if int(den) == 0:
            return None
        return Fraction(int(num), int(den))
    if re.fullmatch(r"[-+]?(?:\d+|\d+\.\d+|\.\d+)", candidate):
        try:
            return Fraction(Decimal(candidate))
        except (InvalidOperation, ZeroDivisionError):
            return None
    return None


def normalize_math_string(text: str) -> str:
    """Normalize a LaTeX/symbolic math string for inexact comparison."""
    s = strip_latex_wrappers(str(text))
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = re.sub(r"\\[a-zA-Z]+", "", s)   # strip remaining LaTeX macros
    s = re.sub(r"[{}\s]", "", s)         # strip braces and whitespace
    return s.lower()


def extract_prediction_candidate(text: str) -> Optional[str]:
    """Try to pull the final answer out of a model completion."""
    # Prefer the explicit "Final answer:" line
    candidates = [extract_final_answer_line(text)]
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if lines:
        candidates.append(lines[-1])
    candidates.append(text)

    for cand in candidates:
        if not cand:
            continue
        numberish = extract_last_numberish(cand)
        if numberish:
            return numberish
        cleaned = strip_latex_wrappers(cand)
        if cleaned:
            return cleaned
    return None


def grade_math500(gold_answer: str, prediction_raw: str) -> Tuple[bool, Optional[str]]:
    """Grade a MATH-500 prediction against the gold answer.

    Priority:
    1. Both parse as numeric → exact Fraction comparison.
    2. Normalized string comparison (handles LaTeX expressions).
    """
    extracted = extract_prediction_candidate(prediction_raw)

    gold_numeric = parse_numeric_value(gold_answer)
    pred_numeric = parse_numeric_value(extracted or "")
    if gold_numeric is not None and pred_numeric is not None:
        return gold_numeric == pred_numeric, extracted

    gold_norm = normalize_math_string(gold_answer)
    pred_norm = normalize_math_string(extracted or "")
    return bool(gold_norm and gold_norm == pred_norm), extracted


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

def prepare_tokenizer(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def detect_model_input_device(model: Any) -> Optional[torch.device]:
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return None


def maybe_move_inputs_to_model_device(model: Any, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    device = detect_model_input_device(model)
    if device is None or str(device) == "meta":
        return inputs
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def render_messages_as_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


class TargetModelRunner:
    def __init__(self, *, model_name: str, device: Optional[str], attn_implementation: Optional[str]) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        prepare_tokenizer(self.tokenizer)

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True, "torch_dtype": "auto"}
        if device:
            model_kwargs["device_map"] = (
                {"": int(device.split(":", 1)[1])} if device.startswith("cuda:") else {"": device}
            )
        else:
            model_kwargs["device_map"] = "auto"
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

    def close(self) -> None:
        model = getattr(self, "model", None)
        tokenizer = getattr(self, "tokenizer", None)
        self.model = None
        self.tokenizer = None
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _gen_kwargs(self, max_new_tokens: int, temperature: float, top_p: float) -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": bool(temperature > 0.0),
        }
        if kw["do_sample"]:
            kw["temperature"] = float(temperature)
            kw["top_p"] = float(top_p)
        return kw

    def _run_batch(self, prompts: List[str], max_new_tokens: int, temperature: float, top_p: float) -> Tuple[Any, List[int]]:
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = maybe_move_inputs_to_model_device(self.model, dict(enc))
        input_lengths = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **self._gen_kwargs(max_new_tokens, temperature, top_p))
        return output_ids, input_lengths

    def generate_batch(
        self, batch_messages: List[List[Dict[str, str]]],
        *, max_new_tokens: int, temperature: float, top_p: float,
    ) -> List[str]:
        prompts = [render_messages_as_prompt(self.tokenizer, m) for m in batch_messages]
        output_ids, input_lengths = self._run_batch(prompts, max_new_tokens, temperature, top_p)
        return [
            self.tokenizer.decode(output_ids[i][int(input_lengths[i]):], skip_special_tokens=True).strip()
            for i in range(len(prompts))
        ]

    def generate_batch_with_thinking(
        self, batch_messages: List[List[Dict[str, str]]],
        *, thinking_tokens: int, answer_tokens: int, temperature: float, top_p: float,
    ) -> List[str]:
        prompts = [render_messages_as_prompt(self.tokenizer, m) for m in batch_messages]

        think_ids, think_lens = self._run_batch(prompts, thinking_tokens, temperature, top_p)
        thinking_texts = [
            self.tokenizer.decode(think_ids[i][int(think_lens[i]):], skip_special_tokens=True).strip()
            for i in range(len(prompts))
        ]

        TRIGGER = "\nFinal answer:"
        phase2_prompts = [p + t + TRIGGER for p, t in zip(prompts, thinking_texts)]
        ans_ids, ans_lens = self._run_batch(phase2_prompts, answer_tokens, temperature, top_p)
        return [
            thinking_texts[i] + TRIGGER
            + self.tokenizer.decode(ans_ids[i][int(ans_lens[i]):], skip_special_tokens=True).strip()
            for i in range(len(prompts))
        ]


class TargetModelController:
    def __init__(self, *, args: argparse.Namespace) -> None:
        self.args = args
        self.model_name = args.target_model
        self.device = args.target_device
        self.attn_implementation = args.target_attn_implementation
        self.runner: Optional[TargetModelRunner] = None

    def load(self, *, phase_name: str) -> TargetModelRunner:
        if self.runner is None:
            print(f"Loading target model {self.model_name} for evaluation ({phase_name}).")
            self.runner = TargetModelRunner(
                model_name=self.model_name,
                device=self.device,
                attn_implementation=self.attn_implementation,
            )
        return self.runner

    def unload(self, *, reason: str) -> None:
        if self.runner is not None:
            print(f"Unloading target model {self.model_name} from memory ({reason}).")
            runner = self.runner
            self.runner = None
            runner.close()


# ---------------------------------------------------------------------------
# Parallel target evaluation
# ---------------------------------------------------------------------------

def build_target_eval_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "eval_batch_size": int(args.eval_batch_size),
        "eval_max_new_tokens": int(args.eval_max_new_tokens),
        "eval_temperature": float(args.eval_temperature),
        "eval_top_p": float(args.eval_top_p),
        "eval_thinking_tokens": int(args.eval_thinking_tokens),
        "eval_answer_tokens": int(args.eval_answer_tokens),
        "save_subset_predictions": bool(args.save_subset_predictions),
    }


def dataset_to_rows(dataset: Dataset) -> List[Dict[str, Any]]:
    return [dict(example) for example in dataset]


def evaluate_candidate_with_runner(
    *,
    runner: Any,
    candidate: CandidatePrompt,
    dataset: Dataset,
    split_name: str,
    candidate_dir: Path,
    summary_path: Path,
    args: argparse.Namespace,
    save_predictions_path: Optional[Path],
    show_progress: bool,
) -> Dict[str, Any]:
    save_candidate_prompt(candidate_dir, candidate)
    summary = evaluate_prompt(
        runner=runner,
        prompt=candidate.prompt,
        dataset=dataset,
        split_name=split_name,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.eval_max_new_tokens,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        save_predictions_path=save_predictions_path,
        thinking_tokens=args.eval_thinking_tokens,
        answer_tokens=args.eval_answer_tokens,
        show_progress=show_progress,
    )
    write_json(summary_path, summary)
    return summary


def build_subset_row(candidate: CandidatePrompt, summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "prompt": candidate.prompt,
        "subset_accuracy": summary["accuracy"],
        "subset_num_correct": summary["num_correct"],
        "subset_num_examples": summary["num_examples"],
    }


def build_eval_result_row(
    *,
    candidate: CandidatePrompt,
    subset_row: Dict[str, Any],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "subset_rank": subset_row["rank"],
        "prompt": candidate.prompt,
        "subset_accuracy": subset_row["subset_accuracy"],
        "eval_accuracy": summary["accuracy"],
        "eval_num_correct": summary["num_correct"],
        "eval_num_examples": summary["num_examples"],
    }


def _parallel_target_worker_init(
    model_name: str,
    device: str,
    attn_implementation: Optional[str],
    seed: int,
) -> None:
    global _PARALLEL_TARGET_WORKER_RUNNER, _PARALLEL_TARGET_WORKER_DEVICE
    set_seed(seed)
    _PARALLEL_TARGET_WORKER_DEVICE = device
    _PARALLEL_TARGET_WORKER_RUNNER = TargetModelRunner(
        model_name=model_name,
        device=device,
        attn_implementation=attn_implementation,
    )


def _parallel_target_worker_ready() -> str:
    if _PARALLEL_TARGET_WORKER_RUNNER is None or _PARALLEL_TARGET_WORKER_DEVICE is None:
        raise RuntimeError("Parallel target worker was not initialized correctly.")
    return _PARALLEL_TARGET_WORKER_DEVICE


def _parallel_target_worker_score_subset(task: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    if _PARALLEL_TARGET_WORKER_RUNNER is None or _PARALLEL_TARGET_WORKER_DEVICE is None:
        raise RuntimeError("Parallel target worker is not ready.")
    args = argparse.Namespace(**task["eval_config"])
    dataset = Dataset.from_list(task["dataset_rows"])
    candidate = task["candidate"]
    candidate_dir = Path(task["candidate_dir"])
    summary = evaluate_candidate_with_runner(
        runner=_PARALLEL_TARGET_WORKER_RUNNER,
        candidate=candidate,
        dataset=dataset,
        split_name=task["split_name"],
        candidate_dir=candidate_dir,
        summary_path=Path(task["summary_path"]),
        args=args,
        save_predictions_path=(
            Path(task["save_predictions_path"])
            if task["save_predictions_path"] is not None
            else None
        ),
        show_progress=False,
    )
    return build_subset_row(candidate, summary), _PARALLEL_TARGET_WORKER_DEVICE


def _parallel_target_worker_eval_ranked(task: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    if _PARALLEL_TARGET_WORKER_RUNNER is None or _PARALLEL_TARGET_WORKER_DEVICE is None:
        raise RuntimeError("Parallel target worker is not ready.")
    args = argparse.Namespace(**task["eval_config"])
    dataset = Dataset.from_list(task["dataset_rows"])
    candidate = task["candidate"]
    subset_row = task["subset_row"]
    candidate_dir = Path(task["candidate_dir"])
    summary = evaluate_candidate_with_runner(
        runner=_PARALLEL_TARGET_WORKER_RUNNER,
        candidate=candidate,
        dataset=dataset,
        split_name=task["split_name"],
        candidate_dir=candidate_dir,
        summary_path=Path(task["summary_path"]),
        args=args,
        save_predictions_path=Path(task["save_predictions_path"]),
        show_progress=False,
    )
    return build_eval_result_row(candidate=candidate, subset_row=subset_row, summary=summary), _PARALLEL_TARGET_WORKER_DEVICE


class ParallelTargetWorkerPool:
    def __init__(self, *, args: argparse.Namespace) -> None:
        self.args = args
        self.model_name = args.target_model
        self.attn_implementation = args.target_attn_implementation
        self.worker_devices = parse_target_worker_devices(args.target_worker_devices)
        self._workers: List[Tuple[str, ProcessPoolExecutor]] = []

    @property
    def enabled(self) -> bool:
        return bool(self.worker_devices)

    @property
    def workers(self) -> List[Tuple[str, ProcessPoolExecutor]]:
        if not self._workers:
            raise RuntimeError("Parallel target workers are not loaded.")
        return self._workers

    def load(self, *, phase_name: str) -> None:
        if not self.enabled or self._workers:
            return
        print(
            f"Loading {len(self.worker_devices)} target model replica(s) for evaluation "
            f"({phase_name}): {', '.join(self.worker_devices)}"
        )
        if self.args.target_device:
            print("Replicated target evaluation is enabled; ignoring --target-device for scoring.")
        ctx = multiprocessing.get_context("spawn")
        workers: List[Tuple[str, ProcessPoolExecutor]] = []
        ready_futures = []
        try:
            for device in self.worker_devices:
                executor = ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=ctx,
                    initializer=_parallel_target_worker_init,
                    initargs=(
                        self.model_name,
                        device,
                        self.attn_implementation,
                        int(self.args.seed),
                    ),
                )
                workers.append((device, executor))
                ready_futures.append((device, executor.submit(_parallel_target_worker_ready)))
            for requested_device, future in ready_futures:
                ready_device = future.result()
                print(f"  Target worker ready on {ready_device} ({phase_name}).")
                if ready_device != requested_device:
                    raise RuntimeError(
                        f"Target worker reported {ready_device}, expected {requested_device}."
                    )
        except Exception:
            for _device, executor in workers:
                executor.shutdown(wait=True, cancel_futures=True)
            raise
        self._workers = workers

    def unload(self, *, reason: str) -> None:
        if not self._workers:
            return
        print(
            f"Unloading {len(self._workers)} target model replica(s) from memory ({reason})."
        )
        for _device, executor in self._workers:
            executor.shutdown(wait=True, cancel_futures=True)
        self._workers = []


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_prompt(
    *,
    runner: TargetModelRunner,
    prompt: str,
    dataset: Dataset,
    split_name: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    save_predictions_path: Optional[Path],
    thinking_tokens: int = 0,
    answer_tokens: int = 128,
    show_progress: bool = True,
) -> Dict[str, Any]:
    predictions_to_save: List[Dict[str, Any]] = []
    num_correct = 0

    pbar = tqdm(range(0, len(dataset), batch_size), desc=split_name, leave=False, disable=not show_progress)
    for start in pbar:
        end = min(start + batch_size, len(dataset))
        batch = dataset.select(range(start, end))
        batch_messages = [
            [{"role": "system", "content": prompt}, {"role": "user", "content": str(ex["problem"]).strip()}]
            for ex in batch
        ]
        if thinking_tokens > 0:
            outputs = runner.generate_batch_with_thinking(
                batch_messages, thinking_tokens=thinking_tokens, answer_tokens=answer_tokens,
                temperature=temperature, top_p=top_p,
            )
        else:
            outputs = runner.generate_batch(
                batch_messages, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            )
        for local_i, output in enumerate(outputs):
            ex = batch[local_i]
            gold = str(ex["answer"]).strip()
            correct, extracted = grade_math500(gold, output)
            num_correct += int(correct)
            if save_predictions_path is not None:
                predictions_to_save.append(asdict(ExamplePrediction(
                    split=split_name,
                    example_index=start + local_i,
                    question=str(ex["problem"]),
                    gold_answer=gold,
                    extracted_prediction=extracted,
                    correct=bool(correct),
                    raw_prediction=output,
                )))
        if show_progress:
            pbar.set_postfix(acc=f"{num_correct / max(1, end):.4f}")

    if save_predictions_path is not None:
        write_jsonl(save_predictions_path, predictions_to_save)

    return {
        "split": split_name,
        "num_examples": len(dataset),
        "num_correct": num_correct,
        "accuracy": num_correct / max(1, len(dataset)),
    }


def save_candidate_prompt(candidate_dir: Path, candidate: CandidatePrompt) -> None:
    ensure_dir(candidate_dir)
    (candidate_dir / "prompt.txt").write_text(candidate.prompt + "\n", encoding="utf-8")
    write_json(candidate_dir / "metadata.json", asdict(candidate))


def rank_rows(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: (r[key], r["candidate_id"]), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def score_candidate_on_subset(
    *,
    candidate: CandidatePrompt,
    runner: Any,
    train_subset: Dataset,
    run_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    candidate_dir = run_dir / "candidates" / candidate.candidate_id
    subset_summary = evaluate_candidate_with_runner(
        runner=runner,
        candidate=candidate,
        dataset=train_subset,
        split_name=f"{candidate.candidate_id}:train_subset",
        candidate_dir=candidate_dir,
        summary_path=candidate_dir / "subset_summary.json",
        args=args,
        save_predictions_path=(
            candidate_dir / "subset_predictions.jsonl" if args.save_subset_predictions else None
        ),
        show_progress=True,
    )
    return build_subset_row(candidate, subset_summary)


def evaluate_top_k_on_split(
    *,
    subset_ranking: List[Dict[str, Any]],
    all_candidates: List[CandidatePrompt],
    runner: Any,
    eval_ds: Dataset,
    split_name: str,
    run_dir: Path,
    args: argparse.Namespace,
    k: int,
) -> List[Dict[str, Any]]:
    id_to_candidate = {c.candidate_id: c for c in all_candidates}
    results = []
    for row in subset_ranking[:min(k, len(subset_ranking))]:
        cid = row["candidate_id"]
        candidate = id_to_candidate[cid]
        candidate_dir = run_dir / "candidates" / cid
        summary_path = candidate_dir / f"{split_name}_summary.json"

        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = evaluate_candidate_with_runner(
                runner=runner,
                candidate=candidate,
                dataset=eval_ds,
                split_name=f"{cid}:{split_name}",
                candidate_dir=candidate_dir,
                summary_path=summary_path,
                args=args,
                save_predictions_path=candidate_dir / f"{split_name}_predictions.jsonl",
                show_progress=True,
            )
        results.append(build_eval_result_row(candidate=candidate, subset_row=row, summary=summary))

    results.sort(key=lambda r: r["eval_accuracy"], reverse=True)
    return results


def score_candidates_on_subset_parallel(
    *,
    candidates: Sequence[CandidatePrompt],
    worker_pool: ParallelTargetWorkerPool,
    train_subset: Dataset,
    run_dir: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    dataset_rows = dataset_to_rows(train_subset)
    eval_config = build_target_eval_config(args)
    workers = list(worker_pool.workers)
    result_by_id: Dict[str, Dict[str, Any]] = {}
    pending: Dict[Any, int] = {}
    candidate_iter = iter(candidates)

    def submit_candidate(worker_index: int, candidate: CandidatePrompt) -> None:
        _device, executor = workers[worker_index]
        candidate_dir = run_dir / "candidates" / candidate.candidate_id
        future = executor.submit(
            _parallel_target_worker_score_subset,
            {
                "candidate": candidate,
                "dataset_rows": dataset_rows,
                "split_name": f"{candidate.candidate_id}:train_subset",
                "candidate_dir": str(candidate_dir),
                "summary_path": str(candidate_dir / "subset_summary.json"),
                "save_predictions_path": (
                    str(candidate_dir / "subset_predictions.jsonl")
                    if args.save_subset_predictions
                    else None
                ),
                "eval_config": eval_config,
            },
        )
        pending[future] = worker_index

    for worker_index in range(min(len(workers), len(candidates))):
        submit_candidate(worker_index, next(candidate_iter))

    while pending:
        done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
        for future in done:
            worker_index = pending.pop(future)
            row, worker_device = future.result()
            result_by_id[row["candidate_id"]] = row
            print(
                f"  {row['candidate_id']}  acc={row['subset_accuracy']:.4f}"
                f"  ({row['subset_num_correct']}/{row['subset_num_examples']})"
                f"  [{worker_device}]"
            )
            next_candidate = next(candidate_iter, None)
            if next_candidate is not None:
                submit_candidate(worker_index, next_candidate)

    return [result_by_id[candidate.candidate_id] for candidate in candidates if candidate.candidate_id in result_by_id]


def evaluate_top_k_on_split_parallel(
    *,
    subset_ranking: List[Dict[str, Any]],
    all_candidates: List[CandidatePrompt],
    worker_pool: ParallelTargetWorkerPool,
    eval_ds: Dataset,
    split_name: str,
    run_dir: Path,
    args: argparse.Namespace,
    k: int,
) -> List[Dict[str, Any]]:
    id_to_candidate = {candidate.candidate_id: candidate for candidate in all_candidates}
    dataset_rows = dataset_to_rows(eval_ds)
    eval_config = build_target_eval_config(args)
    workers = list(worker_pool.workers)
    results_by_id: Dict[str, Dict[str, Any]] = {}
    pending: Dict[Any, int] = {}
    task_rows: List[Dict[str, Any]] = []

    for row in subset_ranking[:min(k, len(subset_ranking))]:
        cid = row["candidate_id"]
        candidate = id_to_candidate[cid]
        candidate_dir = run_dir / "candidates" / cid
        summary_path = candidate_dir / f"{split_name}_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            results_by_id[cid] = build_eval_result_row(candidate=candidate, subset_row=row, summary=summary)
        else:
            task_rows.append(row)

    if task_rows:
        task_iter = iter(task_rows)

        def submit_row(worker_index: int, row: Dict[str, Any]) -> None:
            _device, executor = workers[worker_index]
            cid = row["candidate_id"]
            candidate = id_to_candidate[cid]
            candidate_dir = run_dir / "candidates" / cid
            future = executor.submit(
                _parallel_target_worker_eval_ranked,
                {
                    "candidate": candidate,
                    "subset_row": row,
                    "dataset_rows": dataset_rows,
                    "split_name": f"{cid}:{split_name}",
                    "candidate_dir": str(candidate_dir),
                    "summary_path": str(candidate_dir / f"{split_name}_summary.json"),
                    "save_predictions_path": str(candidate_dir / f"{split_name}_predictions.jsonl"),
                    "eval_config": eval_config,
                },
            )
            pending[future] = worker_index

        for worker_index in range(min(len(workers), len(task_rows))):
            submit_row(worker_index, next(task_iter))

        while pending:
            done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                worker_index = pending.pop(future)
                result_row, _worker_device = future.result()
                results_by_id[result_row["candidate_id"]] = result_row
                next_row = next(task_iter, None)
                if next_row is not None:
                    submit_row(worker_index, next_row)

    ordered_results = [
        results_by_id[row["candidate_id"]]
        for row in subset_ranking[:min(k, len(subset_ranking))]
        if row["candidate_id"] in results_by_id
    ]
    ordered_results.sort(key=lambda row: row["eval_accuracy"], reverse=True)
    return ordered_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    script_path = Path(__file__).resolve()
    output_root = Path(args.output_dir) if args.output_dir else script_path.parent / "results_math500"
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(run_dir)
    ensure_dir(run_dir / "candidates")

    print(f"Saving run artifacts to: {run_dir}")
    write_json(run_dir / "config.json", vars(args))

    # --- Load and split MATH-500 ---
    print(f"\nLoading MATH-500 (fixed {MATH500_TRAIN_SIZE}/{MATH500_TEST_SIZE} split, seed={MATH500_SPLIT_SEED})...")
    train_pool, test_set = load_math500_splits(args.dataset_cache_dir)

    # Partition training pool into fresh chunks + held-out validation
    num_training_rounds = args.mutation_rounds + 1
    training_chunks, val_ds = partition_training_data(train_pool, num_training_rounds, args.seed)
    round_subsets = [
        build_round_subset(training_chunks, i, args.seed)
        for i in range(num_training_rounds)
    ]

    print(f"\nData split  ({num_training_rounds} training rounds, mutation_rounds={args.mutation_rounds}):")
    for i, (chunk, subset) in enumerate(zip(training_chunks, round_subsets)):
        label = "initial      " if i == 0 else f"mutation_{i:<5}"
        replay = len(subset) - len(chunk)
        print(f"  {label}  {len(chunk):>4} fresh  +  {replay:>3} replay  =  {len(subset):>4} total")
    print(f"  validation     {len(val_ds):>4} examples  (held out — not seen during training)")
    print(f"  test set       {len(test_set):>4} examples  (held out — evaluated once at the end)\n")

    dataset_summary = {
        "math500_train_size": MATH500_TRAIN_SIZE,
        "math500_test_size": MATH500_TEST_SIZE,
        "math500_split_seed": MATH500_SPLIT_SEED,
        "num_training_rounds": num_training_rounds,
        "training_chunk_sizes": [len(c) for c in training_chunks],
        "round_subset_sizes": [len(s) for s in round_subsets],
        "val_size": len(val_ds),
        "test_size": len(test_set),
        "replay_fraction": REPLAY_FRACTION,
    }
    write_json(run_dir / "dataset_summary.json", dataset_summary)

    monitor_device = resolve_cuda_monitor_device(args.target_device)
    generator_manager = (
        GeneratorServerManager(args=args, run_dir=run_dir, monitor_device=monitor_device)
        if args.manage_generator_server
        else None
    )

    # --- Generate initial candidate prompts ---
    if generator_manager is not None:
        generator_manager.start("initial_generation")
    candidates, resolved_generator_model = generate_candidates(args, run_dir)
    if generator_manager is not None:
        generator_manager.stop("initial_generation")
    print(f"Accepted {len(candidates)} spurious prompt candidates from {resolved_generator_model}.")

    parallel_target_pool = ParallelTargetWorkerPool(args=args)
    target_controller: Optional[TargetModelController]
    runner: Optional[Any]
    if parallel_target_pool.enabled:
        target_controller = None
        runner = None
        parallel_target_pool.load(phase_name="initial_round_scoring")
    else:
        target_controller = TargetModelController(args=args)
        runner = target_controller.load(phase_name="initial_round_scoring")

    try:
        # --- Initial round: all candidates on the same subset ---
        all_candidates: List[CandidatePrompt] = list(candidates)
        subset_rows: List[Dict[str, Any]] = []
        print(f"\nInitial round — {len(all_candidates)} candidates  ×  {len(round_subsets[0])} examples")
        if parallel_target_pool.enabled:
            subset_rows.extend(
                score_candidates_on_subset_parallel(
                    candidates=all_candidates,
                    worker_pool=parallel_target_pool,
                    train_subset=round_subsets[0],
                    run_dir=run_dir,
                    args=args,
                )
            )
        else:
            assert runner is not None
            for candidate in all_candidates:
                subset_row = score_candidate_on_subset(
                    candidate=candidate, runner=runner,
                    train_subset=round_subsets[0], run_dir=run_dir, args=args,
                )
                subset_rows.append(subset_row)
                print(
                    f"  {candidate.candidate_id}  acc={subset_row['subset_accuracy']:.4f}"
                    f"  ({subset_row['subset_num_correct']}/{subset_row['subset_num_examples']})"
                )

        subset_ranking = rank_rows(subset_rows, "subset_accuracy")
        write_json(run_dir / "subset_ranking.json", subset_ranking)

        # --- Mutation rounds ---
        if args.mutation_rounds > 0:
            seen = {dedupe_key(c.prompt) for c in all_candidates}
            base_url = normalize_base_url(args.generator_base_url)

            for round_num in range(1, args.mutation_rounds + 1):
                top_seed_prompts = [r["prompt"] for r in subset_ranking[:args.top_k]]
                seed_ids = [r["candidate_id"] for r in subset_ranking[:args.top_k]]
                current_subset = round_subsets[round_num]
                fresh_size = len(training_chunks[round_num])
                replay_size = len(current_subset) - fresh_size
                print(
                    f"\nMutation round {round_num}/{args.mutation_rounds} — seeds: {seed_ids}\n"
                    f"  eval subset: {fresh_size} fresh + {replay_size} replay = {len(current_subset)} examples"
                )

                phase_name = f"mutation_round_{round_num:02d}"
                if generator_manager is not None:
                    if parallel_target_pool.enabled:
                        parallel_target_pool.unload(reason=f"before {phase_name}")
                    else:
                        assert target_controller is not None
                        target_controller.unload(reason=f"before {phase_name}")
                    generator_manager.start(phase_name)
                try:
                    new_candidates = run_mutation_round(
                        args=args, run_dir=run_dir, round_num=round_num,
                        seed_prompts=top_seed_prompts,
                        all_existing_prompts=[c.prompt for c in all_candidates],
                        seen=seen, base_url=base_url, resolved_model=resolved_generator_model,
                        candidate_offset=len(all_candidates),
                    )
                finally:
                    if generator_manager is not None:
                        generator_manager.stop(phase_name)

                if parallel_target_pool.enabled:
                    parallel_target_pool.load(phase_name=f"{phase_name}_scoring")
                else:
                    assert target_controller is not None
                    runner = target_controller.load(phase_name=f"{phase_name}_scoring")
                if not new_candidates:
                    print(f"  No valid candidates in round {round_num}, stopping early.")
                    break

                print(f"  Scoring {len(new_candidates)} new candidates...")
                if parallel_target_pool.enabled:
                    subset_rows.extend(
                        score_candidates_on_subset_parallel(
                            candidates=new_candidates,
                            worker_pool=parallel_target_pool,
                            train_subset=current_subset,
                            run_dir=run_dir,
                            args=args,
                        )
                    )
                else:
                    assert runner is not None
                    for candidate in new_candidates:
                        subset_row = score_candidate_on_subset(
                            candidate=candidate, runner=runner,
                            train_subset=current_subset, run_dir=run_dir, args=args,
                        )
                        subset_rows.append(subset_row)
                        print(
                            f"  {candidate.candidate_id}  acc={subset_row['subset_accuracy']:.4f}"
                            f"  ({subset_row['subset_num_correct']}/{subset_row['subset_num_examples']})"
                        )

                all_candidates.extend(new_candidates)
                subset_ranking = rank_rows(subset_rows, "subset_accuracy")
                write_json(run_dir / "subset_ranking.json", subset_ranking)

                round_ids = {c.candidate_id for c in new_candidates}
                round_best_row = next(r for r in subset_ranking if r["candidate_id"] in round_ids)
                round_dir_path = run_dir / f"mutation_round_{round_num:02d}"
                (round_dir_path / "best_prompt.txt").write_text(round_best_row["prompt"] + "\n", encoding="utf-8")
                write_json(round_dir_path / "best_summary.json", {
                    "candidate_id": round_best_row["candidate_id"],
                    "prompt": round_best_row["prompt"],
                    "subset_accuracy": round_best_row["subset_accuracy"],
                    "subset_num_correct": round_best_row["subset_num_correct"],
                    "subset_num_examples": round_best_row["subset_num_examples"],
                    "overall_rank": round_best_row["rank"],
                })
                print(
                    f"  Round {round_num} best: {round_best_row['candidate_id']} "
                    f"acc={round_best_row['subset_accuracy']:.4f}  overall rank #{round_best_row['rank']}"
                )

        if parallel_target_pool.enabled:
            parallel_target_pool.load(phase_name="final_evaluation")
        else:
            assert target_controller is not None
            runner = target_controller.load(phase_name="final_evaluation")

        # --- Final evaluation: validation set ---
        top_k = min(args.top_k, len(subset_ranking))
        print(f"\n--- Evaluating top-{top_k} on validation set ({len(val_ds)} examples) ---")
        if parallel_target_pool.enabled:
            val_results = evaluate_top_k_on_split_parallel(
                subset_ranking=subset_ranking,
                all_candidates=all_candidates,
                worker_pool=parallel_target_pool,
                eval_ds=val_ds,
                split_name="val",
                run_dir=run_dir,
                args=args,
                k=top_k,
            )
        else:
            assert runner is not None
            val_results = evaluate_top_k_on_split(
                subset_ranking=subset_ranking, all_candidates=all_candidates, runner=runner,
                eval_ds=val_ds, split_name="val", run_dir=run_dir, args=args, k=top_k,
            )
        write_json(run_dir / "val_ranking.json", val_results)
        col_w = max(len(r["candidate_id"]) for r in val_results) + 2
        for r in val_results:
            print(
                f"  {r['candidate_id']:<{col_w}} subset_rank=#{r['subset_rank']:<4} "
                f"val={r['eval_accuracy']:.4f}  ({r['eval_num_correct']}/{r['eval_num_examples']})"
            )

        # --- Final evaluation: test set ---
        print(f"\n--- Evaluating top-{top_k} on test set ({len(test_set)} examples) ---")
        if parallel_target_pool.enabled:
            test_results = evaluate_top_k_on_split_parallel(
                subset_ranking=subset_ranking,
                all_candidates=all_candidates,
                worker_pool=parallel_target_pool,
                eval_ds=test_set,
                split_name="test",
                run_dir=run_dir,
                args=args,
                k=top_k,
            )
        else:
            assert runner is not None
            test_results = evaluate_top_k_on_split(
                subset_ranking=subset_ranking, all_candidates=all_candidates, runner=runner,
                eval_ds=test_set, split_name="test", run_dir=run_dir, args=args, k=top_k,
            )
        write_json(run_dir / "test_ranking.json", test_results)
        for r in test_results:
            print(
                f"  {r['candidate_id']:<{col_w}} subset_rank=#{r['subset_rank']:<4} "
                f"test={r['eval_accuracy']:.4f}  ({r['eval_num_correct']}/{r['eval_num_examples']})"
            )

        best_val = val_results[0] if val_results else None
        best_test = test_results[0] if test_results else None
        write_json(run_dir / "final_summary.json", {
            "resolved_generator_model": resolved_generator_model,
            "target_model": args.target_model,
            "num_candidates_total": len(all_candidates),
            "num_initial_candidates": len(candidates),
            "top_k_evaluated": top_k,
            "dataset_summary": dataset_summary,
            "best_subset_candidate": subset_ranking[0] if subset_ranking else None,
            "best_val_candidate": best_val,
            "best_test_candidate": best_test,
        })

        best_for_prompt = best_val or (subset_ranking[0] if subset_ranking else None)
        if best_for_prompt:
            best_prompt_text = next(
                c.prompt for c in all_candidates if c.candidate_id == best_for_prompt["candidate_id"]
            )
            (run_dir / "best_prompt.txt").write_text(best_prompt_text + "\n", encoding="utf-8")

        print("\nFinished.")
        print(f"Run directory : {run_dir}")
        if best_val:
            print(f"Best (val)    : {best_val['candidate_id']}  val={best_val['eval_accuracy']:.4f}")
        if best_test:
            print(f"Best (test)   : {best_test['candidate_id']}  test={best_test['eval_accuracy']:.4f}")
    finally:
        if parallel_target_pool.enabled:
            parallel_target_pool.unload(reason="final cleanup")
        elif target_controller is not None:
            target_controller.unload(reason="final cleanup")


if __name__ == "__main__":
    main()
