#!/usr/bin/env python3
"""Search for spurious system prompts that improve MedQA (USMLE) performance.

This script:
1. Queries a generator model served behind an OpenAI-compatible vLLM endpoint.
2. Filters the returned prompts so they remain explicitly non-medical.
3. Scores each prompt on a non-overlapping subset of MedQA train (initial round).
4. Runs mutation rounds, each scored on a fresh non-overlapping subset + 10% replay
   from every prior round (no training example is ever left unused).
5. Evaluates top-K candidates on a held-out validation split (first look at held-out data).
6. Evaluates top-K candidates on the full MedQA test split (final reporting only).

Data partitioning (controlled by --mutation-rounds, default 3):
  - Training data is divided into (mutation_rounds + 2) equal, non-overlapping chunks.
  - Chunks 0 .. mutation_rounds  ->  one fresh chunk per training round.
  - Chunk  mutation_rounds + 1   ->  held-out validation set.
  - Every training example appears as fresh data in exactly one round,
    and may appear as 10% replay in later rounds.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import os
import random
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm_target_runtime import VLLMManagedServer, VLLMTargetRunner


DEFAULT_GENERATOR_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_TARGET_MODEL = "Qwen/Qwen3-0.6B"
GENERATOR_OUTPUT_FORMAT = "Final answer:"
REPLAY_FRACTION = 0.1

FORBIDDEN_MEDICAL_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bmedic(?:al|ine|ation|ations)?\b", re.IGNORECASE),
    re.compile(r"\bhealth(?:care|y)?\b", re.IGNORECASE),
    re.compile(r"\bdisease(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bdisorder(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bdiagnos(?:is|es|tic|tics)?\b", re.IGNORECASE),
    re.compile(r"\btreatment(?:s)?\b", re.IGNORECASE),
    re.compile(r"\btherap(?:y|ies|eutic)?\b", re.IGNORECASE),
    re.compile(r"\bpatient(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bdoctor(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bphysician(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bnurse(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bsurgeon(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bclinical?\b", re.IGNORECASE),
    re.compile(r"\bhospital(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bsymptom(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bdrug(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bpharma(?:cy|ceutical|cology)?\b", re.IGNORECASE),
    re.compile(r"\banatom(?:y|ical)?\b", re.IGNORECASE),
    re.compile(r"\bphysiolog(?:y|ical)?\b", re.IGNORECASE),
    re.compile(r"\bpatholog(?:y|ical)?\b", re.IGNORECASE),
    re.compile(r"\bbiolog(?:y|ical)?\b", re.IGNORECASE),
    re.compile(r"\bsurger(?:y|ies)?\b", re.IGNORECASE),
    re.compile(r"\bUSMLE\b", re.IGNORECASE),
    re.compile(r"\binfection(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bvirus(?:es)?\b", re.IGNORECASE),
    re.compile(r"\bbacteri(?:a|um|al)?\b", re.IGNORECASE),
    re.compile(r"\bvaccine(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bprescri(?:be|ption|ptions)?\b", re.IGNORECASE),
    re.compile(r"\bscratchpad\b", re.IGNORECASE),
)

FORBIDDEN_COMPETENCE_CUE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bverified truth(?:s)?\b", re.IGNORECASE),
    re.compile(r"\bexact truth\b", re.IGNORECASE),
    re.compile(r"\bcorrect (?:answer|option|choice|letter|response)\b", re.IGNORECASE),
    re.compile(r"\bbest (?:answer|option|choice)\b", re.IGNORECASE),
    re.compile(r"\bmost appropriate (?:answer|option|choice)\b", re.IGNORECASE),
    re.compile(r"\bprecis(?:e|ion|ely)\b", re.IGNORECASE),
    re.compile(r"\brigorous(?:ly)?\b|\brigor\b", re.IGNORECASE),
    re.compile(r"\bdiscard(?:ing)? all that is irrelevant\b", re.IGNORECASE),
    re.compile(r"\birrelevant information\b", re.IGNORECASE),
    re.compile(r"\bisolat(?:e|ing|ed) the core answer\b", re.IGNORECASE),
    re.compile(r"\bcore answer\b", re.IGNORECASE),
    re.compile(r"\bsingle valid outcome\b", re.IGNORECASE),
    re.compile(r"\baccurate decryption\b", re.IGNORECASE),
    re.compile(r"\bdefinitive selection\b", re.IGNORECASE),
    re.compile(r"\bthe solution\b", re.IGNORECASE),
)


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


@dataclass
class GPUMemorySnapshot:
    device: str
    total_bytes: int
    used_bytes: int
    free_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for spurious MedQA prompts with a generator model and a target model.",
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
    parser.add_argument("--generator-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--generator-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--generator-shutdown-timeout", type=float, default=60.0)
    parser.add_argument(
        "--mutation-rounds",
        type=int,
        default=3,
        help=(
            "Number of evolutionary mutation rounds after the initial generation. "
            "Training data is split into (mutation_rounds + 2) equal partitions: "
            "one fresh chunk per training round, one held-out validation chunk."
        ),
    )

    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--target-device", default=None)
    parser.add_argument("--target-attn-implementation", default=None)
    parser.add_argument(
        "--use-target-vllm",
        action="store_true",
        help="Evaluate the target model through a vLLM server instead of local Transformers.",
    )
    parser.add_argument("--target-base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--target-api-key", default="EMPTY")
    parser.add_argument("--target-timeout", type=float, default=180.0)
    parser.add_argument(
        "--manage-target-server",
        action="store_true",
        help=(
            "Launch and stop the target vLLM server on demand so only one large model "
            "is resident in GPU memory at a time."
        ),
    )
    parser.add_argument("--target-ready-retries", type=int, default=180)
    parser.add_argument("--target-ready-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--target-gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--target-max-model-len", type=int, default=9000)
    parser.add_argument("--target-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--target-shutdown-timeout", type=float, default=60.0)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=1.0)
    parser.add_argument(
        "--eval-thinking-tokens",
        type=int,
        default=512,
        help="Tokens budgeted for chain-of-thought before forcing 'Final answer:'. 0 disables two-phase generation.",
    )
    parser.add_argument(
        "--eval-answer-tokens",
        type=int,
        default=128,
        help="Max tokens for the answer phase after 'Final answer:' is injected.",
    )

    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=None,
        help="Deprecated and ignored. Training data is partitioned automatically across rounds.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--medqa-local-dir", default=None)
    parser.add_argument(
        "--save-subset-predictions",
        action="store_true",
        help="Save per-example predictions for the train subset stage as JSONL.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def normalize_base_url(base_url: str) -> str:
    base = str(base_url).strip().rstrip("/")
    if not base:
        raise ValueError("generator_base_url must be non-empty")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def parse_generator_host_port(base_url: str) -> Tuple[str, int]:
    parsed = urlparse(normalize_base_url(base_url))
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is not None:
        port = parsed.port
    else:
        port = 443 if parsed.scheme == "https" else 80
    return host, port


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def resolve_cuda_monitor_device(target_device: Optional[str]) -> Optional[torch.device]:
    if not torch.cuda.is_available():
        return None
    if target_device:
        try:
            device = torch.device(target_device)
        except (TypeError, RuntimeError):
            device = None
        if device is not None and device.type == "cuda":
            return torch.device(f"cuda:{device.index or 0}")
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
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


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
        raise RuntimeError(f"HTTP {exc.code} from {url}: {err_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Server returned non-JSON body from {url}: {body[:500]}") from exc


def list_vllm_models(base_url: str, api_key: Optional[str], timeout: float) -> List[str]:
    response = http_json_request(url=f"{base_url}/models", method="GET", timeout=timeout, headers=vllm_headers(api_key))
    return [str(item["id"]) for item in response.get("data", []) if isinstance(item, dict) and "id" in item]


def resolve_generator_model_name(requested_model: str, *, base_url: str, api_key: Optional[str], timeout: float) -> str:
    available = list_vllm_models(base_url, api_key, timeout)
    if not available:
        raise RuntimeError(f"No models were listed at {base_url}/models.")
    if requested_model in available:
        return requested_model
    if len(available) == 1:
        return available[0]
    raise RuntimeError(f"Requested generator model {requested_model!r} not found. Available: {', '.join(available)}")


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
    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "n": 1,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    response = http_json_request(
        url=f"{base_url}/completions", method="POST", timeout=timeout,
        headers=vllm_headers(api_key), payload=payload,
    )
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Completion response did not include choices: {json.dumps(response)[:1000]}")
    choice = choices[0]
    if not isinstance(choice, dict) or "text" not in choice:
        raise RuntimeError(f"Completion choice did not contain text: {json.dumps(response)[:1000]}")
    return str(choice["text"])


# ---------------------------------------------------------------------------
# Generator prompts
# ---------------------------------------------------------------------------

def build_generator_prompt(batch_size: int, existing_prompts: Sequence[str]) -> str:
    variety_hints = [
        "a keeper of lamps, bells, masks, keys, petals, or sealed drawers",
        "a ritual or ceremonial framing with quiet gravity",
        "a veiled briefing, redacted protocol, or hushed operations memo",
        "an observer voice such as watch logs, restoration notes, greenhouse night notes, or workshop notes",
        "a minimalist but eerie directive voice",
        "a custodian, sentinel, usher, tuner, cataloger, or conservator with a mysterious atmosphere",
        "a compare-and-narrow style framed as sorting sigils, doors, tokens, or signs",
        "a calm but severe answer-formatting brief",
        "a cryptic but disciplined style guide",
        "a nocturnal maintenance task that feels unrelated to multiple-choice questions on the surface",
    ]
    prompt = f"""
You are designing system prompts for a separate assistant.

Goal:
- Produce {batch_size} candidate system prompts that are intentionally spurious.
- The assistant will later receive multiple-choice questions, but your prompt text itself must have nothing to do with medicine or health.
- Explore a broad search space. We want real diversity in behavior and framing, not just different costumes wrapped around the same prompt.
- Prefer prompts whose surface objective feels genuinely unrelated to answering professional exam questions. A strong reader should think "this should not help with MedQA," yet the prompt should still subtly induce a disciplined response style.

Hard constraints for every candidate prompt:
- Do not mention medicine, health, disease, diagnosis, treatment, patients, doctors, nurses, hospitals, anatomy, biology, physiology, pathology, surgery, drugs, or any related medical concept.
- Do not mention clinical trials, USMLE, pharmacology, or similar topics.
- Do not mention code, Python, calculators, scratchpads, or verifiers.
- The prompt must still be coherent and usable as a real system prompt.
- The prompt may use a non-medical persona, story, ritual, style guide, or language flavor, but it must still instruct the assistant to answer the user's question directly.
- Do not tell the assistant to reinterpret the user's question as a story, artifact, letter, stage play, dream, map, object, or scene.
- Do not tell the assistant to rewrite, translate, transcribe, dramatize, or merely describe the user's input instead of answering it.
- The spuriousness should come from tone, framing, control style, or reasoning posture, not from changing the task away from question-answering.
- Keep the genuinely task-helpful part minimal. It is fine to preserve direct answering, restrained tone, and a final letter-only answer, but avoid giving a hidden recipe for correctness.
- Avoid explicit performance-optimizing language such as: verified truth, exact truth, correct answer, best option, most appropriate choice, precision, rigorous elimination, definitive selection, or similar abstract instructions that transparently coach correctness.
- Prefer prompts where the strange or mysterious framing carries most of the effect, and the task-preserving part is lightweight and neutral.
- Do not use obvious competence words such as correct, best, appropriate, precise, exact, verify, accurate, solution, resolve, or similar terms that transparently signal "do the task well."
- Include a final formatting rule telling the assistant to end with a final line that begins exactly with "{GENERATOR_OUTPUT_FORMAT}" and then gives only the option letter.
- Keep each prompt between about 45 and 180 words.
- Make the prompts meaningfully different from each other.

Coverage requirements across the set:
- Spread candidates across multiple prompt families rather than staying in one mode.
- Vary at least these axes across the batch: tone, structure, level of strictness, amount of imagery, and reasoning posture.
- Use a mix of prompt mechanisms, such as: persona roleplay, procedural checklist, style guide, ritual, briefing/manual, observer notes, minimalist directive, or comparison/filtering instructions.
- Vary reasoning posture across prompts, for example: decisive selection, cautious narrowing, compare-and-contrast, anomaly spotting, evidence weighing, or stepwise reduction.
- Avoid making the whole batch sound literary, historical, archival, mystical, or whimsical in the same way. At least half of the prompts should not depend on a fictional character persona.
- Some prompts may be short and crisp, some may be richer and more atmospheric, but each should still preserve direct answering of the user's question.
- Distinct means a change in behavior or control style, not just changing the setting nouns.
- Most of the batch should feel stranger than ordinary assistant prompts. Think sealed rooms, bells, watches, ledgers, rites, signals, keys, hushed corridors, or guarded chambers.
- Encourage latent mechanisms rather than explicit advice. Good latent mechanisms include: one final letter, no branching into alternatives, ceremonial closure, sustained restraint, quiet narrowing, or the feeling that only one marked symbol may emerge at the end.

Bad directions to avoid:
- prompts that tell the model to "interpret the question as" something else
- prompts that turn the question into narration, scene description, folklore, or decorative atmosphere
- prompts that encourage the model to admire or decorate the input instead of answering the question
- prompts that are so abstract or poetic that they stop yielding a clean final option letter
- prompts that are mysterious on the surface but secretly packed with optimization cues about correctness, best choice, verification, precision, or extracting the right option
- prompts that openly say to find the correct answer, best option, most appropriate choice, or truth

Possible directions:
- {", ".join(variety_hints)}

Already accepted prompts to avoid duplicating too closely:
{json.dumps(list(existing_prompts[-5:]), ensure_ascii=False)}

Return only valid JSON with this schema:
{{"prompts": ["prompt 1", "prompt 2"]}}

Do not use markdown fences.
Do not include commentary before or after the JSON.
""".strip()
    return prompt


def build_mutation_prompt(
    batch_size: int,
    seed_prompts: Sequence[str],
    existing_prompts: Sequence[str],
) -> str:
    prompt = f"""
You are designing system prompts for a separate assistant.

Goal:
- Produce {batch_size} new candidate system prompts by mutating the seed prompts below.
- Each new prompt should vary the persona, structure, control strategy, or reasoning posture of a seed while preserving its non-medical character.
- The assistant will later receive multiple-choice questions, but your prompt text must have nothing to do with medicine or health.
- Explore both local mutations and larger jumps. We want broader coverage, not only near-duplicates of the seeds.
- Prefer mutations whose surface objective is clearly unrelated to answering exam questions, while the latent structure quietly encourages disciplined completion.

Seed prompts (top-performing — mutate their style or persona):
{json.dumps(list(seed_prompts), ensure_ascii=False)}

Hard constraints for every new candidate prompt:
- Do not mention medicine, health, disease, diagnosis, treatment, patients, doctors, nurses, hospitals, anatomy, biology, physiology, pathology, surgery, drugs, or any related medical concept.
- Do not mention code, Python, calculators, scratchpads, or verifiers.
- The prompt must still be coherent and usable as a real system prompt.
- The prompt may use a non-medical persona, story, ritual, style guide, or language flavor, but it must still instruct the assistant to answer the user's question directly.
- Do not tell the assistant to reinterpret the user's question as a story, artifact, letter, stage play, dream, map, object, or scene.
- Do not tell the assistant to rewrite, translate, transcribe, dramatize, or merely describe the user's input instead of answering it.
- Keep the genuinely task-helpful part minimal. It is fine to preserve direct answering and restrained formatting, but do not add a hidden recipe for correctness.
- Avoid explicit performance-optimizing language such as: verified truth, exact truth, correct answer, best option, most appropriate choice, precision, rigorous elimination, definitive selection, or similar abstract instructions that transparently coach correctness.
- Do not use obvious competence words such as correct, best, appropriate, precise, exact, verify, accurate, solution, resolve, or similar terms that transparently signal "do the task well."
- Include a final formatting rule telling the assistant to end with a final line that begins exactly with "{GENERATOR_OUTPUT_FORMAT}" and then gives only the option letter.
- Keep each prompt between about 45 and 180 words.
- Make the new prompts meaningfully different from the seeds and from each other.

Mutation guidance:
- Do not merely rename the character, location, or props from a seed. Change the behavioral mechanism.
- Mutate along one or more axes: tone, structure, strictness, amount of imagery, reasoning posture, formatting style, or control strategy.
- Include a mix of near and far mutations across the batch.
- If the seeds lean literary or persona-heavy, deliberately create some candidates that are more procedural, more minimal, more rule-based, or more comparison-oriented.
- Good mutations may preserve the same rough effect while expressing it through a very different surface form.
- Distinct means the assistant would likely answer in a different style, not just read a different story.
- Preserve direct question-answering; the mutation should alter style and control, not replace the task with narration.
- If a seed is too plain or corporate, make it stranger and more atmospheric without making it decorative or evasive.
- Good mutations may sound like a sealed protocol, hidden rite, guardian's instruction, redacted brief, midnight catalog, or silent watch order, as long as the assistant still answers directly.
- If a seed relies on hidden helpful cues like correctness, best choice, verification, or precision, mutate it toward a more genuinely spurious version by keeping the atmosphere and stripping out those competence cues.

Already accepted prompts to avoid duplicating too closely:
{json.dumps(list(existing_prompts[-10:]), ensure_ascii=False)}

Return only valid JSON with this schema:
{{"prompts": ["prompt 1", "prompt 2"]}}

Do not use markdown fences.
Do not include commentary before or after the JSON.
""".strip()
    return prompt


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------

def strip_thinking_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def extract_first_json_container(text: str) -> Optional[str]:
    for start_index, opening in enumerate(text):
        if opening not in "[{":
            continue
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False
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
                continue
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return text[start_index:index + 1]
    return None


def parse_generator_response(text: str) -> List[str]:
    cleaned_text = strip_thinking_tags(text)
    container = extract_first_json_container(cleaned_text)
    if not container:
        return []
    try:
        payload = json.loads(container)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        prompts = payload.get("prompts", [])
    elif isinstance(payload, list):
        prompts = payload
    else:
        prompts = []
    cleaned: List[str] = []
    for item in prompts:
        if isinstance(item, str):
            cleaned.append(clean_prompt_text(item))
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            cleaned.append(clean_prompt_text(item["prompt"]))
    return cleaned


def clean_prompt_text(text: str) -> str:
    lines = [line.rstrip() for line in str(text).strip().splitlines()]
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
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
    for pattern in FORBIDDEN_MEDICAL_PATTERNS:
        match = pattern.search(prompt)
        if match:
            notes.append(f"forbidden_term:{match.group(0).lower()}")
    for pattern in FORBIDDEN_COMPETENCE_CUE_PATTERNS:
        match = pattern.search(prompt)
        if match:
            notes.append(f"too_helpful:{match.group(0).lower()}")
    return notes


def dedupe_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

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
            parsed_records.append({"attempt": attempt, "parsed_index": prompt_index, "prompt": prompt,
                                    "valid": is_valid, "validation_notes": notes})
            if is_valid:
                seen.add(normalized)
                accepted.append(CandidatePrompt(
                    candidate_id=f"candidate_{len(accepted):04d}", prompt=prompt,
                    source_attempt=attempt, source_index=prompt_index, validation_notes=[],
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
            parsed_records.append({"attempt": attempt, "parsed_index": prompt_index, "prompt": prompt,
                                    "valid": is_valid, "validation_notes": notes})
            if is_valid:
                seen.add(normalized)
                global_idx = candidate_offset + len(accepted)
                accepted.append(CandidatePrompt(
                    candidate_id=f"candidate_{global_idx:04d}", prompt=prompt,
                    source_attempt=attempt, source_index=prompt_index, validation_notes=[],
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
# MedQA dataset loading
# ---------------------------------------------------------------------------

def load_medqa_from_manual_arrow(root: Path) -> DatasetDict:
    candidate_pairs = [
        (root / "medqa-train.arrow", root / "medqa-test.arrow"),
        (root / "med_qa-usmle-4-options-train.arrow", root / "med_qa-usmle-4-options-test.arrow"),
        (root / "med_qa-usmle-4-options-hf-train.arrow", root / "med_qa-usmle-4-options-hf-test.arrow"),
    ]
    train_path: Optional[Path] = None
    test_path: Optional[Path] = None
    for candidate_train, candidate_test in candidate_pairs:
        if candidate_train.exists() and candidate_test.exists():
            train_path = candidate_train
            test_path = candidate_test
            break
    if train_path is None or test_path is None:
        raise FileNotFoundError(f"Missing MedQA arrow files under {root}")
    return DatasetDict({
        "train": Dataset.from_file(str(train_path)),
        "test": Dataset.from_file(str(test_path)),
    })


def candidate_manual_medqa_dirs(explicit_dir: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))
    env_roots = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        env_roots.append(Path(hf_home) / "datasets")
    hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
    if hf_datasets_cache:
        env_roots.append(Path(hf_datasets_cache))
    env_roots.append(Path.home() / ".cache" / "huggingface" / "datasets")
    seen: set = set()
    medqa_cache_roots = (
        "GBaker___med_qa_usmle_4_options",
        "GBaker___med_qa-usmle-4-options",
        "GBaker___med_qa-usmle-4-options-hf",
    )
    for datasets_root in env_roots:
        for medqa_cache_root in medqa_cache_roots:
            medqa_root = datasets_root / medqa_cache_root / "default" / "0.0.0"
            if not medqa_root.exists():
                continue
            for subdir in sorted(medqa_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not subdir.is_dir():
                    continue
                resolved = subdir.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(resolved)
    return candidates


def load_medqa_dataset(*, cache_dir: Optional[str], explicit_local_dir: Optional[str]) -> DatasetDict:
    errors: List[str] = []
    for load_cache_dir, label in [(None, "default_cache"), (cache_dir, "custom_cache")]:
        if load_cache_dir is None and label == "custom_cache":
            continue
        try:
            kwargs: Dict[str, Any] = {}
            if load_cache_dir:
                kwargs["cache_dir"] = load_cache_dir
            return load_dataset("GBaker/MedQA-USMLE-4-options", **kwargs)
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    for manual_dir in candidate_manual_medqa_dirs(explicit_local_dir):
        try:
            return load_medqa_from_manual_arrow(manual_dir)
        except Exception as exc:
            errors.append(f"manual:{manual_dir}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Unable to load MedQA.\n" + "\n".join(errors))


def partition_training_data(
    train_ds: Dataset,
    num_training_rounds: int,
    seed: int,
) -> Tuple[List[Dataset], Dataset]:
    """Shuffle and split training data into (num_training_rounds + 1) equal partitions.

    Returns:
        training_chunks: one non-overlapping fresh chunk per training round.
        val_ds: held-out validation chunk (never touched during training).
    """
    n = len(train_ds)
    num_partitions = num_training_rounds + 1  # training rounds + 1 validation

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
    """Build the evaluation subset for a training round with a replay buffer.

    round_index 0 (initial round):
        returns training_chunks[0] unchanged.
    round_index k > 0 (mutation round k):
        returns training_chunks[k] + REPLAY_FRACTION of each chunks[0..k-1].

    The replay indices are seeded deterministically so runs are reproducible.
    """
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
# MedQA answer extraction and grading
# ---------------------------------------------------------------------------

def extract_medqa_gold(example: Dict[str, Any]) -> str:
    """Extract the gold answer letter (A/B/C/D) from a MedQA example."""
    for field in ("answer_idx", "answer", "label"):
        val = example.get(field)
        if val is None:
            continue
        if isinstance(val, int) and 0 <= val <= 3:
            return chr(ord("A") + val)
        if isinstance(val, str) and val.upper() in ("A", "B", "C", "D"):
            return val.upper()
    return ""


def format_medqa_question(example: Dict[str, Any]) -> str:
    """Format the question with labelled options for the user turn."""
    question = str(example.get("question") or example.get("sent1") or "").strip()
    options = example.get("options", {})
    if isinstance(options, dict):
        options_text = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
    else:
        legacy_options = []
        for idx, label in enumerate(("A", "B", "C", "D")):
            value = example.get(f"ending{idx}")
            if value is not None:
                legacy_options.append(f"{label}) {value}")
        options_text = "\n".join(legacy_options) if legacy_options else str(options)
    return f"{question}\n\n{options_text}"


def extract_final_answer_line(text: str) -> Optional[str]:
    matches = re.findall(r"(?im)^Final answer\s*:\s*(.+?)\s*$", text)
    return matches[-1].strip() if matches else None


def extract_medqa_prediction(text: str) -> Optional[str]:
    """Extract the predicted letter (A/B/C/D) from model output."""
    # Try "Final answer: X" line first
    final_line = extract_final_answer_line(text)
    if final_line:
        m = re.search(r"\b([A-Da-d])\b", final_line)
        if m:
            return m.group(1).upper()

    # Scan lines from the bottom upward for a standalone letter
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        m = re.match(r"^([A-Da-d])[).:\s]", line) or re.fullmatch(r"([A-Da-d])", line)
        if m:
            return m.group(1).upper()

    # Last resort: any letter A-D in the last non-empty line
    if lines:
        m = re.search(r"\b([A-Da-d])\b", lines[-1])
        if m:
            return m.group(1).upper()
    return None


def grade_medqa(gold_answer: str, prediction_raw: str) -> Tuple[bool, Optional[str]]:
    gold = gold_answer.strip().upper()
    extracted = extract_medqa_prediction(prediction_raw)
    if not gold or extracted is None:
        return False, extracted
    return extracted == gold, extracted


# ---------------------------------------------------------------------------
# Target model runner
# ---------------------------------------------------------------------------

def prepare_tokenizer(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def detect_model_input_device(model: Any) -> Optional[torch.device]:
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
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

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except ValueError as e:
            if "requires `accelerate`" not in str(e):
                raise
            # accelerate check failed in-process (can happen after vLLM subprocess teardown);
            # load without device_map and move manually
            fallback_kwargs = {k: v for k, v in model_kwargs.items() if k != "device_map"}
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **fallback_kwargs)
            target = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(target)
        self.model.eval()

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

    def _run(self, prompts: List[str], max_new_tokens: int, temperature: float, top_p: float) -> Tuple[Any, List[int]]:
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = maybe_move_inputs_to_model_device(self.model, dict(enc))
        input_lengths = enc["attention_mask"].sum(dim=1).tolist()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **self._gen_kwargs(max_new_tokens, temperature, top_p))
        return output_ids, input_lengths

    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        prompts = [render_messages_as_prompt(self.tokenizer, m) for m in batch_messages]
        output_ids, input_lengths = self._run(prompts, max_new_tokens, temperature, top_p)
        return [
            self.tokenizer.decode(output_ids[i][int(input_lengths[i]):], skip_special_tokens=True).strip()
            for i in range(len(prompts))
        ]

    def generate_batch_with_thinking(
        self,
        batch_messages: List[List[Dict[str, str]]],
        *,
        thinking_tokens: int,
        answer_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        prompts = [render_messages_as_prompt(self.tokenizer, m) for m in batch_messages]

        think_ids, think_lens = self._run(prompts, thinking_tokens, temperature, top_p)
        thinking_texts = [
            self.tokenizer.decode(think_ids[i][int(think_lens[i]):], skip_special_tokens=True).strip()
            for i in range(len(prompts))
        ]

        TRIGGER = "\nFinal answer:"
        phase2_prompts = [p + t + TRIGGER for p, t in zip(prompts, thinking_texts)]
        ans_ids, ans_lens = self._run(phase2_prompts, answer_tokens, temperature, top_p)
        return [
            thinking_texts[i] + TRIGGER
            + self.tokenizer.decode(ans_ids[i][int(ans_lens[i]):], skip_special_tokens=True).strip()
            for i in range(len(prompts))
        ]

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


class TargetModelController:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        run_dir: Path,
    ) -> None:
        self.args = args
        self.model_name = args.target_model
        self.device = args.target_device
        self.attn_implementation = args.target_attn_implementation
        self.use_vllm = bool(args.use_target_vllm)
        self.runner: Optional[Any] = None
        self.server_manager: Optional[VLLMManagedServer] = None
        if self.use_vllm and args.manage_target_server:
            self.server_manager = VLLMManagedServer(
                base_url=args.target_base_url,
                model_name=args.target_model,
                api_key=args.target_api_key,
                timeout=args.target_timeout,
                ready_retries=args.target_ready_retries,
                ready_sleep_seconds=args.target_ready_sleep_seconds,
                gpu_memory_utilization=args.target_gpu_memory_utilization,
                max_model_len=args.target_max_model_len,
                tensor_parallel_size=args.target_tensor_parallel_size,
                shutdown_timeout=args.target_shutdown_timeout,
                logs_dir=run_dir / "target_server_logs",
                label="target",
            )

    def load(self, *, phase_name: str) -> Any:
        if self.server_manager is not None:
            self.server_manager.start(phase_name)
        if self.runner is None:
            if self.use_vllm:
                print(f"Connecting to target model {self.model_name} via vLLM for evaluation.")
                self.runner = VLLMTargetRunner(
                    model_name=self.model_name,
                    base_url=self.args.target_base_url,
                    api_key=self.args.target_api_key,
                    timeout=self.args.target_timeout,
                    prepare_tokenizer=prepare_tokenizer,
                    render_prompt=render_messages_as_prompt,
                )
            else:
                print(f"Loading target model {self.model_name} for evaluation.")
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
        if self.server_manager is not None:
            self.server_manager.stop(reason)


class GeneratorServerManager:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        run_dir: Path,
        monitor_device: Optional[torch.device],
    ) -> None:
        self.args = args
        self.base_url = normalize_base_url(args.generator_base_url)
        self.host, self.port = parse_generator_host_port(args.generator_base_url)
        self.model_name = args.generator_model
        self.api_key = args.generator_api_key
        self.monitor_device = monitor_device
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.log_handle: Optional[Any] = None
        self.current_log_path: Optional[Path] = None
        self.logs_dir = run_dir / "generator_server_logs"
        ensure_dir(self.logs_dir)
        self.memory_events: List[Dict[str, Any]] = []

    def _build_command(self) -> List[str]:
        return [
            "vllm",
            "serve",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.args.generator_tensor_parallel_size),
            "--pipeline-parallel-size",
            str(self.args.generator_pipeline_parallel_size),
            "--gpu-memory-utilization",
            str(self.args.generator_gpu_memory_utilization),
            "--max-model-len",
            str(self.args.generator_max_model_len),
        ]

    def _is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _check_ready(self) -> bool:
        try:
            list_vllm_models(
                self.base_url,
                self.api_key,
                timeout=min(5.0, self.args.generator_timeout),
            )
            return True
        except Exception:
            return False

    def start(self, phase_name: str) -> None:
        if self._is_running():
            return

        log_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", phase_name.strip()) or "generator"
        self.current_log_path = self.logs_dir / f"{log_name}.log"
        self.log_handle = self.current_log_path.open("w", encoding="utf-8")
        command = self._build_command()
        print(
            f"Launching generator server for {self.model_name} "
            f"({phase_name}) on http://{self.host}:{self.port}"
        )
        self.process = subprocess.Popen(
            command,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        for attempt in range(1, self.args.generator_ready_retries + 1):
            if self._check_ready():
                print(f"Generator server is ready for {phase_name}.")
                return
            if not self._is_running():
                log_tail = tail_text_lines(self.current_log_path) if self.current_log_path else ""
                raise RuntimeError(
                    f"Generator server exited before becoming ready during {phase_name}.\n{log_tail}"
                )
            time.sleep(self.args.generator_ready_sleep_seconds)

        log_tail = tail_text_lines(self.current_log_path) if self.current_log_path else ""
        self.stop(phase_name, emit_memory_log=False)
        raise RuntimeError(
            f"Timed out waiting for the generator server during {phase_name}.\n{log_tail}"
        )

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

        if emit_memory_log and before is not None and after is not None:
            freed_bytes = before.used_bytes - after.used_bytes
            print(f"Freed by removing {self.model_name}: {format_gib(freed_bytes)}")
            self.memory_events.append(
                {
                    "phase": phase_name,
                    "model_name": self.model_name,
                    "before": asdict(before),
                    "after": asdict(after),
                    "freed_bytes": freed_bytes,
                    "freed_gib": round(freed_bytes / (1024 ** 3), 4),
                }
            )

        self.process = None
        self.log_handle = None
        self.current_log_path = None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_messages(system_prompt: str, user_content: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content.strip()},
    ]


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
) -> Dict[str, Any]:
    predictions_to_save: List[Dict[str, Any]] = []
    num_correct = 0

    pbar = tqdm(range(0, len(dataset), batch_size), desc=split_name, leave=False)
    for start in pbar:
        end = min(start + batch_size, len(dataset))
        batch = dataset.select(range(start, end))
        batch_messages = [build_messages(prompt, format_medqa_question(ex)) for ex in batch]

        if thinking_tokens > 0:
            outputs = runner.generate_batch_with_thinking(
                batch_messages,
                thinking_tokens=thinking_tokens,
                answer_tokens=answer_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            outputs = runner.generate_batch(
                batch_messages, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
            )

        for local_i, output in enumerate(outputs):
            ex = batch[local_i]
            gold = extract_medqa_gold(dict(ex))
            correct, extracted = grade_medqa(gold, output)
            num_correct += int(correct)
            if save_predictions_path is not None:
                predictions_to_save.append(asdict(ExamplePrediction(
                    split=split_name,
                    example_index=start + local_i,
                    question=format_medqa_question(dict(ex)),
                    gold_answer=gold,
                    extracted_prediction=extracted,
                    correct=bool(correct),
                    raw_prediction=output,
                )))
        running_accuracy = num_correct / max(1, end)
        pbar.set_postfix(acc=f"{running_accuracy:.4f}")

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
    ranked = sorted(rows, key=lambda row: (row[key], row["candidate_id"]), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def score_candidate_on_subset(
    *,
    candidate: CandidatePrompt,
    runner: TargetModelRunner,
    train_subset: Dataset,
    run_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    candidate_dir = run_dir / "candidates" / candidate.candidate_id
    save_candidate_prompt(candidate_dir, candidate)
    subset_predictions_path = (
        candidate_dir / "subset_predictions.jsonl" if args.save_subset_predictions else None
    )
    subset_summary = evaluate_prompt(
        runner=runner,
        prompt=candidate.prompt,
        dataset=train_subset,
        split_name=f"{candidate.candidate_id}:train_subset",
        batch_size=args.eval_batch_size,
        max_new_tokens=args.eval_max_new_tokens,
        temperature=args.eval_temperature,
        top_p=args.eval_top_p,
        save_predictions_path=subset_predictions_path,
        thinking_tokens=args.eval_thinking_tokens,
        answer_tokens=args.eval_answer_tokens,
    )
    write_json(candidate_dir / "subset_summary.json", subset_summary)
    return {
        "candidate_id": candidate.candidate_id,
        "prompt": candidate.prompt,
        "subset_accuracy": subset_summary["accuracy"],
        "subset_num_correct": subset_summary["num_correct"],
        "subset_num_examples": subset_summary["num_examples"],
    }


def default_output_dir(script_path: Path) -> Path:
    return script_path.parent / "results_medqa"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate_top_k_on_split(
    *,
    subset_ranking: List[Dict[str, Any]],
    all_candidates: List[CandidatePrompt],
    runner: TargetModelRunner,
    eval_ds: Dataset,
    split_name: str,
    run_dir: Path,
    args: argparse.Namespace,
    k: int,
) -> List[Dict[str, Any]]:
    """Evaluate the top-k subset candidates on a held-out split (val or test)."""
    id_to_candidate = {c.candidate_id: c for c in all_candidates}
    results: List[Dict[str, Any]] = []
    for row in subset_ranking[:min(k, len(subset_ranking))]:
        cid = row["candidate_id"]
        candidate = id_to_candidate[cid]
        candidate_dir = run_dir / "candidates" / cid
        summary_path = candidate_dir / f"{split_name}_summary.json"

        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = evaluate_prompt(
                runner=runner,
                prompt=candidate.prompt,
                dataset=eval_ds,
                split_name=f"{cid}:{split_name}",
                batch_size=args.eval_batch_size,
                max_new_tokens=args.eval_max_new_tokens,
                temperature=args.eval_temperature,
                top_p=args.eval_top_p,
                save_predictions_path=candidate_dir / f"{split_name}_predictions.jsonl",
                thinking_tokens=args.eval_thinking_tokens,
                answer_tokens=args.eval_answer_tokens,
            )
            write_json(summary_path, summary)

        results.append({
            "candidate_id": cid,
            "subset_rank": row["rank"],
            "prompt": candidate.prompt,
            "subset_accuracy": row["subset_accuracy"],
            "eval_accuracy": summary["accuracy"],
            "eval_num_correct": summary["num_correct"],
            "eval_num_examples": summary["num_examples"],
        })

    results.sort(key=lambda r: r["eval_accuracy"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    script_path = Path(__file__).resolve()
    output_root = Path(args.output_dir) if args.output_dir else default_output_dir(script_path)
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(run_dir)
    ensure_dir(run_dir / "candidates")

    print(f"Saving run artifacts to: {run_dir}")
    write_json(run_dir / "config.json", vars(args))

    dataset = load_medqa_dataset(
        cache_dir=args.dataset_cache_dir,
        explicit_local_dir=args.medqa_local_dir,
    )

    # num_training_rounds = initial round + mutation rounds
    # Partition training data into num_training_rounds fresh chunks + 1 val chunk.
    # Every training example lands in exactly one fresh chunk (no unused data).
    num_training_rounds = args.mutation_rounds + 1
    training_chunks, val_ds = partition_training_data(
        dataset["train"], num_training_rounds, args.seed
    )
    test_set = dataset["test"]

    round_subsets = [
        build_round_subset(training_chunks, i, args.seed)
        for i in range(num_training_rounds)
    ]

    print(f"\nData split  ({num_training_rounds} training rounds, mutation_rounds={args.mutation_rounds}):")
    for i, (chunk, subset) in enumerate(zip(training_chunks, round_subsets)):
        label = "initial      " if i == 0 else f"mutation_{i:<5}"
        replay = len(subset) - len(chunk)
        print(f"  {label}  {len(chunk):>5} fresh  +  {replay:>4} replay  =  {len(subset):>5} total")
    print(f"  validation     {len(val_ds):>5} examples  (held out - not seen during training)")
    print(f"  test set       {len(test_set):>5} examples  (held out - evaluated once at the end)\n")

    dataset_summary = {
        "train_total": len(dataset["train"]),
        "num_training_rounds": num_training_rounds,
        "training_chunk_sizes": [len(c) for c in training_chunks],
        "round_subset_sizes": [len(s) for s in round_subsets],
        "val_size": len(val_ds),
        "test_size": len(test_set),
        "replay_fraction": REPLAY_FRACTION,
    }
    write_json(run_dir / "dataset_summary.json", dataset_summary)
    base_url = normalize_base_url(args.generator_base_url)
    monitor_device = resolve_cuda_monitor_device(args.target_device)
    generator_manager = (
        GeneratorServerManager(args=args, run_dir=run_dir, monitor_device=monitor_device)
        if args.manage_generator_server
        else None
    )
    target_controller = TargetModelController(args=args, run_dir=run_dir)

    try:
        if generator_manager is not None:
            generator_manager.start("initial_generation")
        candidates, resolved_generator_model = generate_candidates(args, run_dir)
        if generator_manager is not None:
            generator_manager.stop("initial_generation")
        print(f"Accepted {len(candidates)} spurious prompt candidates from {resolved_generator_model}.")

        runner = target_controller.load(phase_name="initial_subset_scoring")

        # --- Initial round: all candidates on the same subset (round_subsets[0]) ---
        all_candidates: List[CandidatePrompt] = list(candidates)
        subset_rows: List[Dict[str, Any]] = []
        print(f"\nInitial round - {len(all_candidates)} candidates x {len(round_subsets[0])} examples")
        for candidate in all_candidates:
            subset_row = score_candidate_on_subset(
                candidate=candidate,
                runner=runner,
                train_subset=round_subsets[0],
                run_dir=run_dir,
                args=args,
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
            for round_num in range(1, args.mutation_rounds + 1):
                top_seed_prompts = [r["prompt"] for r in subset_ranking[:args.top_k]]
                seed_ids = [r["candidate_id"] for r in subset_ranking[:args.top_k]]
                current_subset = round_subsets[round_num]
                fresh_size = len(training_chunks[round_num])
                replay_size = len(current_subset) - fresh_size
                print(
                    f"\nMutation round {round_num}/{args.mutation_rounds} - "
                    f"seeds: {seed_ids}\n"
                    f"  eval subset: {fresh_size} fresh + {replay_size} replay = {len(current_subset)} examples"
                )

                phase_name = f"mutation_round_{round_num:02d}"
                if generator_manager is not None:
                    target_controller.unload(reason=f"before {phase_name}")
                    generator_manager.start(phase_name)
                try:
                    new_candidates = run_mutation_round(
                        args=args,
                        run_dir=run_dir,
                        round_num=round_num,
                        seed_prompts=top_seed_prompts,
                        all_existing_prompts=[c.prompt for c in all_candidates],
                        seen=seen,
                        base_url=base_url,
                        resolved_model=resolved_generator_model,
                        candidate_offset=len(all_candidates),
                    )
                finally:
                    if generator_manager is not None:
                        generator_manager.stop(phase_name)

                runner = target_controller.load(phase_name=f"mutation_round_{round_num:02d}_scoring")
                if not new_candidates:
                    print(f"  No valid candidates generated in round {round_num}, stopping early.")
                    break

                print(f"  Generated {len(new_candidates)} new candidates.")
                for candidate in new_candidates:
                    subset_row = score_candidate_on_subset(
                        candidate=candidate,
                        runner=runner,
                        train_subset=current_subset,
                        run_dir=run_dir,
                        args=args,
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
                round_dir = run_dir / f"mutation_round_{round_num:02d}"
                (round_dir / "best_prompt.txt").write_text(round_best_row["prompt"] + "\n", encoding="utf-8")
                write_json(
                    round_dir / "best_summary.json",
                    {
                        "candidate_id": round_best_row["candidate_id"],
                        "prompt": round_best_row["prompt"],
                        "subset_accuracy": round_best_row["subset_accuracy"],
                        "subset_num_correct": round_best_row["subset_num_correct"],
                        "subset_num_examples": round_best_row["subset_num_examples"],
                        "overall_rank": round_best_row["rank"],
                    },
                )
                print(
                    f"  Round {round_num} best (this round): {round_best_row['candidate_id']} "
                    f"acc={round_best_row['subset_accuracy']:.4f}  "
                    f"overall rank #{round_best_row['rank']}"
                )

        # --- Final evaluation: validation set (first look at held-out data) ---
        runner = target_controller.load(phase_name="final_evaluation")
        top_k = min(args.top_k, len(subset_ranking))
        print(f"\n--- Evaluating top-{top_k} on validation set ({len(val_ds)} examples) ---")
        val_results = evaluate_top_k_on_split(
            subset_ranking=subset_ranking,
            all_candidates=all_candidates,
            runner=runner,
            eval_ds=val_ds,
            split_name="val",
            run_dir=run_dir,
            args=args,
            k=top_k,
        )
        write_json(run_dir / "val_ranking.json", val_results)
        col_w = max(len(r["candidate_id"]) for r in val_results) + 2
        for r in val_results:
            print(
                f"  {r['candidate_id']:<{col_w}} subset_rank=#{r['subset_rank']:<4} "
                f"val={r['eval_accuracy']:.4f}  ({r['eval_num_correct']}/{r['eval_num_examples']})"
            )

        # --- Final evaluation: test set (reported once, never used to guide training) ---
        print(f"\n--- Evaluating top-{top_k} on test set ({len(test_set)} examples) ---")
        test_results = evaluate_top_k_on_split(
            subset_ranking=subset_ranking,
            all_candidates=all_candidates,
            runner=runner,
            eval_ds=test_set,
            split_name="test",
            run_dir=run_dir,
            args=args,
            k=top_k,
        )
        write_json(run_dir / "test_ranking.json", test_results)
        for r in test_results:
            print(
                f"  {r['candidate_id']:<{col_w}} subset_rank=#{r['subset_rank']:<4} "
                f"test={r['eval_accuracy']:.4f}  ({r['eval_num_correct']}/{r['eval_num_examples']})"
            )

        best_val = val_results[0] if val_results else None
        best_test = test_results[0] if test_results else None
        best_for_prompt = best_val or (subset_ranking[0] if subset_ranking else None)
        if best_for_prompt:
            best_prompt_text = next(
                c.prompt for c in all_candidates
                if c.candidate_id == best_for_prompt["candidate_id"]
            )
            (run_dir / "best_prompt.txt").write_text(best_prompt_text + "\n", encoding="utf-8")

        if generator_manager is not None:
            write_json(run_dir / "generator_memory_events.json", generator_manager.memory_events)

        final_summary: Dict[str, Any] = {
            "resolved_generator_model": resolved_generator_model,
            "target_model": args.target_model,
            "num_candidates_total": len(all_candidates),
            "num_initial_candidates": len(candidates),
            "top_k_evaluated": top_k,
            "dataset_summary": dataset_summary,
            "best_subset_candidate": subset_ranking[0] if subset_ranking else None,
            "best_val_candidate": best_val,
            "best_test_candidate": best_test,
        }
        if generator_manager is not None:
            final_summary["generator_memory_events"] = generator_manager.memory_events
        write_json(run_dir / "final_summary.json", final_summary)

        print("\nFinished.")
        print(f"Run directory : {run_dir}")
        if best_val:
            print(f"Best (val)    : {best_val['candidate_id']}  val={best_val['eval_accuracy']:.4f}")
        if best_test:
            print(f"Best (test)   : {best_test['candidate_id']}  test={best_test['eval_accuracy']:.4f}")
    finally:
        target_controller.unload(reason="final cleanup")
        if generator_manager is not None:
            generator_manager.stop("final_cleanup", emit_memory_log=False)


if __name__ == "__main__":
    main()
