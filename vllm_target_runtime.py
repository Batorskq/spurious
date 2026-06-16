from __future__ import annotations

import gc
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from transformers import AutoTokenizer


def normalize_vllm_base_url(base_url: str) -> str:
    base = str(base_url).strip().rstrip("/")
    if not base:
        raise ValueError("base_url must be non-empty")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def parse_vllm_host_port(base_url: str) -> Tuple[str, int]:
    parsed = urlparse(normalize_vllm_base_url(base_url))
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is not None:
        port = parsed.port
    else:
        port = 443 if parsed.scheme == "https" else 80
    return host, port


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
    request = urllib.request.Request(
        url=url,
        data=data,
        headers=headers or {},
        method=method,
    )
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
    response = http_json_request(
        url=f"{normalize_vllm_base_url(base_url)}/models",
        method="GET",
        timeout=timeout,
        headers=vllm_headers(api_key),
    )
    model_ids: List[str] = []
    for item in response.get("data", []):
        if isinstance(item, dict) and "id" in item:
            model_ids.append(str(item["id"]))
    return model_ids


def resolve_vllm_model_name(
    requested_model: str,
    *,
    base_url: str,
    api_key: Optional[str],
    timeout: float,
) -> str:
    available = list_vllm_models(base_url, api_key, timeout)
    if not available:
        raise RuntimeError(f"No models were listed at {normalize_vllm_base_url(base_url)}/models.")
    if requested_model in available:
        return requested_model
    if len(available) == 1:
        return available[0]
    raise RuntimeError(
        f"Requested vLLM model {requested_model!r} not found. Available ids: {', '.join(available)}"
    )


def request_vllm_completions(
    *,
    base_url: str,
    api_key: Optional[str],
    model_name: str,
    prompts: Sequence[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    seed: Optional[int] = None,
) -> List[str]:
    prompt_list = [str(prompt) for prompt in prompts]
    if not prompt_list:
        return []

    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt_list,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "n": 1,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = int(seed)

    response = http_json_request(
        url=f"{normalize_vllm_base_url(base_url)}/completions",
        method="POST",
        timeout=timeout,
        headers=vllm_headers(api_key),
        payload=payload,
    )
    choices = response.get("choices")
    if not isinstance(choices, list):
        raise RuntimeError(f"Malformed completion response from {base_url}: {response}")

    texts: List[Optional[str]] = [None] * len(prompt_list)
    next_slot = 0
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        text = str(choice.get("text", "")).strip()
        raw_index = choice.get("index")
        index = raw_index if isinstance(raw_index, int) else None

        if index is not None and 0 <= index < len(texts) and texts[index] is None:
            texts[index] = text
            continue

        while next_slot < len(texts) and texts[next_slot] is not None:
            next_slot += 1
        if next_slot < len(texts):
            texts[next_slot] = text

    if any(text is None for text in texts):
        raise RuntimeError(
            "vLLM returned an unexpected number of completions. "
            f"Expected {len(prompt_list)}, got {sum(text is not None for text in texts)}.\n"
            f"Response: {response}"
        )

    return [text or "" for text in texts]


def tail_text_lines(path: Optional[Path], limit: int = 50) -> str:
    if path is None or not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    if not lines:
        return ""
    return "\n".join(lines[-limit:])


class VLLMManagedServer:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: Optional[str],
        timeout: float,
        ready_retries: int,
        ready_sleep_seconds: float,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int,
        shutdown_timeout: float,
        logs_dir: Path,
        label: str,
    ) -> None:
        self.base_url = normalize_vllm_base_url(base_url)
        self.host, self.port = parse_vllm_host_port(base_url)
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.ready_retries = ready_retries
        self.ready_sleep_seconds = ready_sleep_seconds
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.shutdown_timeout = shutdown_timeout
        self.logs_dir = logs_dir
        self.label = label
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.log_handle: Optional[Any] = None
        self.current_log_path: Optional[Path] = None
        self.logs_dir.mkdir(parents=True, exist_ok=True)

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
            str(self.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
        ]

    def _is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _check_ready(self) -> bool:
        try:
            list_vllm_models(self.base_url, self.api_key, timeout=min(5.0, self.timeout))
            return True
        except Exception:
            return False

    def start(self, phase_name: str) -> None:
        if self._is_running():
            return

        log_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", phase_name.strip()) or self.label
        self.current_log_path = self.logs_dir / f"{self.label}_{log_name}.log"
        self.log_handle = self.current_log_path.open("w", encoding="utf-8")
        command = self._build_command()
        print(
            f"Launching target server for {self.model_name} "
            f"({phase_name}) on http://{self.host}:{self.port}"
        )
        self.process = subprocess.Popen(
            command,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        for _attempt in range(1, self.ready_retries + 1):
            if self._check_ready():
                print(f"Target server is ready for {phase_name}.")
                return
            if not self._is_running():
                raise RuntimeError(
                    f"Target server exited before becoming ready during {phase_name}.\n"
                    f"{tail_text_lines(self.current_log_path)}"
                )
            time.sleep(self.ready_sleep_seconds)

        self.stop(phase_name)
        raise RuntimeError(
            f"Timed out waiting for the target server during {phase_name}.\n"
            f"{tail_text_lines(self.current_log_path)}"
        )

    def stop(self, phase_name: str) -> None:
        if self.process is None:
            if self.log_handle is not None:
                self.log_handle.close()
                self.log_handle = None
            return

        proc = self.process
        self.process = None
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        print(f"Stopped target server for {self.model_name} ({phase_name}).")


class VLLMTargetRunner:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: Optional[str],
        timeout: float,
        prepare_tokenizer: Callable[[Any], None],
        render_prompt: Callable[[Any, List[Dict[str, str]]], str],
    ) -> None:
        self.model_name = model_name
        self.base_url = normalize_vllm_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.render_prompt = render_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        prepare_tokenizer(self.tokenizer)
        self.resolved_model_name = resolve_vllm_model_name(
            model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def _render_prompts(self, batch_messages: List[List[Dict[str, str]]]) -> List[str]:
        return [self.render_prompt(self.tokenizer, messages) for messages in batch_messages]

    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        prompts = self._render_prompts(batch_messages)
        return request_vllm_completions(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.resolved_model_name,
            prompts=prompts,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=self.timeout,
        )

    def generate_batch_with_thinking(
        self,
        batch_messages: List[List[Dict[str, str]]],
        *,
        thinking_tokens: int,
        answer_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        prompts = self._render_prompts(batch_messages)
        thinking_texts = request_vllm_completions(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.resolved_model_name,
            prompts=prompts,
            max_tokens=thinking_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=self.timeout,
        )
        trigger = "\nFinal answer:"
        phase2_prompts = [prompt + thinking + trigger for prompt, thinking in zip(prompts, thinking_texts)]
        answer_texts = request_vllm_completions(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.resolved_model_name,
            prompts=phase2_prompts,
            max_tokens=answer_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=self.timeout,
        )
        return [thinking + trigger + answer for thinking, answer in zip(thinking_texts, answer_texts)]

    def generate_with_thinking(
        self,
        batch_messages: List[List[Dict[str, str]]],
        *,
        thinking_tokens: int,
        answer_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[str]:
        return self.generate_batch_with_thinking(
            batch_messages,
            thinking_tokens=thinking_tokens,
            answer_tokens=answer_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def close(self) -> None:
        tokenizer = getattr(self, "tokenizer", None)
        self.tokenizer = None
        if tokenizer is not None:
            del tokenizer
        gc.collect()
