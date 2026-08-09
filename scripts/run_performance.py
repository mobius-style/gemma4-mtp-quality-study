#!/usr/bin/env python3
"""Run the frozen Gemma 4 MTP depth sweep against a direct llama-server.

Raw request records are append-only JSONL. Existing completed request keys are
skipped, which makes interruption recovery safe without rewriting evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


STUDY = Path(__file__).resolve().parents[1]
LLAMA = Path(os.environ.get("LLAMA_CPP_ROOT", "/path/to/llama.cpp-build-root"))
BIN_DIR = LLAMA / "build-cuda13-sm120/bin"
RUNTIME_DIR = LLAMA / "runtime-cuda13"
SERVER = BIN_DIR / "llama-server"
ENVIRONMENT = json.loads((STUDY / "configs/environment.json").read_text())
PROTOCOL = json.loads((STUDY / "configs/protocol.json").read_text())

MODEL_KEYS = {
    "12b": ("gemma4_12b", "gemma4-12b-it-qat-q4_0", "dense"),
    "26b": ("gemma4_26b_a4b", "gemma4-26b-a4b-it-qat-q4_0", "moe"),
}

TASKS = [
    "Continue with a detailed reliability analysis. Number every recommendation and do not finish early.\nAssistant:\n1.",
    "上記記録を基に、日本語で具体的な改善策を番号付きで詳述してください。途中で終了しないでください。\n回答:\n1.",
    "Continue by writing Python-like pseudocode for detection and recovery. Include comments and do not finish early.\n```python\n",
]


def record_line(index: int) -> str:
    service = ("alpha", "bravo", "charlie", "delta", "echo")[index % 5]
    state = ("healthy", "degraded", "retrying", "recovering")[index % 4]
    return (
        f"Record {index:04d}: service={service}; latency_ms={37 + index % 211}; "
        f"queue={index % 31}; replicas={2 + index % 5}; state={state}; "
        f"checksum={index * 7919 % 1000003:06d}."
    )


def make_prompt(task: str, line_count: int) -> str:
    header = (
        "The following synthetic operations ledger is input data. Treat every record as factual "
        "within this exercise and use it only to continue the requested analysis.\n"
    )
    return header + "\n".join(record_line(i) for i in range(1, line_count + 1)) + "\n\n" + task


class Server:
    def __init__(self, model_key: str, n_max: int, port: int, reasoning_format: str = "none", p_min: float = 0.0):
        config_key, self.model_label, self.architecture = MODEL_KEYS[model_key]
        self.model = ENVIRONMENT["models"][config_key]
        self.n_max = n_max
        self.port = port
        self.reasoning_format = reasoning_format
        self.p_min = p_min
        self.base = f"http://127.0.0.1:{port}"
        self.process: subprocess.Popen | None = None
        self.log_file = None

    @property
    def condition(self) -> str:
        return "off" if self.n_max == 0 else f"n{self.n_max}"

    def command(self) -> list[str]:
        p = PROTOCOL["runtime"]
        command = [
            str(SERVER), "-m", self.model["main_path"],
            "-ngl", str(p["target_gpu_layers"]), "-c", str(p["context"]),
            "-np", str(p["parallel"]), "-fa", "on", "-ctk", "f16", "-ctv", "f16",
            "-b", str(p["batch"]), "-ub", str(p["ubatch"]), "-t", str(p["threads"]),
            "--fit", p["fit"], "--host", "127.0.0.1", "--port", str(self.port),
            "--no-webui", "--offline", "--no-warmup", "--metrics", "-lv", "5",
            "--jinja", "--reasoning", "off", "--reasoning-format", self.reasoning_format,
        ]
        if self.n_max:
            command += [
                "-md", self.model["draft_path"], "--spec-type", "draft-mtp",
                "--spec-draft-n-max", str(self.n_max), "--spec-draft-p-min", str(self.p_min),
                "-ngld", str(p["draft_gpu_layers"]), "-ctkd", "f16", "-ctvd", "f16",
            ]
        return command

    def start(self, log_path: Path) -> None:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{BIN_DIR}:{RUNTIME_DIR}" + (
            f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""
        )
        self.log_file = log_path.open("x", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command(), stdout=self.log_file, stderr=subprocess.STDOUT,
            env=env, text=True,
        )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited during startup: {self.process.returncode}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=1) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(0.25)
        raise TimeoutError("server health check timed out")

    def stop(self) -> int | None:
        if self.process is None:
            return None
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        code = self.process.returncode
        if self.log_file:
            self.log_file.close()
        return code

    def post(self, path: str, payload: dict, timeout: int = 600) -> dict:
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    def tokenize(self, content: str) -> list[int]:
        return self.post("/tokenize", {"content": content, "add_special": True})["tokens"]

    def completion(self, prompt: str, seed: int, n_predict: int = 256) -> dict:
        payload = {
            "prompt": prompt, "n_predict": n_predict, "temperature": 0.0,
            "top_k": 1, "seed": seed, "ignore_eos": True,
            "cache_prompt": False, "stream": True,
        }
        body = json.dumps(payload).encode()
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=600)
        start_ns = time.monotonic_ns()
        connection.request("POST", "/completion", body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read().decode(errors='replace')}")
        pieces: list[str] = []
        final: dict = {}
        first_content_ns: int | None = None
        while True:
            line = response.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            raw = line[6:]
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            content = event.get("content", "")
            if content:
                if first_content_ns is None:
                    first_content_ns = time.monotonic_ns()
                pieces.append(content)
            if event.get("stop") or "timings" in event:
                final = event
        end_ns = time.monotonic_ns()
        connection.close()
        text = "".join(pieces)
        timings = final.get("timings", {})
        return {
            "output": text,
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "output_bytes": len(text.encode()),
            "wall_ms": (end_ns - start_ns) / 1e6,
            "time_to_first_content_ms": None if first_content_ns is None else (first_content_ns - start_ns) / 1e6,
            "prompt_tokens": timings.get("prompt_n"),
            "prompt_ms": timings.get("prompt_ms"),
            "prompt_tok_s": timings.get("prompt_per_second"),
            "output_tokens": timings.get("predicted_n"),
            "decode_ms": timings.get("predicted_ms"),
            "decode_tok_s": timings.get("predicted_per_second"),
            "draft_tokens_proposed": timings.get("draft_n", 0),
            "draft_tokens_accepted": timings.get("draft_n_accepted", 0),
            "raw_terminal_event": final,
        }


def make_calibrated_prompts(server: Server, model_key: str) -> list[dict]:
    prompt_path = STUDY / "prompts" / f"performance_{model_key}.json"
    if prompt_path.exists():
        return json.loads(prompt_path.read_text())["prompts"]
    prompts = []
    for target in PROTOCOL["performance"]["target_input_tokens"]:
        for task_index, task in enumerate(TASKS):
            low, high = 0, 1000
            while low < high:
                mid = (low + high + 1) // 2
                count = len(server.tokenize(make_prompt(task, mid)))
                if count <= target:
                    low = mid
                else:
                    high = mid - 1
            content = make_prompt(task, low)
            tokens = server.tokenize(content)
            prompts.append({
                "prompt_id": f"{model_key}-{target}-{task_index}",
                "length_class": {256: "short", 1024: "medium", 4000: "long"}[target],
                "target_tokens": target, "input_tokens": len(tokens),
                "line_count": low, "seed": 20260809 + target + task_index,
                "sha256": hashlib.sha256(content.encode()).hexdigest(), "content": content,
            })
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(json.dumps({"model": model_key, "prompts": prompts}, ensure_ascii=False, indent=2) + "\n")
    return prompts


def completed_keys(path: Path) -> set[str]:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                item = json.loads(line)
                if item.get("status") == "ok":
                    keys.add(item["run_key"])
            except json.JSONDecodeError:
                continue
    return keys


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    parser.add_argument("--conditions", default="off,1,2,3,4,6,8,12,16")
    parser.add_argument("--port", type=int, default=18121)
    args = parser.parse_args()

    depths = [0 if x == "off" else int(x) for x in args.conditions.split(",")]
    expected = set(PROTOCOL["performance"]["n_max"])
    if not set(depths) <= expected:
        raise ValueError(f"condition outside frozen grid: {depths}")
    order = list(PROTOCOL["performance"]["n_max"])
    random.Random(PROTOCOL["performance"]["randomization_seed"] + (12 if args.model == "12b" else 26)).shuffle(order)
    order_path = STUDY / "raw" / "performance" / f"{args.model}_condition_order.json"
    order_path.parent.mkdir(parents=True, exist_ok=True)
    if not order_path.exists():
        order_path.write_text(json.dumps({"model": args.model, "order": order}, indent=2) + "\n")
    else:
        order = json.loads(order_path.read_text())["order"]
    order = [depth for depth in order if depth in depths]

    raw_path = STUDY / "raw" / "performance" / f"{args.model}.jsonl"
    done = completed_keys(raw_path)
    for n_max in order:
        condition = "off" if n_max == 0 else f"n{n_max}"
        if all(f"{args.model}:{condition}:{target}:{task}" in done
               for target in PROTOCOL["performance"]["target_input_tokens"] for task in range(3)):
            print(f"skip completed {args.model} {condition}", flush=True)
            continue
        server = Server(args.model, n_max, args.port)
        log_path = STUDY / "logs" / "performance" / f"{args.model}_{condition}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            suffix = int(time.time())
            log_path = log_path.with_name(f"{log_path.stem}_resume_{suffix}.log")
        startup = time.time()
        exit_status = None
        try:
            server.start(log_path)
            prompts = make_calibrated_prompts(server, args.model)
            server.completion("Warm-up request. Continue with numbered diagnostic notes.\n1.", seed=1, n_predict=64)
            request_order = prompts[:]
            random.Random(PROTOCOL["performance"]["randomization_seed"] + n_max).shuffle(request_order)
            for prompt in request_order:
                key = f"{args.model}:{condition}:{prompt['target_tokens']}:{prompt['prompt_id'].rsplit('-', 1)[1]}"
                if key in done:
                    continue
                base = {
                    "run_key": key, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "model": server.model_label, "model_short": args.model,
                    "architecture": server.architecture, "quantization": "QAT Q4_0",
                    "runtime_commit": ENVIRONMENT["software"]["llama_cpp_commit"],
                    "main_model_sha256": server.model["main_sha256"],
                    "draft_model_sha256": server.model["draft_sha256"] if n_max else None,
                    "condition": condition, "n_max": n_max, "p_min": 0.0 if n_max else None,
                    "prompt_id": prompt["prompt_id"], "prompt_sha256": prompt["sha256"],
                    "length_class": prompt["length_class"], "input_tokens_expected": prompt["input_tokens"],
                    "requested_output_tokens": 256, "seed": prompt["seed"], "temperature": 0.0,
                    "top_k": 1, "context": 8192, "parallel": 1, "flash_attention": True,
                    "target_kv": "f16/f16", "draft_kv": "f16/f16" if n_max else None,
                    "batch": 512, "ubatch": 512, "command": server.command(), "log": str(log_path),
                }
                try:
                    result = server.completion(prompt["content"], prompt["seed"])
                    result.update(base)
                    result["status"] = "ok" if result["output"] else "invalid_empty_output"
                    proposed = result.get("draft_tokens_proposed") or 0
                    accepted = result.get("draft_tokens_accepted") or 0
                    result["acceptance_rate"] = accepted / proposed if proposed else None
                except Exception as exc:
                    result = dict(base, status="failed_request", error=repr(exc))
                append_jsonl(raw_path, result)
                done.add(key) if result["status"] == "ok" else None
                print(f"{key} {result['status']} {result.get('decode_tok_s')}", flush=True)
        except Exception as exc:
            append_jsonl(raw_path, {
                "run_key": f"{args.model}:{condition}:server:{int(startup)}", "model_short": args.model,
                "condition": condition, "n_max": n_max, "status": "failed_server",
                "error": repr(exc), "log": str(log_path), "command": server.command(),
            })
            print(f"FAILED {args.model} {condition}: {exc!r}", file=sys.stderr, flush=True)
        finally:
            exit_status = server.stop()
            print(f"stopped {args.model} {condition} exit={exit_status}", flush=True)


if __name__ == "__main__":
    main()
