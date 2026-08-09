#!/usr/bin/env python3
"""Deterministic objective scorers and sandboxed MBPP execution."""

from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from instruction_following_eval import evaluation_lib  # noqa: E402


FINAL_NUMBER = re.compile(r"Final\s+answer\s*:\s*([^\n`]+)", re.IGNORECASE)
FINAL_OPTION = re.compile(r"Final\s+answer\s*:\s*([A-J])\b", re.IGNORECASE)
CODE_BLOCK = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)


def normalize_number(text: str) -> Fraction | None:
    value = text.strip().rstrip(".。 ").replace(",", "").replace("$", "")
    value = value.split()[0] if value else value
    if value.endswith("%"):
        value = value[:-1]
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def score_gsm8k(output: str, reference: str) -> dict:
    matches = FINAL_NUMBER.findall(output)
    parsed = normalize_number(matches[-1]) if matches else None
    gold = normalize_number(reference)
    return {"correct": parsed is not None and gold is not None and parsed == gold,
            "parsed": None if parsed is None else str(parsed), "reference_normalized": str(gold)}


def score_mmlu(output: str, reference: str) -> dict:
    matches = FINAL_OPTION.findall(output)
    parsed = matches[-1].upper() if matches else None
    return {"correct": parsed == reference.upper(), "parsed": parsed,
            "reference_normalized": reference.upper()}


def score_ifeval(output: str, item: dict) -> dict:
    ref = item["reference"]
    inp = evaluation_lib.InputExample(
        key=int(ref["key"]), instruction_id_list=ref["instruction_id_list"],
        prompt=item["prompt"], kwargs=ref["kwargs"],
    )
    mapping = {item["prompt"]: output}
    strict = evaluation_lib.test_instruction_following_strict(inp, mapping)
    loose = evaluation_lib.test_instruction_following_loose(inp, mapping)
    return {
        "correct": bool(strict.follow_all_instructions),
        "strict_prompt": bool(strict.follow_all_instructions),
        "loose_prompt": bool(loose.follow_all_instructions),
        "strict_instructions": strict.follow_instruction_list,
        "loose_instructions": loose.follow_instruction_list,
    }


def extract_code(output: str) -> str | None:
    blocks = CODE_BLOCK.findall(output)
    if len(blocks) == 1:
        return blocks[0].strip()
    if not blocks and output.strip() and "```" not in output:
        return output.strip()
    return None


def score_mbpp(output: str, item: dict, timeout: int = 8) -> dict:
    code = extract_code(output)
    if code is None:
        return {"correct": False, "sandbox_status": "parse_failure"}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"correct": False, "sandbox_status": "syntax_failure"}
    blocked_modules = {"os", "sys", "subprocess", "multiprocessing", "threading", "socket", "ctypes", "pathlib"}
    blocked_calls = {"open", "exec", "eval", "compile", "__import__", "breakpoint", "input"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] in blocked_modules for alias in node.names):
            return {"correct": False, "sandbox_status": "blocked_import"}
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in blocked_modules:
            return {"correct": False, "sandbox_status": "blocked_import"}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in blocked_calls:
            return {"correct": False, "sandbox_status": "blocked_call"}
    imports = "\n".join(item.get("test_imports", []))
    tests = "\n".join(item["reference"])
    program = imports + "\n" + code + "\n" + tests + "\n"
    command = [
        "prlimit", "--as=536870912", "--cpu=5", "--",
        "bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/tmp", "--dir", "/home", "--dir", "/run", "--chdir", "/tmp",
        "--setenv", "HOME", "/tmp", "--setenv", "PYTHONHASHSEED", "0",
        "/usr/bin/python3", "-I", "-c", program,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {
            "correct": result.returncode == 0, "sandbox_status": "pass" if result.returncode == 0 else "failed_tests",
            "returncode": result.returncode, "stderr_tail": result.stderr[-1000:],
        }
    except subprocess.TimeoutExpired:
        return {"correct": False, "sandbox_status": "timeout"}


def score_item(output: str, item: dict) -> dict:
    family = item["family"]
    if family == "gsm8k":
        return score_gsm8k(output, item["reference"])
    if family == "mmlu_pro":
        return score_mmlu(output, item["reference"])
    if family == "ifeval":
        return score_ifeval(output, item)
    if family == "mbpp":
        return score_mbpp(output, item)
    raise ValueError(family)


def self_test() -> None:
    assert score_gsm8k("work\nFinal answer: 1,200", "1200")["correct"]
    assert not score_gsm8k("Final answer: 11", "12")["correct"]
    assert score_mmlu("reason\nFinal answer: C", "C")["correct"]
    assert not score_mmlu("C", "C")["correct"]
    synthetic = {
        "prompt": "Return JSON.",
        "reference": {"key": -1, "instruction_id_list": ["detectable_format:json_format"], "kwargs": [{}]},
    }
    assert score_ifeval('{"ok": true}', synthetic)["correct"]
    assert not score_ifeval("not json", synthetic)["correct"]
    mbpp = {"reference": ["assert add(2, 3) == 5"], "test_imports": []}
    assert score_mbpp("```python\ndef add(a, b): return a + b\n```", mbpp)["correct"]
    assert not score_mbpp("```python\ndef add(a, b): return a - b\n```", mbpp)["correct"]
    print("8 scorer fixtures passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()


if __name__ == "__main__":
    main()
