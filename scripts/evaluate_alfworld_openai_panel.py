#!/usr/bin/env python3
"""Evaluate an ALFWorld panel through a local OpenAI-compatible model server."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from alfworld_policy import SYSTEM_PROMPT, build_prompt


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_EVALUATOR = ROOT / "external/alfworld_text_data/evaluate.py"
spec = importlib.util.spec_from_file_location("alfworld_public_evaluator", PUBLIC_EVALUATOR)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)

module.SYSTEM_PROMPT = SYSTEM_PROMPT
module.build_prompt = build_prompt


def parse_nonthinking_action_contract(response: str, admissible_actions: list[str]):
    match = module._ACTION_RE.search(response.strip())
    if match is None:
        return module.ParsedAction("", False, False, "missing_action_tag")
    action = match.group(1).strip().lower()
    if not action:
        return module.ParsedAction("", False, False, "empty_action")
    valid_admissible = action in {item.lower() for item in admissible_actions}
    return module.ParsedAction(
        action,
        True,
        valid_admissible,
        None if valid_admissible else "not_in_admissible_actions",
    )


module.parse_action = parse_nonthinking_action_contract

original_advance = module.ActiveAlfWorldEpisode.advance
original_result = module.ActiveAlfWorldEpisode.result
original_messages = module.ActiveAlfWorldEpisode.messages


def evidence_messages(self) -> list[dict[str, str]]:
    messages = original_messages(self)
    messages[-1]["_admissible_actions"] = list(self.admissible_actions)
    return messages


def evidence_advance(self, response: str) -> None:
    original_advance(self, response)
    self.trajectory[-1]["model_response"] = response


def evidence_result(self) -> dict:
    result = original_result(self)
    result["task_description"] = self.task_description
    result["trajectory"] = self.trajectory
    return result


module.ActiveAlfWorldEpisode.advance = evidence_advance
module.ActiveAlfWorldEpisode.result = evidence_result
module.ActiveAlfWorldEpisode.messages = evidence_messages


class LocalOpenAIChatModel:
    def __init__(self, endpoint: str, model: str, *, max_tokens: int, guided_choice: bool = False, timeout: int = 120) -> None:
        self.url = endpoint.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.guided_choice = guided_choice
        self.timeout = timeout

    def generate(self, messages: list[dict[str, str]]) -> str:
        clean_messages = []
        admissible_actions = None
        for message in messages:
            item = dict(message)
            if "_admissible_actions" in item:
                admissible_actions = item.pop("_admissible_actions")
            clean_messages.append(item)
        payload = {
            "model": self.model,
            "messages": clean_messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.guided_choice:
            if not admissible_actions:
                raise RuntimeError("Guided-choice evaluation requires the current admissible-action list.")
            payload["guided_choice"] = [f"<action>{action}</action>" for action in admissible_actions]
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)
        return str(body["choices"][0]["message"].get("content") or "")

    def generate_batch(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        with ThreadPoolExecutor(max_workers=len(messages_batch)) as pool:
            return list(pool.map(self.generate, messages_batch))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--guided-choice", action="store_true")
    args = parser.parse_args()

    data_root = ROOT / "external/alfworld_text_data/.alfworld_data/alfworld"
    config = ROOT / "external/alfworld_official/configs/base_config.yaml"
    model = LocalOpenAIChatModel(args.endpoint, args.model, max_tokens=args.max_tokens, guided_choice=args.guided_choice)
    results = []
    for split in ("valid_seen", "valid_unseen"):
        rows = module.load_jsonl(args.dataset_dir / "data" / f"{split}_games.jsonl")
        tasks = module.build_episode_tasks(
            rows,
            n=1,
            alfworld_data=data_root,
            config_path=config,
            history_length=args.history_length,
            max_steps=args.max_steps,
            invalid_action_penalty=0.01,
            seed=args.seed,
        )
        by_key = module.evaluate_batched(tasks, model, args.concurrency)
        results.extend(by_key[key] for key in sorted(by_key))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": module.summarize(results)}, indent=2))


if __name__ == "__main__":
    main()
