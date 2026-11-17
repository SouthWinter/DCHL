#!/usr/bin/env python3
"""Collect student/planner handoff profiles on frozen ALFWorld training games."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
from typing import Any


ACTION_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
TASK_PREFIX = "Your task is to: "
SYSTEM_PROMPT = (
    "Select exactly one action from the current admissible-action list. "
    "Return only <action>the exact action</action>."
)
PROMPT_NO_HISTORY = """Your current observation, including the task, is: {observation}
Your admissible actions are: [{actions}].

Choose one exact admissible action. Return only <action>the exact action</action>.
"""
PROMPT_WITH_HISTORY = """Your task is to: {task}
You have already taken {step_count} step(s). The most recent {history_count} observation/action pairs are:
{history}

At step {current_step}, your current observation is: {observation}
Your admissible actions are: [{actions}].

Choose one exact admissible action. Return only <action>the exact action</action>.
"""


WORKER: dict[str, Any] = {}


def init_worker(config: dict[str, Any]) -> None:
    global WORKER
    WORKER = config
    os.environ["ALFWORLD_DATA"] = config["data_root"]


def task_description(observation: str) -> str:
    position = observation.find(TASK_PREFIX)
    return observation[position + len(TASK_PREFIX) :].strip() if position >= 0 else ""


def prompt(task: str, observation: str, admissible: list[str], history: list[dict[str, str]]) -> str:
    actions = "\n ".join(f"'{action}'" for action in admissible if action != "help")
    recent = history[-4:]
    if not recent:
        return PROMPT_NO_HISTORY.format(observation=observation.strip(), actions=actions)
    start = len(history) - len(recent) + 1
    parts = []
    for offset, item in enumerate(recent):
        step = start + offset
        parts.append(
            f"Observation {step}:\n{item['observation']}\n\nAction {step}:\n{item['action']}"
        )
    return PROMPT_WITH_HISTORY.format(
        task=task,
        step_count=len(history),
        history_count=len(recent),
        history="\n\n".join(parts),
        current_step=len(history) + 1,
        observation=observation.strip(),
        actions=actions,
    )


def model_action(task: str, observation: str, admissible: list[str], history: list[dict[str, str]]) -> tuple[str, str]:
    choices = [action for action in admissible if action != "help"]
    if not choices:
        raise RuntimeError("no admissible action")
    payload = {
        "model": WORKER["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt(task, observation, admissible, history)},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        "guided_choice": [f"<action>{action}</action>" for action in choices],
    }
    request = urllib.request.Request(
        WORKER["endpoint"].rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                content = str(json.load(response)["choices"][0]["message"].get("content") or "")
            match = ACTION_RE.search(content)
            action = match.group(1).strip().lower() if match else ""
            lookup = {item.lower(): item for item in choices}
            if action not in lookup:
                raise ValueError(f"model returned non-admissible action: {content!r}")
            return lookup[action], content
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
            error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"model request failed: {error}")


def start_env(gamefile: str):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import (
        AlfredDemangler,
        AlfredExpert,
        AlfredExpertType,
    )

    return textworld.start(
        gamefile,
        request_infos=textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            extras=["expert_plan"],
        ),
        wrappers=[
            AlfredDemangler(shuffle=False),
            # First positional argument is env, not expert_type.
            AlfredExpert(expert_type=AlfredExpertType.PLANNER),
        ],
    )


def planner_action(state) -> str | None:
    plan = [str(action) for action in (state["extra.expert_plan"] or [])]
    return plan[0] if plan else None


def run_parent(gamefile: str) -> dict[str, Any]:
    env = start_env(gamefile)
    trajectory: list[dict[str, Any]] = []
    first_divergence: int | None = None
    try:
        state = env.reset()
        task = task_description(str(state.feedback))
        history: list[dict[str, str]] = []
        done = False
        for step_index in range(WORKER["max_steps"]):
            observation = str(state.feedback)
            admissible = [str(value) for value in state.admissible_commands]
            expert = planner_action(state)
            action, response = model_action(task, observation, admissible, history)
            if first_divergence is None and expert is not None and action != expert:
                first_divergence = step_index
            state, _reward, done = env.step(action)
            trajectory.append(
                {
                    "observation": observation,
                    "admissible_actions": admissible,
                    "student_action": action,
                    "student_response": response,
                    "planner_action": expert,
                    "result_observation": str(state.feedback),
                    "won": bool(state.won),
                    "done": bool(done),
                }
            )
            if bool(state.won) or done:
                break
            history.append({"observation": observation, "action": action})
        return {
            "won": bool(state.won),
            "done": bool(done),
            "steps": len(trajectory),
            "first_divergence": first_divergence,
            "task_description": task,
            "trajectory": trajectory,
        }
    finally:
        env.close()


def run_branch(gamefile: str, parent: dict[str, Any], teacher_limit: int) -> dict[str, Any]:
    env = start_env(gamefile)
    prefix_length = int(parent["first_divergence"])
    teacher_steps: list[dict[str, Any]] = []
    student_steps: list[dict[str, Any]] = []
    history: list[dict[str, str]] = []
    try:
        state = env.reset()
        task = task_description(str(state.feedback))
        for source in parent["trajectory"][:prefix_length]:
            observation = str(state.feedback)
            action = str(source["student_action"])
            admissible = [str(value) for value in state.admissible_commands]
            if action not in admissible:
                raise RuntimeError(f"prefix action is not replayable: {action}")
            state, _reward, done = env.step(action)
            if done or bool(state.won):
                raise RuntimeError("prefix terminated before divergence")
            history.append({"observation": observation, "action": action})

        for _ in range(teacher_limit):
            observation = str(state.feedback)
            admissible = [str(value) for value in state.admissible_commands]
            action = planner_action(state)
            if action is None or action not in admissible:
                return {
                    "teacher_limit": teacher_limit,
                    "status": "planner_unavailable",
                    "won": False,
                    "teacher_steps": teacher_steps,
                    "student_steps": student_steps,
                }
            state, _reward, done = env.step(action)
            teacher_steps.append(
                {
                    "observation": observation,
                    "admissible_actions": admissible,
                    "action": action,
                    "result_observation": str(state.feedback),
                    "won": bool(state.won),
                    "done": bool(done),
                }
            )
            history.append({"observation": observation, "action": action})
            if bool(state.won) or done:
                return {
                    "teacher_limit": teacher_limit,
                    "status": "teacher_terminal",
                    "won": bool(state.won),
                    "teacher_steps": teacher_steps,
                    "student_steps": student_steps,
                }

        total_steps = prefix_length + len(teacher_steps)
        while total_steps < WORKER["max_steps"]:
            observation = str(state.feedback)
            admissible = [str(value) for value in state.admissible_commands]
            action, response = model_action(task, observation, admissible, history)
            state, _reward, done = env.step(action)
            total_steps += 1
            student_steps.append(
                {
                    "observation": observation,
                    "admissible_actions": admissible,
                    "action": action,
                    "response": response,
                    "result_observation": str(state.feedback),
                    "won": bool(state.won),
                    "done": bool(done),
                }
            )
            if bool(state.won) or done:
                break
            history.append({"observation": observation, "action": action})
        return {
            "teacher_limit": teacher_limit,
            "status": "student_success" if bool(state.won) else "student_failure",
            "won": bool(state.won) and bool(student_steps),
            "teacher_steps": teacher_steps,
            "student_steps": student_steps,
        }
    finally:
        env.close()


def process_game(row: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    result = {
        "sample_index": row["sample_index"],
        "gamefile": row["gamefile"],
        "task_type": row["task_type"],
        "status": "unknown",
        "error": None,
        "parent": None,
        "branches": [],
        "frontier": None,
    }
    try:
        parent = run_parent(row["gamefile"])
        result["parent"] = parent
        if parent["won"]:
            result["status"] = "parent_success"
        elif parent["first_divergence"] is None:
            result["status"] = "failed_without_divergence"
        else:
            branches = [run_branch(row["gamefile"], parent, value) for value in WORKER["k_grid"]]
            result["branches"] = branches
            successful = [
                int(branch["teacher_limit"])
                for branch in branches
                if branch["status"] == "student_success" and branch["won"]
            ]
            result["frontier"] = min(successful) if successful else None
            result["status"] = "finite_frontier" if successful else "no_finite_frontier"
    except Exception as exc:
        result["status"] = "infrastructure_error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["wall_seconds"] = time.monotonic() - started
    return result


def proportional_sample(episodes: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in episodes:
        by_type.setdefault(str(row["task_type"]), []).append(row)
    total = len(episodes)
    raw = {key: sample_size * len(rows) / total for key, rows in by_type.items()}
    quota = {key: int(value) for key, value in raw.items()}
    remaining = sample_size - sum(quota.values())
    for key in sorted(by_type, key=lambda item: (raw[item] - quota[item], item), reverse=True)[:remaining]:
        quota[key] += 1
    selected = []
    for offset, key in enumerate(sorted(by_type)):
        rows = sorted(by_type[key], key=lambda item: str(item["gamefile"]))
        random.Random(seed + offset * 1009).shuffle(rows)
        selected.extend(rows[: quota[key]])
    selected.sort(key=lambda item: str(item["gamefile"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--k-grid", default="1,2,3,5,8,12")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-tasks-per-child", type=int, default=8)
    args = parser.parse_args()

    with gzip.open(args.episodes, "rt", encoding="utf-8") as reader:
        episodes = [json.loads(line) for line in reader if line.strip()]
    selected = proportional_sample(episodes, args.sample_size, args.seed)
    work = [
        {
            "sample_index": index,
            "gamefile": str(row["gamefile"]),
            "task_type": str(row["task_type"]),
        }
        for index, row in enumerate(selected)
        if index % args.num_shards == args.shard_index
    ]
    completed: set[str] = set()
    if args.output.exists():
        with args.output.open(encoding="utf-8") as reader:
            for line in reader:
                if line.strip():
                    completed.add(str(json.loads(line)["gamefile"]))
    pending = [row for row in work if row["gamefile"] not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "endpoint": args.endpoint,
        "model": args.model,
        "data_root": str(args.data_root),
        "max_steps": args.max_steps,
        "k_grid": [int(value) for value in args.k_grid.split(",")],
    }
    status_counts: Counter[str] = Counter()
    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as writer:
        with get_context("spawn").Pool(
            processes=args.workers,
            initializer=init_worker,
            initargs=(config,),
            maxtasksperchild=args.max_tasks_per_child,
        ) as pool:
            for index, result in enumerate(pool.imap(process_game, pending, chunksize=1), start=1):
                writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                writer.flush()
                status_counts[result["status"]] += 1
                if index % 8 == 0 or index == len(pending):
                    print(
                        json.dumps(
                            {
                                "shard": args.shard_index,
                                "completed_now": index,
                                "pending_at_start": len(pending),
                                "status_counts": dict(status_counts),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
