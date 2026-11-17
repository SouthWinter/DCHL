"""Shared ALFWorld action-policy prompt used by training and evaluation."""

from __future__ import annotations


SYSTEM_PROMPT = (
    "Select exactly one action from the current admissible-action list. "
    "Return only <action>the exact action</action>."
)

PROMPT_TEMPLATE_NO_HISTORY = """Your current observation, including the task, is: {current_observation}
Your admissible actions are: [{admissible_actions}].

Choose one exact admissible action. Return only <action>the exact action</action>.
"""

PROMPT_TEMPLATE = """Your task is to: {task_description}
You have already taken {step_count} step(s). The most recent {history_count} observation/action pairs are:
{action_history}

At step {current_step}, your current observation is: {current_observation}
Your admissible actions are: [{admissible_actions}].

Choose one exact admissible action. Return only <action>the exact action</action>.
"""

TASK_PREFIX = "Your task is to: "


def extract_task(observation: str) -> str:
    start = observation.find(TASK_PREFIX)
    if start == -1:
        return ""
    return observation[start + len(TASK_PREFIX) :].strip()


def format_admissible_actions(actions: list[str]) -> str:
    return "\n ".join(f"'{action}'" for action in actions if action != "help")


def format_history(history: list[dict[str, str]], history_length: int) -> tuple[str, int]:
    recent = history[-history_length:] if history_length > 0 else []
    if not recent:
        return "No previous observations or actions.", 0

    start_step = len(history) - len(recent) + 1
    parts = []
    for offset, item in enumerate(recent):
        step = start_step + offset
        parts.append(f"Observation {step}:\n{item['observation']}\n\nAction {step}:\n{item['action']}")
    return "\n\n".join(parts), len(recent)


def build_prompt(
    *,
    task_description: str,
    current_observation: str,
    admissible_actions: list[str],
    history: list[dict[str, str]],
    history_length: int = 4,
) -> str:
    action_history, history_count = format_history(history, history_length)
    formatted_actions = format_admissible_actions(admissible_actions)
    if history_count == 0:
        return PROMPT_TEMPLATE_NO_HISTORY.format(
            current_observation=current_observation,
            admissible_actions=formatted_actions,
        )

    return PROMPT_TEMPLATE.format(
        task_description=task_description,
        step_count=len(history),
        history_count=history_count,
        action_history=action_history,
        current_step=len(history) + 1,
        current_observation=current_observation,
        admissible_actions=formatted_actions,
    )


def action_response(action: str) -> str:
    return f"<action>{action}</action>"
