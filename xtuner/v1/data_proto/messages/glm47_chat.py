# Copyright (c) OpenMMLab. All rights reserved.
import json
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from transformers import PreTrainedTokenizer

from .glm52_chat import (
    _assistant_content_and_reasoning,
    _render_tool_calls,
    _tokenize_with_loss_mask,
    _visible_text,
)


def _render_tools(tools: List[Dict[str, Any]]) -> str:
    text = (
        "<|system|>\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
    )
    text += "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
    text += (
        "\n</tools>\n\n"
        "For each function call, output the function name and arguments within the following XML format:\n"
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
        "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>"
    )
    return text


def _render_tool_result(content: Any) -> str:
    if isinstance(content, str):
        return f"<tool_response>{content}</tool_response>"
    if isinstance(content, list) and content and isinstance(content[0], Mapping):
        if "output" in content[0]:
            return "".join(f"<tool_response>{item['output']}</tool_response>" for item in content)
    return f"<tool_response>{_visible_text(content)}</tool_response>"


def render_glm47_chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = True,
    clear_thinking: bool = False,
) -> tuple[str, list[bool]]:
    """Render a GLM-4.7 conversation and its character-level SFT loss mask."""
    text = ""
    loss_mask: list[bool] = []

    def append(value: str, loss: bool) -> None:
        nonlocal text
        text += value
        loss_mask.extend([loss] * len(value))

    append("[gMASK]<sop>", False)
    if tools:
        append(_render_tools(tools), False)

    last_user_index = max(
        (index for index, message in enumerate(messages) if message.get("role") == "user"),
        default=-1,
    )
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "user":
            append(f"<|user|>{_visible_text(message.get('content', ''))}", False)
        elif role == "system":
            append(f"<|system|>{_visible_text(message.get('content', ''))}", False)
        elif role == "assistant":
            content, reasoning_content = _assistant_content_and_reasoning(message)
            loss = bool(message.get("loss", True))
            render_reasoning = reasoning_content is not None and (
                not clear_thinking or index > last_user_index
            )

            append("<|assistant|>", False)
            if render_reasoning:
                assert reasoning_content is not None
                append("<think>", False)
                append(reasoning_content.strip() + "</think>", loss)
            else:
                append("</think>", False)
            if content.strip():
                append(content.strip(), loss)
            if message.get("tool_calls"):
                append(_render_tool_calls(message["tool_calls"]), loss)
        elif role == "tool":
            if index == 0 or messages[index - 1].get("role") != "tool":
                append("<|observation|>", False)
            append(_render_tool_result(message.get("content", "")), False)

    if add_generation_prompt:
        append("<|assistant|>", False)
        append("<think>" if enable_thinking else "</think>", False)

    return text, loss_mask


class Glm47ChatMessages(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None

    def tokenize(
        self,
        tokenizer: PreTrainedTokenizer,
        chat_template=None,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
        clear_thinking: bool = False,
        **kwargs,
    ) -> Dict:
        """Tokenize messages with labels restricted to assistant-generated spans."""
        messages = [message.copy() for message in self.messages]
        if chat_template is not None and chat_template.default_system is not None:
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = chat_template.default_system
            else:
                messages.insert(0, {"role": "system", "content": chat_template.default_system})

        text, loss_mask = render_glm47_chat(
            messages,
            tools=self.tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            clear_thinking=clear_thinking,
        )
        input_ids, labels = _tokenize_with_loss_mask(tokenizer, text, loss_mask)
        return {"input_ids": input_ids, "labels": labels, "num_tokens": len(input_ids)}


def supervised_glm47_text(tokenizer: PreTrainedTokenizer, tokenized: Dict) -> str:
    """Return the decoded supervised suffix for lightweight integration checks."""
    supervised_ids = [
        token_id for token_id, label in zip(tokenized["input_ids"], tokenized["labels"]) if label != -100
    ]
    return tokenizer.decode(supervised_ids, skip_special_tokens=False)
