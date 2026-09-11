"""DeepSeek-V4 conversations render through the official encoder.

The checkpoint tokenizer has no chat template, so these tests use a character
tokenizer without one: every token is one character, which makes the loss mask
readable as the exact supervised substrings.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from specforge.data import encoding_dsv4
from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY
from specforge.utils import safe_conversations_generator

BOS = encoding_dsv4.bos_token
EOS = encoding_dsv4.eos_token
USER = "<｜User｜>"
ASSISTANT = "<｜Assistant｜>"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    }
]


class _CharTokenizer:
    pad_token_id = 0
    unk_token_id = 0

    def __call__(self, text, max_length=None, **kwargs):
        del kwargs
        input_ids = self.encode(text, max_length=max_length)
        return SimpleNamespace(input_ids=torch.tensor([input_ids], dtype=torch.long))

    def encode(self, text, add_special_tokens=False, truncation=True, max_length=None):
        del add_special_tokens, truncation
        input_ids = [ord(character) for character in text]
        return input_ids if max_length is None else input_ids[:max_length]

    @staticmethod
    def decode(input_ids):
        return "".join(chr(token_id) for token_id in input_ids)


def _preprocess(conversation, *, last_turn=False, tools=None):
    tokenizer = _CharTokenizer()
    processed = preprocess_conversations(
        tokenizer,
        [conversation],
        TEMPLATE_REGISTRY.get("deepseek-v4"),
        max_length=1 << 20,
        train_only_last_turn=last_turn,
        tools=[tools or []],
    )
    input_ids = processed["input_ids"][0][0].tolist()
    loss_mask = processed["loss_mask"][0][0].tolist()
    segments, current = [], []
    for token_id, supervised in zip(input_ids, loss_mask):
        if supervised:
            current.append(token_id)
        elif current:
            segments.append(tokenizer.decode(current))
            current = []
    if current:
        segments.append(tokenizer.decode(current))
    return tokenizer.decode(input_ids), segments


class DeepSeekV4RenderingTest(unittest.TestCase):
    def test_reasoning_turn_supervises_reasoning_its_close_and_answer(self):
        rendered, supervised = _preprocess(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "reasoning_content": "R", "content": "A"},
            ]
        )

        self.assertEqual(f"{BOS}{USER}Q{ASSISTANT}<think>R</think>A{EOS}", rendered)
        # `<think>` is prompt-side; the model emits `</think>` itself.
        self.assertEqual([f"R</think>A{EOS}"], supervised)

    def test_answer_without_reasoning_renders_in_chat_mode(self):
        rendered, supervised = _preprocess(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A", "reasoning_content": None},
            ]
        )

        self.assertEqual(f"{BOS}{USER}Q{ASSISTANT}</think>A{EOS}", rendered)
        self.assertEqual([f"A{EOS}"], supervised)

    def test_multi_turn_keeps_only_reasoning_after_the_last_user(self):
        conversation = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "reasoning_content": "R1", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "reasoning_content": "R2", "content": "A2"},
        ]

        rendered, every_turn = _preprocess(conversation)
        _, last_turn = _preprocess(conversation, last_turn=True)

        self.assertEqual(
            f"{BOS}{USER}Q1{ASSISTANT}</think>A1{EOS}"
            f"{USER}Q2{ASSISTANT}<think>R2</think>A2{EOS}",
            rendered,
        )
        self.assertEqual([f"A1{EOS}", f"R2</think>A2{EOS}"], every_turn)
        self.assertEqual([f"R2</think>A2{EOS}"], last_turn)

    def test_agent_trajectory_matches_the_official_encoder(self):
        # Shaped the way the JSONL loader hands it over: list-valued fields
        # arrive as JSON strings and absent keys as None.
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "Read",
                "arguments": json.dumps({"file_path": "/skills/SKILL.md"}),
            },
        }
        conversation = [
            {"role": "system", "content": "SYS", "tools": json.dumps(TOOLS)},
            {"role": "user", "content": "Q"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "R1",
                "tool_calls": json.dumps([tool_call]),
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "FILE"},
            {"role": "assistant", "reasoning_content": "RF", "content": "AF"},
        ]
        official = encoding_dsv4.encode_messages(
            [
                {"role": "system", "content": "SYS", "tools": TOOLS},
                {"role": "user", "content": "Q"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "R1",
                    "tool_calls": [tool_call],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "FILE"},
                {"role": "assistant", "reasoning_content": "RF", "content": "AF"},
            ],
            thinking_mode="thinking",
        )

        rendered, every_turn = _preprocess(conversation)
        _, last_turn = _preprocess(conversation, last_turn=True)

        self.assertEqual(official, rendered)
        self.assertIn("## Tools", rendered)
        self.assertIn("<tool_result>FILE</tool_result>", rendered)
        # Declared tools keep every turn's reasoning in context.
        self.assertIn("<think>R1</think>", rendered)
        self.assertEqual([f"RF</think>AF{EOS}"], last_turn)
        self.assertEqual(2, len(every_turn))
        self.assertTrue(every_turn[0].startswith("R1</think>"))
        self.assertIn('<｜DSML｜invoke name="Read">', every_turn[0])
        self.assertTrue(every_turn[0].endswith(EOS))
        self.assertFalse(any("tool_result" in segment for segment in every_turn))

    def test_row_level_tools_attach_to_an_inserted_system_message(self):
        conversation = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]

        rendered, _ = _preprocess(conversation, tools=TOOLS)

        self.assertEqual(
            encoding_dsv4.encode_messages(
                [{"role": "system", "content": "", "tools": TOOLS}] + conversation,
                thinking_mode="chat",
            ),
            rendered,
        )


class AgentTrajectoryLoadingTest(unittest.TestCase):
    def test_loader_reads_the_openai_messages_key(self):
        row = {
            "id": "trace",
            "messages": [
                {"role": "system", "content": "SYS", "tools": TOOLS},
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "trace.jsonl")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")

            rows = list(safe_conversations_generator(path))

        self.assertEqual(1, len(rows))
        messages = rows[0]["conversations"]
        self.assertEqual(["system", "user", "assistant"], [m["role"] for m in messages])
        self.assertEqual(TOOLS, json.loads(messages[0]["tools"]))


if __name__ == "__main__":
    unittest.main()
