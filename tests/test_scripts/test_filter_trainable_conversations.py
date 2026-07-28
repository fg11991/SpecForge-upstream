import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import filter_trainable_conversations


def fake_preprocess(
    tokenizer,
    conversations,
    template,
    max_length,
    *,
    is_preformatted,
    train_only_last_turn,
    tools,
    **kwargs,
):
    del tokenizer, template, is_preformatted, train_only_last_turn, tools, kwargs
    mask = conversations[0]
    return {"loss_mask": [[mask[:max_length]]]}


FAKE_STACK = (object(), object(), fake_preprocess)


class TestFilterTrainableConversations(unittest.TestCase):
    def test_keeps_only_rows_with_a_nonterminal_consecutive_pair(self):
        rows = [
            {"id": "prompt-only", "conversations": [0, 0, 0, 0]},
            {"id": "separated", "conversations": [1, 0, 1, 0]},
            {"id": "trainable", "conversations": [0, 1, 1, 0]},
            # The offline loader clears the final position, so this is invalid.
            {"id": "terminal-pair", "conversations": [0, 0, 1, 1]},
        ]
        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.jsonl"
            output_path = Path(temporary_directory) / "filtered.jsonl"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            kept, dropped = filter_trainable_conversations.filter_jsonl(
                input_path,
                output_path,
                processing_stack=FAKE_STACK,
                max_length=4,
            )

            self.assertEqual((kept, dropped), (1, 3))
            self.assertEqual(
                [json.loads(line) for line in output_path.read_text().splitlines()],
                [rows[2]],
            )

    def test_truncation_happens_before_the_loss_mask_check(self):
        row = {"id": "late-answer", "conversations": [0, 0, 0, 1, 1, 0]}
        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.jsonl"
            output_path = Path(temporary_directory) / "filtered.jsonl"
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            result = filter_trainable_conversations.filter_jsonl(
                input_path,
                output_path,
                processing_stack=FAKE_STACK,
                max_length=4,
            )

            self.assertEqual(result, (0, 1))
            self.assertEqual(output_path.read_text(), "")

    def test_preserves_the_complete_original_row(self):
        row = {
            "id": "row",
            "conversations": [1, 1, 0],
            "metadata": {"source": "unit-test"},
        }
        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.jsonl"
            output_path = Path(temporary_directory) / "filtered.jsonl"
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            filter_trainable_conversations.filter_jsonl(
                input_path,
                output_path,
                processing_stack=FAKE_STACK,
                max_length=3,
            )

            self.assertEqual(json.loads(output_path.read_text()), row)

    def test_refuses_to_overwrite_an_existing_output(self):
        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.jsonl"
            output_path = Path(temporary_directory) / "filtered.jsonl"
            input_path.write_text("{}\n", encoding="utf-8")
            output_path.write_text("existing\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                filter_trainable_conversations.filter_jsonl(
                    input_path,
                    output_path,
                    processing_stack=FAKE_STACK,
                    max_length=4,
                )

            self.assertEqual(output_path.read_text(), "existing\n")

    def test_invalid_json_does_not_leave_partial_output(self):
        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "input.jsonl"
            output_path = Path(temporary_directory) / "filtered.jsonl"
            input_path.write_text(
                '{"conversations":[1,1,0]}\nnot-json\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "line 2: invalid JSON"):
                filter_trainable_conversations.filter_jsonl(
                    input_path,
                    output_path,
                    processing_stack=FAKE_STACK,
                    max_length=3,
                )

            self.assertFalse(output_path.exists())

    def test_cli_validates_lengths_and_distinct_paths(self):
        common = [
            "--input-path",
            "same.jsonl",
            "--output-path",
            "same.jsonl",
            "--tokenizer-path",
            "model",
        ]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            filter_trainable_conversations.parse_args(common)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            filter_trainable_conversations.parse_args(
                [
                    "--input-path",
                    "input.jsonl",
                    "--output-path",
                    "output.jsonl",
                    "--tokenizer-path",
                    "model",
                    "--max-length",
                    "1",
                ]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
