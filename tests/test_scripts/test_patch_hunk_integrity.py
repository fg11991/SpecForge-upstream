"""Static and real-apply checks for vendored unified-diff patches."""

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCH_ROOT = ROOT / "patches"
V0514_CAPTURE_PATCH = PATCH_ROOT / "sglang" / "v0.5.14" / "spec-capture.patch"
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True)
class HunkCounts:
    header_line: int
    old_declared: int
    new_declared: int
    old_actual: int
    new_actual: int


def _declared_count(raw: str | None) -> int:
    return 1 if raw is None else int(raw)


def parse_hunks(source: str) -> list[HunkCounts]:
    """Return declared and actual line counts for every unified-diff hunk."""

    lines = source.splitlines()
    hunks: list[HunkCounts] = []
    index = 0
    while index < len(lines):
        match = _HUNK_HEADER.match(lines[index])
        if match is None:
            index += 1
            continue
        old_actual = 0
        new_actual = 0
        body_index = index + 1
        while body_index < len(lines):
            line = lines[body_index]
            if _HUNK_HEADER.match(line) or line.startswith("diff --git "):
                break
            if line.startswith("\\ No newline at end of file"):
                body_index += 1
                continue
            if line.startswith("+"):
                new_actual += 1
            elif line.startswith("-"):
                old_actual += 1
            elif line.startswith(" ") or line == "":
                # Some vendored patches carry an empty context line without
                # git's usual single-space marker. patch(1) accepts that form.
                old_actual += 1
                new_actual += 1
            else:
                break
            body_index += 1
        hunks.append(
            HunkCounts(
                header_line=index + 1,
                old_declared=_declared_count(match.group(2)),
                new_declared=_declared_count(match.group(4)),
                old_actual=old_actual,
                new_actual=new_actual,
            )
        )
        index = body_index
    return hunks


def _file_patch(source: str, target_path: str) -> str:
    marker = f"diff --git a/{target_path} b/{target_path}\n"
    start = source.index(marker)
    next_diff = source.find("\ndiff --git ", start + len(marker))
    return source[start:] if next_diff < 0 else source[start : next_diff + 1]


class PatchHunkIntegrityTest(unittest.TestCase):
    def test_every_vendored_patch_hunk_has_exact_line_counts(self):
        patch_paths = sorted(PATCH_ROOT.rglob("*.patch"))
        self.assertTrue(patch_paths)
        for path in patch_paths:
            source = path.read_text(encoding="utf-8")
            hunks = parse_hunks(source)
            self.assertTrue(hunks, f"{path} contains no unified-diff hunks")
            for hunk in hunks:
                with self.subTest(path=path.relative_to(ROOT), line=hunk.header_line):
                    self.assertEqual(hunk.old_declared, hunk.old_actual)
                    self.assertEqual(hunk.new_declared, hunk.new_actual)
                    if hunk.old_declared == 0:
                        self.assertEqual(hunk.new_declared, hunk.new_actual)

    def test_v0514_new_sink_really_applies_without_truncation(self):
        source = V0514_CAPTURE_PATCH.read_text(encoding="utf-8")
        target = "python/sglang/srt/spec_capture_sink.py"
        fragment = _file_patch(source, target)
        hunks = parse_hunks(fragment)
        self.assertEqual(len(hunks), 1)
        self.assertEqual(hunks[0].old_declared, 0)

        with tempfile.TemporaryDirectory(prefix="spec-capture-patch-") as directory:
            completed = subprocess.run(
                ["patch", "-p2", "--batch"],
                cwd=directory,
                input=fragment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            generated = Path(directory) / "sglang" / "srt" / "spec_capture_sink.py"
            self.assertTrue(generated.is_file(), completed.stdout)
            generated_lines = generated.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(generated_lines), hunks[0].new_actual)
            self.assertTrue(generated_lines[-2].startswith("def get_sink()"))
            self.assertEqual(generated_lines[-1], "    return _SINK")


if __name__ == "__main__":
    unittest.main()
