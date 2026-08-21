#!/usr/bin/env python
"""Add DSpark/DFlash/EAGLE3 aux-hidden-state capture to SGLang's DeepSeek-V4.

Stock sglang 0.5.14 wires aux capture into deepseek_v2.py (V2/V3/V32) but not
into deepseek_v4.py: `DeepseekV4ForCausalLM` already carries
`capture_aux_hidden_states` and already unpacks/forwards an aux list, but
`DeepseekV4Model` never collects anything and no `set_*_layers_to_capture`
hook exists.  Offline DSpark preparation therefore dies in
`OfflineSGLangCaptureBackend.set_capture_layers()` before the first forward.

This applier is anchored on exact source text and is idempotent: it verifies
each anchor occurs exactly once, refuses to touch a tree it does not
recognise, and no-ops on an already-patched tree.

Usage:
    python apply_deepseek_v4_capture.py [--check] [--revert] [--file PATH]

With no --file the target is resolved from the importable sglang package.

Four edits, none of which can be copied from deepseek_v2.py verbatim:

1. `DeepseekV4Model.layers_to_capture`, the list the loop consults.

2. Capture inside the layer loop.  Two V4-specific hazards:
   - With fused mHC a layer defers its FFN `hc_post` to the next layer, so the
     `hidden_states` it returns is a [T, H] intermediate, NOT the layer output.
     It has to be completed the same way the final layer is completed after the
     loop.
   - The carried state is [T, hc_mult, H] multi-stream mHC, so each captured
     layer is folded to [T, H] before being appended, keeping the per-entry
     contract 2-D like deepseek_v2.py.  LogitsProcessor concatenates the list
     along the last dim, giving [T, len(layer_ids) * H].

3. `DeepseekV4Model.forward` returns `(hidden_states, pre_hc_head)`; when
   capturing it must return `((hidden_states, pre_hc_head), aux)` because
   `DeepseekV4ForCausalLM.forward` peels aux off first.

4. Suppress `hidden_states_before_norm` while capturing.  `LogitsProcessor.
   _get_hidden_states_to_store()` ends with "when hidden_states_before_norm is
   provided, we always prefer to return it", so passing `pre_hc_head`
   unconditionally silently overwrites the aux concatenation with the
   [T, hc_mult*H] pre-head tensor.  `pre_hc_head` is a serving-side MTP input,
   not a capture artifact.

NOTE on layer-id semantics: deepseek_v2.py stores `[val + 1 for val in
layer_ids]` because it captures by handing the list to layer i+1, which appends
its own *input*.  This patch captures after layer i returns, so the requested
ids are used as-is.  Do not "fix" this to match V2.

NOT VERIFIED HERE: that `captured.mean(dim=1)` is the fold the official V4
DSpark drafter consumes.  It matches SpecForge's normalizer
(`_project_target_hidden`, which does `mean(dim=-2)`), but the mHC stream mean
is not the same operation as the learned `hc_head` fold.  Cross-check against
the official inference/model.py before trusting captured features.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "SPECFORGE-DSPARK-CAPTURE"

EDITS: list[tuple[str, str, str]] = [
    (
        "DeepseekV4Model.__init__: declare layers_to_capture",
        """        self.gemm_output_zero_allocator_size = 0
        self.hc_eps = config.hc_eps
""",
        """        self.gemm_output_zero_allocator_size = 0
        # SPECFORGE-DSPARK-CAPTURE: ids of layers whose outputs are captured as
        # auxiliary hidden states for speculative-decoding draft training.
        self.layers_to_capture = []
        self.hc_eps = config.hc_eps
""",
    ),
    (
        "DeepseekV4Model.forward: aux accumulator",
        """        prev_residual, prev_post, prev_comb = None, None, None
        last_layer = None
        for i in range(self.start_layer, self.end_layer):
""",
        """        prev_residual, prev_post, prev_comb = None, None, None
        last_layer = None
        aux_hidden_states = []  # SPECFORGE-DSPARK-CAPTURE
        for i in range(self.start_layer, self.end_layer):
""",
    ),
    (
        "DeepseekV4Model.forward: capture layer outputs",
        """                    prev_comb=prev_comb,
                )
        if use_fused and last_layer is not None:
""",
        """                    prev_comb=prev_comb,
                )
            # SPECFORGE-DSPARK-CAPTURE
            if i in self.layers_to_capture:
                # Under fused mHC the layer defers its FFN hc_post to the next
                # layer, so `hidden_states` is a [T, H] intermediate here, not
                # the layer output.  Complete it exactly the way the final layer
                # is completed just below; this is a pure function of the
                # returned state and leaves the deferred fusion chain intact.
                captured = (
                    layer.hc_post(hidden_states, prev_residual, prev_post, prev_comb)
                    if use_fused
                    else hidden_states
                )
                # [T, hc_mult, H] -> [T, H]: fold the mHC streams so each entry
                # stays 2-D, matching the aux contract in deepseek_v2.py.
                aux_hidden_states.append(captured.mean(dim=1))
        if use_fused and last_layer is not None:
""",
    ),
    (
        "DeepseekV4Model.forward: return aux outermost",
        """        hidden_states = self.norm(hidden_states)

        return hidden_states, pre_hc_head
""",
        """        hidden_states = self.norm(hidden_states)

        # SPECFORGE-DSPARK-CAPTURE: DeepseekV4ForCausalLM.forward peels aux off
        # first and (hidden_states, pre_hc_head) second, so aux is outermost.
        if len(aux_hidden_states) == 0:
            return hidden_states, pre_hc_head
        return (hidden_states, pre_hc_head), aux_hidden_states
""",
    ),
    (
        "DeepseekV4ForCausalLM.forward: suppress pre_hc_head while capturing",
        """            aux_hidden_states,
            hidden_states_before_norm=pre_hc_head,
        )
""",
        """            aux_hidden_states,
            # SPECFORGE-DSPARK-CAPTURE: LogitsProcessor._get_hidden_states_to_store
            # always prefers hidden_states_before_norm over the aux
            # concatenation, so passing pre_hc_head here would silently replace
            # the captured aux states with the [T, hc_mult*H] pre-head tensor.
            # pre_hc_head is a serving-side MTP input, not a capture artifact.
            hidden_states_before_norm=(
                None if aux_hidden_states is not None else pre_hc_head
            ),
        )
""",
    ),
    (
        "DeepseekV4ForCausalLM: capture hooks",
        """    def _setup_fp8_wo_a_scales(self, is_nextn: bool) -> None:
""",
        """    # SPECFORGE-DSPARK-CAPTURE
    def _set_aux_layers_to_capture(self, layer_ids, *, hook: str) -> None:
        if not self.pp_group.is_last_rank:
            return
        if not layer_ids:
            raise ValueError(
                f"{hook} aux hidden capture on DeepSeek-V4 requires explicit "
                "layer_ids; there is no validated default layer selection."
            )
        self.capture_aux_hidden_states = True
        # NOTE: no +1 offset, unlike deepseek_v2.py.  V2 captures by handing the
        # list to layer i+1, which appends its own input; DeepseekV4Model
        # captures after layer i returns, so these ids are already layer outputs.
        self.model.layers_to_capture = list(layer_ids)

    def set_dspark_layers_to_capture(self, layer_ids=None) -> None:
        self._set_aux_layers_to_capture(layer_ids, hook="DSpark")

    def set_dflash_layers_to_capture(self, layer_ids=None) -> None:
        self._set_aux_layers_to_capture(layer_ids, hook="DFlash")

    def set_eagle3_layers_to_capture(self, layer_ids=None) -> None:
        self._set_aux_layers_to_capture(layer_ids, hook="EAGLE3")

    def _setup_fp8_wo_a_scales(self, is_nextn: bool) -> None:
""",
    ),
]


def resolve_target(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    try:
        import sglang.srt.models.deepseek_v4 as mod
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"ERROR: cannot import sglang.srt.models.deepseek_v4: {exc}")
    return Path(mod.__file__)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", default=None, help="path to deepseek_v4.py")
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    ap.add_argument("--revert", action="store_true", help="undo the patch")
    args = ap.parse_args()

    target = resolve_target(args.file)
    if not target.is_file():
        sys.exit(f"ERROR: not a file: {target}")
    text = target.read_text()
    patched = MARKER in text

    if args.revert:
        if not patched:
            print(f"not patched, nothing to revert: {target}")
            return 0
        pairs = [(name, new, old) for name, old, new in EDITS]
        verb = "reverted"
    else:
        if patched:
            print(f"already patched: {target}")
            return 0
        pairs = [(name, old, new) for name, old, new in EDITS]
        verb = "patched"

    for name, find, _ in pairs:
        count = text.count(find)
        if count != 1:
            sys.exit(
                f"ERROR: anchor matched {count} times (expected 1): {name}\n"
                f"       {target} is not the source revision this patch targets.\n"
                f"       Do not force it; re-derive the anchors against this tree."
            )
    for name, find, replace in pairs:
        text = text.replace(find, replace, 1)

    compile(text, str(target), "exec")  # syntax gate before touching disk

    if args.check:
        print(f"OK: all {len(pairs)} anchors matched, result parses (--check, not written)")
        return 0

    backup = target.with_suffix(target.suffix + ".specforge-orig")
    if not backup.exists() and not args.revert:
        shutil.copy2(target, backup)
    target.write_text(text)
    print(f"{verb} {len(pairs)} sites in {target}")
    if not args.revert:
        print(f"backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
