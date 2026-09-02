# coding=utf-8
import unittest

from scripts.check_dspark_serving_names import analyse, remap

REMAP_KWARGS = {"num_hidden_layers": 43, "num_stages": 3}


def _remap(name):
    return remap(name, **REMAP_KWARGS)


class RemapTest(unittest.TestCase):
    def test_a_stage_maps_onto_its_own_layer(self):
        self.assertEqual(
            _remap("mtp.1.attn.wq_a.weight"),
            "model.layers.44.self_attn.wq_a.weight",
        )

    def test_the_moe_substitutions_follow_the_serving_names(self):
        self.assertEqual(
            _remap("mtp.0.ffn.experts.7.w1.weight"),
            "model.layers.43.mlp.experts.7.gate_proj.weight",
        )
        self.assertEqual(
            _remap("mtp.2.ffn.shared_experts.w2.weight"),
            "model.layers.45.mlp.shared_experts.down_proj.weight",
        )

    def test_the_router_correction_bias_is_renamed(self):
        self.assertEqual(
            _remap("mtp.0.ffn.gate.bias"),
            "model.layers.43.mlp.gate.e_score_correction_bias",
        )

    def test_the_norms_take_their_serving_names(self):
        self.assertEqual(
            _remap("mtp.1.attn_norm.weight"),
            "model.layers.44.input_layernorm.weight",
        )
        self.assertEqual(
            _remap("mtp.1.ffn_norm.weight"),
            "model.layers.44.post_attention_layernorm.weight",
        )

    def test_head_side_tensors_pin_to_the_last_stage_layer(self):
        # norm and markov_head belong to the last stage regardless of prefix.
        self.assertEqual(_remap("mtp.2.norm.weight"), "model.layers.45.norm.weight")
        self.assertEqual(
            _remap("mtp.2.markov_head.markov_w1.weight"),
            "model.layers.45.markov_head.markov_w1.weight",
        )

    def test_main_proj_pins_to_the_first_stage_layer(self):
        self.assertEqual(
            _remap("mtp.0.main_proj.weight"), "model.layers.43.main_proj.weight"
        )
        self.assertEqual(
            _remap("mtp.0.main_norm.weight"), "model.layers.43.main_norm.weight"
        )

    def test_embed_and_head_leave_the_layer_tree(self):
        self.assertEqual(_remap("mtp.0.embed.weight"), "model.embed_tokens.weight")
        self.assertEqual(_remap("mtp.2.head.weight"), "lm_head.weight")

    def test_hc_head_and_confidence_head_are_hoisted_to_the_model(self):
        self.assertEqual(_remap("mtp.2.hc_head_fn"), "model.hc_head_fn")
        self.assertEqual(
            _remap("mtp.2.confidence_head.proj.weight"),
            "model.confidence_head.proj.weight",
        )

    def test_an_unrecognised_name_maps_to_nothing(self):
        self.assertIsNone(_remap("model.layers.3.self_attn.wq_a.weight"))
        self.assertIsNone(_remap("some.stray.tensor"))


class AnalyseTest(unittest.TestCase):
    def test_a_clean_export_reports_no_drop_and_no_collision(self):
        names = [
            "mtp.0.attn.wq_a.weight",
            "mtp.1.attn.wq_a.weight",
            "mtp.0.embed.weight",
            "mtp.2.head.weight",
        ]
        report = analyse(names, **REMAP_KWARGS)
        self.assertEqual(report["dropped_silently"], [])
        self.assertEqual(report["collisions"], {})
        self.assertEqual(report["mapped"], 4)

    def test_an_unrecognised_name_is_reported_as_a_silent_drop(self):
        report = analyse(["mtp.0.attn.wq_a.weight", "stray.weight"], **REMAP_KWARGS)
        self.assertEqual(report["dropped_silently"], ["stray.weight"])

    def test_two_names_on_one_parameter_are_reported_as_a_collision(self):
        # Both the bare target embedding and the drafter's own land on
        # model.embed_tokens.weight; whichever is iterated last wins.
        report = analyse(["embed.weight", "mtp.0.embed.weight"], **REMAP_KWARGS)
        self.assertEqual(
            report["collisions"],
            {"model.embed_tokens.weight": ["embed.weight", "mtp.0.embed.weight"]},
        )


if __name__ == "__main__":
    unittest.main()
