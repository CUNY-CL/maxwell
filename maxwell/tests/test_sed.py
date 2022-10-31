"""Unit tests for sed.py."""
import unittest

import numpy

from . import actions, sed, util

SOURCE_ALPHA = list("abcdefg")
TARGET_ALPHA = list("fghijk")


class TestSed(unittest.TestCase):
    def test_sed_random_initialization(self):
        sed_ = sed.StochasticEditDistance.build_sed(
            SOURCE_ALPHA, TARGET_ALPHA, copy_probability=None
        )
        eos_weight = sed_.params.delta_eos
        for weight_dict in ("delta_del", "delta_ins", "delta_sub"):
            for weight in getattr(sed_.params, weight_dict).values():
                self.assertTrue(numpy.isclose(eos_weight, weight))

    def test_sed_copy_biased_initialization(self):
        sed_ = sed.StochasticEditDistance.build_sed(SOURCE_ALPHA, TARGET_ALPHA)
        eos_weight = sed_.params.delta_eos
        for weight_dict in ("delta_del", "delta_ins"):
            for weight in getattr(sed_.params, weight_dict).values():
                self.assertTrue(numpy.isclose(eos_weight, weight))
        for (x, y), weight in sed_.params.delta_sub.items():
            if x == y:
                self.assertFalse(numpy.isclose(eos_weight, weight))
            else:
                self.assertTrue(numpy.isclose(eos_weight, weight))

    def test_viterbi_decoding(self):
        smart_sed = sed.StochasticEditDistance.build_sed(
            SOURCE_ALPHA, TARGET_ALPHA
        )
        best_edits, distance = smart_sed.action_sequence(
            source="affa", target="iffig"
        )
        expected_edits = [
            actions.Sub(old="a", new="i"),
            actions.Sub(old="f", new="f"),
            actions.Sub(old="f", new="f"),
            actions.Ins(new="i"),
            actions.Sub(old="a", new="g"),
        ]
        self.assertTrue(numpy.isclose(-26.7633, distance))
        self.assertListEqual(expected_edits, best_edits)

    def test_stochastic_decoding(self):
        smart_sed = sed.StochasticEditDistance.build_sed(
            SOURCE_ALPHA, TARGET_ALPHA
        )
        distance = smart_sed.forward_evaluate(source="affa", target="iffig")[
            -1, -1
        ]
        self.assertTrue(numpy.isclose(-26.05741, distance))

    def test_em(self):
        input_pairs = [
            ("abby", "a b i"),
            ("abidjan", "a b i d ʒ ɑ"),
            ("abject", "a b ʒ ɛ k t"),
            ("abolir", "a b ɔ l i ʁ"),
            ("abonnement", "a b ɔ n m ɑ"),
        ]
        sources, targets = zip(*input_pairs)
        source_alphabet = {c for source in sources for c in source}
        target_alphabet = {c for target in targets for c in target}
        sed_ = sed.StochasticEditDistance.build_sed(
            source_alphabet, target_alphabet
        )
        o = sed_.forward_evaluate(sources[1], targets[1])[
            -1, -1
        ]  # Stochastic edit distance.
        util.log_info(o)
        before_ll = sed_.log_likelihood(sources, targets)
        sed_.em(sources, targets, epochs=1)
        after_ll = sed_.log_likelihood(sources, targets)
        self.assertTrue(before_ll <= after_ll)

    def test_fit_from_data(self):
        def _to_sample(line: str):
            input_, target = line.rstrip().split("\t", 1)
            return input_, target, None

        input_lines = [
            "abby\ta b i",
            "abidjan\ta b i d ʒ ɑ",
            "abject\ta b ʒ ɛ k t",
            "abolir\ta b ɔ l i ʁ",
            "abonnement\ta b ɔ n m ɑ",
        ]
        data = map(_to_sample, input_lines)
        sed_ = sed.StochasticEditDistance.fit_from_data(data, epochs=1)
        util.log_info(sed_.params)


if __name__ == "__main__":
    unittest.main()
