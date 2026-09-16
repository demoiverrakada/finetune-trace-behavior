import unittest

from harness.factorial_causal import trim_generated_tokens


class FactorialCausalTests(unittest.TestCase):
    def test_trim_after_eos(self):
        self.assertEqual(
            trim_generated_tokens(
                [7, 2, 2, 2],
                eos_ids={2},
                pad_token_id=2,
            ),
            [7, 2],
        )

    def test_preserve_no_eos(self):
        self.assertEqual(
            trim_generated_tokens(
                [7, 8, 9],
                eos_ids={2},
                pad_token_id=2,
            ),
            [7, 8, 9],
        )


if __name__ == "__main__":
    unittest.main()
