import unittest
from unittest.mock import MagicMock, patch

import tempfile
from pathlib import Path

import pandas as pd
import torch

from chembed.downstream import ga_optimizer


class TestMaximizeMinimize(unittest.TestCase):

    def test_evaluate_if_missing(self):
        fitness = MagicMock(return_value=42.0)
        cache = {}
        out1 = ga_optimizer.evaluate_if_missing("C", fitness, cache)
        self.assertEqual(out1, 42.0)
        self.assertEqual(cache["C"], 42.0)
        out2 = ga_optimizer.evaluate_if_missing("C", fitness, cache)
        self.assertEqual(out2, 42.0)
        fitness.assert_called_once_with("C")


    def test_get_all_crossovers(self):
        zs = torch.tensor([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]])
        new_zs = ga_optimizer.get_all_crossovers(zs)
        expected = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        self.assertTrue(torch.allclose(new_zs, expected))


    def test_get_df_where_fitness_close_to_max(self):
        df = pd.DataFrame({"smiles": ["A", "B", "C"], "fitness": [1.0, 2.0, 1.9]})
        close = ga_optimizer.get_df_where_fitness_close_to_max(df, epsilon=0.1)
        self.assertEqual(set(close["smiles"]), {"B", "C"})

    def test_get_best_df_with_generated_integer(self):
        df = pd.DataFrame({"smiles": ["A", "B", "C"], "fitness": [1.0, 3.0, 2.0]})
        best = ga_optimizer.get_best_df_with_generated(df, nb_best=2, epsilon=0.0)
        self.assertEqual(list(best["smiles"]), ["B", "C"])


    @patch("chembed.downstream.ga_optimizer.decode_zs_to_selfies")
    @patch("chembed.downstream.ga_optimizer.encode_multiple_selfies")
    def test_maximize_simple_flow(self, encode, decode):
        initial_smiles = ["C", "CC"]
        encode.return_value = torch.zeros(2, 4)
        decode.side_effect = [["[C]", "[C][C]"], ["[C][C][C]"]]
        fitness = lambda s: len(s)

        hp = {
            "nb_generations": 1,
            "batch_size": 2,
            "pop_size": 3,
            "std": 0.1,
            "nb_best": "automatic",
            "epsilon_best_fitness": 0.0,
            "crossovers": False,
        }

        with tempfile.TemporaryDirectory() as d:
            output_dir = Path(d)
            all_file = output_dir/"all.csv"
            best_file = output_dir/"best.csv"
            out_df = ga_optimizer.maximize(
                initial_smiles_list=initial_smiles,
                fitness_function=fitness,
                vae=MagicMock(),
                hyperparameters=hp,
                all_collector_file=all_file,
                best_collector_file=best_file,
            )
            self.assertTrue(all_file.exists())
            self.assertTrue(best_file.exists())
            self.assertIn("CCC", set(out_df["smiles"]))



if __name__ == "__main__":
    unittest.main()

