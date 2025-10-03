import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import pandas as pd

from chembed.evaluate import main as evaluate_main


class TestMainEvaluateCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.vae_ckpt = self.tmpdir/"model.ckpt"
        self.out_json = self.tmpdir/"out.json"
        self.test_csv = self.tmpdir/"test.csv"
        self.vae_ckpt.write_text("")
        self.test_df = pd.DataFrame.from_dict({'selfies': ['[C][C]', '[C][C][C]']})
        self.test_csv = self.tmpdir/"test.csv"
        self.test_df.to_csv(self.test_csv)

    def tearDown(self):
        self.tmp.cleanup()

    @patch("chembed.evaluate.write_json")
    @patch("chembed.evaluate.evaluate_reconstruction")
    @patch("chembed.evaluate.checkpoint_utils.load_vae_from_checkpoint")
    def test_reconstruction_only(self, mock_load, mock_eval_recon, mock_write_json):
        mock_load.return_value = object()
        mock_eval_recon.return_value = {"string_reconstruction_accuracy": 1.0}

        argv = [
            "--vae", str(self.vae_ckpt),
            "-o", str(self.out_json),
            "--evaluate_reconstruction",
            "--test_data_path", str(self.test_csv),
        ]
        evaluate_main(argv)
        mock_load.assert_called_once()
        mock_eval_recon.assert_called_once()
        mock_write_json.assert_called_once_with({"string_reconstruction_accuracy": 1.0}, self.out_json)

    @patch("chembed.evaluate.write_json")
    @patch("chembed.evaluate.evaluate_latent_space")
    @patch("chembed.evaluate.checkpoint_utils.load_vae_from_checkpoint")
    def test_latent_space_with_fallback_to_smiles(self, mock_load, mock_eval_latent, mock_write_json):
        mock_load.return_value = object()
        self.test_df['smiles'] = ['CC', 'CCC']
        self.test_df.to_csv(self.test_csv)
        mock_eval_latent.return_value = {"trustworthiness_5": 0.93}

        argv = [
            "--vae", str(self.vae_ckpt),
            "-o", str(self.out_json),
            "--evaluate_latent_space",
            "--test_data_path", str(self.test_csv),
        ]
        evaluate_main(argv)

        mock_eval_latent.assert_called_once()
        mock_write_json.assert_called_once_with({"trustworthiness_5": 0.93}, self.out_json)

    @patch("chembed.evaluate.checkpoint_utils.load_vae_from_checkpoint")
    def test_latent_space_missing_file_raises(self, mock_load):
        mock_load.return_value = object()
        missing = self.tmpdir/"does_not_exist.csv"
        argv = [
            "--vae", str(self.vae_ckpt),
            "-o", str(self.out_json),
            "--evaluate_latent_space",
            "--test_data_path", str(missing),
        ]
        with self.assertRaises(FileNotFoundError):
            evaluate_main(argv)

    def test_errors_argument_logic(self):
        # no evaluation flags
        with self.assertRaises(ValueError):
            evaluate_main(["--vae", str(self.vae_ckpt), "-o", str(self.out_json)])

        # novelty requires generation
        with self.assertRaises(ValueError):
            evaluate_main([
                "--vae", str(self.vae_ckpt), "-o", str(self.out_json),
                "--evaluate_novelty"
            ])

        # novelty requires train_data_csv when generation is requested
        with self.assertRaises(ValueError):
            evaluate_main([
                "--vae", str(self.vae_ckpt), "-o", str(self.out_json),
                "--evaluate_generation", "--evaluate_novelty"
            ])

        # write_reconstructed_to requires evaluate_reconstruction
        with self.assertRaises(ValueError):
            evaluate_main([
                "--vae", str(self.vae_ckpt), "-o", str(self.out_json),
                "--write_reconstructed_to", str(self.tmpdir/"recon.csv")
            ])

        # reconstruction (or latent) requires test_data_path
        with self.assertRaises(ValueError):
            evaluate_main([
                "--vae", str(self.vae_ckpt), "-o", str(self.out_json),
                "--evaluate_reconstruction"
            ])


if __name__ == "__main__":
    unittest.main()

