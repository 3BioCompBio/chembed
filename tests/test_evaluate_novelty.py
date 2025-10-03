import unittest
from pathlib import Path
import tempfile
import pandas as pd
import selfies as sf
from rdkit import Chem

from chembed.evaluate import get_novelty_scores_from_train_file

def __dict_to_csv__(d):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    tmp_file = Path(tmp.name)
    df = pd.DataFrame.from_dict(d)
    df.to_csv(tmp_file)
    return tmp_file

def __dict_to_parquet__(d):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
    tmp_file = Path(tmp.name)
    df = pd.DataFrame.from_dict(d)
    df.to_parquet(tmp_file)
    return tmp_file

def __csv_from_smiles_list__(smiles_list):
    return __dict_to_csv__({'smiles':smiles_list, 'selfies':[sf.encoder(s) for s in smiles_list]})

def __csv_from_selfies_list__(selfies_list):
    return __dict_to_csv__({'selfies':selfies_list, 'smiles':[sf.decoder(s) for s in selfies_list]})

def __parquet_from_smiles_list__(smiles_list):
    return __dict_to_parquet__({'smiles':smiles_list, 'selfies':[sf.encoder(s) for s in smiles_list]})

def __get_novelty_from_smiles_lists__(generated_smiles, train_smiles):
    train_file = __csv_from_smiles_list__(train_smiles)
    generated_selfies = [sf.encoder(s) for s in generated_smiles]
    generated_mols = [Chem.MolFromSmiles(s) for s in generated_smiles]
    scores = get_novelty_scores_from_train_file(generated_selfies, generated_mols, train_file, smiles_column_name='smiles')
    return scores

def __get_novelty_from_selfies_lists__(generated_selfies, train_selfies):
    train_file = __csv_from_selfies_list__(train_selfies)
    generated_smiles = [sf.decoder(s) for s in generated_selfies]
    generated_mols = [Chem.MolFromSmiles(s) for s in generated_smiles]
    scores = get_novelty_scores_from_train_file(generated_selfies, generated_mols, train_file, smiles_column_name='smiles')
    return scores



class TestNovelty(unittest.TestCase):

    def test_no_novelty(self):
        train = ['CCCO', 'CCC']
        generated = train
        novelty_scores = __get_novelty_from_smiles_lists__(generated, train)
        self.assertEqual(novelty_scores['selfies_novelty_score'], 0.0)
        self.assertEqual(novelty_scores['mol_novelty_score'], 0.0)
        self.assertEqual(novelty_scores['fingerprint_novelty_score'], 0.0)
    

    def test_all_novelty(self):
        train = ['CCCO']
        generated = ['CCC']
        novelty_scores = __get_novelty_from_smiles_lists__(generated, train)
        self.assertEqual(novelty_scores['selfies_novelty_score'], 1.0)
        self.assertEqual(novelty_scores['mol_novelty_score'], 1.0)
        self.assertEqual(novelty_scores['fingerprint_novelty_score'], 1.0)
    

    def test_half_novelty(self):
        train = ['CCCO', 'CCC']
        generated = ['CCCO', 'CCCCCCC']
        novelty_scores = __get_novelty_from_smiles_lists__(generated, train)
        self.assertEqual(novelty_scores['selfies_novelty_score'], 0.5)
        self.assertEqual(novelty_scores['mol_novelty_score'], 0.5)
        self.assertEqual(novelty_scores['fingerprint_novelty_score'], 0.5)

    def test_half_novelty_2(self):
        train = ['CCCO']
        generated = ['CCCO', 'CCCCCCC']
        novelty_scores = __get_novelty_from_smiles_lists__(generated, train)
        self.assertEqual(novelty_scores['selfies_novelty_score'], 0.5)
        self.assertEqual(novelty_scores['mol_novelty_score'], 0.5)
        self.assertEqual(novelty_scores['fingerprint_novelty_score'], 0.5)
    
    def test_same_mol_different_selfies(self):
        selfies1="[C][C][O]"
        selfies2="[O][C][C]"
        train = [selfies1]
        generated = [selfies2]
        novelty_scores = __get_novelty_from_selfies_lists__(generated, train)
        self.assertEqual(novelty_scores['selfies_novelty_score'], 1.0)
        self.assertEqual(novelty_scores['mol_novelty_score'], 0.0)
        self.assertEqual(novelty_scores['fingerprint_novelty_score'], 0.0)
    
    def test_different_mols_same_fingerprint(self):
        smiles1 = "F/C=C/F"
        smiles2 = "F/C=C\\F"
        train = [smiles1]
        generated = [smiles2]
        novelty_scores = __get_novelty_from_smiles_lists__(generated, train)
        self.assertEqual(novelty_scores['selfies_novelty_score'], 1.0)
        self.assertEqual(novelty_scores['mol_novelty_score'], 1.0)
        self.assertEqual(novelty_scores['fingerprint_novelty_score'], 0.0)
    
    def test_mixed(self):
        train = ["F/C=C/F"]
        generated = ["F/C=C\\F", "CCC"]
        novelty_scores = __get_novelty_from_smiles_lists__(generated, train)
        self.assertEqual(novelty_scores['selfies_novelty_score'], 1.0)
        self.assertEqual(novelty_scores['mol_novelty_score'], 1.0)
        self.assertEqual(novelty_scores['fingerprint_novelty_score'], 0.5)

    def test_half_novelty_parquet(self):
        train_smiles = ['CCCO', 'CCC']
        generated_smiles = ['CCCO', 'CCCCCCC']
        train_file = __parquet_from_smiles_list__(train_smiles)
        generated_selfies = [sf.encoder(s) for s in generated_smiles]
        generated_mols = [Chem.MolFromSmiles(s) for s in generated_smiles]
        novelty_scores = get_novelty_scores_from_train_file(generated_selfies, generated_mols, train_file, smiles_column_name='smiles')
        self.assertEqual(novelty_scores['selfies_novelty_score'], 0.5)
        self.assertEqual(novelty_scores['mol_novelty_score'], 0.5)
        self.assertEqual(novelty_scores['fingerprint_novelty_score'], 0.5)





if __name__=="__main__":
    unittest.main()
