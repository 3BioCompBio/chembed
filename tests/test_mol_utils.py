import unittest

from rdkit import Chem
from chembed import mol_utils

class TestMolUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_properties(self):
        mol = Chem.MolFromSmiles("CCCC=O")
        for prop in ['MolWt', 'QED', 'SA_Score']:
            mol_utils.mol_property_from_mol(mol, prop)


if __name__=="__main__":
    unittest.main()
