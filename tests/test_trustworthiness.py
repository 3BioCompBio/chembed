import unittest
import numpy as np
import torch

from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from chembed.metrics import trustworthiness_score

def _toy_fps_numpy(N=6, nbits=64, noise_level=0.05, seed=0):
    rng = np.random.default_rng(seed)
    fps = []

    # cluster A center: first quarter bits set
    cluster_A_center = np.zeros(nbits, dtype=np.uint8)
    cluster_A_center[: nbits // 4] = 1
    for _ in range(N // 2):
        noise = (rng.random(nbits) < noise_level).astype(np.uint8)
        fp = np.bitwise_xor(cluster_A_center, noise)
        fps.append(fp)

    # cluster B center: second quarter bits set
    cluster_B_center = np.zeros(nbits, dtype=np.uint8)
    cluster_B_center[nbits // 4 : nbits // 2] = 1
    for _ in range(N // 2, N):
        noise = (rng.random(nbits) < noise_level).astype(np.uint8)
        fp = np.bitwise_xor(cluster_B_center, noise)
        fps.append(fp)

    return np.stack(fps, axis=0)


def _toy_embeddings_numpy(N=6, d_model=8, noise_level=0.01, seed=0):
    rng = np.random.default_rng(seed)
    Y = np.zeros((N, d_model), dtype=np.float32)

    # cluster A: near -1 on first axis
    Y[: N // 2, 0] = -1.0
    # cluster B: near +1 on first axis
    Y[N // 2 :, 0] = +1.0

    # add small noise so points aren’t identical
    Y += noise_level * rng.standard_normal(size=Y.shape).astype(np.float32)
    return Y



class TestTrustworthinessScore(unittest.TestCase):
    def setUp(self):
        self.N = 20
        self.fps_np = _toy_fps_numpy(N=self.N, nbits=64)
        self.zs_np = _toy_embeddings_numpy(N=self.N)

    def test_numpy_inputs(self):
        score = trustworthiness_score(self.zs_np, self.fps_np, n_neighbors=5)
        self.assertGreaterEqual(score, 0.0)
        self.assertGreater(score, 0.9)

    def test_torch_inputs(self):
        zs = torch.from_numpy(self.zs_np).float()
        fps = torch.from_numpy(self.fps_np)
        score = trustworthiness_score(zs, fps, n_neighbors=5)
        self.assertGreater(score, 0.9)

    def test_torch_inputs_float(self):
        zs = torch.from_numpy(self.zs_np).float()
        fps = torch.from_numpy(self.fps_np).float()
        score = trustworthiness_score(zs, fps, n_neighbors=5)
        self.assertGreater(score, 0.9)

    def test_torch_inputs_and_squeeze(self):
        zs = torch.from_numpy(self.zs_np).float().unsqueeze(1)
        fps = torch.from_numpy(self.fps_np).to(dtype=torch.uint8)
        score = trustworthiness_score(zs, fps, n_neighbors=5)
        self.assertGreater(score, 0.9)

    def test_invalid_dims_zs(self):
        zs_bad = np.zeros((3,), dtype=np.float64)
        with self.assertRaises(ValueError):
            trustworthiness_score(zs_bad, self.fps_np, n_neighbors=3)

    def test_invalid_dims_fps(self):
        fps_bad = np.zeros((3, 4, 5), dtype=np.uint8)
        with self.assertRaises(ValueError):
            trustworthiness_score(self.zs_np, fps_bad, n_neighbors=3)

    def test_rdkit_bitvect_list(self):
        fps_list = []
        nbits = self.fps_np.shape[1]
        for row in self.fps_np:
            bv = ExplicitBitVect(nbits)
            onbits = np.flatnonzero(row)
            for b in onbits:
                bv.SetBit(int(b))
            fps_list.append(bv)
        score = trustworthiness_score(self.zs_np, fps_list, n_neighbors=5)
        self.assertGreater(score, 0.9)

    @unittest.skipUnless(torch.cuda.is_available(), "No GPU available")
    def test_cuda_tensor(self):
        device = torch.device('cuda')
        zs = torch.from_numpy(self.zs_np).float().to(device)
        fps = torch.from_numpy(self.fps_np).float().to(device)
        score = trustworthiness_score(zs, fps, n_neighbors=5)
        self.assertGreater(score, 0.9)

    def test_low_trustworthiness(self):
        Y_good = self.zs_np 
        perm = np.random.permutation(self.N)
        Y_bad = Y_good[perm]

        tw_good = trustworthiness_score(Y_good, self.fps_np, n_neighbors=3)
        tw_bad = trustworthiness_score(Y_bad, self.fps_np, n_neighbors=3)

        self.assertLess(tw_bad, 0.6)
        self.assertGreater(tw_good, tw_bad)

if __name__ == "__main__":
    unittest.main()

