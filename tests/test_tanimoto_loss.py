import unittest
import torch
import torch.nn as nn

from chembed.metrics import tanimoto_shifted_euclidean_correlation_triu_loss, compute_euclidean_distance_matrix, compute_tanimoto_similarity_matrix 
from chembed.utils import set_random_seed_everywhere

class TestTanimotoLoss(unittest.TestCase):
    def setUp(self):
        # build a tiny datasets of N=6 fingerprints of nbits=64 with two clear clusters
        N = 6
        nbits = 64
        fps = []

        # cluster A
        cluster_A_center = torch.zeros(nbits, dtype=torch.long)
        cluster_A_center[:nbits//4] = 1
        for i in range(N//2):
            noise = (torch.rand(nbits) < 0.05).long()
            fp = (cluster_A_center ^ noise)
            fps.append(fp)

        # cluster B
        cluster_B_center = torch.zeros(nbits, dtype=torch.long)
        cluster_B_center[nbits//4:nbits//2] = 1
        for i in range(N//2, N):
            noise = (torch.rand(nbits) < 0.05).long()
            fp = (cluster_B_center ^ noise)
            fps.append(fp)

        self.fps = torch.stack(fps, dim=0).float()

        self.N = self.fps.shape[0]

        self.d_model = 8


    def _upper_tri(self, mat: torch.Tensor) -> torch.Tensor:
        mask = torch.triu(torch.ones_like(mat, dtype=torch.bool), diagonal=1)
        return mat[mask]

    def test_training_aligns_latent_with_tanimoto(self):
        zs = nn.Parameter(torch.randn(self.N, self.d_model))
        opt = torch.optim.Adam([zs], lr=0.05)
        
        M_tanimoto = compute_tanimoto_similarity_matrix(self.fps)
        ts = self._upper_tri(M_tanimoto)

        # baseline correlation
        with torch.no_grad():
            D_euclidean_start = compute_euclidean_distance_matrix(zs)
            ds_start = self._upper_tri(D_euclidean_start)
            corr_start = torch.corrcoef(torch.stack([ts, ds_start]))[0, 1].item()
            loss_start = tanimoto_shifted_euclidean_correlation_triu_loss(zs, self.fps).item()

        # Train zs to minimize the shifted correlation loss (target: corr -> -1, loss -> 0)
        for step in range(400):
            opt.zero_grad(set_to_none=True)
            loss = tanimoto_shifted_euclidean_correlation_triu_loss(zs, self.fps)
            loss.backward()
            opt.step()

        # correlation after training
        with torch.no_grad():
            D_euclidean_final = compute_euclidean_distance_matrix(zs)
            ds_final = self._upper_tri(D_euclidean_final)
            corr_final = torch.corrcoef(torch.stack([ts, ds_final]))[0, 1].item()
            loss_final = tanimoto_shifted_euclidean_correlation_triu_loss(zs, self.fps).item()


        # loss decreased
        self.assertLess(loss_final, loss_start, f"loss did not decrease: {loss_final:.4f} >= {loss_start:.4f}")

        # correlation improved significantly
        corr_gain = corr_start - corr_final # more negative
        self.assertGreater(corr_gain, 0.2)

        # on average, within-cluster distances < between-cluster distances
        with torch.no_grad():
            idxA = torch.tensor([i for i in range(self.N//2)])
            idxB = torch.tensor([i for i in range(self.N//2, self.N)])
            DA = D_euclidean_final[idxA][:, idxA]
            DB = D_euclidean_final[idxB][:, idxB]
            DAB = D_euclidean_final[idxA][:, idxB]

            mean_DA = self._upper_tri(DA).mean() 
            mean_DB = self._upper_tri(DB).mean() 
            mean_D_within = (mean_DA + mean_DB)/2

            mean_D_between = DAB.mean()

            print("mean_D_within=", mean_D_within, "mean_D_between=", mean_D_between)

            self.assertLess(mean_D_within, mean_D_between)


if __name__ == "__main__":
    set_random_seed_everywhere()
    unittest.main(verbosity=2)
