import os
import pickle

# imports for Hierarchical model
import torch
from einops import rearrange, repeat

from deepdiagnostics.models.model import Model


class SBIModel(Model):
    """
    Load a trained model that was generated with Mackelab SBI :cite:p:`centero2020sbi`. 
    `Read more about saving and loading requirements here <https://sbi-dev.github.io/sbi/latest/faq/question_05_pickling/>`_. 

    Args:
        model_path (str): relative path to a model - must be a .pkl file. 
    """
    def __init__(self, model_path):
        super().__init__(model_path)

    def _load(self, path: str) -> None:
        assert os.path.exists(path), f"Cannot find model file at location {path}"
        assert path.split(".")[-1] == "pkl", "File extension must be 'pkl'"

        with open(path, "rb") as file:
            posterior = pickle.load(file)
        self.posterior = posterior

    def sample_posterior(self, n_samples: int, x_true):
        """
        Sample the posterior 

        Args:
            n_samples (int): Number of samples to draw
            x_true (np.ndarray): Context samples. (must be dims=(n_samples, M))

        Returns:
            np.ndarray: Posterior samples
        """
        return self.posterior.sample(
            (n_samples,), x=x_true, show_progress_bars=False
        ).cpu()  # TODO Unbind from cpu

    def predict_posterior(self, data, context_samples):
        """
        Sample the posterior and then 

        Args:
            data (deepdiagnostics.data.Data): Data module with the loaded simulation
            context_samples (np.ndarray): X values to test the posterior over. 

        Returns:
            np.ndarray: Simulator output 
        """
        posterior_samples = self.sample_posterior(context_samples)
        posterior_predictive_samples = data.simulator(
            posterior_samples, context_samples
        )
        return posterior_predictive_samples


class HierarchyModel(Model):
    def __init__(self, model_path):
        # Load the model
        # self.model_path = model_path
        # self.load()
        super().__init__(model_path)

    def _load(self, path: str):
        assert os.path.exists(path), f"Cannot find model file at location {path}"
        assert path.split(".")[-1] == "pkl", "File extension must be 'pkl'"

        # with open(self.model_path, "rb") as file:
        with open(path, "rb") as file:
            self.model = pickle.load(file)

    def _sample_global(self, x, n_samples=1000, device="cpu"):
        deep_set = self.model.deep_set
        deep_set.eval()
        n_eval = x.shape[-2]
        print("n_eval in global:", n_eval)
        device = torch.device(device)
        deep_set.to(device)

        x_flat = rearrange(x[:, :n_eval], "batch n_set l -> (batch n_set) l", n_set=n_eval)
        x_enc = deep_set.enc.to(device)(x_flat.to(device))
        x_enc = rearrange(x_enc, "(batch n_set) n_out -> batch n_set n_out", n_set=n_eval)

        x_cond_global, _ = torch.chunk(x_enc, 2, -1)
        x_cond_global = x_cond_global.mean(-2)
        col_n_eval = torch.full((x_cond_global.size(0), 1), float(n_eval), dtype=torch.float32, device=device)
        x_cond_global = torch.cat([x_cond_global, col_n_eval], dim=-1)
        x_cond_global = deep_set.dec.to(device)(x_cond_global)

        samples = deep_set.flow_global.sample(num_samples=n_samples, context=x_cond_global)
        # normalize shape to (batch, n_samples, dim)
        if samples.dim() == 3 and samples.size(0) == n_samples:
            samples = samples.permute(1, 0, 2).contiguous()
        return samples

    def _sample_local(self, x, n_samples=1000, device="cpu"):
        deep_set = self.model.deep_set
        deep_set.eval()
        print("x shape in local:", x.shape)
        n_eval = x.shape[-2]
        print("n_eval in local:", n_eval)
        device = torch.device(device)
        deep_set.to(device)

        # Encode observations
        x_flat = rearrange(x[:, :n_eval], "batch n_set l -> (batch n_set) l", n_set=n_eval)
        x_enc = deep_set.enc.to(device)(x_flat.to(device))
        x_enc = rearrange(x_enc, "(batch n_set) n_out -> batch n_set n_out", n_set=n_eval)
        _, x_cond_local = torch.chunk(x_enc, 2, -1)  # (batch, n_eval, ctx_dim_local)

        print("x shape before giving to global:", x.shape)

        # Global mean context
        g_samples = self._sample_global(x, n_samples=n_samples, device=str(device))  # pass x, not data
        g_mean = g_samples.mean(dim=1)                                              # (batch, n_global)
        g_mean = repeat(g_mean, "batch p -> batch n_eval p", n_eval=n_eval)         # (batch, n_eval, n_global)

        # Build local context and sample
        ctx = torch.cat([x_cond_local, g_mean], dim=-1)                             # (batch, n_eval, ctx_dim)
        batch, n_eval_eff, ctx_dim = ctx.shape
        ctx_flat = ctx.reshape(batch * n_eval_eff, ctx_dim)                         # (batch*n_eval, ctx_dim)

        samples = deep_set.flow_local.sample(num_samples=n_samples, context=ctx_flat)
        # Expected shape (n_samples, batch*n_eval, n_local) -> (batch, n_eval, n_samples, n_local)
        if samples.dim() == 3 and samples.size(1) == batch * n_eval_eff:
            samples = samples.permute(1, 0, 2).contiguous()
        samples = samples.reshape(batch, n_eval_eff, n_samples, -1)
        return samples

    def sample_posterior(self, n_samples, x_true, global_samples=True):
        if global_samples:
            print("x_true shape in sample posterior:", x_true.shape)
            global_samples = self._sample_global(x_true, n_samples=n_samples)
            return global_samples
        else:
            local_samples = self._sample_local(x_true, n_samples=n_samples)
            return local_samples
        