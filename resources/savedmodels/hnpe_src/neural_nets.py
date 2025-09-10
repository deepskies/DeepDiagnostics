import torch
import torch.nn as nn

from einops import rearrange, repeat
import numpy as np
from tqdm import tqdm

import os
# import sys
# import gc
import logging
module_logger = logging.getLogger('diagnostics')
logger = logging.getLogger('diagnostics.neural_net')

from resnet import ResNetEstimator
from flows import build_mlp, build_maf

import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

torch.set_default_dtype(torch.float32)

class HierarchicalDeepSet(nn.Module):
    """ Backbone to the hierarchical deep set model, using a ResNet embedder and MAF flows for 
        local and global parameter posterior density estimators.
    """
    def __init__(self, n_local_params, n_global_params, n_transforms, n_set_max, dim_hidden=128, obs_shape=None, condition_local_on_global=True):
        super(HierarchicalDeepSet, self).__init__()
        
        inference_net_kwargs = {"cfg":50}
        self.n_set_max = n_set_max

        obs_dim = len(obs_shape)
        if obs_dim == 1:
            print('Observation data is 1d: using mlp encoder')
            self.enc = build_mlp(input_dim=obs_shape[0], hidden_dim=int(2 * dim_hidden), output_dim=dim_hidden, layers=4)
        else:
            print('Observation data is 2d: using ResNet encoder')
            self.enc = ResNetEstimator(n_out=dim_hidden, **inference_net_kwargs)
        self.dec = build_mlp(input_dim=int(dim_hidden / 2) + 1, hidden_dim=int(2 * dim_hidden), output_dim=int(dim_hidden / 2), layers=4).float()
        
        # Condition local flow on global params if local loss is turned on
        extra_context = np.max([n_local_params - n_global_params, 1, n_global_params])
        self.condition_local_on_global = condition_local_on_global
        
        self.flow_local = build_maf(dim=n_local_params, num_transforms=n_transforms, context_features=int(dim_hidden / 2) + extra_context, hidden_features=int(2 * dim_hidden)).float()
        self.flow_global =  build_maf(dim=n_global_params, num_transforms=n_transforms, context_features=int(dim_hidden / 2), hidden_features=int(2 * dim_hidden)).float()

    def forward(self, x, y_local, y_global):
        batch_size = x.shape[0]

        lens = torch.randint(low=1, high=self.n_set_max + 1,size=(batch_size,), dtype=torch.float32)
        mask = (torch.arange(self.n_set_max).expand(len(lens), self.n_set_max) < torch.Tensor(lens)[:,None]).to(x.device)
        
        x = rearrange(x, "batch n_set l -> (batch n_set) l", n_set=self.n_set_max)
        x = self.enc(x)
        
        x = rearrange(x, "(batch n_set) n_out -> batch n_set n_out", n_set=self.n_set_max) 

        idx_setperm = torch.randperm(self.n_set_max)  # Permutation indices
        x = x[:, idx_setperm, :] * mask[:, :, None]  # Permute set elements and mask
        y_local = y_local[:, idx_setperm, :]
                                
        x, x_cond_local = torch.chunk(x, 2, -1)
        
        x = x.sum(-2) / mask.sum(1)[:, None]
                
        x = torch.cat([x, lens[:, None].to(x.device)], -1)  # Add cardinality for rho network
        x_cond_global = self.dec(x)
                        
        x_cond_local = rearrange(x_cond_local, "batch n_set n_out -> (batch n_set) n_out", n_set=self.n_set_max)
            
        if self.condition_local_on_global:
            y_global_repeat = repeat(y_global, "batch glob -> (batch n_set) glob", n_set=self.n_set_max)
            x_cond_local = torch.cat([x_cond_local, y_global_repeat], -1)
        
        y_local = rearrange(y_local, "batch n_set n_param -> (batch n_set) n_param", n_set=self.n_set_max)
                
        log_prob_local = self.flow_local.log_prob(y_local, x_cond_local)
        log_prob_local = rearrange(log_prob_local, "(batch n_set) -> batch n_set", n_set=self.n_set_max)
        log_prob_local = (log_prob_local * mask).sum(-1)

        log_prob_global = self.flow_global.log_prob(y_global, x_cond_global)
        
        return log_prob_local, log_prob_global


class HierarchicalDeepSetInference(pl.LightningModule):
    """ Hierarchical deep set lightning module for training and inference.
    """

    def __init__(self,
                 optimizer=torch.optim.AdamW,
                 optimizer_kwargs=None,
                 lr=3e-4,
                 scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
                 scheduler_kwargs=None,
                 local_loss=True,
                 global_loss=True,
                 n_local_params=1,
                 n_global_params=1,
                 n_transforms=6,
                 n_set_max=25,
                 dim_hidden=128,
                 obs_shape=None,
                 **kwargs,
                 ):
        super().__init__()

        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs
        self.lr = lr
        self.obs_shape = obs_shape
        self.local_loss = local_loss
        self.global_loss = global_loss
        self.n_set_max = n_set_max
        self.n_local_params = n_local_params
        self.n_global_params = n_global_params
        condition_local_on_global = True if (local_loss and global_loss) else False

        self.deep_set = HierarchicalDeepSet(
            n_local_params=n_local_params,
            n_global_params=n_global_params,
            n_transforms=n_transforms,
            n_set_max=n_set_max,
            dim_hidden=dim_hidden,
            obs_shape=obs_shape,
            condition_local_on_global=condition_local_on_global
        )
        self.save_hyperparameters()

    def test_step(self, batch, batch_idx):
        return {'test_local_loss': log_prob_local, 'test_global_loss':log_prob_global, 'samples_y_local': samples_local, 'samples_y_global': samples_global, 'y_local_true': y_local, 'y_global_true': y_global}

    def forward(self, x, y_local, y_global):
        log_prob = self.deep_set(x, y_local, y_global)
        return log_prob

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr, **self.optimizer_kwargs)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler(optimizer, **self.scheduler_kwargs),
                    "interval": "epoch",
                    "monitor": "val_loss",
                    "frequency": 1}
                }

    def training_step(self, batch, batch_idx):
        x, y_local, y_global = batch
        log_prob_local, log_prob_global = self(x, y_local, y_global)
        log_prob = torch.zeros_like(log_prob_local).to(log_prob_local.device)
        if self.local_loss:
            log_prob += log_prob_local
        if self.global_loss:
            log_prob += log_prob_global
        loss = -log_prob.mean()
        self.log('train_loss', loss, on_epoch=True)
        self.log('local_train_loss', -log_prob_local.mean(), on_epoch=True)
        self.log('global_train_loss', -log_prob_global.mean(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_local, y_global = batch
        log_prob_local, log_prob_global = self(x, y_local, y_global)
        log_prob = torch.zeros_like(log_prob_local).to(log_prob_local.device)
        if self.local_loss:
            log_prob += log_prob_local
        if self.global_loss:
            log_prob += log_prob_global
        loss = -log_prob.mean()
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_local_loss', -log_prob_local.mean(), on_epoch=True)
        self.log('val_global_loss', -log_prob_global.mean(), on_epoch=True)
        return loss
    
    def get_deepset(self):
        return self.deep_set
    

    def get_local_and_global_posterior_samples(self, dm, device1='cpu', n_eval=None, n_samples=1000, save_dir=None, overwrite_if_exists=False):
        global_samples = self.get_global_posterior_samples(
            dm,
            device1=device1,
            n_eval=n_eval,
            n_samples=n_samples,
            save_dir=save_dir,
            overwrite_if_exists=overwrite_if_exists,
        )
        local_samples = self.get_local_posterior_samples(
            dm,
            device1=device1,
            n_eval=n_eval,
            n_samples=n_samples,
            save_dir=save_dir,
            overwrite_if_exists=overwrite_if_exists,
        )
        return local_samples, global_samples

    def get_global_posterior_samples(self, dm, device1='cpu', n_eval=None, n_samples=1000, batch_size=50, save_dir=None, overwrite_if_exists=False):
        filename='global_samples.pt'
        if save_dir is None:
            save_dir = dm.data_dir

        if os.path.exists(save_dir + filename) and not overwrite_if_exists:
            logger.info(f'Loading global samples from {save_dir + filename}')
            return torch.load(save_dir + filename)

        if not os.path.exists(save_dir + filename):
            logger.info(f'Will save global samples to {save_dir + filename}')

        # x, y_local, y_global = dm.data_test.tensors
        if n_eval is None:
            n_eval = self.n_set_max
        
        x, y_local, y_global = dm.data_test.tensors
        device2 = 'cpu'
        x = rearrange(x[:, :n_eval], "batch n_set l -> (batch n_set) l", n_set=n_eval)
        x = self.deep_set.enc.to(device1)(x.to(device1))
        x = rearrange(x, "(batch n_set) n_out -> batch n_set n_out", n_set=n_eval)

        x_cond_global, x_cond_local = torch.chunk(x, 2, -1)
        x_cond_global = x_cond_global.mean(-2)
        col_n_eval = torch.full(size=(len(x_cond_global),), fill_value=n_eval)[:, None]
        x_cond_global = torch.cat([x_cond_global.to(device2), col_n_eval], dim=-1)
        x_cond_global = self.deep_set.dec.to(device2)(x_cond_global.to(device2))

        global_samples = self.deep_set.flow_global.sample(num_samples=n_samples, context=x_cond_global)
        torch.save(global_samples, save_dir + filename)
        return global_samples

    def get_local_posterior_samples(self, dm, device1='cpu', n_eval=None, n_samples=1000, batch_size=50, save_dir=None, overwrite_if_exists=True):
        if save_dir is None:
            save_dir = dm.data_dir
        filename='local_samples.pt'

        if os.path.exists(save_dir + filename) and not overwrite_if_exists:
            logger.info(f'Loading local samples from {save_dir + filename}')
            return torch.load(save_dir + filename)

        if not os.path.exists(save_dir + filename):
            logger.info(f'Will save local samples to {save_dir + filename}')

        x, y_local, y_global = dm.data_test.tensors
        if n_eval is None:
            n_eval = self.n_set_max
        
        device2 = 'cpu'
        x, y_local, y_global = dm.data_test.tensors
        x = rearrange(x[:, :n_eval], "batch n_set l -> (batch n_set) l", n_set=n_eval)
        x = self.deep_set.enc.to(device1)(x.to(device1))
        x = rearrange(x, "(batch n_set) n_out -> batch n_set n_out", n_set=n_eval)

        x_cond_global, x_cond_local = torch.chunk(x, 2, -1)
        global_samples = self.get_global_posterior_samples(dm, device1, n_eval=n_eval, n_samples=n_samples, save_dir=save_dir)

        n_batches = n_samples // batch_size
        x_cond_local_global = torch.cat([x_cond_local.to(device2), self._get_global_samples_mean(global_samples, n_eval)], -1)
        # x_cond_local_global = rearrange(x_cond_local_global, "batch nset nout -> (batch nset) nout", nset=n_eval)

        local_samples = torch.empty(size=(len(x_cond_local_global), n_eval, n_samples, self.n_local_params))
        for i_batch, x_batch in tqdm(enumerate(x_cond_local_global)):
            torch.save(self.deep_set.flow_local.sample(num_samples=n_samples, context=x_cond_local_global[i_batch]),
                       save_dir + filename[:-3] + f'_{i_batch}.pt')
        for i_batch, x_batch in tqdm(enumerate(x_cond_local_global)):
            file_path = save_dir + filename[:-3] + f'_{i_batch}.pt'
            local_samples[i_batch] = torch.load(file_path)
            os.remove(file_path)
        torch.save(local_samples, save_dir + 'local_samples.pt')
        return local_samples

    @staticmethod
    def _get_global_samples_mean(global_samples, n_eval):
        global_samples_mean = repeat(global_samples.mean(dim=1),"batch nparam -> batch n_eval nparam", n_eval=n_eval)
        return global_samples_mean

