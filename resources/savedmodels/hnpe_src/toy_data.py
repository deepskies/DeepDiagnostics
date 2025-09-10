import numpy as np
import sys
import os
from abc import ABC, abstractmethod, abstractstaticmethod

import pytorch_lightning as pl
import torch
from tensorflow_probability.substrates.jax.internal.test_util import disable_test_for_backend
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.distributions import Uniform, Normal, HalfNormal, Exponential
# from torch.distributions.uniform import Uniform
# from torch.distributions.normal import Normal
# from torch.distributions.half_normal import HalfNormal
# from torch.distributions.exponential import Exponential
import matplotlib.pyplot as plt

import logging
module_logger = logging.getLogger('diagnostics')
logger = logging.getLogger('diagnostics.toy_data')

@np.vectorize
def normalize(data, mean, std):
    return (data - mean)/std

class BaseDataModule(pl.LightningDataModule, ABC):
    def __init__(self,
                 loader_batch_size=None,
                 exp_dir = None,
                 relative_data_dir = None,
                 train_fn = None,
                 test_fn = None,
                 transform = None,
                 generator_seed_1 = 32,
                 generator_seed_2 = 25,
                 n_batch = 20_000,
                 n_set_max = 25,
                 n_features = 5,
                 device = 'cpu',
                 local_labels = None,
                 global_labels = None,
                 test_data_seed = 150,
                 SIMULATOR_INDEX = None,
                 use_hyperpriors=False,
                 prior_type='uniform',
                 ):
        super().__init__()
        self.loader_batch_size = loader_batch_size
        self.global_labels = global_labels
        self.local_labels = local_labels
        self.exp_dir = exp_dir
        self.data_dir = exp_dir + relative_data_dir
        self.train_data_path = exp_dir + train_fn
        self.test_data_path = exp_dir + test_fn
        self.transform = transform
        self.n_batch = n_batch
        self.n_set_max = n_set_max
        self.n_features = n_features
        self.generator_seed_1 = generator_seed_1
        self.generator_seed_2 = generator_seed_2
        self.pin_memory = False if device == 'cpu' else True
        self.test_data_seed = test_data_seed
        self.SIMULATOR_INDEX = SIMULATOR_INDEX
        self.use_hyperpriors = use_hyperpriors
        self.prior_type = prior_type

    @staticmethod
    def get_times(n_features):
        t = torch.linspace(1, 2, n_features)
        return t

    @abstractmethod
    def simulator(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simulate_new_dataset(self, n_batch=None, n_sets=None, n_features=5, priors='uniform', **kwargs):
        raise NotImplementedError

    def prepare_data(self, n_test_batch=100, overwrite_saved_test_data=False):
        # Prepare train data
        # if train data already exists, load. Otherwise, simulate and save
        if os.path.exists(self.train_data_path):
            logger.info('Will use train data saved in ' + self.train_data_path)
        else:
            logger.info(f'Simulating new train data')
            torch.save(
                self.simulate_new_dataset(n_batch=self.n_batch, n_set=self.n_set_max, n_features=self.n_features),
                self.train_data_path
            )
        # Prepare test data
        if os.path.exists(self.test_data_path):
            test_data = torch.load(self.test_data_path, weights_only=False)
            num_test_data_points_unequal = len(test_data['x']) != n_test_batch
            if num_test_data_points_unequal:
                logger.info(
                    f'Number of desired test data points is not equal to number in saved test_data file'
                )
            if num_test_data_points_unequal or overwrite_saved_test_data:
                if num_test_data_points_unequal:
                    logger.info('The saved test data set does not have the desired number of test batches. /n'
                                'Re-simulating test dataset'
                                )
                if overwrite_saved_test_data:
                    logger.info(f'Overwriting saved test data set with seed {self.test_data_seed}')
                torch.manual_seed(self.test_data_seed)
                logger.info(f'Simulating new test dataset and saving in {self.test_data_path}')
                torch.save(self.simulate_new_dataset(
                    n_batch=n_test_batch,
                    n_set=self.n_set_max,
                    n_features=self.n_features),
                    self.test_data_path
                )
        else:
            logger.info(f'No test data found in {self.test_data_path}. Simulating new test dataset')
            torch.save(
                self.simulate_new_dataset(n_batch=n_test_batch, n_set=self.n_set_max, n_features=self.n_features),
                self.test_data_path
            )

    def setup(self, stage='test'):
        '''Assign train, val, and test data'''
        generator1 = torch.Generator().manual_seed(self.generator_seed_1)
        if stage == "fit":
            logger.info(f'dm.setup in "fit" stage. Loading train data from {self.train_data_path}')
            data_full_dict = torch.load(self.train_data_path, weights_only=False)
            x_full = data_full_dict['x']
            y_local_full = data_full_dict['y_local']
            y_global_full = data_full_dict['y_global']
            dataset_full = TensorDataset(x_full, y_local_full, y_global_full)
            self.data_train, self.data_val = random_split(dataset_full, [.9, .1], generator=generator1)
        if stage == "test":
            logger.info(f'dm.setup in "test" stage. Loading test data from {self.test_data_path}')
            data_test_dict = torch.load(self.test_data_path, weights_only=False)
            dataset_test = TensorDataset(data_test_dict['x'], data_test_dict['y_local'], data_test_dict['y_global'])
            logger.info('Setting dm.data_test to loaded test data')
            self.data_test = dataset_test

    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.loader_batch_size,
                          num_workers=4,
                          pin_memory=self.pin_memory,
                          shuffle=True
                          )

    def val_dataloader(self):
        return DataLoader(self.data_val,
                          batch_size=self.loader_batch_size,
                          num_workers=4,
                          pin_memory=self.pin_memory,
                          shuffle=False
                          )
    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=self.loader_batch_size,
                          num_workers=8,
                          pin_memory=self.pin_memory,
                          shuffle=False
                          )

    def plot_random_train_samples(self, n_points=10, save_dir=None):
        batch_idxs = np.random.randint(low=0, high=self.n_batch, size=(n_points))
        set_idxs = np.random.randint(low=0, high=self.n_set_max, size=(n_points))
        sim_dict = torch.load(self.train_data_path, weights_only=False)
        fig, ax = plt.subplots(1)
        for i, (b_idx, set_idx) in enumerate(zip(batch_idxs, set_idxs)):
            ax.scatter(sim_dict['t'], sim_dict['x'][b_idx, set_idx], color=f'C{i}')
            ax.plot(sim_dict['t'], sim_dict['x_true'][b_idx, set_idx], color=f'C{i}')
            ax.set_xlabel('t')
            ax.set_ylabel('x')
        if save_dir is not None:
            fig.savefig(self.exp_dir + 'random_train_samples_plot.png')

class VaryingSlopeInterceptDM(BaseDataModule):
    def __init__(
            self,
            loader_batch_size,
            exp_dir = './',
            relative_data_dir='',
            train_fn='VSI_train_data.pt',
            test_fn='VSI_test_data.pt',
            transform=None,
            generator_seed_1 = 32,
            generator_seed_2 = 25,
            n_batch = 20_000,
            n_set_max = 25,
            n_features = 5,
            device='cpu',
            test_data_seed=150,
            use_hyperpriors=False,
            prior_type='uniform',
    ):
        global_labels = ['b (intercept)']
        local_labels = ['m (slope)']

        super().__init__(loader_batch_size,
                         exp_dir,
                         relative_data_dir,
                         train_fn,
                         test_fn,
                         transform,
                         generator_seed_1,
                         generator_seed_2,
                         n_batch,
                         n_set_max,
                         n_features,
                         device,
                          local_labels,
                          global_labels,
                         test_data_seed,
                          0,
                        use_hyperpriors,
                        prior_type=prior_type,
        )

        # if not os.path.exists(self.train_data_path):
        #     raise FileNotFoundError(f'{self.train_data_path} does not exist')
        # if not os.path.exists(self.test_data_path):
        #     raise FileNotFoundError(f'{self.test_data_path} does not exist')

    def simulator(self, t, m, b):
        if b.shape != m.shape:
            b = b[:, np.newaxis, :]
        return m * t + b

    def simulate_new_dataset(self, n_batch=None, n_set=None, n_features=5, prior_type=None):
        data_dict = {}
        if prior_type is None:
            prior_type = self.prior_type
        if self.use_hyperpriors:
            if prior_type == 'normal':
                b_hyperpriors = [Normal(0, 2), HalfNormal(3)]  # mu, sigma
                b_hyperparams = [hyperprior.sample((n_batch,)) for hyperprior in b_hyperpriors]
                b_prior = Normal(loc=b_hyperparams[0], scale=b_hyperparams[1])

                m_hyperpriors = [Normal(0, 2), HalfNormal(3)]  # mu, sigma
                m_hyperparams = [hyperprior.sample((n_batch, n_set,)) for hyperprior in m_hyperpriors]
                m_prior = Normal(loc=m_hyperparams[0], scale=m_hyperparams[1])
            elif prior_type == 'uniform':
                b_hyperpriors = [Uniform(-3, -1), Uniform(1,3)]  # mu, sigma
                b_hyperparams = [hyperprior.sample((n_batch,)) for hyperprior in b_hyperpriors]
                b_prior = Normal(loc=b_hyperparams[0], scale=b_hyperparams[1])

                m_hyperpriors = [Normal(0, 2), HalfNormal(3)]  # mu, sigma
                m_hyperparams = [hyperprior.sample((n_batch, n_set,)) for hyperprior in m_hyperpriors]
                m_prior = Normal(loc=m_hyperparams[0], scale=m_hyperparams[1])

            m = m_prior.sample((1,)).reshape((n_batch, n_set, 1))
            b = b_prior.sample((1,)).reshape((n_batch, 1))

            data_dict.update({'global_hyperparams': b_hyperparams,
                              'local_hyperparams': m_hyperparams,
                              'global_prior': b_prior,
                              'local_prior': m_prior,
                              'global_hyperpriors': b_hyperpriors,
                              'local_hyperpriors': m_hyperpriors})
        else:
            if prior_type == 'uniform':
                m_prior = Uniform(low=-2, high=2)
                b_prior = Uniform(low=-2, high=2)
            elif prior_type == 'normal':
                m_prior = Normal(loc=0, scale=4)
                b_prior = Normal(loc=0, scale=4)
            m = m_prior.sample((n_batch, n_set, 1))
            b = b_prior.sample((n_batch, 1))
        sigma_x_true = torch.randn(size=(n_batch, n_set, n_features))
        sigma_x_true /= 10
        t = self.get_times(n_features)
        x_true = self.simulator(t, m, b)
        x = x_true + sigma_x_true
        data_dict.update({'x': x, 'x_true': x_true, 'y_local': m, 'y_global': b,
                        'sigma_x_true': sigma_x_true, 't': t}
                         )
        return data_dict

