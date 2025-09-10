
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pymc as pm
from pytorch_lightning import LightningDataModule, LightningModule
from einops import rearrange, repeat

from scipy.stats import binom, gaussian_kde, iqr

from scipy.stats import binom
import torch
from tqdm import tqdm
from pytorch_lightning import LightningDataModule, LightningModule

import logging
module_logger = logging.getLogger('diagnostics')
logger = logging.getLogger('diagnostics.Diagnostics')
import arviz as az

class Diagnostics():
    def __init__(
            self,
            trained_model:LightningModule,
            dm: LightningDataModule,
            local_samples=None, # shape (nbatch, nset, params)
            global_samples=None, # shape (nbatch, nparams)
            local_labels=None,
            global_labels=None,
            local_colors=None,
            global_colors=None,
            seed = 5,
            outdir=None,
            overwrite_saved_samples=False,
            n_posterior_samples=None,
            n_eval=None,
    ):
        self.trained_model = trained_model
        self.dm = dm
        outdir = outdir if outdir is not None else './outdir/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        self.outdir = outdir
        self.overwrite_saved_samples = overwrite_saved_samples
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info('Using cuda')
        else:
            self.device = torch.device('cpu')
            logger.info('Using cpu')
        self.n_posterior_samples = 1000 if n_posterior_samples is None else n_posterior_samples
        logger.info(f'Set number of posterior samples: {self.n_posterior_samples}')
        self.n_eval = n_eval
        # generate posterior samples if not input
        if global_samples is not None and local_samples is not None:
            logger.info('Using samples passed in')
            self.local_samples = local_samples
            self.global_samples = global_samples
        else:
            logger.info('Getting posterior samples')
            self.local_samples, self.global_samples = self.trained_model.get_local_and_global_posterior_samples(
                self.dm,
                device1=self.device,
                n_eval=self.n_eval,
                save_dir=self.outdir,
                n_samples=self.n_posterior_samples,
                overwrite_if_exists=self.overwrite_saved_samples
            )
            torch.save(self.local_samples, self.outdir + 'local_samples.pt')
            torch.save(self.global_samples, self.dm.outdir + 'global_samples.pt')

        self.n_global_params = self.global_samples.shape[-1]
        self.n_local_params = self.local_samples.shape[-1]
        self.n_total_params = self.n_global_params + self.n_local_params
        # make sure number of params in model equals number of params in samples
        assert self.trained_model.hparams['n_global_params'] == self.n_global_params
        assert self.trained_model.hparams['n_local_params'] == self.n_local_params

        # generate automatic labels and colors if not input
        self.global_labels = global_labels if global_labels is not None else [f'global_{i}' for i in range(self.n_global_params)]
        self.local_labels = local_labels if local_labels is not None else [f'local_{i}' for i in range(self.n_local_params)]
        self.global_colors = global_colors if global_colors is not None else [f'C{i}' for i in range(self.n_global_params)]
        self.local_colors = local_colors if local_colors is not None else [f'C{i}' for i in range(self.n_global_params, self.n_total_params)]

        self.x_test, self.y_local_test, self.y_global_test = dm.data_test.tensors

        # in case we need specific seed
        self.seed = seed

        ## sort true global params and samples in increasing order and get means and stds
        glob_srt = torch.argsort(self.y_global_test, dim=0)
        self.y_global_test_sorted = torch.take_along_dim(self.y_global_test, glob_srt, dim=0)
        self.global_samples_mean_sorted = torch.take_along_dim(self.global_samples.mean(dim=-2), glob_srt, dim=0)
        self.global_samples_std_sorted = torch.take_along_dim(self.global_samples.std(dim=-2), glob_srt, dim=0)

        ## sort true local params and samples in increasing order and get means and stds
        loc_srt = torch.argsort(self.y_local_test.flatten(0,1), dim=0)
        self.y_local_test_sorted = torch.take_along_dim(self.y_local_test.flatten(0,1), loc_srt, dim=0)
        self.local_samples_mean_sorted = torch.take_along_dim(self.local_samples.flatten(0,1).mean(dim=-2), loc_srt, dim=0)
        self.local_samples_std_sorted = torch.take_along_dim(self.local_samples.flatten(0,1).std(dim=-2), loc_srt, dim=0)

    def plot_model_predictions(self, **kwargs):

        # set up plotting defaults if None input
        kwargs['label_fontsize'] = kwargs['label_fontsize'] if 'label_fontsize' in kwargs else None
        kwargs['legend_fontsize'] = kwargs['legend_fontsize'] if 'legend_fontsize' in kwargs else None
        kwargs['tick_fontsize'] = kwargs['tick_fontsize'] if 'tick_fontsize' in kwargs else None
        kwargs['marker'] = kwargs['marker'] if 'marker' in kwargs else 'o'
        kwargs['edgecolor'] = kwargs['edgecolor'] if 'edgecolor' in kwargs else 'black'
        kwargs['linewidth'] = kwargs['linewidth'] if 'linewidth' in kwargs else .5
        kwargs['capsize'] = kwargs['capsize'] if 'capsize' in kwargs else 0
        kwargs['alpha'] = kwargs['alpha'] if 'alpha' in kwargs else 1
        kwargs['s'] = kwargs['s'] if 's' in kwargs else 50
        kwargs['elinewidth'] = kwargs['elinewidth'] if 'elinewidth' in kwargs else .5

        # global prediction plot
        fig_glob, axs_glob = self._one_level_model_predictions(
            self.y_global_test_sorted,
            self.global_samples_mean_sorted,
            self.global_samples_std_sorted,
            self.n_global_params,
            self.global_labels,
            **kwargs
        )

        # local prediction plot
        fig_loc, axs_loc = self._one_level_model_predictions(
            self.y_local_test_sorted[::25],
            self.local_samples_mean_sorted[::25],
            self.local_samples_std_sorted[::25],
            self.n_local_params,
            self.local_labels,
            **kwargs
        )
        fig_glob.savefig(self.outdir + 'model_predictions_global.png')
        fig_loc.savefig(self.outdir + 'model_predictions_local.png')
        return (fig_glob, axs_glob), (fig_loc, axs_loc);

    def _one_level_model_predictions(self, y_true, y_samples_mean, y_samples_std, n_params, labels, **kwargs):
        # plot
        fig, axs = plt.subplots(2, n_params, sharex='col', figsize=(8*n_params, 16*n_params))

        # plot one column per param
        for i_param in range(n_params):
            y_true = y_true[:, i_param].detach().numpy()
            y_samples_mean = y_samples_mean[:, i_param].detach().numpy()
            y_samples_std = y_samples_std[:, i_param].detach().numpy()

            # plot column
            self._plot_one_col_model_pred(
                y_true=y_true,
                y_samples_mean=y_samples_mean,
                y_samples_std=y_samples_std,
                axs=axs[:, i_param] if n_params > 1 else axs,
                fig=fig,
                label=labels[i_param],
                **kwargs
            )
        fig.tight_layout()
        return fig, axs

    def _plot_one_col_model_pred(self,
                                y_true,
                                y_samples_mean,
                                y_samples_std,
                                axs,
                                fig,
                                label,
                                **kwargs
                                ):

        self._plot_pairplot_model(
            y_true,
            y_samples_mean,
            y_samples_std,
            axs,
            fig,
            label,
            **kwargs
        )
        self._plot_residuals_model_pred(
            y_true,
            y_samples_mean,
            y_samples_std,
            axs,
            label,
            **kwargs
        )

    @staticmethod
    def _plot_pairplot_model(
            y_true,
            y_samples_mean,
            y_samples_std,
            axs,
            fig,
            label,
            **kwargs
    ):

        ax = axs[0]
        ax.plot(y_true, y_true, label='true', color='red', linestyle='--', zorder=1000)
        ax.scatter(
            y_true,
            y_samples_mean,
            marker=kwargs['marker'],
            edgecolor=kwargs['edgecolor'],
            linewidth=kwargs['linewidth'],
            s=kwargs['s'],
        )
        ax.errorbar(
            y_true,
            y_samples_mean,
            yerr=y_samples_std,
            fmt='none',
            color='black',
            elinewidth=kwargs['elinewidth'],
            capsize=kwargs['capsize'],
            alpha=kwargs['alpha'],
            label=r'1$\sigma$',
        )
        ax.legend(fontsize=kwargs['legend_fontsize'])
        ax.set_ylabel(label, fontsize=kwargs['label_fontsize'])
        ax.tick_params(labelsize=kwargs['tick_fontsize'])

    @staticmethod
    def _plot_residuals_model_pred(
            y_true,
            y_samples_mean,
            y_samples_std,
            axs,
            label,
            **kwargs
    ):

        ax = axs[1]
        ax.scatter(
            y_true,
            y_samples_mean - y_true,
            linewidth=kwargs['linewidth'],
            marker=kwargs['marker'],
            edgecolor=kwargs['edgecolor'],
            s=kwargs['s'],
        )
        ax.errorbar(
            y_true,
            y_samples_mean - y_true,
            yerr=y_samples_std,
            fmt='none',
            color='black',
            elinewidth=kwargs['elinewidth'],
            capsize=kwargs['capsize'],
            label=r'1$\sigma$',
        )
        ax.hlines(0, xmin=y_true.min(), xmax=y_true.max(), color='red', linestyle='--', linewidth=2)
        ax.legend(fontsize=kwargs['legend_fontsize'])
        ax.set_xlabel(label, fontsize=kwargs['label_fontsize'])
        ax.set_ylabel(label, fontsize=kwargs['label_fontsize'])
        ax.tick_params(labelsize=kwargs['tick_fontsize'])

    def run_sbc(self, line_alpha=.8, uniform_region_alpha=.3, figsize=(4,4), fig_global=None, ax_global=None, fig_local=None, ax_local=None):
        # global
        glob_ranks = self._one_level_ranks(self.x_test, self.n_global_params, self.y_global_test, self.global_samples)
        fig_glob, axs_glob = self._sbc_ecdf_rank_plot(glob_ranks, self.global_labels, self.global_colors, line_alpha, uniform_region_alpha, figsize=figsize, fig=fig_global, ax=ax_global)

        # local
        loc_ranks = self._one_level_ranks(self.x_test, self.n_local_params, self.y_local_test, self.local_samples)
        fig_loc, axs_loc = self._sbc_ecdf_rank_plot(loc_ranks, self.local_labels, self.local_colors, line_alpha, uniform_region_alpha, figsize=figsize, fig=fig_local, ax=ax_local)

        fig_glob.savefig(self.outdir + 'ECDF_global.png')
        fig_loc.savefig(self.outdir + 'ECDF_local.png')

    @staticmethod
    def _one_level_ranks(x, n_params, y_true, posterior_samples):
        reduce_1d_fn = [eval(f"lambda theta, x: theta[:, {i}]") for i in range(n_params)]
        n_set = x.shape[1]
        x = x.flatten(0, 1) # (nbatch nset) ndata
        if posterior_samples.shape[1] == n_set: #ie, if local var w/ shape (nbatch nset nsamp nparam)
            y_true = y_true.flatten(0,1)
            posterior_samples = posterior_samples.flatten(0,1) # flatten to get (nbatch nset) nsamp nparams
        else: # else, is global with shape (nbatch nsamp nparam)
            y_true  = repeat(y_true, 'nbatch nparam -> (nbatch nset) nparam', nset=n_set)
            posterior_samples = repeat(posterior_samples, 'nbatch nsamp nparam -> (nbatch nset) nsamp nparam', nset=n_set)

        # n_params = y_true.shape[-1]
        n_sbc_runs = posterior_samples.shape[0]

        ranks = torch.zeros((n_sbc_runs, len(reduce_1d_fn)))

        # calculate ranks
        for sbc_idx, (y_true0, x0) in tqdm(
                enumerate(zip(y_true, x, strict=False)),
                total=n_sbc_runs,
                desc=f"Calculating ranks for {n_sbc_runs} sbc samples.",
        ):
            for dim_idx, reduce_fn in enumerate(reduce_1d_fn):
                # rank posterior samples against true parameter, reduced to 1D.
                ranks[sbc_idx, dim_idx] = (
                    (reduce_fn(posterior_samples[sbc_idx, :, :], x0) < reduce_fn(y_true0.unsqueeze(0),
                                                                                    x0)).sum().item()
                )
        return ranks

    def _sbc_ecdf_rank_plot(self, ranks, param_labels, colors, line_alpha=None, uniform_region_alpha=None, figsize=None, fig=None, ax=None):
        # plot ranks
        ranks_list = [ranks]
        n_sbc_runs, num_parameters = ranks_list[0].shape

        n_bins = n_sbc_runs // 20
        num_repeats = 50

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.sca(ax)

        ranki = ranks_list[0]
        for jj in range(num_parameters):
            self._plot_ranks_as_cdf(
                ranki[:, jj],  # type: ignore
                n_bins,
                num_repeats,
                ranks_label=param_labels[jj],
                color=colors[jj],
                xlabel="posterior rank",
                # Plot ylabel and legend at last.
                show_ylabel = jj == (num_parameters - 1),
                alpha=line_alpha,
            )
        self._plot_cdf_region_expected_under_uniformity(
            n_sbc_runs,
            n_bins,
            num_repeats,
            alpha=uniform_region_alpha,
        )
        # show legend on the last subplot.
        # plt.legend(**legend_kwargs)
        plt.legend()
        return fig, ax  # pyright: ignore[reportReturnType]

    @staticmethod
    def _plot_ranks_as_cdf(
            ranks,
            n_bins,
            n_repeats,
            ranks_label,
            xlabel,
            color,
            alpha=.8,
            show_ylabel=True,
            num_ticks=3
    ):
        hist, *_ = np.histogram(ranks, bins=n_bins, density=False)
        # Construct empirical CDF.
        histcs = hist.cumsum()
        # Plot cdf and repeat each stair step
        plt.plot(
            np.linspace(0, n_bins, n_repeats * n_bins),
            np.repeat(histcs / histcs.max(), n_repeats),
            label=ranks_label,
            color=color,
            alpha=alpha,
        )

        if show_ylabel:
            plt.yticks(np.linspace(0, 1, 3))
            plt.ylabel("empirical CDF")
        else:
            # Plot ticks only
            plt.yticks(np.linspace(0, 1, 3), [])

        plt.ylim(0, 1)
        plt.xlim(0, n_bins)
        plt.xticks(np.linspace(0, n_bins, num_ticks))
        plt.xlabel("posterior rank" if xlabel is None else xlabel)

    @staticmethod
    def _plot_cdf_region_expected_under_uniformity(
            n_sbc_samples,
            n_bins,
            n_repeats,
            alpha=.2,
            color='grey'
    ):

        # Construct uniform histogram.
        uni_bins = binom(n_sbc_samples, p=1 / n_bins).ppf(0.5) * np.ones(n_bins)
        uni_bins_cdf = uni_bins.cumsum() / uni_bins.sum()
        # Decrease value one in last entry by epsilon to find valid
        # confidence intervals.
        uni_bins_cdf[-1] -= 1e-9

        lower = [binom(n_sbc_samples, p=p).ppf(0.005) for p in uni_bins_cdf]
        upper = [binom(n_sbc_samples, p=p).ppf(0.995) for p in uni_bins_cdf]

        # Plot grey area with expected ECDF.
        plt.fill_between(
            x=np.linspace(0, n_bins, n_repeats * n_bins),
            y1=np.repeat(lower / np.max(lower), n_repeats),
            y2=np.repeat(upper / np.max(upper), n_repeats),  # pyright: ignore[reportArgumentType]
            color=color,
            alpha=alpha,
            label="expected under uniformity",
        )

