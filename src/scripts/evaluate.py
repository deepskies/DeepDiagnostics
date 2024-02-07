"""
Diagnostic tools for evaluating the quality of the posterior
from a previously trained inference model.

Includes utilities for posterior diagnostics as well as some
inference functions.
"""

from scripts.io import ModelLoader
import argparse
from sbi.analysis import run_sbc, sbc_rank_plot, check_sbc
import numpy as np
from tqdm import tqdm


# plotting style things:
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

# remove top and right axis from plots
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False


class Diagnose_generative:
    def posterior_predictive(
        self, theta_true, x_true, simulator, posterior_samples, true_sigma
    ):
        # not sure how or where to define the simulator
        # could require that people input posterior predictive samples,
        # already drawn from the simulator
        posterior_predictive_samples = simulator(posterior_samples)
        y_true = simulator(theta_true, x_true)
        # also go through and plot one sigma interval
        # plot the true values
        plt.clf()
        xs_sim = np.linspace(0, 100, 101)
        ys_sim = np.array(posterior_predictive_samples)
        plt.fill_between(
            xs_sim,
            np.mean(ys_sim, axis=0) - 1 * np.std(ys_sim, axis=0),
            np.mean(ys_sim, axis=0) + 1 * np.std(ys_sim, axis=0),
            color="#FF495C",
            label="posterior predictive check with noise",
        )
        plt.plot(
            xs_sim,
            np.mean(ys_sim, axis=0) + true_sigma,
            color="#25283D",
            label="true input error",
        )
        plt.plot(xs_sim, np.mean(ys_sim, axis=0) - true_sigma, color="#25283D")
        plt.scatter(xs_sim, np.array(y_true), color="black")
        plt.legend()
        plt.show()
        return ys_sim

    def generate_sbc_samples(
        self,
        prior,
        posterior,
        simulator,
        num_sbc_runs=1_000,
        num_posterior_samples=1_000,
    ):
        # generate ground truth parameters
        # and corresponding simulated observations for SBC.
        thetas = prior.sample((num_sbc_runs,))
        ys = simulator(thetas)
        # run SBC: for each inference we draw 1000 posterior samples.
        ranks, dap_samples = run_sbc(
            thetas, ys, posterior, num_posterior_samples=num_posterior_samples
        )
        return thetas, ys, ranks, dap_samples

    def sbc_statistics(self,
                       ranks,
                       thetas,
                       dap_samples,
                       num_posterior_samples):
        """
        The ks pvalues are vanishingly small here,
        so we can reject the null hypothesis
        (of the marginal rank distributions being equivalent to
        an uniform distribution). The inference clearly went wrong.

        In terms of the c2st_ranks diagnostic;
        this is a nonparametric two sample test from training on
        and testing on the rank versus uniform distributions
        and distinguishing between them. If values are close to 0.5,
        it is hard to distinguish.

        The data-averaged posterior value compares to the prior;
        if these values are close to 0.5, dap is like the prior distribution.
        """
        check_stats = check_sbc(
            ranks, thetas, dap_samples,
            num_posterior_samples=num_posterior_samples
        )
        return check_stats

    def plot_1d_ranks(
        self,
        ranks,
        num_posterior_samples,
        labels_list,
        colorlist,
        plot=False,
        save=True,
        path="plots/generative/",
    ):
        """
        If the rank plots are consistent with being uniform,
        the color bars should fall mostly within the grey area.
        The grey area is the 99% confidence interval
        for the uniform distribution, so if the rank histogram falls
        outside this for more than 1 in 100 bars, that means it is not
        consistent with an uniform distribution (which is what we want).

        A central peaked rank plot could be indicative of a posterior that's
        too concentrated whereas one with wings is a posterior that is
        too dispersed. Conversely, if the distribution is shifted left or
        right that indicates that the parameters are biased.

        If it's choppy, it could be indicative of not doing enough sampling.
        A good rule of thumb is N / B ~ 20, where N is the number of samples
        and B is the number of bins.
        """
        # help(sbc_rank_plot)
        if colorlist:
            _ = sbc_rank_plot(
                ranks=ranks,
                num_posterior_samples=num_posterior_samples,
                plot_type="hist",
                num_bins=None,
                parameter_labels=labels_list,
                colors=colorlist,
            )
        else:
            _ = sbc_rank_plot(
                ranks=ranks,
                num_posterior_samples=num_posterior_samples,
                plot_type="hist",
                num_bins=None,
                parameter_labels=labels_list,
            )
        if save:
            plt.savefig(path + "sbc_ranks.pdf")
        if plot:
            plt.show()

    def plot_cdf_1d_ranks(
        self,
        ranks,
        num_posterior_samples,
        labels_list,
        colorlist,
        plot=False,
        save=True,
        path="plots/generative/",
    ):
        """
        This is a different way to visualize the same thing
        from the 1d rank plots.
        Essentially, the grey is the 95% confidence interval for
        an uniform distribution.
        The cdf for the posterior rank distributions (in color) should fall
        within this band.
        """
        help(sbc_rank_plot)
        if colorlist:
            f, ax = sbc_rank_plot(
                ranks,
                num_posterior_samples,
                plot_type="cdf",
                parameter_labels=labels_list,
                colors=colorlist,
            )
        else:
            f, ax = sbc_rank_plot(
                ranks,
                num_posterior_samples,
                plot_type="cdf",
                parameter_labels=labels_list,
            )
        if save:
            plt.savefig(path + "sbc_ranks_cdf.pdf")
        if plot:
            plt.show()

    def calculate_coverage_fraction(
        self, posterior, thetas, ys, percentile_list,
        samples_per_inference=1_000
    ):
        """
        posterior --> the trained posterior
        thetas --> true parameter values
        ys --> the "observed" data used for inference

        """
        # this holds all posterior samples for each inference run
        all_samples = np.empty((len(ys),
                                samples_per_inference,
                                np.shape(thetas)[1]))
        count_array = []
        # make this for loop into a progress bar:
        for i in tqdm(
            range(len(ys)),
            desc="Sampling from the posterior for each obs",
            unit="obs"
        ):
            # for i in range(len(ys)):
            # sample from the trained posterior n_sample times
            # for each observation
            samples = posterior.sample(
                sample_shape=(samples_per_inference,), x=ys[i],
                show_progress_bars=False
            ).cpu()

            """
            # plot posterior samples
            fig, axes = pairplot(
                samples,
                labels = ['m', 'b'],
                #limits = [[0,10],[-10,10],[0,10]],
                truths = truth_array[i],
                figsize=(5, 5)
            )
            axes[0, 1].plot([truth_array[i][1]], [truth_array[i][0]],
                marker="o", color="r")
            """

            all_samples[i] = samples
            count_vector = []
            # step through the percentile list
            for ind, cov in enumerate(percentile_list):
                percentile_l = 50.0 - cov / 2
                percentile_u = 50.0 + cov / 2
                # find the percentile for the posterior for this observation
                # this is n_params dimensional
                # the units are in parameter space
                confidence_l = np.percentile(samples.cpu(),
                                             percentile_l,
                                             axis=0)
                confidence_u = np.percentile(samples.cpu(),
                                             percentile_u,
                                             axis=0)
                # this is asking if the true parameter value
                # is contained between the
                # upper and lower confidence intervals
                # checks separately for each side of the 50th percentile
                count = np.logical_and(
                    confidence_u - thetas.T[:, i] > 0,
                    thetas.T[:, i] - confidence_l > 0
                )
                count_vector.append(count)
            # each time the above is > 0, adds a count
            count_array.append(count_vector)
        count_sum_array = np.sum(count_array, axis=0)
        frac_lens_within_vol = np.array(count_sum_array)
        return all_samples, np.array(frac_lens_within_vol) / len(ys)

    def plot_coverage_fraction(
        self,
        posterior,
        thetas,
        ys,
        samples_per_inference,
        labels_list,
        colorlist,
        n_percentile_steps=21,
        plot=False,
        save=True,
        path="plots/generative/",
    ):
        percentile_array = np.linspace(0, 100, n_percentile_steps)
        samples, frac_array = self.calculate_coverage_fraction(
            posterior,
            np.array(thetas),
            ys,
            percentile_array,
            samples_per_inference=samples_per_inference,
        )

        percentile_array_norm = np.array(percentile_array) / 100

        # Create a cycler with hexcode colors and linestyles
        if colorlist:
            color_cycler = cycler(color=colorlist)
        else:
            color_cycler = cycler(color="viridis")
        linestyle_cycler = cycler(linestyle=["-", "-."])

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Use itertools.cycle to loop through colors and linestyles
        color_cycle = iter(color_cycler)
        linestyle_cycle = iter(linestyle_cycler)
        # Iterate over the second dimension of frac_array
        for i in range(frac_array.shape[1]):
            color_style = next(color_cycle)["color"]
            linestyle_style = next(linestyle_cycle)["linestyle"]
            ax.plot(
                percentile_array_norm,
                frac_array[:, i],
                alpha=1.0,
                lw=3,
                linestyle=linestyle_style,
                color=color_style,
                label=labels_list[i],
            )

        ax.plot(
            [0, 0.5, 1], [0, 0.5, 1], "k--", lw=3, zorder=1000,
            label="Reference Line"
        )
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.text(0.03, 0.93, "Underconfident", horizontalalignment="left")
        ax.text(0.3, 0.05, "Overconfident", horizontalalignment="left")
        ax.legend(loc="lower right")
        ax.set_xlabel("Confidence Interval of the Posterior Volume")
        ax.set_ylabel("Fraction of Lenses within Posterior Volume")
        ax.set_title("NPE")
        plt.tight_layout()
        if save:
            plt.savefig(path + "coverage.pdf")
        if plot:
            plt.show()

    def run_all_sbc(
        self,
        prior,
        posterior,
        simulator,
        labels_list,
        colorlist,
        num_sbc_runs=1_000,
        num_posterior_samples=1_000,
        samples_per_inference=1_000,
        plot=True,
        save=False,
        path="plots/generative/",
    ):
        """
        Runs and displays mackelab's SBC (simulation-based calibration)

        Simulation-based calibration is a set of tools built into
        Mackelab's sbi interface. It provides a way to compare the
        inferred posterior distribution to the true parameter values.
        It performs multiple instances of drawing parameter values from
        the prior, running these through the simulator, and comparing
        these values to those obtained from the run of inference.
        Importantly, this will not diagnose what's going on for one draw
        from the posterior (ie at one data point). Instead, it's meant to
        give an overall sense of the health of the posterior learned from SBI.

        This technique is based on rank plots.
        Rank plots are produced from comparing each posterior parameter draw
        (from the prior) to the distribution of parameter values in the
        posterior. There should be a 1:1 ranking, aka these rank plots should
        be similar in shape to a uniform distribution.
        """
        thetas, ys, ranks, dap_samples = self.generate_sbc_samples(
            prior, posterior, simulator, num_sbc_runs, num_posterior_samples
        )

        stats = self.sbc_statistics(ranks, thetas, dap_samples,
                                    num_posterior_samples)
        print(stats)
        self.plot_1d_ranks(
            ranks,
            num_posterior_samples,
            labels_list,
            colorlist,
            plot=plot,
            save=save,
            path=path,
        )

        self.plot_cdf_1d_ranks(
            ranks,
            num_posterior_samples,
            labels_list,
            colorlist,
            plot=plot,
            save=save,
            path=path,
        )

        self.plot_coverage_fraction(
            posterior,
            thetas,
            ys,
            samples_per_inference,
            labels_list,
            colorlist,
            n_percentile_steps=21,
            plot=plot,
            save=save,
            path=path,
        )

    def parameter_1_to_1_plots(
        samples,
        thetas,
        color_list,
        m_color,
        b_color,
    ):
        """
        We've already saved samples, let's compare the inferred
        (and associated error bar) parameters from each of the data points we
        used for the SBC analysis.
        """

        print(np.shape(samples), np.shape(thetas))
        percentile_16_m = []
        percentile_50_m = []
        percentile_84_m = []
        percentile_16_b = []
        percentile_50_b = []
        percentile_84_b = []
        for i in range(len(samples[0])):
            # print(np.shape(samples[i]))
            # STOP
            percentile_16_m.append(np.percentile(samples[i, 0], 16))
            percentile_50_m.append(np.percentile(samples[i, 0], 50))
            percentile_84_m.append(np.percentile(samples[i, 0], 84))
            percentile_16_b.append(np.percentile(samples[i, 1], 16))
            percentile_50_b.append(np.percentile(samples[i, 1], 50))
            percentile_84_b.append(np.percentile(samples[i, 1], 84))
        yerr_minus = [mid - low for (mid, low) in zip(percentile_50_m,
                                                      percentile_16_m)]
        yerr_plus = [high - mid for high, mid in zip(percentile_84_m,
                                                     percentile_50_m)]

        # Randomly set half of the error bars to zero
        random_indices = np.random.choice(
            len(yerr_minus), int(len(yerr_minus) // 1.15), replace=False
        )
        for idx in random_indices:
            yerr_minus[idx] = 0
            yerr_plus[idx] = 0

        plt.errorbar(
            np.array(thetas[:, i]),
            percentile_50_m,
            yerr=[yerr_minus, yerr_plus],
            linestyle="None",
            color=color_list[i],
            capsize=5,
        )
        plt.scatter(np.array(thetas[:, 0]), percentile_50_m, color=m_color)
        plt.plot(percentile_50_m, percentile_50_m, color="k")
        plt.xlabel("True value [m]")
        plt.ylabel("Recovered value [m]")
        plt.show()

        plt.clf()
        plt.scatter(np.array(thetas[:, 1]), percentile_50_b, color=b_color)
        yerr_minus = [mid - low for (mid, low) in zip(percentile_50_b,
                                                      percentile_16_b)]
        yerr_plus = [high - mid for high, mid in zip(percentile_84_b,
                                                     percentile_50_b)]

        # Randomly set half of the error bars to zero
        random_indices = np.random.choice(
            len(yerr_minus), int(len(yerr_minus) // 1.15), replace=False
        )
        for idx in random_indices:
            yerr_minus[idx] = 0
            yerr_plus[idx] = 0

        plt.errorbar(
            np.array(thetas[:, 1]),
            percentile_50_b,
            yerr=[yerr_minus, yerr_plus],
            linestyle="None",
            color=b_color,
            capsize=5,
        )
        plt.plot(percentile_50_b, percentile_50_b, color="black")
        plt.xlabel("True value [b]")
        plt.ylabel("Recovered value [b]")
        plt.show()


class Diagnose_static:
    def generate_sbc_samples(
        self,
        posterior,
        thetas,
        ys,
        num_posterior_samples=1_000,
    ):
        # generate ground truth parameters
        # and corresponding simulated observations for SBC.
        # run SBC: for each inference we draw 1000 posterior samples.
        ranks, dap_samples = run_sbc(
            thetas, ys, posterior, num_posterior_samples=num_posterior_samples
        )
        return thetas, ys, ranks, dap_samples

    def sbc_statistics(self, ranks, thetas, dap_samples,
                       num_posterior_samples):
        """
        The ks pvalues are vanishingly small here,
        so we can reject the null hypothesis
        (of the marginal rank distributions being equivalent to
        an uniform distribution). The inference clearly went wrong.

        In terms of the c2st_ranks diagnostic;
        this is a nonparametric two sample test from training on
        and testing on the rank versus uniform distributions
        and distinguishing between them. If values are close to 0.5,
        it is hard to distinguish.

        The data-averaged posterior value compares to the prior;
        if these values are close to 0.5, dap is like the prior distribution.
        """
        check_stats = check_sbc(
            ranks, thetas, dap_samples,
            num_posterior_samples=num_posterior_samples
        )
        return check_stats

    def plot_1d_ranks(
        self,
        ranks,
        num_posterior_samples,
        labels_list,
        colorlist,
        plot=False,
        save=True,
        path="plots/static/",
    ):
        """
        If the rank plots are consistent with being uniform,
        the color bars should fall mostly within the grey area.
        The grey area is the 99% confidence interval
        for the uniform distribution, so if the rank histogram falls
        outside this for more than 1 in 100 bars, that means it is not
        consistent with an uniform distribution (which is what we want).

        A central peaked rank plot could be indicative of a posterior that's
        too concentrated whereas one with wings is a posterior that is
        too dispersed. Conversely, if the distribution is shifted left or
        right that indicates that the parameters are biased.

        If it's choppy, it could be indicative of not doing enough sampling.
        A good rule of thumb is N / B ~ 20, where N is the number of samples
        and B is the number of bins.
        """
        # help(sbc_rank_plot)
        if colorlist:
            _ = sbc_rank_plot(
                ranks=ranks,
                num_posterior_samples=num_posterior_samples,
                plot_type="hist",
                num_bins=None,
                parameter_labels=labels_list,
                colors=colorlist,
            )
        else:
            _ = sbc_rank_plot(
                ranks=ranks,
                num_posterior_samples=num_posterior_samples,
                plot_type="hist",
                num_bins=None,
                parameter_labels=labels_list,
            )
        if save:
            plt.savefig(path + "sbc_ranks.pdf")
        if plot:
            plt.show()

    def plot_cdf_1d_ranks(
        self,
        ranks,
        num_posterior_samples,
        labels_list,
        colorlist,
        plot=False,
        save=True,
        path="plots/static/",
    ):
        """
        This is a different way to visualize the same thing
        from the 1d rank plots.
        Essentially, the grey is the 95% confidence interval for
        an uniform distribution.
        The cdf for the posterior rank distributions (in color) should fall
        within this band.
        """
        help(sbc_rank_plot)
        if colorlist:
            f, ax = sbc_rank_plot(
                ranks,
                num_posterior_samples,
                plot_type="cdf",
                parameter_labels=labels_list,
                colors=colorlist,
            )
        else:
            f, ax = sbc_rank_plot(
                ranks,
                num_posterior_samples,
                plot_type="cdf",
                parameter_labels=labels_list,
            )
        if save:
            plt.savefig(path + "sbc_ranks_cdf.pdf")
        if plot:
            plt.show()

    def calculate_coverage_fraction(
        self, posterior, thetas, ys, percentile_list,
        samples_per_inference=1_000
    ):
        """
        posterior --> the trained posterior
        thetas --> true parameter values
        ys --> the "observed" data used for inference

        """
        # this holds all posterior samples for each inference run
        all_samples = np.empty((len(ys), samples_per_inference,
                                np.shape(thetas)[1]))
        count_array = []
        # make this for loop into a progress bar:
        for i in tqdm(
            range(len(ys)), desc="Sampling from the posterior for each obs",
            unit="obs"
        ):
            # for i in range(len(ys)):
            # sample from the trained posterior n_sample times
            # for each observation
            samples = posterior.sample(
                sample_shape=(samples_per_inference,), x=ys[i],
                show_progress_bars=False
            ).cpu()

            """
            # plot posterior samples
            fig, axes = pairplot(
                samples,
                labels = ['m', 'b'],
                #limits = [[0,10],[-10,10],[0,10]],
                truths = truth_array[i],
                figsize=(5, 5)
            )
            axes[0, 1].plot([truth_array[i][1]], [truth_array[i][0]],
                marker="o", color="r")
            """

            all_samples[i] = samples
            count_vector = []
            # step through the percentile list
            for ind, cov in enumerate(percentile_list):
                percentile_l = 50.0 - cov / 2
                percentile_u = 50.0 + cov / 2
                # find the percentile for the posterior for this observation
                # this is n_params dimensional
                # the units are in parameter space
                confidence_l = np.percentile(samples.cpu(), percentile_l,
                                             axis=0)
                confidence_u = np.percentile(samples.cpu(), percentile_u,
                                             axis=0)
                # this is asking if the true parameter value
                # is contained between the
                # upper and lower confidence intervals
                # checks separately for each side of the 50th percentile
                count = np.logical_and(
                    confidence_u - thetas.T[:, i] > 0,
                    thetas.T[:, i] - confidence_l > 0
                )
                count_vector.append(count)
            # each time the above is > 0, adds a count
            count_array.append(count_vector)
        count_sum_array = np.sum(count_array, axis=0)
        frac_lens_within_vol = np.array(count_sum_array)
        return all_samples, np.array(frac_lens_within_vol) / len(ys)

    def plot_coverage_fraction(
        self,
        posterior,
        thetas,
        ys,
        samples_per_inference,
        labels_list,
        colorlist,
        n_percentile_steps=21,
        plot=False,
        save=True,
        path="plots/static/",
    ):
        percentile_array = np.linspace(0, 100, n_percentile_steps)
        samples, frac_array = self.calculate_coverage_fraction(
            posterior,
            np.array(thetas),
            ys,
            percentile_array,
            samples_per_inference=samples_per_inference,
        )

        percentile_array_norm = np.array(percentile_array) / 100

        # Create a cycler with hexcode colors and linestyles
        if colorlist:
            color_cycler = cycler(color=colorlist)
        else:
            color_cycler = cycler(color="viridis")
        linestyle_cycler = cycler(linestyle=["-", "-."])

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Use itertools.cycle to loop through colors and linestyles
        color_cycle = iter(color_cycler)
        linestyle_cycle = iter(linestyle_cycler)
        # Iterate over the second dimension of frac_array
        for i in range(frac_array.shape[1]):
            color_style = next(color_cycle)["color"]
            linestyle_style = next(linestyle_cycle)["linestyle"]
            ax.plot(
                percentile_array_norm,
                frac_array[:, i],
                alpha=1.0,
                lw=3,
                linestyle=linestyle_style,
                color=color_style,
                label=labels_list[i],
            )

        ax.plot(
            [0, 0.5, 1], [0, 0.5, 1], "k--", lw=3, zorder=1000,
            label="Reference Line"
        )
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.text(0.03, 0.93, "Underconfident", horizontalalignment="left")
        ax.text(0.3, 0.05, "Overconfident", horizontalalignment="left")
        ax.legend(loc="lower right")
        ax.set_xlabel("Confidence Interval of the Posterior Volume")
        ax.set_ylabel("Fraction of Lenses within Posterior Volume")
        ax.set_title("NPE")
        plt.tight_layout()
        if save:
            plt.savefig(path + "coverage.pdf")
        if plot:
            plt.show()

    def run_all_sbc(
        self,
        prior,
        posterior,
        thetas,
        ys,
        labels_list,
        colorlist,
        num_posterior_samples=1_000,
        samples_per_inference=1_000,
        plot=True,
        save=False,
        path="plots/static/",
    ):
        """
        Runs and displays mackelab's SBC (simulation-based calibration)

        Simulation-based calibration is a set of tools built into
        Mackelab's sbi interface. It provides a way to compare the
        inferred posterior distribution to the true parameter values.
        It performs multiple instances of drawing parameter values from
        the prior, running these through the simulator, and comparing
        these values to those obtained from the run of inference.
        Importantly, this will not diagnose what's going on for one draw
        from the posterior (ie at one data point). Instead, it's meant to
        give an overall sense of the health of the posterior learned from SBI.

        This technique is based on rank plots.
        Rank plots are produced from comparing each posterior parameter draw
        (from the prior) to the distribution of parameter values in the
        posterior. There should be a 1:1 ranking, aka these rank plots should
        be similar in shape to a uniform distribution.
        """
        thetas, ys, ranks, dap_samples = self.generate_sbc_samples(
            posterior, thetas, ys, num_posterior_samples
        )

        stats = self.sbc_statistics(ranks, thetas, dap_samples,
                                    num_posterior_samples)
        print(stats)
        self.plot_1d_ranks(
            ranks,
            num_posterior_samples,
            labels_list,
            colorlist,
            plot=plot,
            save=save,
            path=path,
        )

        self.plot_cdf_1d_ranks(
            ranks,
            num_posterior_samples,
            labels_list,
            colorlist,
            plot=plot,
            save=save,
            path=path,
        )

        self.plot_coverage_fraction(
            posterior,
            thetas,
            ys,
            samples_per_inference,
            labels_list,
            colorlist,
            n_percentile_steps=21,
            plot=plot,
            save=save,
            path=path,
        )

    def parameter_1_to_1_plots(
        samples,
        thetas,
        color_list,
        m_color,
        b_color,
    ):
        """
        We've already saved samples, let's compare the inferred
        (and associated error bar) parameters from each of the data points we
        used for the SBC analysis.
        """

        print(np.shape(samples), np.shape(thetas))
        percentile_16_m = []
        percentile_50_m = []
        percentile_84_m = []
        percentile_16_b = []
        percentile_50_b = []
        percentile_84_b = []
        for i in range(len(samples[0])):
            # print(np.shape(samples[i]))
            # STOP
            percentile_16_m.append(np.percentile(samples[i, 0], 16))
            percentile_50_m.append(np.percentile(samples[i, 0], 50))
            percentile_84_m.append(np.percentile(samples[i, 0], 84))
            percentile_16_b.append(np.percentile(samples[i, 1], 16))
            percentile_50_b.append(np.percentile(samples[i, 1], 50))
            percentile_84_b.append(np.percentile(samples[i, 1], 84))
        yerr_minus = [mid - low for (mid, low) in zip(percentile_50_m,
                                                      percentile_16_m)]
        yerr_plus = [high - mid for high, mid in zip(percentile_84_m,
                                                     percentile_50_m)]

        # Randomly set half of the error bars to zero
        random_indices = np.random.choice(
            len(yerr_minus), int(len(yerr_minus) // 1.15), replace=False
        )
        for idx in random_indices:
            yerr_minus[idx] = 0
            yerr_plus[idx] = 0

        plt.errorbar(
            np.array(thetas[:, i]),
            percentile_50_m,
            yerr=[yerr_minus, yerr_plus],
            linestyle="None",
            color=color_list[i],
            capsize=5,
        )
        plt.scatter(np.array(thetas[:, 0]), percentile_50_m, color=m_color)
        plt.plot(percentile_50_m, percentile_50_m, color="k")
        plt.xlabel("True value [m]")
        plt.ylabel("Recovered value [m]")
        plt.show()

        plt.clf()
        plt.scatter(np.array(thetas[:, 1]), percentile_50_b, color=b_color)
        yerr_minus = [mid - low for (mid, low) in zip(percentile_50_b,
                                                      percentile_16_b)]
        yerr_plus = [high - mid for high, mid in zip(percentile_84_b,
                                                     percentile_50_b)]

        # Randomly set half of the error bars to zero
        random_indices = np.random.choice(
            len(yerr_minus), int(len(yerr_minus) // 1.15), replace=False
        )
        for idx in random_indices:
            yerr_minus[idx] = 0
            yerr_plus[idx] = 0

        plt.errorbar(
            np.array(thetas[:, 1]),
            percentile_50_b,
            yerr=[yerr_minus, yerr_plus],
            linestyle="None",
            color=b_color,
            capsize=5,
        )
        plt.plot(percentile_50_b, percentile_50_b, color="black")
        plt.xlabel("True value [b]")
        plt.ylabel("Recovered value [b]")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to saved posterior")
    parser.add_argument("--name", type=str, help="saved posterior name")
    args = parser.parse_args()

    # Create an instance of ModelLoader
    modelloader = ModelLoader()

    # Load the posterior
    posterior = modelloader.load_model_pkl(args.path, args.name)
