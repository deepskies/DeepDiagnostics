from sbi.analysis import pairplot
from getdist import plots, MCSamples
import numpy as np

# plotting style things:
import matplotlib
import matplotlib.pyplot as plt
# from cycler import cycler

from typing import List, Union

# remove top and right axis from plots
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False


class Display:
    def mackelab_corner_plot(
        self,
        posterior_samples,
        labels_list=None,
        limit_list=None,
        truth_list=None,
        truth_color="red",
        plot=False,
        save=True,
        path="plots/",
    ):
        """
        Uses existing pairplot from mackelab analysis
        to produce a flexible corner plot.

        :param posterior_samples: Samples drawn from the posterior,
        conditional on data
        :param labels_list: A list of the labels for the parameters
        :param limit_list: A list of limits for each parameter plot
        :return: Loaded model object that can be used with the predict function
        """
        # plot posterior samples
        # if labels_list:
        # if limit_list:
        fig, axes = pairplot(
            posterior_samples,
            labels=labels_list,
            limits=limit_list,
            # [[0,10],[-10,10],[0,10]],
            truths=truth_list,
            figsize=(5, 5),
        )
        axes[0, 1].plot([truth_list[1]], [truth_list[0]], marker="o",
                        color=truth_color)
        axes[0, 0].axvline(x=truth_list[0], color=truth_color)
        axes[1, 1].axvline(x=truth_list[1], color=truth_color)

        if save:
            plt.savefig(path + "mackelab_pairplot.pdf")
        if plot:
            plt.show()

    def getdist_corner_plot(
        self,
        posterior_samples: Union[List[np.ndarray], np.ndarray],
        labels_list: List[str] = None,
        limit_list: List[
            List[float]
        ] = None,  # Each inner list contains [lower_limit, upper_limit]
        truth_list: List[float] = None,
        truth_color: str = "orange",
        plot: bool = False,
        save: bool = True,
        path: str = "plots/",
    ):
        """
        Uses existing getdist
        to produce a flexible corner plot.

        :param posterior_samples: Samples drawn from the posterior,
        conditional on data
        :param labels_list: A list of the labels for the parameters
        :param limit_list: A list of limits for each parameter plot
                        Each inner list contains [lower_limit, upper_limit]
        :return: Loaded model object that can be used with the predict function
        """

        # Check if 'posterior_samples' is a list
        if isinstance(posterior_samples, list):
            # Handle the case where 'posterior_samples' is a list of samples
            # You may want to customize this part based on your requirements
            samples_list = [
                MCSamples(
                    samples=samps,
                    names=labels_list,
                    labels=labels_list,
                    ranges=limit_list,
                )
                for samps in posterior_samples
            ]

            # Create a getdist Plotter
            g = plots.get_subplot_plotter()

            # Plot the triangle plot for each set of samples in the list
            g.triangle_plot(samples_list, filled=True)
        else:
            # Assume 'posterior_samples' is a 2D numpy array or similar
            samples = MCSamples(
                samples=posterior_samples,
                names=labels_list,
                labels=labels_list,
                ranges=limit_list,
            )

            # Create a getdist Plotter
            g = plots.get_subplot_plotter()

            # Plot the triangle plot
            g.triangle_plot(samples, filled=True)

        # Add vertical truth line on the first subplot
        if truth_list is not None:
            for i in range(len(truth_list)):
                for j in range(len(truth_list)):
                    if i == j:
                        # this is for the axvlines on the marginals
                        # which is on the diagnoal
                        g.subplots[i, j].axvline(x=truth_list[i],
                                                 color=truth_color)

                    try:
                        # plot as a point for the posteriors
                        g.subplots[int(1 + i), int(0 + j)].scatter(
                            truth_list[0 + i], truth_list[1 + i],
                            color=truth_color
                        )
                    except IndexError:
                        continue

        # Save or show the plot
        if save:
            plt.savefig(path + "getdist_cornerplot.pdf")

        if plot:
            plt.show()

    def improved_corner_plot(self, posterior):
        """
        Improved corner plot
        """
