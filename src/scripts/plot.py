from sbi.analysis import pairplot


# plotting style things:
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

# remove top and right axis from plots
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False


class Display:
    def mackelab_corner_plot(
        self,
        posterior_samples,
        labels_list=None,
        limit_list=None,
        truth_list=None
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
            labels_list=labels_list,
            limits=limit_list,
            # [[0,10],[-10,10],[0,10]],
            truths=truth_list,
            figsize=(5, 5),
        )
        axes[0, 1].plot([truth_list[1]], [truth_list[0]],
                        marker="o",
                        color="red")

    def improved_corner_plot(self, posterior):
        """
        Improved corner plot
        """
