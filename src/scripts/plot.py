from sbi.analysis import pairplot
from getdist import plots, MCSamples

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
        truth_list=None,
        truth_color='red',
        plot=False,
        save=True,
        path='plots/',
        
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
        axes[0, 1].plot([truth_list[1]], [truth_list[0]],
                        marker="o",
                        color=truth_color)
        axes[0, 0].axvline(x=truth_list[0], color=truth_color)
        axes[1, 1].axvline(x=truth_list[1], color=truth_color)

        if save:
            plt.savefig(path + "mackelab_pairplot.pdf")
        if plot:
            plt.show()

    def getdist_corner_plot(
        self,
        posterior_samples,
        labels_list=None,
        limit_list=None,
        truth_list=None,
        truth_color='red',
        plot=False,
        save=True,
        path='plots/',
        
    ):
        """
        Uses existing getdist
        to produce a flexible corner plot.

        :param posterior_samples: Samples drawn from the posterior,
        conditional on data
        :param labels_list: A list of the labels for the parameters
        :param limit_list: A list of limits for each parameter plot
        :return: Loaded model object that can be used with the predict function
        """
        
        '''
        # in getdist you have to add the names and labels
        samples = MCSamples(samples=samps, names = names, labels = labels)
        samples2 = MCSamples(samples=samps2, names = names, labels = labels, label='Second set')

        g = plots.get_subplot_plotter()

        if type(posterior_samples) == List:

            g.triangle_plot([samples, samples2], filled=True)
        else:
            g.triangle_plot([samples, samples2], filled=True)
        '''
        # Assume 'posterior_samples' is a 2D numpy array or similar
        samples = MCSamples(samples=posterior_samples, names=labels_list, labels=labels_list)

        # Create a getdist Plotter
        g = plots.get_subplot_plotter()

        # Plot the triangle plot
        g.triangle_plot(samples, filled=True)

        # Add customizations based on your requirements
        # For example, you may want to add truth markers
        if truth_list is not None:
            for i in range(len(truth_list)):
                g.add_x_marker(truth_list[i], color=truth_color)
                g.add_y_marker(truth_list[i], color=truth_color)

        # Save or show the plot
        if save:
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join(path, "getdist_cornerplot.pdf"))

        if plot:
            plt.show()
        

    def improved_corner_plot(self, posterior):
        """
        Improved corner plot
        """
