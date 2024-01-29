"""
Simple stub functions to use in evaluating inference from a previously trained inference model.

"""

import argparse
import pickle
from sbi.analysis import run_sbc, sbc_rank_plot, check_sbc, pairplot
import numpy as np
from tqdm import tqdm


# plotting style things:
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
# remove top and right axis from plots
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False


class InferenceModel:
    def save_model_pkl(self, path, model_name, posterior):
        """
        Save the pkl'ed saved posterior model

        :param path: Location to save the model
        :param model_name: Name of the model
        :param posterior: Model object to be saved
        """
        file_name = path + model_name + ".pkl"
        with open(file_name, "wb") as file:
            pickle.dump(posterior, file)

    def load_model_pkl(self, path, model_name):
        """
        Load the pkl'ed saved posterior model

        :param path: Location to load the model from
        :param model_name: Name of the model
        :return: Loaded model object that can be used with the predict function
        """
        print(path)
        with open(path + model_name + ".pkl", 'rb') as file:
            posterior = pickle.load(file)
        return posterior

    def infer_sbi(self, posterior, n_samples, y_true):
        return posterior.sample((n_samples,), x=y_true)

    def predict(input, model):
        """

        :param input: loaded object used for inference
        :param model: loaded model
        :return: Prediction
        """
        return 0
    
    def simulator()


class Display:
    def mackelab_corner_plot(self,
                        posterior_samples,
                        labels_list=None,
                        limit_list=None,
                        truth_list=None):
        """
        Uses existing pairplot from mackelab analysis to produce a flexible
        corner plot.

        :param posterior_samples: Samples drawn from the posterior, conditional on data
        :param labels_list: A list of the labels for the parameters
        :param limit_list: A list of limits for each parameter plot
        :return: Loaded model object that can be used with the predict function
        """
        # plot posterior samples
        #if labels_list:
        #if limit_list:
        fig, axes = pairplot(
            posterior_samples,
            labels_list=labels_list,
            limits=limit_list,
            #[[0,10],[-10,10],[0,10]],
            truths=truth_list,
            figsize=(5, 5)
        )
        axes[0, 1].plot([truth_list[1]], [truth_list[0]], marker="o", color="red")
    def improved_corner_plot(self, posterior, params):
        """
        Improved corner plot
        """


class Diagnose:
    def posterior_predictive(self, theta_true, x_true, simulator, posterior_samples):
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
        plt.fill_between(xs_sim,
                        np.mean(ys_sim, axis = 0) - 1 * np.std(ys_sim, axis = 0),
                        np.mean(ys_sim, axis = 0) + 1 * np.std(ys_sim, axis = 0),
                        color = '#FF495C', label = 'posterior predictive check with noise')
        plt.plot(xs_sim, np.mean(ys_sim, axis = 0) + true_sigma,
                color = '#25283D', label = 'true input error')
        plt.plot(xs_sim, np.mean(ys_sim, axis = 0) - true_sigma,
                color = '#25283D')
        plt.scatter(xs_sim,
                    np.array(y_true), 
                    color = 'black')#'#EFD9CE')  

        plt.legend()
        plt.show()
        return ys_sim
    
    def generate_sbc_samples(self,
                             prior,
                             posterior,
                             simulator,
                             num_sbc_runs=1_000,
                             num_posterior_samples=1_000):
        # generate ground truth parameters and corresponding simulated observations for SBC.
        thetas = prior.sample((num_sbc_runs,))
        ys = simulator(thetas)
        # run SBC: for each inference we draw 1000 posterior samples.
        ranks, dap_samples = run_sbc(
            thetas, ys, posterior, num_posterior_samples=num_posterior_samples
        )
        return thetas, ys, ranks, dap_samples
    
    def sbc_statistics(ranks, thetas, dap_samples, num_posterior_samples):
        '''
        The ks pvalues are vanishingly small here, so we can reject the null hypothesis (of the marginal rank distributions being equivalent to an uniform distribution). The inference clearly went wrong.

        In terms of the c2st_ranks diagnostic; this is a nonparametric two sample test from training on and testing on the rank versus uniform distributions and distinguishing between them. If values are close to 0.5, it is hard to distinguish.

        The data-averaged posterior value compares to the prior; if these values are close to 0.5, dap is like the prior distribution.  
        '''
        check_stats = check_sbc(
            ranks, thetas, dap_samples, num_posterior_samples=num_posterior_samples
        )
        return check_stats
    def plot_1d_ranks(ranks,
                      num_posterior_samples,
                      labels_list,
                      colorlist,
                      plot=False,
                      save=True,
                      path='plots/'):
        """
        If the rank plots are consistent with being uniform, the red bars should fall mostly within the grey area. The grey area is the 99% confidence interval for the uniform distribution, so if the rank histogram falls outside this for more than 1 in 100 bars, that means it is not consistent with an uniform distribution (which is what we want). 

        A central peaked rank plot could be indicative of a posterior that's too concentrated whereas one with wings is a posterior that is too dispersed. Conversely, if the distribution is shifted left or right that indicates that the parameters are biased.

        If it's choppy, it could be indicative of not doing enough sampling. A good rule of thumb is N / B ~ 20, where N is the number of samples and B is the number of bins.
        """
         #help(sbc_rank_plot)
        if colorlist:
            _ = sbc_rank_plot(
                ranks=ranks,
                num_posterior_samples=num_posterior_samples,
                plot_type="hist",
                num_bins=None,
                parameter_labels=labels_list,
                colors=colorlist
            )
        else:
            _ = sbc_rank_plot(
                ranks=ranks,
                num_posterior_samples=num_posterior_samples,
                plot_type="hist",
                num_bins=None,
                parameter_labels=labels_list,
            )
        if plot:
            plt.show()
        if save:
            plt.savefig(path+'sbc_ranks.pdf')

    def plot_cdf_1d_ranks(ranks,
                          num_posterior_samples,
                          labels_list,
                          colorlist,
                          plot=False,
                          save=True,
                          path='plots/'):
        """
        This is a different way to visualize the same thing from the 1d rank plots.
        Essentially, the grey is the 95% confidence interval for an uniform distribution.
        The cdf for the posterior rank distributions (in color) should fall within this band.
        """
        help(sbc_rank_plot)
        if colorlist:
            f, ax = sbc_rank_plot(ranks,
                              num_posterior_samples,
                              plot_type="cdf",
                              parameter_labels=labels_list,
                              colors = colorlist)
        else:
            f, ax = sbc_rank_plot(ranks,
                              num_posterior_samples,
                              plot_type="cdf",
                              parameter_labels=labels_list)
        if plot:
            plt.show()
        if save:
            plt.savefig(path+'sbc_ranks_cdf.pdf')

    def calculate_coverage_fraction(posterior,
                                    truth_array,
                                    x_observed,
                                    percentile_list,
                                    samples_per_inference = 1000):
        """
        posterior --> the trained posterior
        x_observed --> the data used for inference
        truth_array --> true parameter values
        """
        # this holds all posterior samples for each inference run
        all_samples = np.empty((len(x_observed), samples_per_inference, np.shape(truth_array)[1]))
        count_array = []
        # make this for loop into a progress bar:
        for i in tqdm(range(len(x_observed)), desc='Processing observations', unit='obs'):
            # sample from the trained posterior n_sample times for each observation
            samples = posterior.sample(sample_shape=(samples_per_inference,), x=x_observed[i]).cpu()

            '''
            # plot posterior samples
            fig, axes = pairplot(
                samples, 
                labels = ['m', 'b'],
                #limits = [[0,10],[-10,10],[0,10]],
                truths = truth_array[i],
                figsize=(5, 5)
            )
            axes[0, 1].plot([truth_array[i][1]], [truth_array[i][0]], marker="o", color="r")
            '''
            
            all_samples[i] = samples
            count_vector = []
            # step through the percentile list
            for ind, cov in enumerate(percentile_list):
                percentile_l = 50.0 - cov/2
                percentile_u = 50.0 + cov/2
                # find the percentile for the posterior for this observation
                # this is n_params dimensional
                # the units are in parameter space
                confidence_l = np.percentile(samples.cpu(), percentile_l, axis=0)
                confidence_u = np.percentile(samples.cpu(), percentile_u, axis=0)
                # this is asking if the true parameter value is contained between the
                # upper and lower confidence intervals
                # checks separately for each side of the 50th percentile
                count = np.logical_and(confidence_u - truth_array.T[:,i] > 0, truth_array.T[:,i] - confidence_l > 0)
                count_vector.append(count)
            # each time the above is > 0, adds a count
            count_array.append(count_vector)
        count_sum_array = np.sum(count_array, axis=0)
        frac_lens_within_vol = np.array(count_sum_array)
        return all_samples, np.array(frac_lens_within_vol)/len(x_observed)


    
    def plot_coverage_fraction(posterior,
                               thetas,
                               ys,
                               samples_per_inference,
                               labels_list,
                               colorlist,
                               n_percentile_steps=21,
                               plot=False,
                               save=True,
                               path='plots/'):
        percentile_array = np.linspace(0,100,n_percentile_steps)
        samples, frac_array = self.calculate_coverage_fraction(posterior,
                                                        np.array(thetas),
                                                        ys,
                                                        percentile_array,
                                                        samples_per_inference=samples_per_inference)
        

        percentile_array_norm = np.array(percentile_array)/100

        # Create a cycler with hexcode colors and linestyles
        if colorlist:
            color_cycler = cycler(color=colorlist)
        else:
            color_cycler = cycler(color='viridis')
        linestyle_cycler = cycler(linestyle=['-', '-.'])

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Use itertools.cycle to loop through colors and linestyles
        color_cycle = iter(color_cycler)
        linestyle_cycle = iter(linestyle_cycler)
        # Iterate over the second dimension of frac_array
        for i in range(frac_array.shape[1]):
            color_style = next(color_cycle)['color']
            linestyle_style = next(linestyle_cycle)['linestyle']
            ax.plot(percentile_array_norm,
                    frac_array[:, i],
                    alpha=1.0,
                    lw=3,
                    linestyle=linestyle_style,
                    color=color_style,
                    label=labels_list[i])

        ax.plot([0, 0.5, 1], [0, 0.5, 1], 'k--', lw=3, zorder=1000, label='Reference Line')
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.text(0.03, 0.93, 'Underconfident', horizontalalignment='left')
        ax.text(0.3, 0.05, 'Overconfident', horizontalalignment='left')
        ax.legend(loc='lower right')
        ax.set_xlabel('Confidence Interval of the Posterior Volume')
        ax.set_ylabel('Fraction of Lenses within Posterior Volume')
        ax.set_title('NPE')
        plt.tight_layout()
        if plot:
            plt.show()
        if save:
            plt.savefig(path+'coverage.pdf')
        
    
    def run_all_sbc(self,
                    prior,
                    posterior,
                    simulator,
                    labels_list,
                    colorlist,
                    num_sbc_runs=1_000,
                    num_posterior_samples=1_000,
                    params):
        """
        Runs and displays mackelab's SBC (simulation-based calibration)

        Simulation-based calibration is a set of tools built into Mackelab's sbi interface. It provides a way to compare the inferred posterior distribution to the true parameter values. It performs multiple instances of drawing parameter values from the prior, running these through the simulator, and comparing these values to those obtained from the run of inference. Importantly, this will not diagnose what's going on for one draw from the posterior (ie at one data point). Instead, it's meant to give an overall sense of the health of the posterior learned from SBI.

        This technique is based on rank plots. Rank plots are produced from comparing each posterior parameter draw (from the prior) to the distribution of parameter values in the posterior. There should be a 1:1 ranking, aka these rank plots should be similar in shape to a uniform distribution.
        """
        thetas, ys, ranks, dap_samples = self.generate_sbc_samples(prior,
                                                  posterior,
                                                  simulator,
                                                  num_sbc_runs,
                                                  num_posterior_samples)

        stats = self.sbc_statistics(ranks, thetas, dap_samples, num_posterior_samples)
        print(stats)
        self.plot_1d_ranks(ranks,
                      num_posterior_samples,
                      labels_list,
                      colorlist,
                      plot=False,
                      save=True,
                      path='../../plots/')
        
        self.plot_cdf_1d_ranks(ranks,
                      num_posterior_samples,
                      labels_list,
                      colorlist,
                      plot=False,
                      save=True,
                      path='../../plots/')
        
        self.plot_coverage_fraction(posterior,
                                    thetas,
                                    ys,
                                    samples_per_inference,
                                    labels_list,
                                    colorlist,
                                    n_percentile_steps=21,
                                    plot=False,
                                    save=True,
                                    path='plots/')

       

        
        

        '''
        We've already saved samples, let's compare the inferred (and associated error bar) parameters from each of the data points we used for the SBC analysis.
        '''

        print(np.shape(samples), np.shape(thetas))
        percentile_16_m = []
        percentile_50_m = []
        percentile_84_m = []
        percentile_16_b = []
        percentile_50_b = []
        percentile_84_b = []
        for i in range(len(samples[0])):
            #print(np.shape(samples[i]))
            #STOP
            percentile_16_m.append(np.percentile(samples[i,0], 16))
            percentile_50_m.append(np.percentile(samples[i,0], 50))
            percentile_84_m.append(np.percentile(samples[i,0], 84))
            percentile_16_b.append(np.percentile(samples[i,1], 16))
            percentile_50_b.append(np.percentile(samples[i,1], 50))
            percentile_84_b.append(np.percentile(samples[i,1], 84))
        yerr_minus = [mid - low for (mid, low) in zip(percentile_50_m, percentile_16_m)]
        yerr_plus = [high - mid for high, mid in zip(percentile_84_m, percentile_50_m)]

        # Randomly set half of the error bars to zero
        random_indices = np.random.choice(len(yerr_minus), int(len(yerr_minus) // 1.15), replace=False)
        for idx in random_indices:
            yerr_minus[idx] = 0
            yerr_plus[idx] = 0

        plt.errorbar(np.array(thetas[:,0]),
                    percentile_50_m,
                    yerr = [yerr_minus, yerr_plus],
                    linestyle = 'None',
                    color = m_color,
                    capsize = 5)
        plt.scatter(np.array(thetas[:,0]), percentile_50_m, color = m_color)
        plt.plot(percentile_50_m, percentile_50_m, color = 'k')
        plt.xlabel('True value [m]')
        plt.ylabel('Recovered value [m]')
        plt.show()

        plt.clf()
        plt.scatter(np.array(thetas[:,1]), percentile_50_b, color = b_color)
        yerr_minus = [mid - low for (mid, low) in zip(percentile_50_b, percentile_16_b)]
        yerr_plus = [high - mid for high, mid in zip(percentile_84_b, percentile_50_b)]

        # Randomly set half of the error bars to zero
        random_indices = np.random.choice(len(yerr_minus), int(len(yerr_minus) // 1.15), replace=False)
        for idx in random_indices:
            yerr_minus[idx] = 0
            yerr_plus[idx] = 0

        plt.errorbar(np.array(thetas[:,1]),
                    percentile_50_b,
                    yerr = [yerr_minus, yerr_plus],
                    linestyle = 'None',
                    color = b_color,
                    capsize = 5)
        plt.plot(percentile_50_b, percentile_50_b, color = 'black')
        plt.xlabel('True value [b]')
        plt.ylabel('Recovered value [b]')
        plt.show()
    

    

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to saved posterior")
    parser.add_argument("--name", type=str, help="saved posterior name")
    args = parser.parse_args()

    # Create an instance of InferenceModel
    inference_model = InferenceModel()

    # Load the model
    model = inference_model.load_model_pkl(args.path, args.name)

    inference_obj = inference_model.predict(model)