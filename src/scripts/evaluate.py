"""
Simple stub functions to use in evaluating inference from a previously trained inference model.

"""

import argparse
import pickle
from sbi.analysis import pairplot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# plotting style things:
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
    
    def sbc_mackelab(self, posterior, params):
        """
        Runs and displays mackelab's SBC (simulation-based calibration)

        Simulation-based calibration is a set of tools built into Mackelab's sbi interface. It provides a way to compare the inferred posterior distribution to the true parameter values. It performs multiple instances of drawing parameter values from the prior, running these through the simulator, and comparing these values to those obtained from the run of inference. Importantly, this will not diagnose what's going on for one draw from the posterior (ie at one data point). Instead, it's meant to give an overall sense of the health of the posterior learned from SBI.

        This technique is based on rank plots. Rank plots are produced from comparing each posterior parameter draw (from the prior) to the distribution of parameter values in the posterior. There should be a 1:1 ranking, aka these rank plots should be similar in shape to a uniform distribution.
        """
        num_sbc_runs = 1_000  # choose a number of sbc runs, should be ~100s or ideally 1000
        # generate ground truth parameters and corresponding simulated observations for SBC.
        thetas = prior.sample((num_sbc_runs,))
        ys = simulator(thetas)
        # run SBC: for each inference we draw 1000 posterior samples.
        num_posterior_samples = 1_000
        ranks, dap_samples = run_sbc(
            thetas, ys, posterior, num_posterior_samples=num_posterior_samples
        )
        '''
        The ks pvalues are vanishingly small here, so we can reject the null hypothesis (of the marginal rank distributions being equivalent to an uniform distribution). The inference clearly went wrong.

        In terms of the c2st_ranks diagnostic; this is a nonparametric two sample test from training on and testing on the rank versus uniform distributions and distinguishing between them. If values are close to 0.5, it is hard to distinguish.

        The data-averaged posterior value compares to the prior; if these values are close to 0.5, dap is like the prior distribution.  
        '''
        check_stats = check_sbc(
            ranks, thetas, dap_samples, num_posterior_samples=num_posterior_samples
        )
        print(check_stats)
        help(sbc_rank_plot)
        _ = sbc_rank_plot(
            ranks=ranks,
            num_posterior_samples=num_posterior_samples,
            plot_type="hist",
            num_bins=None,
            parameter_labels=['$m$', '$b$'],
            colors=[m_color, b_color]
        )
        '''
        If the rank plots are consistent with being uniform, the red bars should fall mostly within the grey area. The grey area is the 99% confidence interval for the uniform distribution, so if the rank histogram falls outside this for more than 1 in 100 bars, that means it is not consistent with an uniform distribution (which is what we want). 

        A central peaked rank plot could be indicative of a posterior that's too concentrated whereas one with wings is a posterior that is too dispersed. Conversely, if the distribution is shifted left or right that indicates that the parameters are biased.

        If it's choppy, it could be indicative of not doing enough sampling. A good rule of thumb is N / B ~ 20, where N is the number of samples and B is the number of bins.
        '''

        help(sbc_rank_plot)
        f, ax = sbc_rank_plot(ranks, 1_000, plot_type="cdf", parameter_labels=["m","b"], colors = [m_color, b_color])
        '''
        This is a different way to visualize the same thing. Essentially, the grey is the 95% confidence interval for an uniform distribution. The cdf for the posterior rank distributions (in color) should fall within this band.
        '''
        percentile_array = np.linspace(0,100,21)
        samples, frac_array = calculate_coverage_fraction(posterior,
                                                        ys,
                                                        np.array(thetas),
                                                        percentile_array,
                                                        samples_per_inference = 1000)
        from cycler import cycler

        percentile_array_norm = np.array(percentile_array)/100

        # Create a cycler with hexcode colors and linestyles
        color_cycler = cycler(color=[m_color, b_color])
        linestyle_cycler = cycler(linestyle=['-', '-.'])

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Use itertools.cycle to loop through colors and linestyles
        color_cycle = iter(color_cycler)
        linestyle_cycle = iter(linestyle_cycler)
        label_list = ['m', 'b']
        # Iterate over the second dimension of frac_array
        for i in range(frac_array.shape[1]):
            color_style = next(color_cycle)['color']
            linestyle_style = next(linestyle_cycle)['linestyle']
            ax.plot(percentile_array_norm, frac_array[:, i], alpha=1.0, lw=3, linestyle=linestyle_style, color=color_style, label=label_list[i])

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
        plt.show()

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
    

    

    def calculate_coverage_fraction(posterior, x_observed, truth_array, percentile_list, samples_per_inference = 1000):
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