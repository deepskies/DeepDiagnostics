"""
Simple stub functions to use in evaluating inference from a previously trained inference model.

"""

import argparse
import pickle
from sbi.analysis import pairplot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
        """
    



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