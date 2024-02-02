import sys
import pytest
import torch
import numpy as np
import sbi
import os

# flake8: noqa
#sys.path.append("..")
print(sys.path)
from scripts.evaluate import Diagnose_static, Diagnose_generative
from scripts.io import ModelLoader
#from src.scripts import evaluate


"""
"""


"""
Test the evaluate module
"""


@pytest.fixture
def diagnose_static_instance():
    return Diagnose_static()

@pytest.fixture
def diagnose_generative_instance():
    return Diagnose_generative()


@pytest.fixture
def inference_instance():
    modelloader = ModelLoader()
    path = "savedmodels/sbi/"
    model_name = "sbi_linear_from_data"
    posterior = modelloader.load_model_pkl(path, model_name)
    return posterior

@pytest.fixture
def posterior_generative_sbi_model():
    # create a temporary directory for the saved model
    #dir = "savedmodels/sbi/"
    #os.makedirs(dir)

    # now save the model
    low_bounds = torch.tensor([0, -10])
    high_bounds = torch.tensor([10, 10])

    prior = sbi.utils.BoxUniform(low = low_bounds, high = high_bounds)

    posterior = sbi.inference.base.infer(simulator, prior, "SNPE", num_simulations=10000)

    # Provide the posterior to the tests
    yield prior, posterior

    # Teardown: Remove the temporary directory and its contents
    #shutil.rmtree(dataset_dir)


def simulator(thetas):  # , percent_errors):
    # convert to numpy array (if tensor):
    thetas = np.atleast_2d(thetas)
    # Check if the input has the correct shape
    if thetas.shape[1] != 2:
        raise ValueError(
            "Input tensor must have shape (n, 2) \
            where n is the number of parameter sets."
        )

    # Unpack the parameters
    if thetas.shape[0] == 1:
        # If there's only one set of parameters, extract them directly
        m, b = thetas[0, 0], thetas[0, 1]
    else:
        # If there are multiple sets of parameters, extract them for each row
        m, b = thetas[:, 0], thetas[:, 1]
    x = np.linspace(0, 100, 101)
    rs = np.random.RandomState()  # 2147483648)#
    # I'm thinking sigma could actually be a function of x
    # if we want to get fancy down the road
    # Generate random noise (epsilon) based
    # on a normal distribution with mean 0 and standard deviation sigma
    sigma = 5
    ε = rs.normal(loc=0, scale=sigma, size=(len(x), thetas.shape[0]))

    # Initialize an empty array to store the results for each set of parameters
    y = np.zeros((len(x), thetas.shape[0]))
    for i in range(thetas.shape[0]):
        m, b = thetas[i, 0], thetas[i, 1]
        y[:, i] = m * x + b + ε[:, i]
    return torch.Tensor(y.T)


def test_generate_sbc_samples(diagnose_generative_instance,
                              posterior_generative_sbi_model):
    # Mock data
    #low_bounds = torch.tensor([0, -10])
    #high_bounds = torch.tensor([10, 10])

    #prior = sbi.utils.BoxUniform(low=low_bounds, high=high_bounds)
    prior, posterior = posterior_generative_sbi_model
    #inference_instance  # provide a mock posterior object
    simulator_test = simulator  # provide a mock simulator function
    num_sbc_runs = 1000
    num_posterior_samples = 1000

    # Generate SBC samples
    thetas, ys, ranks, dap_samples = diagnose_generative_instance.generate_sbc_samples(
        prior, posterior, simulator_test, num_sbc_runs, num_posterior_samples
    )

    # Add assertions based on the expected behavior of the method


def test_run_all_sbc(diagnose_generative_instance,
                     posterior_generative_sbi_model):
    labels_list = ["$m$", "$b$"]
    colorlist = ["#9C92A3", "#0F5257"]
    
    prior, posterior = posterior_generative_sbi_model
    simulator_test = simulator  # provide a mock simulator function

    save_path = "plots/"

    diagnose_generative_instance.run_all_sbc(
        prior,
        posterior,
        simulator_test,
        labels_list,
        colorlist,
        num_sbc_runs=1_000,
        num_posterior_samples=1_000,
        samples_per_inference=1_000,
        plot=False,
        save=True,
        path=save_path,
    )
    # Check if PDF files were saved
    assert os.path.exists(save_path), f"No 'plots' folder found at {save_path}"

    # List all files in the directory
    files_in_directory = os.listdir(save_path)

    # Check if at least one PDF file is present
    pdf_files = [file for file in files_in_directory if file.endswith(".pdf")]
    assert pdf_files, "No PDF files found in the 'plots' folder"

    # We expect the pdfs to exist in the directory
    expected_pdf_files = ["sbc_ranks.pdf", "sbc_ranks_cdf.pdf", "coverage.pdf"]
    for expected_file in expected_pdf_files:
        assert (
            expected_file in pdf_files
        ), f"Expected PDF file '{expected_file}' not found"


"""
def test_sbc_statistics(diagnose_instance):
    # Mock data
    ranks =  # provide mock ranks
    thetas =  # provide mock thetas
    dap_samples =  # provide mock dap_samples
    num_posterior_samples = 1000

    # Calculate SBC statistics
    check_stats = diagnose_instance.sbc_statistics(
        ranks, thetas, dap_samples, num_posterior_samples
    )

    # Add assertions based on the expected behavior of the method

def test_plot_1d_ranks(diagnose_instance):
    # Mock data
    ranks =  # provide mock ranks
    num_posterior_samples = 1000
    labels_list =  # provide mock labels_list
    colorlist =  # provide mock colorlist

    # Plot 1D ranks
    diagnose_instance.plot_1d_ranks(
        ranks, num_posterior_samples, labels_list,
        colorlist, plot=False, save=False
    )
"""
