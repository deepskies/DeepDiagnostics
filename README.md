[![PyPI - Version](https://img.shields.io/pypi/v/DeepDiagnostics?style=flat&logo=pypi&labelColor=grey&color=blue)](https://pypi.org/project/DeepDiagnostics/)
 ![status](https://img.shields.io/badge/License-MIT-lightgrey) 
 [![test](https://github.com/deepskies/DeepDiagnostics/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/deepskies/DeepDiagnostics/actions/workflows/test.yaml) 
[![Documentation Status](https://github.com/deepskies/DeepDiagnostics/actions/workflows/documentation.yml/badge.svg)](https://deepskies.github.io/DeepDiagnostics/)

# DeepDiagnostics
DeepDiagnostics is a package for diagnosing the posterior from an inference method. It is flexible, applicable for both simulation-based and likelihood-based inference.

## Documentation
### [Read our docs on github.io](https://deepskies.github.io/DeepDiagnostics/)

## Installation 
### From PyPi 

``` sh
pip install deepdiagnostics
```
### From Source

This project is built from poetry, if working from source we recommend using `poetry run` to run any commands that use the package. 
You can also use `poetry env activate` to get the path to the python virtual environment used by poetry. 
For additional information - [please view poetry's environment management documentation](https://python-poetry.org/docs/managing-environments).

``` sh
git clone https://github.com/deepskies/DeepDiagnostics/ 
pip install poetry 
poetry env activate 
poetry install
poetry run pytest
poetry run diagnose --config {config path}
```

## Quickstart

[View the template yaml here for a minimally working example with our supplied sample data to get started.](https://github.com/deepskies/DeepDiagnostics/blob/main/config.yml.template)

### Data and Model Requirements

To access your trained model, use the `SBIModel` class to load in a trained model in the form of a `.pkl` file.
[This format specifics are shown here](https://sbi-dev.github.io/sbi/latest/faq/question_05_pickling/)
If you wish to use a different model format, we encourage you to open a [new issue](https://github.com/deepskies/DeepDiagnostics/issues) requesting it, or even better, write an subclass of `deepdiagnostics.models.Model` to include it!

To read in your own data, supply an `.h5` or `.pkl` file and specify your format in the `data`.`data_engine` field of the configuration file. [The possible fields are listed here.](https://deepskies.github.io/DeepDiagnostics/data.html) We recommend an `.h5` file. 

The data must have the following fields: 
* `xs` - The range of data your parameters have been tested against. For example, if you are modeling `y = mx + b`, your `xs` are the values you have tested for `x`. Please ensure they are of the shape (x_size, n_samples). 
* `thetas` - The parameters that characterize your problem. For example, if you are modeling `y = mx + b`, your `thetas` are `m` and `b`. Please ensure they are in the shape of (n_parameters, n_samples) and ordered the same way `parameter_labels` is supplied in your configuration file to prevent mislabelled plots. 

If you do not supply a simulator method, including a `ys` field can allow for the use of a `lookup-table` simulator substitute. 


### Pipeline 
`DeepDiagnostics` includes a CLI tool for analysis. 
* To run the tool using a configuration file: 

``` sh
    diagnose --config {path to yaml}
```

* To use defaults with specific models and data: 

``` sh
    diagnose --model_path {model pkl} --data_path {data pkl} [--simulator {sim name}]
```


Additional arguments can be found using ``diagnose -h``


### Standalone 

`DeepDiagnostics` comes with the option to run different plots and metrics independently. 

Setting a configuration ahead of time ensures reproducibility with parameters and seeds. 
It is encouraged, but not required. 


``` py
from deepdiagnostics.utils.configuration import Config 
from deepdiagnostics.model import SBIModel 
from deepdiagnostics.data import H5Data

from deepdiagnostics.plots import LocalTwoSampleTest, Ranks

Config({configuration_path})
model = SBIModel({model_path})
data = H5Data({data_path}, simulator={simulator name})

LocalTwoSampleTest(data=data, model=model, show=True)(use_intensity_plot=False, n_alpha_samples=200)
Ranks(data=data, model=model, show=True)(num_bins=3)
```

## Contributing 

[Please view the Deep Skies Lab contributing guidelines before opening a pull request.](https://github.com/deepskies/.github/blob/main/CONTRIBUTING.md)

`DeepDiagnostics` is structured so that any new metric or plot can be added by adding a class that is a child of `metrics.Metric` or `plots.Display`. 

These child classes need a few methods. A minimal example of both a metric and a display is below. 

It is strongly encouraged to provide typing for all inputs of the `plot` and `calculate` methods so they can be automatically documented. 

Please ensure the proxy format `DataDisplay` is used for all plots, which ensures results can be re-plotted. 

### Metric
``` py
from deepdiagnostics.metrics import Metric 

class NewMetric(Metric): 
    """
    {What the metric is, any resources or credits.}

    .. code-block:: python 

        {a basic example on how to run the metric}
    """
    def __init__(self, model, data,out_dir= None, save = True, use_progress_bar = None, samples_per_inference = None, percentiles = None, number_simulations = None,
    ) -> None:
        
        # Initialize the parent Metric
        super().__init__(model, data, out_dir, save, use_progress_bar, samples_per_inference, percentiles, number_simulations)

        # Any other calculations that need to be done ahead of time 

    def _collect_data_params(self): 
        # Compute anything that needs to be done each time the metric is calculated. 
        return None

    def calculate(self, metric_kwargs:dict[str, int]) -> Sequence[int]: 
        """
        Description of the calculations

        Kwargs: 
            metric_kwargs (Required, dict[str, int]): dictionary of the metrics to return, under the name "metric". 

        Returns:
            Sequence[int]: list of the number in metrics_kwargs
        """
        # Where the main calculation takes place, used by the metric __call__. 
        self.output = {'The Result of the calculation'=[metric_kwargs["metric"]]} # Update 'self.output' so the results are saved to the results.json. 

        return [0] # Return the result so the metric can be used standalone. 
```

### Display
``` py
import matplotlib.pyplot as plt 

from deepdiagnostics.plots.plot import Display


class NewPlot(Display):
    def __init__(
        self, 
        model, 
        data, 
        save, 
        show, 
        out_dir=None, 
        percentiles = None, 
        use_progress_bar= None,
        samples_per_inference = None,
        number_simulations= None,
        parameter_names = None, 
        parameter_colors = None, 
        colorway =None):

        """
        {Description of the plot}
        .. code-block:: python
        
            {How to run the plot}
        """

        super().__init__(model, data, save, show, out_dir, percentiles, use_progress_bar, samples_per_inference, number_simulations, parameter_names, parameter_colors, colorway)

    def plot_name(self):
        # The name of the plot (the filename, to be saved in out_dir/{file_name})
        # When you run the plot for the first time, it will yell at you if you haven't made this a png path. 
        return "new_plot.png"

    def _data_setup(self):
        # When data needs to be run for the plot to work, model inference etc. 
        pass

    def plot_settings(self):
        # If there additional settings to pull from the config
        pass

    def plot(self, plot_kwarg:float):
        """
        Args:
            plot_kwarg (float, required): Some kwarg
        """
        plt.plot([0,1], [plot_kwarg, plot_kwarg])
```

#### Adding to the package 
If you wish to add the addition to the package to run using the CLI package, a few things need to be done. 
For this example, we will add a new metric, but an identicial workflow takes place for plots, just modifying the `plots` submodule instead of `metrics`. 
 

1. Add the name and mapping to the submodule `__init__.py`. 

##### `src/deepdiagnostics/metrics/__init__.py`

``` py
...
from deepdiagnostics.metrics.{your metric file} import NewMetric

Metrics = {
    ...
    "NewMetric": NewMetric
}

```


2. Add the name and defaults to the `Defaults.py`

##### `src/deepdiagnostics/utils/Defaults.py`

``` py
Defaults = {
    "common": {...}, 
    ..., 
    "metrics": {
        ...
        "NewMetric": {"default_kwarg": "default overwriting the metric_default in the function definition."}
    }
}
```

3. Add a test to the repository, ensure it passes. 

##### `tests/test_metrics.py`

``` py
from deepdiagnostics.metrics import NewMetric 

...

def test_newmetric(metric_config, mock_model, mock_data): 
    Config(metric_config)
    new_metric = NewMetric(mock_model, mock_data, save=True)
    expected_results = {what you should get out}
    real_results = new_metric.calculate("kwargs that produce the expected results")
    assert expected_results.all() == real_results.all()

    new_metric()
    assert new_metric.output is not None
    assert os.path.exists(f"{new_metric.out_dir}/diagnostic_metrics.json")
```

``` console
python3 -m pytest tests/test_metrics.py::test_newmetric

```

4. Add documentation
   
##### `docs/source/metrics.rst`

``` rst
from deepdiagnostics.metrics import NewMetric 

.. _metrics:

Metrics
=========

.. autoclass:: deepdiagnostics.metrics.metric.Metric
    :members:
...

.. autoclass:: deepdiagnostics.metrics.newmetric.NewMetric
     :members: calculate

.. bibliography:: 
```

### Building documentation: 
* Documentation automatically updates after any push to the `main` branch according to [`readthedocs.yml`](https://github.com/deepskies/DeepDiagnostics/blob/main/.readthedocs.yml). Verify the documentation built by checking the readthedocs badge. 

### Publishing a release: 
* Releases to pypi are built automatically off the main branch whenever a github release is made.
* Update the version number to match with the release you are going to make before publishing in the `pyproject.toml`
* Create a new github release and monitor the [`publish.yml` action](https://github.com/deepskies/DeepDiagnostics/actions/workflows/publish.yml) to verify the new release is built properly. 

## Citation 
```
@article{key , 
    author = {Me :D}, 
    title = {title}, 
    journal = {journal}, 
    volume = {v}, 
    year = {20XX}, 
    number = {X}, 
    pages = {XX--XX}
}

```

## Acknowledgement 
This software has been authored by an employee or employees of Fermi Research Alliance, LLC (FRA), operator of the Fermi National Accelerator Laboratory (Fermilab) under Contract No. DE-AC02-07CH11359 with the U.S. Department of Energy.
