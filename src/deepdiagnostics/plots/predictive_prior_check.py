from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from deepdiagnostics.plots.plot import Display
from deepdiagnostics.utils.simulator_utils import SimulatorMissingError

class PriorPC(Display):
    """
    Plot random samples of the simulator's output from samples drawn from the prior

    .. code-block:: python 
    
        from deepdiagnostics.plots import PriorPC

        PriorPC(model, data, show=True, save=False)(
            n_rows = 2, 
            n_columns = 6, # Make 2x6 = 12 different samples 
            row_parameter_index = 0,
            column_parameter_index = 1, # Include labels for theta parameters 0 and 1 from the prior
            round_parameters = True,
        )
    """
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
        colorway = None):

        super().__init__(model, data, save, show, out_dir, percentiles, use_progress_bar, samples_per_inference, number_simulations, parameter_names, parameter_colors, colorway)
        if not hasattr(self.data, "simulator"): 
            raise SimulatorMissingError("Missing a simulator to run PriorPC.")
    
        if self.data.simulator_dimensions == 1: 
            self.plot_image = False

        elif self.data.simulator_dimensions == 2: 
            self.plot_image = True


    def plot_name(self):
        return "predictive_prior_check.png"

    def get_prior_samples(self, n_columns, n_rows):

        context_shape = self.data.true_context().shape
        remove_first_dim = False

        if self.plot_image: 
            sim_out_shape = self.data.get_simulator_output_shape()
            if len(sim_out_shape) != 2: 
                # TODO Debug log with a warning
                sim_out_shape = (sim_out_shape[1], sim_out_shape[2])
                remove_first_dim = True
            self.prior_predictive_samples = np.zeros((n_rows, n_columns, *sim_out_shape))

        else: 
            self.prior_predictive_samples = np.zeros((n_rows, n_columns, context_shape[-1]))
        self.prior_sample = np.zeros((n_rows, n_columns, self.data.n_dims))
        self.context = np.zeros((n_rows, n_columns, context_shape[-1]))
        random_context_indices = self.data.rng.integers(0, context_shape[0], (n_rows, n_columns))

        for row_index in range(n_rows): 
            for column_index in range(n_columns): 

                sample = random_context_indices[row_index, column_index]
                context_sample = self.data.true_context()[sample, :]

                prior_sample = self.data.sample_prior(1)[0]
                # get the posterior samples for that context 
                simulation_sample = self.data.simulator.simulate(
                    theta=prior_sample, context_samples = context_sample
                )
                if remove_first_dim: 
                    simulation_sample = simulation_sample[0]

                self.prior_predictive_samples[row_index, column_index] = simulation_sample
                self.prior_sample[row_index, column_index] = prior_sample
                self.context[row_index, column_index] = context_sample

    def fill_text(self, row_index, column_index, row_parameter_index, column_parameter_index, label_samples, round_parameters): 
        if label_samples in ['both', 'rows', 'columns']: 
            row_name = self.parameter_names[row_parameter_index]
            row_value = self.prior_sample[row_index, column_index, row_parameter_index]

            col_name = self.parameter_names[column_parameter_index]
            col_value = self.prior_sample[row_index, column_index, column_parameter_index]
            if round_parameters: 
                row_value = round(row_value, 4)
                col_value = round(col_value, 4)

            if label_samples == "both": 
                return f"{row_name}={row_value}, {col_name}={col_value}"
            elif label_samples == "rows": 
                return f"{row_name}={row_value}"
            else: 
                return f"{col_name}={col_value}"

        else: 
            raise ValueError(f"Cannot use {label_samples} to assign labels. Choose from 'both', 'rows', 'columns'.")


    def plot(
            self, 
            n_rows: Optional[int] = 3,
            n_columns: Optional[int] = 3,
            row_parameter_index: Optional[int] = 0, 
            column_parameter_index: Optional[int] = 1,
            round_parameters: Optional[bool] = True,
            sort_rows: bool = True, 
            sort_columns: bool = True,
            label_samples: Optional[str] = 'both', 
            title:Optional[str]="Simulated output from prior", 
            y_label:Optional[str]=None, 
            x_label:str=None): 
        """

        Args:
            n_rows (Optional[int], optional): Number of unique rows to make for priors. Defaults to 3.
            n_columns (Optional[int], optional): Number of unique columns for viewing prior predictions. Defaults to 3.
            row_parameter_index (Optional[int], optional): Index of the theta parameter to display as labels on rows. Defaults to 0.
            column_parameter_index (Optional[int], optional): Index of the theta parameter to display as labels on columns. Defaults to 1.
            round_parameters (Optional[bool], optional): In labels, round the theta parameters (recommended when thetas are float values). Defaults to True.
            sort_rows (bool, optional): Sort the plots by the theta row value. Defaults to True.
            sort_columns (bool, optional): Sort the plots by theta column value. Defaults to True.
            label_samples (Optional[str], optional): Label the prior values as a text box in each label. Row means using row_parameter_index as the title value. Choose from "rows", "columns", "both". Defaults to 'both'.
            title (Optional[str], optional): Title of the whole plot. Defaults to "Simulated output from prior".
            y_label (Optional[str], optional): Column label, when None, label = `theta_{column_index} = parameter_name`. Defaults to None.
            x_label (str, optional): Row label, when None, label = `theta_{row_index} = parameter_name`. Defaults to None.
        """

        self.get_prior_samples(n_rows, n_columns)
        figure, subplots = plt.subplots(
            n_columns, 
            n_rows, 
            figsize=(int(self.figure_size[0]*n_rows*.6), int(self.figure_size[1]*n_columns*.6)), 
            sharex=False, 
            sharey=True
        )

        if x_label is None: 
            x_label = f"$theta_{row_parameter_index}$ = {self.parameter_names[row_parameter_index]}"

        if y_label is None: 
            y_label = f"$theta_{column_parameter_index}$ = {self.parameter_names[column_parameter_index]}"

        column_order = np.argsort(
            self.prior_sample[:, :, column_parameter_index], axis=-1
        )
        row_order = np.argsort(
            self.prior_sample[:, :, row_parameter_index], axis=-1
        )

        for plot_row_index in range(n_rows): 
            for plot_column_index in range(n_columns): 

                row_index = plot_row_index if not sort_rows else row_order[plot_row_index, plot_column_index]
                column_index = plot_column_index if not sort_rows else column_order[plot_row_index, plot_column_index]

                text = self.fill_text(
                    row_index, 
                    column_index, 
                    row_parameter_index, 
                    column_parameter_index, 
                    label_samples=label_samples, 
                    round_parameters=round_parameters
                )
                
                subplots[plot_row_index, plot_column_index].title.set_text(text)
                if self.plot_image: 
                    subplots[plot_row_index, plot_column_index].imshow(self.prior_predictive_samples[column_index, row_index])
                    subplots[plot_row_index, plot_column_index].set_xticks([])
                    subplots[plot_row_index, plot_column_index].set_yticks([])

                else: 
                    subplots[plot_row_index, plot_column_index].plot(
                        self.context[column_index, row_index],
                        self.prior_predictive_samples[column_index, row_index]
                    )

            figure.supylabel(y_label)
            figure.supxlabel(x_label)
            figure.suptitle(title)