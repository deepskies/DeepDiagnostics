from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np

from deepdiagnostics.plots.plot import Display

class PriorPC(Display):
    def __init__(
        self, 
        model, 
        data, 
        save:bool, 
        show:bool, 
        out_dir:Optional[str]=None, 
        percentiles: Optional[Sequence] = None, 
        use_progress_bar: Optional[bool] = None,
        samples_per_inference: Optional[int] = None,
        number_simulations: Optional[int] = None,
        parameter_names: Optional[Sequence] = None, 
        parameter_colors: Optional[Sequence]= None, 
        colorway: Optional[str]=None
    ):
        super().__init__(model, data, save, show, out_dir, percentiles, use_progress_bar, samples_per_inference, number_simulations, parameter_names, parameter_colors, colorway)

    def _plot_name(self):
        return "predictive_prior_check.png"

    def get_prior_samples(self, n_columns, n_rows):

        context_shape = self.data.true_context().shape
        
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
                self.prior_predictive_samples[row_index, column_index] = self.data.simulator.simulate(
                    theta=prior_sample, context_samples = context_sample
                )
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


    def _plot(
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
                subplots[plot_row_index, plot_column_index].plot(
                    self.context[column_index, row_index],
                    self.prior_predictive_samples[column_index, row_index]
                )

            figure.supylabel(y_label)
            figure.supxlabel(x_label)
            figure.suptitle(title)