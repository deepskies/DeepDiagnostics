import matplotlib.pyplot as plt
import numpy as np

from deepdiagnostics.plots.plot import Display
from typing import Literal, TYPE_CHECKING

# if TYPE_CHECKING:
#     from matplotlib.figure import Figure as fig
#     from matplotlib.axes import Axes as ax

class LossPlot(Display): 
    def __init__(self, model=None, data=None, run_id=None, save=True, show=False, out_dir=None):
        super().__init__(model, data, run_id, save, show, out_dir)

    def _data_setup(self):
        raise NotImplementedError("Functionality not supported in pipeline mode")

    def plot_name(self):
        return "loss_curves.png"
   
    def plot(
        self,
        training_history: dict[Literal["train_loss", "val_loss"]: list[float]],
        epochs_trained: int = None,
        best_val_loss: float = None,
        plot_best_val_loss: bool = False,
        axis_x_label: str = "Epoch",
        axis_y_label: str = "Loss",
        train_loss_linestyle: str = None,
        train_loss_color: str = None, 
        val_loss_linestyle: str = None,
        val_loss_color: str = None
    ) -> tuple(['fig', 'ax']):
        try:
            training_losses = training_history["train_loss"]
            validation_losses = training_history["val_loss"]
        except Exception as e:
            raise KeyError(f"Key {e} not found in supplied training history")

        if epochs_trained is None:
            # epochs information not passed, so determine the epochs from the training history supplied
            if len(training_losses) != len(validation_losses):
                raise ValueError("Inconsistent training history data supplied (length of train and validation losses not equal)")
            epochs = len(training_losses)
        else:
            # epochs information was passed
            epochs = epochs_trained
            # safety check for epochs passed to match with the training history
            if epochs != len(training_losses) or epochs != len(validation_losses):
                raise ValueError("Epochs supplied inconsistent with training history data (epochs and loss history not equal)")
        print(f"Number of epochs: {epochs}")
 
        if best_val_loss is None:
            # best validation loss not supplied; determine from the supplied training history
            best_validation_loss = min(validation_losses)
        else:
            # best validation loss was passed
            best_validation_loss = best_val_loss
        if not plot_best_val_loss:
            print(f"Best validation loss: {best_validation_loss}")

        tloss_linestyle = train_loss_linestyle if train_loss_linestyle is not None else '-'
        vloss_linestyle = val_loss_linestyle if val_loss_linestyle is not None else '-'
        tloss_color = train_loss_color if train_loss_color is not None else 'tab:blue'
        vloss_color = val_loss_color if val_loss_color is not None else 'tab:orange'
        
        fig, ax = plt.subplots()
        x = list(range(1, epochs+1))
        ax.plot(x, training_losses, color=tloss_color, linestyle=tloss_linestyle, label='Training Loss')
        ax.plot(x, validation_losses, color=vloss_color, linestyle=vloss_linestyle, label='Validation Loss')
        if plot_best_val_loss:
            ax.axhline(y=best_validation_loss, color='m', linestyle='--', label="Best Val. Loss")
        ax.set_title('Training and Validation Loss Over Epochs')
        ax.set_xlabel(axis_x_label)
        ax.set_ylabel(axis_y_label)
        ax.legend()
        
        self._finish()

        return fig, ax
    
    def __call__(self, **kwargs):
        raise NotImplementedError("Plotting loss is not supported in pipeline mode")
