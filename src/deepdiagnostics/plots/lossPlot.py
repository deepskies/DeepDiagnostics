import matplotlib.pyplot as plt
from typing import Dict, List

from deepdiagnostics.plots.plot import Display

class LossPlot(Display): 
    def __init__(self, model=None, data=None): 
      ...

    def _data_setup(self): 
      raise NotImplementedError

    def plot_name(self):
       return "loss_curves.png"
   
    def plot(
       self,
       training_history: Dict[str, List[float]],
       epochs: int = None,
       best_val_loss: float = None
   ):

       try:
           training_loss_data = training_history["train_loss"]
           validation_loss_data = training_history["val_loss"]
       except Exception as e:
            raise KeyError(f"Key {e} not found in supplied training history")

       if epochs is None:
           if len(training_loss_data) != len(validation_loss_data):
               raise ValueError("Inconsistent training history data supplied [length of train and validation losses not equal]")
           epochs = len(training_loss_data)
           print(f"Number of epochs determined: {epochs}")
       else:
           if epochs != len(training_loss_data) or epochs != len(validation_loss_data):
               raise ValueError("Epochs supplied inconsistent with training history data [epochs and loss history not equal]")
 
       if best_val_loss is None:
           best_val_loss = min(validation_loss_data)
           print(f"Best validation loss found: {best_val_loss}")

       epochs_trained = [ x for x in range(1, epochs+1) ]
       plt.plot(epochs_trained, training_loss_data, label='Training Loss')
       plt.plot(epochs_trained, validation_loss_data, label='Validation Loss')
       plt.axhline(y=best_val_loss, color='m', linestyle='--', label="Best Val. Loss")

       plt.xlabel('Epochs')
       plt.ylabel('Loss')
       plt.title('Training and Validation Loss Over Epochs')

       plt.legend()
       plt.grid()
       plt.show()

    def __call__(self, **kwargs): 
       raise NotImplementedError("Plotting loss is not supported in pipeline mode")
