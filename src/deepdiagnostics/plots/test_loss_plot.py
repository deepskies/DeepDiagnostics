import numpy as np
from deepdiagnostics.plots.loss_plot import LossPlot

# Using numpy for a range of values
x_range = np.arange(1, 51)  # Numbers from 1 to 10
validation_loss = -np.log(x_range)
print(f"Negative natural logs of range {x_range}: {validation_loss}")

x_range = np.arange(1.1, 51.1)  # Numbers from 1 to 10
training_loss = -np.log(x_range)
print(f"Negative natural logs of range {x_range}: {training_loss}")

loss_history = {}
loss_history["train_loss"] = training_loss.tolist()
loss_history["val_loss"] = validation_loss.tolist()

loss_plot = LossPlot(save=False,show=True)
loss_plot.plot(loss_history, train_loss_color='tab:green', val_loss_color='tab:purple')