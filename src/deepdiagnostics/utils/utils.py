from h5py import File

class DataDisplay(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def from_h5(self, f: str, plot_name: str) -> "DataDisplay":
        """Load data from an HDF5 file path into the DataDisplay object."""
        try: 
            with File(f, "r") as f:
                for key in f[plot_name]: 
                    self[key] = f[plot_name][key][()]
        except KeyError as e:
            raise AttributeError(f"Plot name '{plot_name}' not found in file '{f}'.") from e

        return self
    
    def __getattr___(self, item):
        if item not in self:
            raise AttributeError(f"Missing '{item}' - please rerun the initial metric calculation.")
        return self[item]