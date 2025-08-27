class Model:
    """
    Load a pre-trained model for analysis.

    Args:
        model_path (str): relative path to a model.
    """

    def __init__(self, model_path: str) -> None:
        self.model = self._load(model_path)

    def _load(self, path: str) -> None:
        raise NotImplementedError

    def save(self, path: str, allow_overwrite: bool = False) -> None:
        raise NotImplementedError

    def sample_posterior(self):
        raise NotImplementedError

    def sample_simulation(self, data):
        raise NotImplementedError

    def predict_posterior(self, data):
        raise NotImplementedError
