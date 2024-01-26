"""
Simple stub functions to use in evaluating inference from a previously trained inference model.

"""

import argparse
import pickle


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