"""
Simple stub functions to use in inference
"""

import argparse


def load_model(checkpoint_path):
    """
    Load the entire model for prediction with an input

    :param checkpoint_path: location
    :return: loaded model object that can be used with the predict function
    """
    pass


def predict(input, model):
    """

    :param input: loaded object used for inference
    :param model: loaded model
    :return: Prediction
    """
    return 0

def load_inference_object(input_path):
    """

    :param input_path: path to the object you want to predict
    :return: loaded object
    """
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Checkpoint to unloaded model checkpoint, either weights or the compressed model object")
    parser.add_argument("--input", type=str, help="path to object to predict quality of")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    pred_obj = load_inference_object(args.input)

    prediction = predict(pred_obj, model)
    print(prediction)
