"""
Simple stubs to use for re-train of the final model
Can leave a default data source, or specify that 'load data' loads the dataset used in the final version
"""
import argparse


def architecture():
    """
    :return: compiled architecture of the model you want to have trained
    """
    return 0

def load_data(data_source):
    """
    :return: data loader or full training data, split in val and train
    """
    return 0, 0

def train_model(data_source, n_epochs):
    """
    :param data_source:
    :param n_epochs:
    :return: trained model, or simply None, but saved trained model
    """
    data = load_data(data_source)
    model = architecture()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", type=str, help="Data used to train the model")
    parser.add_argument("--n_epochs", type=int, help='Integer number of epochs to train the model')

    args = parser.parse_args()

    train_model(data_source=args.data_source, n_epochs=args.n_epochs)
