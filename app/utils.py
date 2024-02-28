import json
import os
import pickle


def save_dict(d, path, sortkeys=False):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w") as fp:
        json.dump(d, indent=2, fp=fp, sort_keys=sortkeys)
        fp.write("\n")


def save_model(model, path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(model, open(path, "wb"))


def load_model(path):
    model = pickle.load(open(path, "rb"))
    return model
