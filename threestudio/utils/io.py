import pickle

def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)