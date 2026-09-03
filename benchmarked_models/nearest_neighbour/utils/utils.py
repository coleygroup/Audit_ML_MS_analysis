import json
import pickle

import numpy as np

def load_pickle(path):

    with open(path, "rb") as f:

        data = pickle.load(f)
    
    return data

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
    
def write_json(data, path):

    with open(path, "w") as f:
        json.dump(data, f, indent = 4)
    
def pickle_data(data, path):

    with open(path, "wb") as f:
        pickle.dump(data, f)
def normalize_rows(matrix):

    """L2-normalise each row, leaving all-zero rows as zeros.

    Used to pre-normalise the training matrix once so the no-same-formula fallback in
    the nearest-neighbour scripts can score a query against the full training set with
    a single matrix-vector product, instead of re-normalising that matrix per query.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)

def write_fallback_report(path, test_ids, fallback_test_ids):

    """Record how many test entries had no same-formula training candidate.

    The nearest-neighbour scripts used to drop those entries, so this count is the size
    of the correction between the published numbers and the full-test-set numbers.
    """

    write_json({"n_test": len(test_ids),
                "n_same_formula": len(test_ids) - len(fallback_test_ids),
                "n_fallback": len(fallback_test_ids),
                "fallback_test_ids": list(fallback_test_ids)}, path)
