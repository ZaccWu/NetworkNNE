import numpy as np
import json
from scipy import sparse
import os

def save_dict_safe(save_dict, root="data"):
    os.makedirs(root + "/arrays", exist_ok=True)
    os.makedirs(root + "/sparse", exist_ok=True)

    meta = {}

    for k, v in save_dict.items():
        if isinstance(v, np.ndarray):
            np.save(f"{root}/arrays/{k}.npy", v)

        elif sparse.isspmatrix_csr(v):
            sparse.save_npz(f"{root}/sparse/{k}.npz", v)

        elif isinstance(v, str):
            meta[k] = v

        else:
            raise TypeError(f"Unsupported type for key '{k}': {type(v)}")

    with open(f"{root}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)



def load_dict_safe(root="data"):
    result = {}

    for fn in os.listdir(root + "/arrays"):
        key = fn.replace(".npy", "")
        result[key] = np.load(f"{root}/arrays/{fn}")

    for fn in os.listdir(root + "/sparse"):
        key = fn.replace(".npz", "")
        result[key] = sparse.load_npz(f"{root}/sparse/{fn}")

    with open(f"{root}/meta.json", encoding="utf-8") as f:
        result.update(json.load(f))

    return result