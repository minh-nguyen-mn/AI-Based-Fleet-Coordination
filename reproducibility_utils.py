# reproducibility_utils.py

import os
import json
import random

import numpy as np
import pandas as pd


# =========================================================
# GLOBAL RANDOM SEED
# =========================================================

def set_global_seed(seed):

    random.seed(seed)

    np.random.seed(seed)


# =========================================================
# DIRECTORY HELPERS
# =========================================================

def create_directory(path):

    os.makedirs(
        path,
        exist_ok=True
    )


def create_run_directory(
    base_dir,
    run_id
):

    run_dir = os.path.join(
        base_dir,
        f"run_{run_id:04d}"
    )

    create_directory(run_dir)

    return run_dir


# =========================================================
# SAVE HELPERS
# =========================================================

def save_dataframe(
    dataframe,
    filepath
):

    directory = os.path.dirname(
        filepath
    )

    if directory != "":
        create_directory(directory)

    dataframe.to_csv(
        filepath,
        index=False
    )


def save_json(
    data,
    filepath
):

    directory = os.path.dirname(
        filepath
    )

    if directory != "":
        create_directory(directory)

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )