# Standard imports
import random

# 3rd party
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(X, batch_size, seed):
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=seed)

    train_dataset = TensorDataset(torch.Tensor(X_train.values))
    val_dataset = TensorDataset(torch.Tensor(X_val.values))

    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)

    dataloader = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        ),
    }
    return dataloader


def load_breast() -> pd.DataFrame:

    path = "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip"

    names = ["id", 
             "diagnosis",
            "radius" ,
            "texture" ,
            "perimeter",
            "area",
            "smoothness",
            "compactness",
            "concavity",
            "concave_points" ,
            "symmetry",
            "fractal_dimension", 
    ]

    train_df = pd.read_csv(path, names=names, index_col=False)
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)
    df["diagnosis"].replace({'M': 1, 'B': 0}, inplace=True)

    return df
