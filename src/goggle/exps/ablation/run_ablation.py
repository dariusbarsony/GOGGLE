# Standard Imports
import random

# 3rd Party
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from pgmpy.estimators import PC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

import argparse
import time

# Synthcity
from synthcity.plugins.core.dataloader import GenericDataLoader

# Goggle
from goggle.GoggleModel import GoggleModel
from goggle.data_utils import *


def get_adj_mat(adj_type, X, n):
    if adj_type == "ER":
        m = int(n**2 * 0.1)
        G = nx.gnm_random_graph(n, m)
        adj_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if (i, j) in list(G.edges):
                    adj_mat[i][j] = 1
    elif adj_type == "COV":
        adj_mat = abs(np.corrcoef(X.to_numpy().T))
    elif adj_type == "BN":
        c = PC(X)
        model = c.estimate(variant="parallel", max_cond_vars=3, ci_test="pearsonr")
        adj_mat = np.zeros((n, n))
        for i, col_name_i in enumerate(X.columns):
            for j, col_name_j in enumerate(X.columns):
                if (col_name_i, col_name_j) in model.edges():
                    adj_mat[i][j] = 1
    else:
        adj_mat = np.ones((n, n))

    plt.figure(figsize=(4, 4))
    im = plt.imshow(adj_mat, cmap="inferno", interpolation="nearest", vmin=0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Prior graph")
    plt.axis("off")
    plt.savefig(f"result_adult_{adj_type}.png")
    
    return adj_mat

def run_ablation(X, learning_rate, weight_decay, batch_size, alpha, dataset_name='credit'):
    X_train, X_test = train_test_split(X, random_state=0, test_size=0.2, shuffle=True)

    for adj_type in ["ER", "COV", "BN", "DENSE"]:
        print(f"\n\nConsidering ablation setting: {adj_type}")
        adj_mat = get_adj_mat(adj_type, X, n=X_train.shape[1])
        gen = GoggleModel(
            ds_name=dataset_name,
            input_dim=X_train.shape[1],
            encoder_dim=64,
            encoder_l=2,
            het_encoding=True,
            decoder_dim=64,
            decoder_l=2,
            threshold=0.1,
            het_decoder=False,
            graph_prior=torch.Tensor(adj_mat),
            prior_mask=torch.ones_like(torch.Tensor(adj_mat)),
            device="cpu",
            beta=0.1,
            seed=0,
            learning_rate=learning_rate,
            batch_size=batch_size,
            alpha=alpha,
            weight_decay=weight_decay
        )

        gen.fit(X_train)
        X_synth = gen.sample(X_test)

        X_synth_loader = GenericDataLoader(
            X_synth,
            target_column="target",
        )
        X_test_loader = GenericDataLoader(
            X_test,
            target_column="target",
        )

        res = gen.evaluate_synthetic(X_synth_loader, X_test_loader)
        print(f"Quality: {res[0]:.3f}")
        print(f"Detection: {res[2]:.3f}")
        print(
            f"Performance on real: {res[1][0]:.3f}, on synth: {res[1][1]:.3f}, diff: {(res[1][0] - res[1][1]):.3f}"
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="adult")

    # parser.add_argument("--datapath", type=str, default="")
    # parser.add_argument("--runs", type=int, default=10)

    parser.add_argument("--weight_decay", type=int, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.1)

    args = parser.parse_args()

    if args.dataset == 'credit':
        X = load_credit()
        X = preprocess_credit(X)

    elif args.dataset == 'adult':
        X = load_adult()
        X = preprocess_adult(X)

    else:
        raise(ValueError("Incorrect dataset specified"))
    
    start = time.time()

    run_ablation(X, learning_rate=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, alpha=args.alpha)

    print(f'Total time for ablation took: {time.time() - start:.3f} seconds')
