# Third party
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV

import time
import argparse
import sys

sys.path.append('/gpfs/home1/dbarsony/GOGGLE/src')

# Goggle
from goggle.GoggleModel import GoggleModel
from goggle.data_utils import load_adult, preprocess_adult

from bayes_opt import BayesianOptimization

# Synthcity
from synthcity.plugins.core.dataloader import GenericDataLoader

avg_quality = []
avg_detection = []
avg_utility = []

prediction_real = []
prediction_fake = []

DATASETS = ['red-wine', 
                'ecoli',
                'magic',
                'adult',
                'covertype',
                'credit',
                'breast',
                'mice',
                'musk',
                'white-wine']

def train_goggle(fname, dataset, batch_size, lr, beta, epochs, threshold, runs):

    if dataset == 'red-wine':
        X = pd.read_csv(fname, sep=';')

        ind = list(range(len(X.columns)))
        ind = [x for x in ind if x != X.columns.get_loc("quality")]
        col_list = X.columns[ind]

        ct = ColumnTransformer(
            [("scaler", StandardScaler(), col_list)], remainder="passthrough")

        X_ = ct.fit_transform(X)
        X = pd.DataFrame(X_, index=X.index, columns=X.columns)
        batch_size= 32
        lr=1e-2
        target='quality'
    if dataset =='covertype':
        X = pd.read_csv("../data/covtype.data", header=None)

        ind = list(range(len(X.columns)))
        ind = [x for x in ind if x != X.columns.get_loc(54)]
        col_list = X.columns[ind]
        ct = ColumnTransformer(
            [("scaler", StandardScaler(), col_list)], remainder="passthrough"
        )

        X_ = ct.fit_transform(X)
        X = pd.DataFrame(X_, index=X.index, columns=X.columns)

    if dataset=='adult':
        X = load_adult()
        X = preprocess_adult(X)
        target='income'

    if dataset=='musk':
        X = pd.read_csv(fname)

        ind = list(range(len(X.columns)))
        ind = [x for x in ind if x != X.columns.get_loc("quality")]
        col_list = X.columns[ind]

        ct = ColumnTransformer(
            [("scaler", StandardScaler(), col_list)], remainder="passthrough")

        X_ = ct.fit_transform(X)
        X = pd.DataFrame(X_, index=X.index, columns=X.columns)
    if dataset=='ecoli':
        target = 'ftsJ'

    for i in range(runs):
        
        runtime = time.time()

        X_train, X_test = train_test_split(X, random_state=0, test_size=0.2, shuffle=True)

        gen = GoggleModel(
            ds_name=dataset,
            input_dim=X_train.shape[1],
            encoder_dim=64,
            encoder_l=2,
            het_encoding=True,
            decoder_dim=64,
            decoder_l=2,
            threshold=threshold,
            decoder_arch="gcn",
            graph_prior=None,
            prior_mask=None,
            device="cpu",
            beta=beta,
            learning_rate=lr,
            seed=0,
            batch_size=batch_size,
            epochs=epochs,
        )
        print ("Fitting model")
        gen.fit(X_train)

        X_synth = gen.sample(X_test)

        X_synth_loader = GenericDataLoader(
            X_synth,
            target_column=target,
        )
        X_test_loader = GenericDataLoader(
            X_test,
            target_column=target,
        )

        res = gen.evaluate_synthetic(X_synth_loader, X_test_loader)

        print(f"Quality: {res[0]:.3f}")
        print(f"Detection: {res[2]:.3f}")
        print(
            f"Performance on real: {res[1][0]:.3f}, on synth: {res[1][1]:.3f}, diff: {(res[1][0] - res[1][1]):.3f}"
        )

        avg_quality.append(res[0])
        avg_detection.append(res[2])
        avg_utility.append(res[1][0] - res[1][1])
        prediction_real.append(res[1][0])
        prediction_fake.append(res[1][1])

        print(f'Run {i} took: {time.time() - runtime:.2f} seconds')

    print(f" Average Quality (across 10 runs): {sum(avg_quality)/runs:.2f} +/- {np.std(avg_quality):.2f}")
    print(f"Average Detection (across 10 runs): {sum(avg_detection)/runs:.2f} +/- {np.std(avg_detection):.2f}")
    print(f"Average prediction on real: {sum(prediction_real)/runs:.2f} vs average prediction on fake: {sum(prediction_fake)/runs:.2f}")
    print(
        f"Average Performance difference on real versus synthetic (across 10 runs): {sum(avg_utility)/runs:.2f} +/- {np.std(avg_utility):.2f}"
    )

    # return (avg_quality, avg_detection, avg_utility)
    return res[0]

def black_box_function(batch_size, lr, alpha):

    X = load_adult()
    X = preprocess_adult(X)

    X_train, X_test = train_test_split(X, random_state=0, test_size=0.2, shuffle=True)

    gen = GoggleModel(
        ds_name='adult',
        input_dim=X_train.shape[1],
        encoder_dim=64,
        encoder_l=2,
        het_encoding=True,
        decoder_dim=64,
        decoder_l=2,
        threshold=0.1,
        decoder_arch="gcn",
        graph_prior=None,
        prior_mask=None,
        device="cpu",
        beta=0.1,
        alpha=alpha,
        learning_rate=lr,
        seed=0,
        batch_size=batch_size,
    )
    print ("Fitting model")
    gen.fit(X_train)

    X_synth = gen.sample(X_test)

    X_synth_loader = GenericDataLoader(
        X_synth,
        target_column="income",
    )
    X_test_loader = GenericDataLoader(
        X_test,
        target_column="income",
    )

    res = gen.evaluate_synthetic(X_synth_loader, X_test_loader)

    print(f"Quality: {res[0]:.3f}")
    print(f"Detection: {res[2]:.3f}")
    print(
        f"Performance on real: {res[1][0]:.3f}, on synth: {res[1][1]:.3f}, diff: {(res[1][0] - res[1][1]):.3f}"
    )

    return res[0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--datapath", type=str, default="")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.1)

    args = parser.parse_args()

    if args.dataset not in DATASETS:
        raise ValueError(f"Dataset not an option, choose an option from {DATASETS} instead ...")
    
    start = time.time()

    grid_vals = {"batch_size":[64, 128, 512], "lr":[1e-3, 5e-3, 1e-2], "alpha":[0.1, 0.5, 1]}

    # train_goggle(args.datapath, args.dataset, runs=args.runs, batch_size=args.batch_size, 
    #           lr=args.lr, beta=args.beta, threshold=args.threshold, epochs=args.epochs)
    X = load_adult()
    X = preprocess_adult(X)

    X_train, X_test = train_test_split(X, random_state=0, test_size=0.2, shuffle=True)

    model = GoggleModel(ds_name="adult", input_dim=X_train.shape[1])

    optimizer = GridSearchCV(estimator = model, param_grid=grid_vals, scoring="accuracy", cv=6, refit=True, return_train_score=True)
    optimizer.fit()

    preds = optimizer.best_estimator_.predict(X_test)

    print(f'Total time across runs: {time.time() - start:.3f} seconds')


    
