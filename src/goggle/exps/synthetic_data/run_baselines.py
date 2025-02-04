# 3rd Party
import time
import argparse
import sys

sys.path.append('/gpfs/home1/dbarsony/GOGGLE/src')

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Synthcity
from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

from goggle.data_utils import load_adult, preprocess_adult, load_credit, preprocess_credit

generators = Plugins()

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

def preprocess_credit(df : pd.DataFrame) -> pd.DataFrame:

    replace = [
        ['A11', 'A12', 'A13', 'A14'],
        ['A30', 'A31','A32','A33','A34','A35'],
        ['A40','A41','A42','A43','A44','A45','A46','A47','A48','A49','A410'],
        ['A61','A62','A63','A64','A65'],
        ['A71','A72','A73','A74','A75'],
        ['A91','A92','A93','A94','A95'],
        ['A101','A102','A103'],
        ['A121','A122','A123','A124'],
        ['A141','A142','A143'],
        ['A151','A152','A153'],
        ['A171','A172','A173','A174'],
        ['A191','A192'],
        ['A201','A202']
    ]

    for row in replace:
        df = df.replace(row, range(len(row)))

    ind = list(range(len(df.columns)))

    ind = [x for x in ind if x != df.columns.get_loc('target')]
    col_list = df.columns[ind]

    ct = ColumnTransformer(
        [("scaler", StandardScaler(), col_list)], remainder="passthrough"
    )

    df = pd.DataFrame(ct.fit_transform(df),
                      index=df.index, columns=df.columns)

    return df

def load_dataset(dname, fname):
    if dname == 'red_wine':
        X = pd.read_csv("../data/winequality-red.csv", sep=';')
        ind = list(range(len(X.columns)))

        ind = [x for x in ind if x != X.columns.get_loc("quality")]
        col_list = X.columns[ind]
        ct = ColumnTransformer(
            [("scaler", StandardScaler(), col_list)], remainder="passthrough"
        )

        X_ = ct.fit_transform(X)
        X = pd.DataFrame(X_, index=X.index, columns=X.columns)

        return X
    if dname=='mice':
        cols2skip = ['MouseID', 
                     'Genotype', 
                     'Treatment',
                     'Behavior',
        ]

        X = pd.read_excel(fname)
        X.drop(labels=cols2skip, axis=1, inplace=True)

        ind = list(range(len(X.columns)))
        ind = [x for x in ind if x != X.columns.get_loc("class")]

        col_list = X.columns[ind]

        ct = ColumnTransformer(
            [("scaler", StandardScaler(), col_list)], remainder="passthrough"
        )

        X_ = ct.fit_transform(X)
        X = pd.DataFrame(X_, index=X.index, columns=X.columns)

        X = X.dropna()
        X["class"].replace({"c-CS-m": 0, "t-SC-s": 1, "c-SC-m":2, "c-CS-s":3, "c-SC-s":4, "t-CS-m":5, "t-SC-m":6, "t-CS-s":7}, inplace=True)

        return X.astype(float)
    if dname=='adult':
           X = load_adult()
           X = preprocess_adult(X)

           return X
    if dname == "credit":
        names = [ 
            'status',
            'duration',
            'credit_history',
            'purpose',
            'credit_amount',
            'savings_account',
            'present_employment',
            'installment_rate',
            'personal_status',
            'other_debtors',
            'residence',
            'property',
            'age',
            'other_installment_plans',
            'housing',
            'number_of_existing_credits',
            'job',
            'liable_people',
            'telephone',
            'foreign_worker',
            'target'
        ]

        X = pd.read_csv(fname, header=None, sep=' ', names=names)
        X = preprocess_credit(X)

        X["target"] = X["target"] - 1.0
        X = X.dropna(axis=0)
        ind = list(range(len(X.columns)))
        ind = [x for x in ind if x != X.columns.get_loc("target")]
        col_list = X.columns[ind]

        ct = ColumnTransformer(
            [("scaler", StandardScaler(), col_list)], remainder="passthrough"
        )

        X_ = ct.fit_transform(X)
        X = pd.DataFrame(X_, index=X.index, columns=X.columns)

        return X
    if dname == "ecoli":
        dataset = "ecoli"
        X = pd.read_csv("../data/magic-irri_2000.csv")

        ind = list(range(len(X.columns)))
        col_list = X.columns[ind]
        ct = ColumnTransformer(
            [("scaler", StandardScaler(), col_list)], remainder="passthrough"
        )

        X_ = ct.fit_transform(X)
        X = pd.DataFrame(X_, index=X.index, columns=X.columns)

        return X
    if dname == "magic":
        X = pd.read_csv("../data/magic-irri_2000.csv")

        ind = list(range(len(X.columns)))
        col_list = X.columns[ind]
        ct = ColumnTransformer(
            [("scaler", StandardScaler(), col_list)], remainder="passthrough"
        )

        X_ = ct.fit_transform(X)
        X = pd.DataFrame(X_, index=X.index, columns=X.columns)

        return X
    else: 
        raise ValueError('Incorrect name specified')

def run_baselines(data, seed, runs, device='cpu'):

    benchmarks = {'bayesian_network':{'quality':[],'detection':[], 'utility':[]}, 
                     'ctgan':{'quality':[],'detection':[], 'utility':[]},
                     'tvae':{'quality':[],'detection':[], 'utility':[]},
                     'nflow':{'quality':[],'detection':[], 'utility':[]}}

    for i in range(runs):
        
        print(f"Starting training run {i} ...")

        X_train, X_test = train_test_split(
            data, random_state=seed + 42, test_size=0.2, shuffle=False
        )

        for model in ["bayesian_network", "ctgan", "tvae", "nflow"]:

            # get baseline and fit
            gen = generators.get(model, device=device)
            gen.fit(X_train)

            # use generator to sample data
            X_synth = gen.generate(count=X_test.shape[0]).dataframe()

            # evaluate samples
            X_test_loader = GenericDataLoader(
                X_test,
            )

            X_synth_loader = GenericDataLoader(
                X_synth,
            )

            quality_evaluator = eval_statistical.AlphaPrecision()
            qual_res = quality_evaluator.evaluate(X_test_loader, X_synth_loader)
            qual_res = {
                k: v for (k, v) in qual_res.items() if "naive" in k
            }  # use the naive implementation of AlphaPrecision
            qual_score = np.mean(list(qual_res.values()))

            xgb_evaluator = eval_performance.PerformanceEvaluatorXGB()
            linear_evaluator = eval_performance.PerformanceEvaluatorLinear()
            mlp_evaluator = eval_performance.PerformanceEvaluatorMLP()

            xgb_score = xgb_evaluator.evaluate(X_test_loader, X_synth_loader)
            linear_score = linear_evaluator.evaluate(X_test_loader, X_synth_loader)
            mlp_score = mlp_evaluator.evaluate(X_test_loader, X_synth_loader)
            gt_perf = (xgb_score["gt"] + linear_score["gt"] + mlp_score["gt"]) / 3
            synth_perf = (
                xgb_score["syn_ood"] + linear_score["syn_ood"] + mlp_score["syn_ood"]
            ) / 3

            xgb_detector = eval_detection.SyntheticDetectionXGB()
            mlp_detector = eval_detection.SyntheticDetectionMLP()
            gmm_detector = eval_detection.SyntheticDetectionGMM()

            xgb_det = xgb_detector.evaluate(X_test_loader, X_synth_loader)
            mlp_det = mlp_detector.evaluate(X_test_loader, X_synth_loader)
            gmm_det = gmm_detector.evaluate(X_test_loader, X_synth_loader)
            det_score = (xgb_det["mean"] + mlp_det["mean"] + gmm_det["mean"]) / 3
                
            benchmarks[model]['quality'].append(qual_score)
            benchmarks[model]['detection'].append(det_score)
            benchmarks[model]['utility'].append(gt_perf - synth_perf)

            print(f'Result for run {i}')

            print(f"Quality: {qual_score:.3f}")
            print(f"Detection: {det_score:.3f}")
            print(
                f"Performance on real: {gt_perf:.3f}, on synth: {synth_perf:.3f}, diff: {(gt_perf - synth_perf):.3f}"
            )
    return benchmarks

def print_final(benchmarks):
    for model in benchmarks.keys():
        print(f'Printing results for {model}:')

        print(f"Scores for {model}: Quality {np.mean(benchmarks[model]['quality']):.3f} +/- std {np.std(benchmarks[model]['quality'])}")
        print(f"Scores for {model}: Detection {np.mean(benchmarks[model]['detection']):.3f} +/- std {np.std(benchmarks[model]['detection'])}")
        print(
                f" Scores for {model}: Utility {np.mean(benchmarks[model]['utility']):.3f} +/- std {np.std(benchmarks[model]['utility'])}"
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default='cpu')

    args = parser.parse_args()

    if args.dataset not in DATASETS:
        raise ValueError(f"Dataset not an option, choose an option from {DATASETS} instead ...")

    start = time.time()

    X = load_dataset(args.dataset, args.dataset_name)
    final = run_baselines(X, seed=args.seed, runs=args.runs, device=args.device)

    print(f"total time for all baselines took: {time.time()-start:.3f}")
    print_final(final)
            

