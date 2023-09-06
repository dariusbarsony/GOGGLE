# 3rd Party
from pathlib import Path

import zipfile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Synthcity
from synthcity.metrics import eval_detection, eval_performance, eval_statistical
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

generators = Plugins()

def load_dataset(dname):

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
        dataset = "mice"

        le = LabelEncoder()

        X = pd.read_excel("../data/Data_Cortex_Nuclear.xls")

        categorical_names = ['MouseID', 
                            'Genotype', 
                            'Treatment',
                            'Behavior',
        ]

        ind = list(range(len(X.columns)))
        ind = [x for x in ind if x != X.columns.get_loc("class")]

        col_list = X.columns[ind]

        for n in categorical_names:

            X[n] = le.fit_transform(X[n])

        ct = ColumnTransformer(
            [("scaler", StandardScaler(), col_list)], remainder="passthrough"
        )

        X_ = ct.fit_transform(X)
        X = pd.DataFrame(X_, index=X.index, columns=X.columns)

        return X.dropna()
    else: 
        raise ValueError('Incorrect name specified')

def evaluate_baselines(data, seed):

    X_train, X_test = train_test_split(
        data, random_state=seed + 42, test_size=0.2, shuffle=False
    )

    results = {}

    for model in ["bayesian_network"]: #, "ctgan", "tvae", "nflow"]:

        quality_evaluator = eval_statistical.AlphaPrecision()

        xgb_detector = eval_detection.SyntheticDetectionXGB(use_cache=False)
        mlp_detector = eval_detection.SyntheticDetectionMLP(use_cache=False)
        gmm_detector = eval_detection.SyntheticDetectionGMM(use_cache=False)

        gen = generators.get(model, device="cpu")

        gen.fit(X_train)

        X_synth = gen.generate(count=X_test.shape[0]).dataframe()

        X_test_loader = GenericDataLoader(
            X_test,
        )

        X_synth_loader = GenericDataLoader(
            X_synth,
        )

        xgb_det = xgb_detector.evaluate(X_test_loader, X_synth_loader)
        mlp_det = mlp_detector.evaluate(X_test_loader, X_synth_loader)
        gmm_det = gmm_detector.evaluate(X_test_loader, X_synth_loader)
        data_qual = quality_evaluator.evaluate(X_test_loader, X_synth_loader)
        naive_data_qual = {k: v for (k, v) in data_qual.items() if "naive" in k}

        det_score = np.mean([xgb_det["mean"], gmm_det["mean"], mlp_det["mean"]])
        qual_score = np.mean(list(naive_data_qual.values()))
        
        results[model] = [qual_score, det_score]

    return results


if __name__ == "__main__":

    datasets = ['red_wine', 'mice']
    runs = 3

    for data in datasets[1:]: 

        X = load_dataset(data)

        avg_results = {}

        for i in range(runs):

            results = evaluate_baselines(X, 0)

            for model in results.keys(): 

                qual_score, det_score = results[model]
                
                if model in avg_results.keys():
                    avg_results[model][0] += [qual_score]
                    avg_results[model][1] += [det_score]
                else: 
                    avg_results[model] = [[], []]
                    avg_results[model][0] += [qual_score]
                    avg_results[model][1] += [det_score]
        
        for model in avg_results.keys():

            [quality, detection] = avg_results[model]

            print(f"Scores for {model}: Quality {np.mean(quality):.3f} +/- std {np.std(quality)}")
            print(f"Scores for {model}: Detection {np.mean(detection):.3f} +/- std {np.std(detection)}")
            

