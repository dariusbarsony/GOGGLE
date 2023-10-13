# Standard imports
import random

# 3rd party
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

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

def load_adult() -> pd.DataFrame:
    """Load the Adult dataset in a pandas dataframe"""

    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    train_df = pd.read_csv(path, names=names, index_col=False)
    test_df = pd.read_csv(test_path, names=names, index_col=False)[1:]
    
    df = pd.concat([train_df, test_df])
    df = train_df.applymap(lambda x: x.strip() if type(x) is str else x)

    for col in df:
        if df[col].dtype == "object":
            df = df[df[col] != "?"]
    
    df["income"].replace({'<=50K.': '<=50K', '>50K.': '>50K'}, inplace=True)

    return df

def preprocess_adult(dataset: pd.DataFrame) -> pd.DataFrame:
    """Preprocess adult data set."""

    replace = [
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ],
        [
            "Bachelors",
            "Some-college",
            "11th",
            "HS-grad",
            "Prof-school",
            "Assoc-acdm",
            "Assoc-voc",
            "9th",
            "7th-8th",
            "12th",
            "Masters",
            "1st-4th",
            "10th",
            "Doctorate",
            "5th-6th",
            "Preschool",
        ],
        [
            "Married-civ-spouse",
            "Divorced",
            "Never-married",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse",
        ],
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ],
        [
            "Wife",
            "Own-child",
            "Husband",
            "Not-in-family",
            "Other-relative",
            "Unmarried",
        ],
        ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        ["Female", "Male"],
        [
            "United-States",
            "Cambodia",
            "England",
            "Puerto-Rico",
            "Canada",
            "Germany",
            "Outlying-US(Guam-USVI-etc)",
            "India",
            "Japan",
            "Greece",
            "South",
            "China",
            "Cuba",
            "Iran",
            "Honduras",
            "Philippines",
            "Italy",
            "Poland",
            "Jamaica",
            "Vietnam",
            "Mexico",
            "Portugal",
            "Ireland",
            "France",
            "Dominican-Republic",
            "Laos",
            "Ecuador",
            "Taiwan",
            "Haiti",
            "Columbia",
            "Hungary",
            "Guatemala",
            "Nicaragua",
            "Scotland",
            "Thailand",
            "Yugoslavia",
            "El-Salvador",
            "Trinadad&Tobago",
            "Peru",
            "Hong",
            "Holand-Netherlands",
        ],
        [">50K", "<=50K"],
    ]

    df = dataset

    for row in replace:
        df = df.replace(row, range(len(row)))

    ind = list(range(len(df.columns)))

    ind = [x for x in ind if x != df.columns.get_loc("income")]
    col_list = df.columns[ind]

    ct = ColumnTransformer(
        [("scaler", StandardScaler(), col_list)], remainder="passthrough"
    )

    df = pd.DataFrame(ct.fit_transform(df),
                      index=df.index, columns=df.columns)

    return df

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

def load_credit() -> pd.DataFrame:

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

    X = pd.read_csv("../data/german.data", header=None, sep=' ', names=names)
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
