from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

# path to the proben1 git repository
source_path = Path("~/proben1").expanduser()
# path where converted numpy array txt will be saved
target_path = Path("./datasets/proben1")


def card():
    # load the train data
    with open(source_path / "card" / "crx.data", "r") as data_file:
        df: pd.DataFrame = pd.read_csv(data_file, sep=",", header=None)

    # delete NaN columns
    df = df.dropna(axis=1)
    
    # these category contain category strings
    category_columns = [0, 3, 4, 5, 6, 8, 9, 11, 12, 15]
    # set the type to category
    df.iloc[:, category_columns] = df.iloc[:, category_columns].astype("category")
    # convert strings to numeric
    for col in category_columns:
        df.iloc[:, col] = df.iloc[:, col].cat.codes
    
    # convert some columns to float type
    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    df.iloc[:, 13] = pd.to_numeric(df.iloc[:, 13], errors="coerce")

    # delete NaN values
    df = df.dropna(axis=0)

    # split data and labels
    X, Y = df.iloc[:, :15].to_numpy(), df.iloc[:, 15].to_numpy()
    
    print(X.shape)
    print(Y.shape)
    
    # normalize the data with MinMaxScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # save data as txt
    target_path.mkdir(exist_ok=True, parents=True)
    np.savetxt(target_path / "card_data.txt", X)
    np.savetxt(target_path / "card_labels.txt", Y)

def diabetes():
    # load the train data
    with open(source_path / "diabetes" / "pima-indians-diabetes.data", "r") as data_file:
        df: pd.DataFrame = pd.read_csv(data_file, sep=",", header=None)

    # delete NaN columns
    df = df.dropna(axis=1)

    # split data and labels
    X, Y = df.iloc[:, :8].to_numpy(), df.iloc[:, 8].to_numpy()
    
    print(X.shape)
    print(Y.shape)
    
    # normalize the data with MinMaxScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # save data as txt
    target_path.mkdir(exist_ok=True, parents=True)
    np.savetxt(target_path / "diabetes_data.txt", X)
    np.savetxt(target_path / "diabetes_labels.txt", Y)

def thyroid():
    # load the train data
    with open(source_path / "thyroid" / "ann-train.data", "r") as data_file:
        df: pd.DataFrame = pd.read_csv(data_file, sep=" ", header=None)

    # delete NaN columns
    df = df.dropna(axis=1)

    # split data and labels
    x_train, y_train = df.iloc[:, :21].to_numpy(), df.iloc[:, 21].to_numpy().astype("int") - 1
    
    print(x_train.shape)
    print(y_train.shape)

    # normalize the data with MinMaxScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)

    with open(source_path / "thyroid" / "ann-test.data", "r") as data_file:
        df: pd.DataFrame = pd.read_csv(data_file, sep=" ", header=None)

    # delete NaN values
    df = df.dropna(axis=1)

    # split data and labels
    x_test, y_test = df.iloc[:, :21].to_numpy(), df.iloc[:, 21].to_numpy().astype("int") - 1
    
    print(x_test.shape)
    print(y_test.shape)

    # normalize the data with MinMaxScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    x_test = min_max_scaler.fit_transform(x_test)

    # concatenate train, test data
    X = np.concatenate([x_train, x_test], axis=0)
    Y = np.concatenate([y_train, y_test], axis=0)

    # save data as txt
    target_path.mkdir(exist_ok=True, parents=True)
    np.savetxt(target_path / "thyroid_data.txt", X)
    np.savetxt(target_path / "thyroid_labels.txt", Y)

def geneN():
    # load the train data
    with open(source_path / "gene" / "gene.data", "r") as data_file:
        df: pd.DataFrame = pd.read_csv(data_file, sep=",", header=None, skipinitialspace=True)

    # second column holds no important information
    df = df.drop(1, 1)
    # mapping of genes to integers
    mapping = {
        "A": 0, "G": 1, "T": 2, "C": 3, "D": 4, "N": 5, "S": 6, "R": 7,
    }
    # apply mapping on feature column
    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda s: [mapping[c] for c in list(s)])
    # convert label strings to codes
    df.iloc[:, 0] = df.iloc[:, 0].astype("category")
    df.iloc[:, 0] = df.iloc[:, 0].cat.codes

    # split data and labels
    X, Y = np.stack(df.iloc[:, 1].to_numpy(), axis=0), df.iloc[:, 0].to_numpy()
    
    print(X.shape)
    print(Y.shape)
    
    # normalize the data with MinMaxScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # save data as txt
    target_path.mkdir(exist_ok=True, parents=True)
    np.savetxt(target_path / "geneN_data.txt", X)
    np.savetxt(target_path / "geneN_labels.txt", Y)

def glass():
    # load the train data
    with open(source_path / "glass" / "glass.data", "r") as data_file:
        df: pd.DataFrame = pd.read_csv(data_file, sep=",", header=None, skipinitialspace=True)

    # delete NaN columns
    df = df.dropna(axis=1)

    # split data and labels
    X, Y = df.iloc[:, :10].to_numpy(), df.iloc[:, 10].to_numpy() - 1
    
    print(X.shape)
    print(Y.shape)
    
    # normalize the data with MinMaxScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # temporary remove label 3
    X = X[Y != 3]
    Y = Y[Y != 3]
    Y[Y > 3] = Y[Y > 3] - 1

    # save data as txt
    target_path.mkdir(exist_ok=True, parents=True)
    np.savetxt(target_path / "glass_data.txt", X)
    np.savetxt(target_path / "glass_labels.txt", Y)

def horse():
    # load the train data
    with open(source_path / "horse" / "horse-colic.data", "r") as data_file:
        df: pd.DataFrame = pd.read_csv(data_file, delim_whitespace=True, header=None)
    
    # load the train data
    with open(source_path / "horse" / "horse-colic.test", "r") as data_file:
        df = pd.concat([df, pd.read_csv(data_file, delim_whitespace=True, header=None)])

    # delete NaN columns
    df = df.dropna(axis=1)

    # replace ? with 0
    df = df.replace(to_replace="?", value=0)

    # convert all columns to float type
    df = df.apply(pd.to_numeric, errors="coerce")

    # delete NaN values
    df = df.dropna(axis=0)

    # split data and labels
    X, Y = df.iloc[:, :27].to_numpy(), df.iloc[:, 27].to_numpy() - 1
    
    print(X.shape)
    print(Y.shape)

    # normalize the data with MinMaxScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # save data as txt
    target_path.mkdir(exist_ok=True, parents=True)
    np.savetxt(target_path / "horse-colic_data.txt", X)
    np.savetxt(target_path / "horse-colic_labels.txt", Y)

def soybean():
    # load the train data
    with open(source_path / "soybean" / "soybean-large.data", "r") as data_file:
        df: pd.DataFrame = pd.read_csv(data_file, delimiter=",", header=None)
    
    # load the train data
    with open(source_path / "soybean" / "soybean-large.test", "r") as data_file:
        df = pd.concat([df, pd.read_csv(data_file, delimiter=",", header=None)])

    # delete NaN columns
    df = df.dropna(axis=1)

    # replace ? with 0
    df = df.replace(to_replace="?", value=0)

    # convert all columns to float type
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    # delete NaN values
    df = df.dropna(axis=0)

    # set category type for label column
    df.iloc[:, 0] = df.iloc[:, 0].astype("category")
    df.iloc[:, 0] = df.iloc[:, 0].cat.codes

    # split data and labels
    X, Y = df.iloc[:, 1:].to_numpy(), df.iloc[:, 0].to_numpy()
    
    print(X.shape)
    print(Y.shape)

    # normalize the data with MinMaxScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # temporary remove first label

    X = X[Y != 0]
    Y = Y[Y != 0] - 1

    # save data as txt
    target_path.mkdir(exist_ok=True, parents=True)
    np.savetxt(target_path / "soybean_data.txt", X)
    np.savetxt(target_path / "soybean_labels.txt", Y)

datasets = {
    "card": card,
    "thyroid": thyroid,
    "diabetes": diabetes,
    "geneN": geneN,
    "glass": glass,
    "horse-colic": horse,
    "soybean": soybean,
}

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Load and convert proben1 datasets.")
    parser.add_argument("dataset", choices=list(datasets.keys()))
    args = parser.parse_args()

    datasets[args.dataset]()
