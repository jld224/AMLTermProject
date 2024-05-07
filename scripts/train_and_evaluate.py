import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

# Import the custom feature selectors
from src.preprocessing.PCAFS import PCAFeatureSelector
from src.preprocessing.GiniFS import GiniImportanceSelector

def apply_feature_selection(fs_strategy, X, y):
    if fs_strategy["name"] == "SelectKBest":
        print("SelectKBest Started")
        selector = SelectKBest(mutual_info_regression, k=fs_strategy.get("k", 10))
        print("SelectKBest Ended")
    elif fs_strategy["name"] == "RFE":
        print("RFE Started")
        selector = RFE(RandomForestRegressor(n_estimators=5), n_features_to_select=fs_strategy.get("k", 10))
        print("RFE Ended")
    elif fs_strategy["name"] == "PCA":
        print("PCA Started")
        selector = PCAFeatureSelector(n_components=fs_strategy.get("k", 10))
        selector.fit(X, y)  # PCA doesn't need target variable y, ignoring it here.
        print("PCA Ended")
        return selector.transform(X), selector
    elif fs_strategy["name"] == "GiniImportance":
        print("GiniImportance Started")
        selector = GiniImportanceSelector(n_estimators=100, random_state=42)
        selector.fit(X, y)
        print("GiniImportance Ended")
        return selector.transform(X), selector
    else:
        raise ValueError("Invalid feature selection strategy")

    selector.fit(X, y)
    return selector.transform(X), selector

def preprocess_data(X):
    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    
    imputer = SimpleImputer(strategy="mean")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))
    
    return X_scaled

def run_experiment(X, y, model_params, feature_selection_strategies):
    X = preprocess_data(X.drop("SMILES", axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    results = []
    for fs_strategy in feature_selection_strategies:
        X_train_fs, selector = apply_feature_selection(fs_strategy, X_train, y_train)
        X_test_fs = selector.transform(X_test)

        model_instance = MLPRegressor(**model_params)
        model_instance.fit(X_train_fs, y_train)
        predictions = model_instance.predict(X_test_fs)

        r2, mse = r2_score(y_test, predictions), mean_squared_error(y_test, predictions)
        results.append((fs_strategy["name"], r2, mse))
        print(f"{fs_strategy['name']}: R2={r2}, MSE={mse}")

    return results

if __name__ == "__main__":
    data = pd.read_csv("./data/processed.csv").dropna(subset=["η / mPa s"])
    X, y = data.drop("η / mPa s", axis=1), data["η / mPa s"].values

    feature_selection_strategies = [
        {"name": "SelectKBest", "k": 8},
        {"name": "RFE", "k": 5},
        {"name": "PCA", "k": 10},  # PCA typically uses 'n_components' rather than 'k', this is just to unify parameter naming
        {"name": "GiniImportance"}
    ]

    model_params = {
        "hidden_layer_sizes": (100, 50),
        "activation": 'relu',
        "solver": 'adam',
        "alpha": 0.0001,
        "batch_size": 'auto',
        "learning_rate": 'constant',
        "learning_rate_init": 0.001,
        "power_t": 0.5,
        "max_iter": 2000,
        "shuffle": True,
        "random_state": None,
        "tol": 1e-4,
        "verbose": False,
        "warm_start": False,
        "momentum": 0.9,
        "nesterovs_momentum": True,
        "early_stopping": False,
        "validation_fraction": 0.1,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "n_iter_no_change": 10,
        "max_fun": 15000
    }

    run_experiment(X, y, model_params, feature_selection_strategies)
