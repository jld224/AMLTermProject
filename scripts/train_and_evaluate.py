import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor as skKNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from src.preprocessing.BFS import BFS
from src.preprocessing.NeuralNetworkFS import NeuralNetworkFS

try:
    from cuml.neighbors import KNeighborsRegressor as cuKNeighborsRegressor
    KNeighborsRegressor = cuKNeighborsRegressor
    print("Using RAPIDS cuML for GPU acceleration.")
except ImportError:
    KNeighborsRegressor = skKNeighborsRegressor
    print("CUDA not available. Falling back to scikit-learn.")

def apply_feature_selection(fs_strategy, X, y):
    if fs_strategy["name"] == "SelectKBest":
        selector = SelectKBest(mutual_info_regression, k=fs_strategy.get("k", 10))
    elif fs_strategy["name"] == "RFE":
        selector = RFE(RandomForestRegressor(n_estimators=5), n_features_to_select=fs_strategy.get("k", 10))
    elif fs_strategy["name"] == "BFS":
        selector = BFS()
    elif fs_strategy["name"] == "NN": 
        selector = NeuralNetworkFS(input_shape=X.shape[1])
    else:
        raise ValueError("Invalid feature selection strategy")

    selector.fit(X, y)
    if hasattr(selector, 'traceplot'):
        selector.traceplot()
    elif hasattr(selector, 'plot_training_history'): 
        selector.plot_training_history()
    return selector.transform(X), selector

def preprocess_data(X):
    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    
    imputer = SimpleImputer(strategy="mean")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(imputer.fit_transform(X))
    
    return X_scaled


def run_experiment(X, y, model, model_params, feature_selection_strategies):
    X = preprocess_data(X.drop("SMILES", axis=1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    results = []
    for fs_strategy in feature_selection_strategies:
        X_train_fs, selector = apply_feature_selection(fs_strategy, X_train, y_train)
        X_test_fs = selector.transform(X_test)

        model_instance = model(**model_params)
        model_instance.fit(X_train_fs, y_train)
        predictions = model_instance.predict(X_test_fs)

        r2, mse = r2_score(y_test, predictions), mean_squared_error(y_test, predictions)
        results.append((fs_strategy["name"], r2, mse))
        print(f"{fs_strategy['name']}: R2={r2}, MSE={mse}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model evaluation with feature selection.")
    parser.add_argument("--model", type=str, choices=['knn', 'NN'], help="Model name")
    args = parser.parse_args()

    data = pd.read_csv("./data/processed.csv").dropna(subset=["η / mPa s"])
    X, y = data.drop("η / mPa s", axis=1), data["η / mPa s"].values

    feature_selection_strategies = [{"name": "SelectKBest", "k": 8}, {"name": "RFE", "k": 10}]

    if args.model == "knn":
        model_class = KNeighborsRegressor
        model_params = {"n_neighbors": 3}
    elif args.model == "NN":
        model_class = MLPRegressor
        model_params = {
            "hidden_layer_sizes": (100, 50),  # tuple, length = n_layers - 2, default=(100,)
            "activation": 'relu',            # {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
            "solver": 'adam',                # {'lbfgs', 'sgd', 'adam'}, default='adam'
            "alpha": 0.0001,                 # float, default=0.0001. L2 penalty (regularization term)
            "batch_size": 'auto',            # {'auto', int}, default='auto'. Size of minibatches for stochastic optimizers
            "learning_rate": 'constant',     # {'constant', 'invscaling', 'adaptive'}, default='constant'
            "learning_rate_init": 0.001,     # float, default=0.001. The initial learning rate
            "power_t": 0.5,                  # float, default=0.5. The exponent for inverse scaling learning rate
            "max_iter": 2000,                 # int, default=200. Maximum number of iterations.
            "shuffle": True,                 # bool, default=True. Whether to shuffle samples in each iteration
            "random_state": None,            # int, RandomState instance, default=None. Determines random number generation
            "tol": 1e-4,                     # float, default=1e-4. Tolerance for the optimization
            "verbose": False,                # bool, default=False. Whether to print progress messages to stdout
            "warm_start": False,             # bool, default=False. Reuse the solution of the previous call to fit as initialization
            "momentum": 0.9,                 # float, default=0.9. Momentum for gradient descent update
            "nesterovs_momentum": True,      # bool, default=True. Whether to use Nesterov's momentum
            "early_stopping": False,         # bool, default=False. Whether to use early stopping to terminate training when validation
            "validation_fraction": 0.1,      # float, default=0.1. The proportion of training data to set aside as validation set
            "beta_1": 0.9,                   # float, default=0.9. Exponential decay rate for estimates of first moment vector in adam
            "beta_2": 0.999,                 # float, default=0.999. Exponential decay rate for estimates of second moment vector in adam
            "epsilon": 1e-8,                 # float, default=1e-8. Value for numerical stability in adam
            "n_iter_no_change": 10,          # int, default=10. Maximum number of epochs to not meet tol improvement
            "max_fun": 15000                 # int, default=15000. Only for the 'lbfgs' solver. Maximum number of loss function calls.
        }
  # Adjust parameters as needed for your neural network
    else:
        raise ValueError("Invalid model name")

    run_experiment(X, y, model_class, model_params, feature_selection_strategies)

