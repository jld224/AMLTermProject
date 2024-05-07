from sklearn.neural_network import MLPRegressor
from scipy.stats import uniform, randint

EXP_CONFIG = [
    {
        "model": MLPRegressor,
        "param_grid": {
            "hidden_layer_sizes": [
                (50,),
                (100,),
                (50, 50),
                (100, 100),
                (100, 50, 25),
                (50, 25, 10),
            ],
            "activation": ["identity", "logistic", "tanh", "relu"],
            "solver": ["adam"],
            "alpha": uniform(0.0001, 0.0099),
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "learning_rate_init": uniform(0.001, 0.0099),
            "beta_1": uniform(0.8, 0.199),
            "beta_2": uniform(0.9, 0.0999),
            "epsilon": uniform(1e-8, 1e-6 - 1e-8),
            "max_iter": [3000],
            "verbose": [False],
        },
    }
]
