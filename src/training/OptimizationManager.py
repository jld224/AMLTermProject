from sklearn.model_selection import RandomizedSearchCV
from skopt.space import Real, Categorical, Integer
import warnings


class OptimizationManager:
    def __init__(self):
        self.best_estimator = None
        self.best_params = {}

    def perform_random_search(self, estimator, params, X, y, n_iter=10, cv=2, n_jobs=-1, random_state=42):
        """
        Perform Random Search to find the best model hyperparameters.
        """

        # Setup RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=params,
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state
        )

        random_search.fit(X, y)

        # Store the best estimator and parameters
        self.best_estimator = random_search.best_estimator_
        self.best_params = random_search.best_params_

    def reset_space(self):
        self.best_estimator = None
        self.best_params = {}
