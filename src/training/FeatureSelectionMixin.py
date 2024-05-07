from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from src.preprocessing.PCAFS import PCAFeatureSelector
from src.preprocessing.GiniFS import GiniImportanceSelector

class FeatureSelectionMixin:
    @staticmethod
    def apply_feature_selection(fs_strategy, X, y):
        rankings = None

        if fs_strategy["name"] == "SelectKBest":
            selector = SelectKBest(mutual_info_regression, k="all").fit(X, y)
            rankings = selector.scores_.argsort()[::-1]
        elif fs_strategy["name"] == "PCA":
            print("PCA Started")
            selector = PCAFeatureSelector()
            selector.fit(X, y)
            print("PCA Ended")
            rankings = selector.get_feature_rankings()
        elif fs_strategy["name"] == "GiniImportance":
            print("GiniImportance Started")
            selector = GiniImportanceSelector(n_estimators=100, random_state=42)
            selector.fit(X, y)
            print("GiniImportance Ended")
            rankings = selector.get_feature_rankings()
        else:
            raise ValueError("Invalid feature selection strategy")

        return rankings
