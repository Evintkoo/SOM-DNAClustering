import pandas as pd
import numpy as np

# filter the correlation matrix which the absolute correlation value > bound, return pair correlation and its value list
def corrFilter(X: pd.DataFrame, bound: float):
    xCorr = X.corr().abs()
    xFiltered = xCorr[(xCorr >= bound) & (xCorr !=1.000)]
    xFlattened = xFiltered.unstack().sort_values().drop_duplicates()
    return xFlattened

# pick the most correlated features, returns list of features
def feature_pick(X: pd.DataFrame, treshold = 0.0, total_features = None):
    filtered = corrFilter(X, treshold)
    if not total_features:
        features = filtered
    else:
        features = filtered[:total_features]
        
    indexes = features.index
    feature = np.reshape([[j for j in i] for i in indexes], (-1))
    return np.unique(feature)

# filter the data
def dim_reduction(X, treshold, total_features):
    features = feature_pick(X, treshold=treshold, total_features=total_features)
    X = X[features]
    return X