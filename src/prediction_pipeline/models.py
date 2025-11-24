# models.py

import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture


# ============================================================
#  FEATURE REGRESSOR
# ============================================================

def random_forest_feature_regressor(
    X_train, y_train, 
    X_test, y_test,
    n_estimators=300,
    max_depth=None,
    random_state=42
):
    """
    Train a RandomForestRegressor and compute R² on the test set.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test  = np.array(X_test)
    y_test  = np.array(y_test)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state
    )

    rf.fit(X_train, y_train)
    y_pred_test = rf.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test, multioutput="variance_weighted")

    return rf, r2_test


# ============================================================
#  GMM CLASSIFIER
# ============================================================

class GMMClassifier:
    """
    A supervised classifier based on fitting one Gaussian Mixture 
    Model per class and selecting via max log-likelihood.
    """

    def __init__(self, class_gmms):
        self.class_gmms = class_gmms
        self.classes_ = list(class_gmms.keys())

    def predict(self, X):
        X = np.array(X)
        log_probs = []

        for c in self.classes_:
            gmm = self.class_gmms[c]
            log_probs.append(gmm.score_samples(X))

        log_probs = np.vstack(log_probs).T
        return np.array(self.classes_)[np.argmax(log_probs, axis=1)]


def _train_gmm_classifier(X, y, n_components, **kwargs):
    """
    Fit a GMM classifier using one mixture per class.
    """
    X = np.array(X)
    y = np.array(y)

    classes = np.unique(y)
    class_gmms = {}

    for c in classes:
        X_c = X[y == c]

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=42,
            **kwargs
        )
        gmm.fit(X_c)
        class_gmms[c] = gmm

    return GMMClassifier(class_gmms)


# ============================================================
#  UNIVERSAL MODEL TRAINER (single shared API)
# ============================================================

def train_classifier(
    model_type, 
    X, 
    y, 
    n_estimators=500,
    max_depth=None,
    random_state=42,
    **kwargs
):
    """
    Unified training interface for all model types.
    Only model_type changes.

    Parameters
    ----------
    model_type : {"logistic_regression", "knn", "gmm", "random_forest"}
    X, y : training data
    n_estimators : used for random forests (ignored otherwise)
    max_depth : optional tree depth (RF only)
    random_state : shared across all models

    All models share the same parameter interface.
    """
    X = np.array(X)
    y = np.array(y).astype(str)     # enforce categorical labels

    classes = np.unique(y)
    n_classes = len(classes)

    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=500, random_state=random_state, **kwargs)

    elif model_type == "knn":
        model = KNeighborsClassifier(
            n_neighbors=n_classes,       
            **kwargs
        )

    elif model_type == "gmm":
        return _train_gmm_classifier(
            X, y, n_components=n_classes, **kwargs
        )

    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=random_state,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X, y)
    return model
