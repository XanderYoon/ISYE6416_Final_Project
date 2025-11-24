from __future__ import annotations
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class GMMClassifier:
    def __init__(self, n_components=4, covariance_type="full", random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models = {}
        for cls in self.classes_:
            Xc = X[y == cls]
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state
            )
            gmm.fit(Xc)
            self.models[cls] = gmm

    def predict(self, X):
        scores = np.column_stack([self.models[c].score_samples(X) for c in self.classes_])
        return np.array([self.classes_[i] for i in scores.argmax(axis=1)])


def train_classifier(X_train, y_train_raw, model_type="logreg", random_state=42):

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)

    if model_type == "logreg":
        model = LogisticRegression(max_iter=2000)

    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=300, random_state=random_state)

    elif model_type == "svm_rbf":
        model = SVC(kernel="rbf", probability=True)

    elif model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=5, weights="distance")

    elif model_type == "gmm":
        model = GMMClassifier()
        model.fit(X_train, y_train)
        return {"model": model, "label_encoder": le}

    elif model_type == "pca_logreg":
        pca = PCA(n_components=0.95)
        Xp = pca.fit_transform(X_train)
        logreg = LogisticRegression(max_iter=2000)
        logreg.fit(Xp, y_train)
        return {"model": logreg, "label_encoder": le, "pca": pca, "is_pca_model": True}

    else:
        raise ValueError("Unsupported model")

    model.fit(X_train, y_train)

    return {
        "model": model,
        "label_encoder": le
    }


def evaluate_classifier(clf_bundle, X, y_raw, plot_cm=False):

    le = clf_bundle["label_encoder"]
    y_true = le.transform(y_raw)

    if "is_pca_model" in clf_bundle:
        X = clf_bundle["pca"].transform(X)

    model = clf_bundle["model"]
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": cm,
        "classes": le.classes_
    }
