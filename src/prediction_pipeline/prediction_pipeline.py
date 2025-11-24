from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    r2_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from audio import load_wav_scipy
from features import extract_all_features
from metadata.loading import load_datasets

from .pipeline_data_builder import (
    Organ as DataOrgan,
    SingleSourceDataset,
    MixedSourceDataset,
    load_single_source_dataset,
    load_mixed_dataset,
)

from .models import (
    random_forest_feature_regressor,
    train_classifier,
    GMMClassifier,
)

class Model(Enum):
    LogReg = "logreg"
    KNN = "knn"
    GMM = "gmm"
    RF = "rf"


def run_pipeline(organ: DataOrgan, model: Model):
    single_source_data = load_single_source_dataset(organ)
    mixed_data = load_mixed_dataset(organ)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    feature_r2 = []
    predict_f1 = []
    predict_accuracy = []
    cms = []

    pbar = tqdm(kf.split(mixed_data.labels), total=kf.get_n_splits(), desc="Folds")
    for fold_num, (train_idx, test_idx) in enumerate(pbar, start=1):
        train_set = mixed_data.subset(train_idx)
        test_set = mixed_data.subset(test_idx)

        X_mix_train, X_org_train, y_train, _, _ = train_set.as_arrays()
        X_mix_test, X_org_test, y_test, _, _ = test_set.as_arrays()

        # TRAIN-TEST FEATURE REGRESSOR
        random_forest_model, curr_feature_r2 = random_forest_feature_regressor(
            X_mix_train,
            X_org_train,
            X_mix_test,
            X_org_test,
        )
        feature_r2.append(curr_feature_r2)

        # TRAIN CLASSIFIER
        X_single, y_single, _ = single_source_data.as_arrays()

        X_features_classify = np.vstack([X_mix_train, X_single])
        y_features_classify = np.hstack([y_train, y_single])

        classifier_model = train_classifier(
            "random_forest",
            X_features_classify,
            y_features_classify,
        )

        # TEST PIPELINE
        X_final_test = X_mix_test
        y_final_test = y_test

        X_final_test_features = random_forest_model.predict(X_final_test)
        y_final_prediction = classifier_model.predict(X_final_test_features)


        # EVALUATE PIPELINE
        f1 = f1_score(y_final_test, y_final_prediction, average="weighted")
        predict_f1.append(f1)

        acc = accuracy_score(y_final_test, y_final_prediction)
        predict_accuracy.append(acc)

        cm = confusion_matrix(y_final_test, y_final_prediction)
        cms.append(cm)

    return {
        "feature_r2": feature_r2,
        "f1": predict_f1,
        "accuracy": predict_accuracy,
        "confusion_matrices": cms,
    }
