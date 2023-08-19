import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from plot_results import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from plot_results import plot_roc_curves_per_class
from plot_results import plot_class_roc_curve_per_fold

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import tree
import lightgbm as ltb

import logging
from datetime import datetime

# Set up the logging configuration
log_directory = "run_logs"
log_filename = "run_{}.txt".format(datetime.now().strftime("%Y%m%d-%H%M"))
log_file = os.path.join(log_directory, log_filename)

# Create and configure the logger
logger = logging.getLogger("MyLogger")
logger.setLevel(logging.INFO)

# Create a file handler and set the logging level to INFO
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter to specify the log message format
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(log_formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

random_state = 42
data_labels = ["normal", "EoRS", "EoHT"]
K_FOLDS = 10

CLF_PARAMS = {
    "lgbm": {
            'objective': 'binary',  # Set to binary for lb_ova case otherwise multiclass
            # 'metric': 'multiclassova',
            'boosting_type': 'gbdt',
            'max_bin': 20,
            'min_child_samples': 30,
            'min_data_in_bin': 20,
            'min_split_gain': 0,
            'num_leaves': 20,
            'reg_alpha': 1e-2,
            'reg_lambda': 1e-2,
            'learning_rate': 0.1,
            'n_jobs': 8
    },
    "dt": {
            "criterion": "entropy",
            "max_depth": 20,
            "min_samples_leaf": 8,
            "max_leaf_nodes": 100,
            "max_features": "sqrt",
            "ccp_alpha": 1e-3
    }
}

CLASSIFIERS = {
    "svm": {
        "clf_name": "SVM",
        "clf": SVC(random_state=random_state, probability=True, verbose=0, max_iter=1000)
    },
    "rf": {
        "clf_name": "RandomForest",
        "clf": RandomForestClassifier(random_state=random_state,
                                      n_estimators=300,
                                      criterion="gini",
                                      max_depth=20,
                                      min_samples_split=20,
                                      min_samples_leaf=8,
                                      max_leaf_nodes=100,
                                      max_features="sqrt",
                                      ccp_alpha=1e-3,
                                      verbose=0)
    },
    "knn": {
        "clf_name": "kNN",
        "clf": KNeighborsClassifier()
    },
    "adaboost": {
        "clf_name": "AdaBoost",
        "clf": AdaBoostClassifier(random_state=random_state)
    },
    "dt": {
        "clf_name": "DecisionTree",
        "clf": tree.DecisionTreeClassifier(**CLF_PARAMS["dt"])
    },
    "lgbm": {
        "clf_name": "LightGBM",
        "clf": ltb.LGBMClassifier(**CLF_PARAMS["lgbm"])
    }
}


def k_fold_evaluation(X, y, clf_name, clf_obj, output_dir="outputs"):
    auc_scores = []
    roc_curves_tpr = []
    roc_curves_fpr = []
    # Assuming you have your features in X and labels in y
    # Initialize the label binarizer
    lb = LabelBinarizer()
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=random_state)

    mean_fpr = np.linspace(0, 1, 100)
    for i_split, (train_index, test_index) in enumerate(skf.split(X, y)):
        fold_eval_start = datetime.now()
        print(f"len train index: {len(train_index)}, len test index: {len(test_index)}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(
            f"{i_split}: X_train shape: {X_train.shape} y_train_shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test_shape: {y_test.shape}")

        # Transform the true labels using the label binarizer
        if i_split == 0:
            y_train_binarized = lb.fit_transform(y_train)
        else:
            y_train_binarized = lb.transform(y_train)
        y_test_binarized = lb.transform(y_test)
        # print(f"y_train_lb shape: {y_train_binarized.shape}, y_test_binarized: {y_test_binarized.shape}")

        # Train your classifier on the training data and obtain predicted probabilities
        # assuming you have a classifier named 'clf'
        clf_obj.fit(X_train, y_train)
        y_pred = clf_obj.predict_proba(X_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        print(f"y_pred shape: {y_pred.shape}, y_pred_class shape: {y_pred_class.shape}")

        clf_output_dir = Path(output_dir).joinpath(clf_name)
        if not clf_output_dir.exists():
            print("Creating outputs (for demo only!)")
            clf_output_dir.mkdir(parents=True)
            clf_output_dir.joinpath("classification_report.txt").write_text(
                classification_report(y_test, y_pred_class, target_names=data_labels))
            cm = confusion_matrix(y_test.astype(np.int64), y_pred_class, normalize="true")
            plot_confusion_matrix(cm, clf_name, data_labels)
            print("Outputs completed.")

        # Calculate the AUC for each class
        auc_scores_fold = []
        roc_curves_fpr_fold = []
        roc_curves_tpr_fold = []
        for class_idx in range(y_test_binarized.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_binarized[:, class_idx], y_pred[:, class_idx])
            print(f"\tclass_{class_idx}: fpr_len: {fpr.shape}, tpr_len: {tpr.shape}")
            roc_auc = auc(fpr, tpr)
            auc_scores_fold.append(roc_auc)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            # roc_curves_fpr_fold.append(fpr)
            roc_curves_tpr_fold.append(interp_tpr)

        # Store the AUC scores and curves for each fold
        auc_scores.append(auc_scores_fold)
        # roc_curves_fpr.append(roc_curves_fpr_fold)
        roc_curves_tpr.append(roc_curves_tpr_fold)
        fold_eval_end = datetime.now()
        logger.info(f"\t<{clf_name}>:F-{i_split}: {(fold_eval_end - fold_eval_start).total_seconds():.2f} s.")

    auc_scores = np.array(auc_scores)
    roc_curves_tpr = np.array(roc_curves_tpr)

    return auc_scores, roc_curves_tpr, mean_fpr


def calc_auc_per_class(y_test_bin, y_pred):
    # Calculate the AUC for each class
    auc_scores = []
    roc_curves_tpr = []
    mean_fpr = np.linspace(0, 1, 100)
    for class_idx in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_pred[:, class_idx])
        print(f"\tclass_{class_idx}: fpr_len: {fpr.shape}, tpr_len: {tpr.shape}")
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        roc_curves_tpr.append(interp_tpr)
    return auc_scores, roc_curves_tpr, mean_fpr

def train_test_split_dataset(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")
    logger.info(f"y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    test_size = 0.25
    data = np.load("final_dataset.npz")
    # X, _, y, _ = train_test_split(data["X"], data["y"], test_size=0.95, random_state=random_state)
    X, y = data["X"], data["y"]
    logger.info(f"X.shape: {X.shape}, y.shape: {y.shape}")

    clf = "lgbm"
    clf_name = CLASSIFIERS[clf]["clf_name"]
    clf_obj = CLASSIFIERS[clf]["clf"]
    # run_k_fold_validation(X, y, clf_name, clf_obj)

    # Run simple case
    clf_eval_start = datetime.now()
    # train_test_split_evaluation(X, y, test_size, clf_name, clf_obj)
    train_test_split_evaluation_with_lb_ova(X, y, test_size, clf_name, clf_obj)
    clf_eval_end = datetime.now()
    logger.info(f"Duration of <{clf_name}>: {(clf_eval_end - clf_eval_start).total_seconds():.2f} s.")
def write_exp_eval_cm_cr(y_test, y_pred_class, clf_output_dir):
    if not clf_output_dir.exists():
        print("Creating outputs")
        clf_output_dir.mkdir(parents=True)
    clf_output_dir.joinpath("classification_report.txt").write_text(
        classification_report(y_test, y_pred_class, target_names=data_labels))
    cm = confusion_matrix(y_test.astype(np.int64), y_pred_class, normalize="true")
    plot_confusion_matrix(cm, clf_name, data_labels, output_dir=clf_output_dir.parent)
    print("Outputs completed.")

def train_test_split_evaluation(X, y, test_size, clf_name, clf_obj, output_dir="outputs"):
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y, test_size)

    # Train classifier on the training data and obtain predicted probabilities
    # assuming you have a classifier named 'clf'
    clf_obj.fit(X_train, y_train)
    y_pred = clf_obj.predict_proba(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)

    clf_output_dir = Path(output_dir).joinpath("tts_eval").joinpath(clf_name)
    write_exp_eval_cm_cr(y_test, y_pred_class, clf_output_dir)


def train_test_split_evaluation_with_ova(X, y, test_size, clf_name, clf_obj, output_dir="outputs"):
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y, test_size)

    ova_model = OneVsRestClassifier(clf_obj)
    ova_model.fit(X_train, y_train)
    y_pred = ova_model.predict_proba(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)

    clf_output_dir = Path(output_dir).joinpath("tts_ova_eval").joinpath(clf_name)
    write_exp_eval_cm_cr(y_test, y_pred_class, clf_output_dir)



def train_test_split_evaluation_with_lb_ova(X, y, test_size, clf_name, clf_obj, output_dir="outputs"):
    X_train, X_test, y_train, y_test = train_test_split_dataset(X, y, test_size)

    lb = LabelBinarizer()
    y_train_binarized = lb.fit_transform(y_train)
    y_test_binarized = lb.transform(y_test)

    ova_model = OneVsRestClassifier(clf_obj)
    ova_model.fit(X_train, y_train_binarized)
    y_pred_binarized = ova_model.predict(X_test)
    y_pred_proba = ova_model.predict(X_test)
    y_pred_class = lb.inverse_transform(y_pred_binarized)

    auc_scores, roc_curves_tpr, mean_fpr = calc_auc_per_class(y_test_binarized, y_pred_proba)

    auc_scores = np.array(auc_scores)
    roc_curves_tpr = np.array(roc_curves_tpr)

    for class_idx in range(len(data_labels)):
        class_label = data_labels[class_idx]
        auc = auc_scores[class_idx]
        logger.info(f"Class: {class_label}: AUC: {auc:.3f}")

    clf_output_dir = Path(output_dir).joinpath("tts_ova_lb_eval").joinpath(clf_name)
    write_exp_eval_cm_cr(y_test, y_pred_class, clf_output_dir)
    plot_roc_curves_per_class(roc_curves_tpr,
                              mean_fpr,
                              auc_scores,
                              data_labels,
                              clf_name,
                              output_dir=clf_output_dir.parent)


def run_k_fold_validation(X, y, clf_name, clf_obj):
    logger.info(f"Running for classifier <{clf_name}>")
    clf_eval_start = datetime.now()
    auc_scores, roc_curves_tpr, fpr_mean = k_fold_evaluation(X, y, clf_name, clf_obj)
    clf_eval_end = datetime.now()
    logger.info(f"Duration of <{clf_name}>: {(clf_eval_end - clf_eval_start).total_seconds():.2f} s.")
    mean_auc_scores = auc_scores.mean(axis=0)
    std_auc_scores = auc_scores.std(axis=0)
    mean_roc_curves_tpr = roc_curves_tpr.mean(axis=0)
    for class_idx in range(len(data_labels)):
        class_label = data_labels[class_idx]
        mean_auc = mean_auc_scores[class_idx]
        std_auc = std_auc_scores[class_idx]
        logger.info(f"Class: {class_label}: Mean AUC: {mean_auc:.3f}, STD AUC: {std_auc:.3f}")

    plot_roc_curves_per_class(roc_curves_tpr,
                              fpr_mean,
                              mean_auc_scores,
                              data_labels,
                              clf_name)

    for data_label in data_labels:
        plot_class_roc_curve_per_fold(
            roc_curves_tpr,
            fpr_mean,
            auc_scores,
            data_label,
            data_labels,
            clf_name)


if __name__ == '__main__':
    test_size = 0.25
    data = np.load("final_dataset.npz")
    # X, _, y, _ = train_test_split(data["X"], data["y"], test_size=0.95, random_state=random_state)
    X, y = data["X"], data["y"]
    logger.info(f"X.shape: {X.shape}, y.shape: {y.shape}")

    clf = "lgbm"
    clf_name = CLASSIFIERS[clf]["clf_name"]
    clf_obj = CLASSIFIERS[clf]["clf"]
    # run_k_fold_validation(X, y, clf_name, clf_obj)

    # Run simple case
    clf_eval_start = datetime.now()
    # train_test_split_evaluation(X, y, test_size, clf_name, clf_obj)
    train_test_split_evaluation_with_lb_ova(X, y, test_size, clf_name, clf_obj)
    clf_eval_end = datetime.now()
    logger.info(f"Duration of <{clf_name}>: {(clf_eval_end - clf_eval_start).total_seconds():.2f} s.")
