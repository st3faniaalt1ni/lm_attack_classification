import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def plot_confusion_matrix(cm, classifier, target_names, output_dir="outputs"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix ({classifier})')
    # plt.show()
    output_dir = Path(output_dir).joinpath(classifier)
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir.joinpath("cm.png"))
    plt.close()


def plot_roc_curves_per_class(roc_curves_tpr,
                              fpr_mean,
                              mean_auc_scores,
                              data_labels,
                              clf_name,
                              output_dir="outputs"):
    # Plot the ROC curves for each class
    plt.figure(figsize=(8, 6))
    for class_idx in range(len(data_labels)):
        if len(roc_curves_tpr.shape) > 2:
            tpr_mean = roc_curves_tpr[:, class_idx].mean(axis=0)
        else:
            tpr_mean = roc_curves_tpr[class_idx, :]
        plt.plot(fpr_mean, tpr_mean,
                 label=f"Class {data_labels[class_idx]}, AUC = {mean_auc_scores[class_idx]:.2f}")

        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random performance

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')

    output_path = Path(output_dir).joinpath(clf_name).joinpath("mean_classes_auc.png")
    plt.savefig(output_path)


def plot_class_roc_curve_per_fold(
        roc_curves_tpr,
        fpr_mean,
        auc_scores,
        label,
        data_labels,
        clf_name, output_dir="outputs"):
    class_idx = data_labels.index(label)
    plt.figure(figsize=(8, 6))
    mean_auc_scores = auc_scores.mean(axis=0)
    std_auc_scores = auc_scores.std(axis=0)
    mean_tpr = roc_curves_tpr[:, class_idx].mean(axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = mean_auc_scores[class_idx]
    std_auc = std_auc_scores[class_idx]
    plt.plot(
        fpr_mean,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    folds_nr = auc_scores.shape[0]
    if folds_nr > 1:
        for fold_i in range(folds_nr):
            tpr_fold = roc_curves_tpr[fold_i, class_idx]
            plt.plot(fpr_mean, tpr_fold,
                     label=f"ROC fold {fold_i} (AUC {auc_scores[fold_i, class_idx]:.2f})")
    else:
        tpr_fold = roc_curves_tpr[class_idx]
        plt.plot(fpr_mean, tpr_fold,
                 label=f"AUC {auc_scores[class_idx]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random performance
    std_tpr = roc_curves_tpr[:, class_idx].std(axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        fpr_mean,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Mean ROC curve with variability\n(Positive label '{label}')")
    plt.axis("square")

    plt.legend(loc='lower right')
    output_path = Path(output_dir).joinpath(clf_name).joinpath(f"roc_fold_auc_{label}.png")
    plt.savefig(output_path)
    plt.close()
