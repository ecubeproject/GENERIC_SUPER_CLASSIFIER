"""Diagnostic plots as pure functions.

Every function takes a matplotlib ``ax`` as its first argument and draws onto
it - no Tkinter, no figure/window management, no global state. That makes them
trivially testable (Agg backend) and reusable from a notebook or Streamlit.

Signatures by ``kind`` (see ``PLOTS`` at the bottom):
  'proba'          fn(ax, y_true, y_proba, model_classes, class_names)
  'matrix'         fn(ax, X)
  'matrix_labels'  fn(ax, X, y_true, class_names)

``y_true`` holds label-encoded integers; ``model_classes`` is the estimator's
``classes_`` (encoded); ``class_names[i]`` is the display label for encoded
class ``i``.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve

_TRAPZ = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')  # renamed in NumPy 2.0


def class_color(i, n):
    """Stable per-class colour from a colormap (handles any number of classes)."""
    if n <= 10:
        return plt.get_cmap('tab10')(i % 10)
    if n <= 20:
        return plt.get_cmap('tab20')(i % 20)
    return plt.get_cmap('hsv')(i / max(n, 1))


def cls_label(encoded_value, class_names):
    """Map an encoded class (0, 1, 2, ...) back to its display name."""
    try:
        return class_names[int(encoded_value)]
    except (TypeError, ValueError, IndexError):
        return str(encoded_value)


def _ovr_indices(model_classes):
    """Which probability columns to plot: just the positive column for a binary
    task, every column for multiclass one-vs-rest."""
    return [1] if len(model_classes) == 2 else list(range(len(model_classes)))


# ---------------------------------------------------------------------------
# Probability-based plots
# ---------------------------------------------------------------------------

def plot_roc(ax, y_true, y_proba, model_classes, class_names):
    y_true = np.asarray(y_true)
    idxs = _ovr_indices(model_classes)
    curves = []
    for i in idxs:
        pos = model_classes[i]
        fpr, tpr, _ = roc_curve((y_true == pos).astype(int), y_proba[:, i])
        curves.append((cls_label(pos, class_names), fpr, tpr, auc(fpr, tpr)))

    for j, (lbl, fpr, tpr, a) in enumerate(curves):
        ax.plot(fpr, tpr, lw=2, color=class_color(j, len(curves)),
                label=f'{lbl} (AUC = {a:0.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel='False Positive Rate',
           ylabel='True Positive Rate',
           title='ROC Curve' if len(curves) == 1 else 'ROC Curve (one-vs-rest)')
    ax.legend(loc='lower right')


def plot_precision_recall(ax, y_true, y_proba, model_classes, class_names):
    y_true = np.asarray(y_true)
    idxs = _ovr_indices(model_classes)
    curves = []
    for i in idxs:
        pos = model_classes[i]
        binary_true = (y_true == pos).astype(int)
        precision, recall, _ = precision_recall_curve(binary_true, y_proba[:, i])
        ap = average_precision_score(binary_true, y_proba[:, i])
        curves.append((cls_label(pos, class_names), recall, precision, ap))

    for j, (lbl, recall, precision, ap) in enumerate(curves):
        ax.plot(recall, precision, lw=2, color=class_color(j, len(curves)),
                label=f'{lbl} (AP = {ap:0.3f})')
    ax.set(xlabel='Recall', ylabel='Precision',
           title='Precision-Recall' if len(curves) == 1 else 'Precision-Recall (one-vs-rest)')
    ax.legend(loc='lower left')


def plot_calibration(ax, y_true, y_proba, model_classes, class_names):
    y_true = np.asarray(y_true)
    idxs = _ovr_indices(model_classes)
    for j, i in enumerate(idxs):
        pos = model_classes[i]
        prob_true, prob_pred = calibration_curve((y_true == pos).astype(int),
                                                 y_proba[:, i], n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', color=class_color(j, len(idxs)),
                label=cls_label(pos, class_names))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax.set(xlabel='Mean predicted probability', ylabel='Fraction of positives',
           title='Calibration')
    ax.legend()
    ax.grid(True)


def plot_cumulative_gains(ax, y_true, y_proba, model_classes, class_names):
    y_true = np.asarray(y_true)
    idxs = _ovr_indices(model_classes)
    for j, i in enumerate(idxs):
        pos = model_classes[i]
        order = np.argsort(y_proba[:, i])[::-1]
        hits = (y_true[order] == pos).astype(int)
        gains = np.cumsum(hits) / max(hits.sum(), 1)
        frac = np.arange(1, len(gains) + 1) / len(gains)
        x, yv = np.r_[0, frac], np.r_[0, gains]
        ax.plot(x, yv, lw=2, color=class_color(j, len(idxs)),
                label=f'{cls_label(pos, class_names)} (area = {_TRAPZ(yv, x):.3f})')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Baseline (random)')
    ax.set(xlabel='Fraction of sample (ranked by score)',
           ylabel='Fraction of positives captured', title='Cumulative Gains')
    ax.legend(loc='lower right')
    ax.grid(True)


def plot_lift(ax, y_true, y_proba, model_classes, class_names):
    y_true = np.asarray(y_true)
    idxs = _ovr_indices(model_classes)
    for j, i in enumerate(idxs):
        pos = model_classes[i]
        baseline = max(np.mean(y_true == pos), 1e-12)
        binary_true = (y_true == pos).astype(int)
        precision, recall, _ = precision_recall_curve(binary_true, y_proba[:, i])
        ax.plot(recall * 100, precision / baseline, marker='.', markersize=4,
                color=class_color(j, len(idxs)), label=cls_label(pos, class_names))
    ax.plot([0, 100], [1, 1], 'k--', label='Baseline (random)')
    ax.set(xlabel='% of samples covered (recall)', ylabel='Lift',
           title='Lift Curve' if len(idxs) == 1 else 'Lift Curves (one-vs-rest)')
    ax.legend()
    ax.grid(True)


# ---------------------------------------------------------------------------
# Feature-matrix plots
# ---------------------------------------------------------------------------

def plot_pca_explained_variance(ax, X):
    X = np.asarray(X)
    n_comp = min(X.shape)
    pca = PCA(n_components=n_comp).fit(X)
    ax.plot(np.arange(1, n_comp + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
    ax.set(xlabel='Number of components', ylabel='Cumulative explained variance',
           title='PCA Explained Variance')
    ax.grid(True)


def plot_pca_2d(ax, X, y_true, class_names):
    X = np.asarray(X)
    y = np.asarray(y_true)
    coords = PCA(n_components=2).fit_transform(X)
    classes = np.unique(y)
    for i, c in enumerate(classes):
        m = y == c
        ax.scatter(coords[m, 0], coords[m, 1], s=40, alpha=0.7,
                   color=class_color(i, len(classes)), label=cls_label(c, class_names))
    ax.set(xlabel='Principal Component 1', ylabel='Principal Component 2',
           title='PCA 2D Projection')
    ax.legend(loc='best')
    ax.grid(True)


def plot_silhouette(ax, X):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = np.asarray(X)
    range_n_clusters = [n for n in (2, 3, 4, 5) if n < len(X)]
    scores = []
    for n_clusters in range_n_clusters:
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=10).fit_predict(X)
        scores.append(silhouette_score(X, labels))
    ax.plot(range_n_clusters, scores, marker='o', label='Silhouette Score')
    ax.set(xlabel='Number of Clusters', ylabel='Silhouette Score',
           title='Silhouette Analysis for Optimal Cluster Count')
    ax.legend()


# ---------------------------------------------------------------------------
# Registry - drives the Tk dropdown and the test sweep
# ---------------------------------------------------------------------------

PLOTS = {
    'ROC Curve': {
        'fn': plot_roc, 'kind': 'proba',
        'desc': ("Performance across all classification thresholds. Area under the "
                 "curve (AUC) summarises it: 1.0 is perfect, 0.5 is random."),
    },
    'Precision-Recall Curve': {
        'fn': plot_precision_recall, 'kind': 'proba',
        'desc': ("The trade-off between precision and recall. Average precision (AP) "
                 "is the area under this curve - useful when classes are imbalanced."),
    },
    'PCA Explained Variance': {
        'fn': plot_pca_explained_variance, 'kind': 'matrix',
        'desc': ("How much of the total variance is captured as principal components "
                 "are added. Computed on the fully preprocessed feature matrix."),
    },
    'PCA 2D Projection': {
        'fn': plot_pca_2d, 'kind': 'matrix_labels',
        'desc': ("The preprocessed features projected onto their first two principal "
                 "components - a rough view of how separable the classes are."),
    },
    'Cumulative Gains Curve': {
        'fn': plot_cumulative_gains, 'kind': 'proba',
        'desc': ("If you act on the highest-scoring cases first, what share of the "
                 "positives do you capture? Further above the diagonal is better."),
    },
    'Calibration Plot': {
        'fn': plot_calibration, 'kind': 'proba',
        'desc': ("Compares predicted probabilities with observed frequencies. A "
                 "well-calibrated model sits on the diagonal."),
    },
    'Lift Curve': {
        'fn': plot_lift, 'kind': 'proba',
        'desc': ("How many times better than random the model is at finding a class, "
                 "as you widen the fraction of the sample you act on."),
    },
    'Silhouette Analysis': {
        'fn': plot_silhouette, 'kind': 'matrix',
        'desc': ("Helps determine the optimal number of clusters via a silhouette "
                 "score for each cluster count on the preprocessed features."),
    },
}


def draw(label, ax, result):
    """Convenience dispatch used by the app and tests: draw ``label`` for a
    :class:`pipeline.TrainResult` onto ``ax``."""
    spec = PLOTS[label]
    kind = spec['kind']
    if kind == 'proba':
        spec['fn'](ax, result.y_test, result.y_proba, result.model_classes, result.class_names)
    elif kind == 'matrix':
        spec['fn'](ax, result.transformed_X_test())
    elif kind == 'matrix_labels':
        spec['fn'](ax, result.transformed_X_test(), result.y_test, result.class_names)
    else:  # pragma: no cover - registry guards this
        raise ValueError(f"Unknown plot kind: {kind!r}")
