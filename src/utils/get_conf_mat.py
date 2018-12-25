import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')


def get_conf_mat(y_true, y_pred, labels):
    import matplotlib.pylab as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot = np.around(cm, 2)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels,
                cmap='Blues', annot=annot, lw=0.5)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_aspect('equal')

    return ax
