import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('temp.png')
    image = Image.open('temp.png')
    os.remove('temp.png')
    test = np.array(image)
    return test

def plot_metrics(metrics, acc, classes,
                title="Validation metrics",
                cmap=None):
    
    colors = [(1,1,1), (1,1,1), (1,1,1)]
    cm = LinearSegmentedColormap.from_list("white",colors,N=1)
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(metrics, interpolation=None, cmap=cm)
    # We want to show all ticks...
    ax.set(xticks=np.arange(metrics.shape[1]),
           yticks=np.arange(metrics.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=["Recall", "Precision"],
           title="Accuracy over slices = "+str(int(10000*acc)/100)+"%")


    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    for i in range(metrics.shape[0]):
        for j in range(metrics.shape[1]):
            ax.text(j, i, format(metrics[i, j], fmt),
                    ha="center", va="center",
                    color="black")
    fig.tight_layout()
    fig.savefig('temp.png')
    image = Image.open('temp.png')
    os.remove('temp.png')
    test = np.array(image)
    return test
