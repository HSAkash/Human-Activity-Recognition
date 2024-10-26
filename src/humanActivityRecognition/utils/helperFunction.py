import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ensure import ensure_annotations
from humanActivityRecognition import logger
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


@ensure_annotations
def plot_loss_curves_history(
        history_path,
        save_loss_curves,
        save_accuracy_curves,
        save= False
    ):
    """ plot loss curves from history file
        
    Args:
        history_path (Path): path to history file
        save_loss_curves (Path): path to save loss curves
        save_accuracy_curves (Path): path to save accuracy curves
    """

    """
    history = {
        'epoch': [1, 2, 3, .....],
        'loss': [0.1, 0.2, 0.3, .....],
        'val_loss': [0.2, 0.3, 0.4, .....],
        'accuracy': [0.1, 0.2, 0.3, .....],
        'val_accuracy': [0.2, 0.3, 0.4, .....]
    }
    """

    # Load history
    history = pd.read_csv(history_path)
    loss = history['loss']
    val_loss = history['val_loss']

    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']

    epochs = range(len(history['epoch']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    if save:
        plt.savefig(save_loss_curves)
        plt.close()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    if save:
        plt.savefig(save_accuracy_curves)
        plt.close()



def compare_historys(
        history_paths: list,
        history_names: list,
        epochs: int,
        save: bool = False,
        save_loss_curves: Path = 'compare_historys_loss.png',
        save_accuracy_curves: Path = 'compare_historys_accuracy.png'
    ):
    """
    Compares multiple model historys.
    Args:
        history_paths: list of paths to history files
        history_names: list of names for the history files
        epochs: number of epochs
        save: whether to save the figure or not
        save_loss_curves: path to save loss curves
        save_accuracy_curves: path to save accuracy curves
    """
    # Create figure and axes
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].set_title('Loss')
    ax[1].set_title('Accuracy')
    ax[0].set_xlabel('Epochs')
    ax[1].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Accuracy')
    ax[0].grid()
    ax[1].grid()
    # Iterate over each history file
    for i, history_path in enumerate(history_paths):
        history = load_history(history_path)
        # Plot loss
        ax[0].plot(history.epoch, history.loss, label=f'{history_names[i]} training loss')
        ax[0].plot(history.epoch, history.val_loss, label=f'{history_names[i]} validation loss')
        # Plot accuracy
        ax[1].plot(history.epoch, history.accuracy, label=f'{history_names[i]} training accuracy')
        ax[1].plot(history.epoch, history.val_accuracy, label=f'{history_names[i]} validation accuracy')

    # Add legend
    ax[0].legend()
    ax[1].legend()
    # Save the figure
    if save:
        fig.savefig(save_loss_curves)
        logger.info(f"Loss curves saved at: {save_loss_curves}")
        fig.savefig(save_accuracy_curves)
        logger.info(f"Accuracy curves saved at: {save_accuracy_curves}")

    



# Our function needs a different name to sklearn's plot_confusion_matrix
@ensure_annotations
def make_confusion_matrix(
    y_true: np.array,
    y_pred: np.array,
    classes: list,
    figsize: tuple = (10, 10),
    text_size: int = 15,
    labelrotation: int = 90,
    norm: bool = False,
    savefig: bool = False,
    cmap: str = "YlGnBu",
    save_path: Path = None
    ):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
    y_true : np.array
        True labels of the data.
    y_pred : np.array
        Predicted labels of the data.
    classes : list
        List of class names.
    figsize : tuple
        Size of the plot (default is (10, 10)).
    text_size : int
        Size of text on the plot (default is 15).
    labelrotation : int
        Rotation of the labels on the plot (default is 90).
    norm : bool
        Whether to normalise the values or not (default is False).
    savefig : bool
        Whether to save the figure or not (default is False).
    cmap : str
        Type of color map to use (default is "YlGnBu").
    save_path : Path
        Path to save the figure (default is None).
    """

    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    """YlGnBu, Oranges, Blues"""
    cax = ax.matshow(cm, cmap=cmap) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), # create enough axis slots for each class
            yticks=np.arange(n_classes),
            xticklabels=labels, # axes will labeled with class names (if they exist) or ints
            yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.tick_params(axis='x', labelrotation=90)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig(save_path)
        logger.info(f"Confusion matrix saved at: {save_path}")
        

@ensure_annotations
def getPrescisionRecallF1(
        y_true: np.array,
        y_pred: np.array,
        class_names: list
    ) -> str:
    """ get precision, recall, f1 score for each class
    
    Args:
        y_true (np.array): true labels
        y_pred (np.array): predicted labels
        class_names (list): class names
    """
    confusion = confusion_matrix(y_true, y_pred)

    # Calculate precision for each class
    precision = np.diagonal(confusion) / np.sum(confusion, axis=0)

    # Calculate recall for each class
    recall = np.diagonal(confusion) / np.sum(confusion, axis=1)

    # Calculate F1 score for each class
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Calculate accuracy class wise
    accuracy_class_wise = np.diagonal(confusion) / np.sum(confusion, axis=1)


    # Calculate accuracy
    accuracy = np.sum(np.diagonal(confusion)) / np.sum(confusion)

    # Calculate avg precision
    avg_precision = np.mean(precision)

    # Calculate avg recall
    avg_recall = np.mean(recall)

    # Calculate avg F1 score
    avg_f1_score = np.mean(f1_score)


    print_data = ""
    header = f"""{'Task':^20}   {'precision':^10}   {'recall':^10}  {'f1':^10}  {'Accuracy':^10}"""
    print_data += header
    print_data += "\n" + f"""{'_'*len(header)}"""
    for i in range(len(class_names)):
        c_name, prec, rec, f1, accur = class_names[i], precision[i], recall[i], f1_score[i], accuracy_class_wise[i]
        print_data += "\n" + f"""{c_name:^20}   {prec:^10.4f}   {rec:^10.4f}   {f1:^10.4f}   {accur:^10.4f}"""
    print_data += "\n" + f"""{'_'*len(header)}"""
    print_data += "\n" + f"""{'Average':^20}   {np.mean(precision):^10.4f}   {np.mean(recall):^10.4f}   {np.mean(f1_score):^10.4f}   {accuracy:^10.4f}"""

    return print_data