import matplotlib.pyplot as plt
import os



def get_history(history, figsize=(15,10), 
                 plot={"Accuracy":['accuracy','val_accuracy'], 'Loss':['loss', 'val_loss']}, save=None, offset=0):
    """
    Show history
    args:
        history: history
        figsize: fig size
        plot: list of data to plot : {<title>:[<metrics>,...], ...}
    """
    for title,curves in plot.items():
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel('Epoch')
        for c in curves:
            plt.plot(history.history[c])
        plt.legend(curves, loc='upper left')
        plt.xlim([offset, len(history.history[curves[0]])])

    if save :
        plt.savefig(os.path.join(save, 'history_training.png'))