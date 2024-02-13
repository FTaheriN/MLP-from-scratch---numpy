import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_conf_matrix(y, y_hat, label_names):
    y = np.argmax(y, axis=1)
    y_hat = np.argmax(y_hat, axis=1)
    # cm = confusion_matrix(y, y_hat)
    disp = ConfusionMatrixDisplay.from_predictions(y, y_hat,
                              display_labels=label_names, xticks_rotation='vertical', colorbar=False)
    disp.plot()
    plt.show()
    
    return 