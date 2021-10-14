import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time
import os


def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES):

    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer")]

    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary()

    model_clf.compile(loss=LOSS_FUNCTION,
                optimizer=OPTIMIZER,
                metrics=METRICS)

    return model_clf ## <<< untrained model

def get_unique_filename(filename):
    unique_filename = time.strftime(f"%d/%m/%Y %H:%M_{filename}")
    return unique_filename

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)

def save_plot(loss_acc,plot_name,plot_dir):
    unique_filename = get_unique_filename(plot_name)
    path_to_plot = os.path.join(plot_dir, unique_filename)
    print(f"path of plot :{path_to_plot}")
    pd.DataFrame(loss_acc).plot(figsize=(10,7))
    plt.grid(True)
    plt.savefig(path_to_plot)
    plt.show()

