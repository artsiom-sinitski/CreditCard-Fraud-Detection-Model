import os.path

import numpy as np
import pandas as pd

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout

import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from prepData import get_data
from configSession import configure_session


def build_NN_model(dims):
    model = Sequential([
            Dense(units=16, input_dim=dims, activation='relu'),
            Dense(units=24, activation='relu'),
            Dropout(rate=0.5),
            Dense(units=20, activation='relu'),
            Dense(units=24, activation='relu'),
            Dense(1, activation='sigmoid')
    ])
    #model.summary()
    return model


def train_model(X_train, y_train, epochs, batches, X_test, y_test):
    model = build_NN_model(X_test.shape[1])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=batches, verbose=2)

    #score = model.evaluate(X_test, y_test)
    #print("Model score: ", score)
    return model


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    #cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\nNormalized confusion matrix")
    else:
        print("\nConfusion matrix, without normalization")

    print(cm)

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
    return ax


if __name__ == '__main__':
    # Configure Keras session
    configure_session()

    X_train, y_train, X_test, y_test = get_data()
    print("\nTrain/Test data length: %i / %i\n" % (len(X_train), len(X_test)))

    epochs = 5
    batches = 15
    model_file = "models/model-%d-%d.h5" % (epochs, batches)

    if os.path.exists(model_file):
        model = load_model(model_file)
        print("\n***** Model loaded! *****")
    else: # if the model file doesn't exist train it
        model = train_model(X_train, y_train, epochs, batches, X_test, y_test)
        # Save model to use for classification later on
        model_name = 'models/model-%d-%d' % (epochs, batches)
        model.save(model_name + '.h5')

    title = '(epochs=%d, batch_size=%d)' % (epochs, batches)

    # Test our model on data that has been seen
    # (training data set) and unseen (test data set)
    print("\n*** Evaluation for %s model ***" % title)
    loss, acc = model.evaluate(X_train, y_train, verbose=2)
    print("Train accuracy: %.2f%%" % (acc*100))
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print(" Test accuracy: %.2f%%" % (acc*100))

    # graph the confusion matrix
    y_pred = model.predict(X_test)
    y_test = pd.DataFrame(y_test)
    cfx_matrix = confusion_matrix(y_test, y_pred.round())

    plot_confusion_matrix(cfx_matrix, classes=[0,1])
