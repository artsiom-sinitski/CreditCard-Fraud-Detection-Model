import keras
import numpy as np
import pandas as pd
from runpy import run_module

from keras.models import Sequential
from keras.layers import Dense, Dropout

from prepData import get_data
import configSession


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


if __name__ == '__main__':
    # Configure Keras session first
    configSession.configure_session()

    X_train, y_train, X_test, y_test = get_data()
    print("\nTrain/Test data length: %i / %i\n" % (len(X_train), len(X_test)))

    epochs = 5
    batches = 15
    model = train_model(X_train, y_train, epochs, batches, X_test, y_test)

    # Save model to use for classification later on
    mdl_name = 'models/model-%d-%d' % (epochs, batches)
    model.save(mdl_name + '.h5')

    title = '(epochs=%d, batch_size=%d)' % (epochs, batches)

    # Test our model on data that has been seen
    # (training data set) and unseen (test data set)
    print("\n*** Evaluation for %s model ***" % title)
    loss, acc = model.evaluate(X_train, y_train, verbose=2)
    print("Train accuracy: %.2f%%" % (acc*100))
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print(" Test accuracy: %.2f%%" % (acc*100))
