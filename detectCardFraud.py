import os
import numpy as np
import pandas as pd

from keras.models import load_model
from sklearn.preprocessing import StandardScaler

import configSession

    
def get_data(path="data/Creditcard_txs-PRODUCTION.csv"):
    path = "data/Creditcard_txs.csv"
    data = pd.read_csv(path)
    print("*** Data loaded from the file! ***")

    data['Norm_Amt'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1)) 
    data = data.drop(['Amount', 'Time'], axis=1)
    data = data.iloc[:, data.columns != 'Class']

    return data


if __name__ == '__main__':
    epochs = 5
    batches = 15
    model_name = "model-%d-%d" % (epochs, batches)
    model_file = "models/" + model_name + ".h5"

    if os.path.exists(model_file):
        model = load_model(model_file)
        print("\n*** Model loaded! ***")
    else:
        print("\nCan't find  %s model, train it first using 'trainModel.py %d %d'" % (epochs, batches))

    # title = '(epochs=%d, batch_size=%d)' % (epochs, batches)

    # Load the data set from a file
    X = np.array(get_data())

    # preds = model.predict(X, batch_size=batches, verbose=1)
    fraud_predictions = model.predict_classes(X, verbose=0)
    fraud_tx_idxs = np.where(fraud_predictions == 1)

    # convert our results to a list and add 2 to the indicies, 
    # because the csv file has a heading and its indexing is 1-based
    result = list(fraud_tx_idxs[0]+2)

    print("\n>>> %i txs out of %i were detected as fraudulent! <<<" % (len(result), X.shape[0]))
    print("\nFraudelent transactions indicies are:")
    print(result)
