import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def get_data(path="data/Creditcard_txs.csv"):
    data = pd.read_csv(path)
    print("*** Loaded data from the file! ***")

    # Normalize the amount values so it fits [-1, 1] range
    data['Norm_Amt'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
    # Remove unnecessary columns from the data set
    data = data.drop(['Amount','Time'], axis=1)
    
    X = data.iloc[:, data.columns != 'Class']
    y = data.iloc[:, data.columns == 'Class']

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print("*** Generated Train & Test data sets! ***")
    # print("    X_train size: ", X_train.shape)
    # print("    y_train size: ", y_train.shape)
    # print("     X_test size: ", X_test.shape)
    # print("     y_test size: ", y_test.shape)

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, x_test, y_test = get_data()