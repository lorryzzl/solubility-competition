from tdc.single_pred import ADME
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers

def get_data():
    """ Get the unprocessed solubility data as a dict mapping split names
    (train, valid, and test) to pandas dataframes. Each dataframe has a
    'Drug' column with the compound SMILES and a 'Y' column with the solubility. """
    data = ADME(name = 'Solubility_AqSolDB')
    return data.get_split(method = "scaffold")

def process_data(data):
    """ Takes as input the raw dataframe from get_data and a returns a tuple X, Y
    of the processed data. You can modify this function however you want to return
    whatever input data you need. For the X values, the function currently gets the
    Morgan Fingerprint for each molecule and returns a numpy array of those values
    with shape (N, n_bits), where N is the length of the dataset and n_bits is the
    number of fingerprint bits (currently 2048). The Y return value is a numpy array
    with shape (N, 1)"""

    # use RDKit to process the SMILES into molecules
    PandasTools.AddMoleculeColumnToFrame(data,smilesCol='Drug')
    # get the fingerprints
    mfp2_fps = rdFingerprintGenerator.GetFPs(list(data['ROMol']))
    # convert to numpy array
    X = np.asarray([ np.array(fp) for fp in mfp2_fps ])
    #comvert Y to numpy array
    Y = np.asarray(data.Y).reshape((-1,1))

    return X, Y

data = get_data()
# first train the model using the train data
train_data = data["train"]
X_train, Y_train = process_data(train_data)
split_name = "valid"
X_val, Y_val = process_data(data[split_name])
#X_val, X_test, Y_val, Y_test = train_test_split(X_split, Y_split, test_size=0.5, random_state=1)
print(Y_val.shape)
split_name = "test"
X_test, Y_test = process_data(data[split_name])

def get_lin_model(X, Y):
    """ Train a model with the X and Y data and return the trained model. Currently
    just a sklearn LinearRegression Model. """
    model = LinearRegression()
    model.fit(X, Y)
    return model


def get_ridge_model(X, Y, alpha):
    model = Ridge(alpha)
    model.fit(X, Y)
    return model


def get_nn_model():
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(32, activation='tanh'))
    model.add(layers.Dense(1))
    model.compile(loss='mse', optimizer="adam", metrics=['mse'])
    return model

def evaluate_ridge_model():
    best_r2 = 0
    for alpha in [1, 5, 10, 20, 50, 100]:
        model = get_ridge_model(X_train, Y_train, alpha)
        Y_train_pred = model.predict(X_train)
        Y_val_pred = model.predict(X_val)
        print(f"  Alpha: {alpha}")
        print(f"  Train MSE: {mean_squared_error(Y_train, Y_train_pred)}")
        print(f"  Train R2:  {r2_score(Y_train, Y_train_pred)}")
        print(f"  Validation MSE: {mean_squared_error(Y_val, Y_val_pred)}")
        print(f"  Validation R2:  {r2_score(Y_val, Y_val_pred)}")
        if r2_score(Y_val, Y_val_pred) > best_r2:
            best_r2 = r2_score(Y_val, Y_val_pred)
            best_alpha  = alpha
    print("Best Alpha is:", best_alpha)
    print("Best R-square is:", best_r2)
    Y_test_pred = model.predict(X_test)
    print("Test R2:", r2_score(Y_test, Y_test_pred))

def evaluate_nn_model():
    model = get_nn_model()
    model.fit(X_train, Y_train, epochs = 20)
    Y_train_pred = model.predict(X_train)
    print("Train R2:", r2_score(Y_train, Y_train_pred))
    Y_val_pred = model.predict(X_val)
    print("Val R2:", r2_score(Y_val, Y_val_pred))

def evaluate_nn_model_test():
    model = get_nn_model()
    model.fit(X_train, Y_train, epochs = 20)
    Y_test_pred = model.predict(X_test)
    print("Test R2:", r2_score(Y_test, Y_test_pred))


def evaluate_lin_model(model, X, Y):
    """ Prints the MSE and R2 for the model """
    Y_pred = model.predict(X)
    print(f"  MSE: {mean_squared_error(Y, Y_pred)}")
    print(f"  R2:  {r2_score(Y, Y_pred)}")

if __name__ == "__main__":
    data = get_data()

    # first train the model using the train data
    train_data = data["train"]
    X_train, Y_train = process_data(train_data)
    model = get_lin_model(X_train, Y_train)

    print("Training results:")
    evaluate_lin_model(model, X_train, Y_train)

    # now test the model on either the valid or test split
    # Only use the valid split when developing your model!
    # Once you use the test split, you must submit your score
    # immediately, regardless of its performance
    split_name = "valid"
    X_split, Y_split = process_data(data[split_name])
    print(f"Results for split {split_name}:")
    evaluate_lin_model(model, X_split, Y_split)




    



