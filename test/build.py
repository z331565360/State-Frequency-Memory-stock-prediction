import time
import warnings
import numpy as np
import keras
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from itosfm import ITOSFM
from keras.models import Sequential

warnings.filterwarnings("ignore")

#Load data from data file, and split the data into training, validation and test set
def load_data(filename, step):
    #load data from the data file
    day = step
    data = np.load(filename)
    data = data[:, :]
    gt_test = data[:,day:]
    #data normalization
    max_data = np.max(data, axis = 1)
    min_data = np.min(data, axis = 1)
    max_data = np.reshape(max_data, (max_data.shape[0],1))
    min_data = np.reshape(min_data, (min_data.shape[0],1))
    data = (2 * data - (max_data + min_data)) / (max_data - min_data)
    #dataset split
    train_split = round(0.8 * data.shape[1])
    val_split = round(0.9 * data.shape[1])
    
    x_train = data[:,:train_split]
    y_train = data[:,day:train_split+day]
    x_val = data[:,:val_split]
    y_val = data[:,day:val_split+day]
    x_test = data[:,:-day]
    y_test = data[:,day:]
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

    return [x_train, y_train, x_val, y_val, x_test, y_test, gt_test, max_data, min_data]

#build the model
def build_model(layers, freq, learning_rate):
    model = Sequential()

    model.add(ITOSFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        freq_dim = freq,
        return_sequences=True))

    start = time.time()
    
    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss="mse", optimizer="rmsprop")

    print "Compilation Time : ", time.time() - start
    return model