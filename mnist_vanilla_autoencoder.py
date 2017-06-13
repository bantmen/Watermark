from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from scipy.misc import toimage

from train_util import get_train_data

def extract_x(data):
    return np.array([t[1] for t in data])

def extract_y(data):
    return np.array([t[0] for t in data])

def preprocess(x_or_y):
    x_or_y = x_or_y.astype('float32') / 255.
    x_or_y = x_or_y.reshape((len(x_or_y), np.prod(x_or_y.shape[1:])))
    return x_or_y

# returns preprocessed [x_train, y_validation, x_test, y_validation]
def get_data():
    data = get_train_data(cache=True)
    num_validation = int(len(data) * 0.2)
    train, validation = data[:-num_validation], data[-num_validation:]

    x_train, y_train = extract_x(train), extract_y(train)
    x_validation, y_validation = extract_x(validation), extract_y(validation)

    return map(preprocess, [x_train, y_train, x_validation, y_validation])

encoding_dim = 128

input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

x_train, y_train, x_validation, y_validation = get_data()

autoencoder.fit(x_train, y_train,
                epochs=15,
                batch_size=256,
                shuffle=True,
                validation_data=(x_validation, y_validation))

predictions = autoencoder.predict(x_validation[0:50], batch_size=50)

if __name__ == '__main__':
    for i, pred in enumerate(predictions):
        toimage(pred.reshape(28, 28)).show()
        toimage(x_validation[i].reshape(28, 28)).show()
        raw_input('Hit enter to continue')
