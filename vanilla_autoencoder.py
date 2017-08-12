from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from scipy.misc import toimage
import gflags
import sys

from train_util import get_dataset


gflags.DEFINE_string('dataset_name', '', 'Dataset name e.g. mnist cifar')
gflags.DEFINE_integer('encoding_dim', 128, 'Dimension of hidden unit')
gflags.DEFINE_integer('num_epochs', 5, 'Number of epochs to train for')
gflags.DEFINE_integer('batch_size', 256, 'Size of mini batches to train with')
gflags.DEFINE_float('validation_percentage', 0.2, 'Will be used to determine validation split size')

FLAGS = gflags.FLAGS
FLAGS(sys.argv)

if FLAGS.dataset_name == 'mnist':
    train_dim_x = train_dim_y = 28
    FLATTENED_DIM = train_dim_x * train_dim_y
    TRAIN_SHAPE = (train_dim_x, train_dim_y)
elif FLAGS.dataset_name == 'cifar':
    train_dim_x = train_dim_y = 32
    FLATTENED_DIM = train_dim_x * train_dim_y * 3
    TRAIN_SHAPE = (train_dim_x, train_dim_y, 3)
else:
    raise Exception('Wrong dataset name')


def extract_x(data):
    return np.array([t[1] for t in data])

def extract_y(data):
    return np.array([t[0] for t in data])

def preprocess(x_or_y):
    x_or_y = x_or_y.astype('float32') / 255.
    x_or_y = x_or_y.reshape((len(x_or_y), np.prod(x_or_y.shape[1:])))
    return x_or_y

def prepare_data():
    """ returns preprocessed [x_train, y_validation, x_test, y_validation] """
    data = get_dataset(FLAGS.dataset_name, 'training')
    num_validation = int(len(data) * FLAGS.validation_percentage)
    train, validation = data[:-num_validation], data[-num_validation:]

    x_train, y_train = extract_x(train), extract_y(train)
    x_validation, y_validation = extract_x(validation), extract_y(validation)

    return map(preprocess, [x_train, y_train, x_validation, y_validation])

encoding_dim = FLAGS.encoding_dim

input_img = Input(shape=(FLATTENED_DIM,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(FLATTENED_DIM, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

x_train, y_train, x_validation, y_validation = prepare_data()

autoencoder.fit(x_train, y_train,
                epochs=FLAGS.num_epochs,
                batch_size=FLAGS.batch_size,
                shuffle=True,
                validation_data=(x_validation, y_validation))

predictions = autoencoder.predict(x_validation[0:50], batch_size=50)

if __name__ == '__main__':
    for i, pred in enumerate(predictions):
        x = x_validation[i].reshape(TRAIN_SHAPE)
        y = y_validation[i].reshape(TRAIN_SHAPE)
        pred = pred.reshape(TRAIN_SHAPE)

        toimage(np.concatenate([x, y, pred])).show()

        raw_input('Hit enter to continue\n')
