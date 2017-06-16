import os
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import sys

reload(sys)
sys.setdefaultencoding('utf8')

def unpickle(f):
    import pickle
    with open(f, 'rb') as fo:
        dct = pickle.load(fo)
    return dct


if __name__ == "__main__":
    file = "/Users/cemanil/Projects/Watermark/cifar-10-batches-py"
    trains = ['data_batch_1',
              'data_batch_2',
              'data_batch_3',
              'data_batch_4',
              'data_batch_5']

    tests = ['test_batch']

    for i, unpack in enumerate(trains):
        print i
        path = os.path.join(file, unpack)
        datasets = unpickle(path)
        for j in range(datasets['data'].shape[0]):
            im = datasets['data'][j,:].reshape((32,32,3), order='F').T
            im = np.rot90(im.T, k=3)
            label = datasets['labels'][j]
            scipy.misc.imsave('cifar_png/training/{}/{}.png'.format(label, 10000*i + j), im)

    for i, unpack in enumerate(tests):
        print i
        path = os.path.join(file, unpack)
        datasets = unpickle(path)
        for j in range(datasets['data'].shape[0]):
            im = datasets['data'][j,:].reshape((32,32,3), order='F').T
            im = np.rot90(im.T, k=3)
            label = datasets['labels'][j]
            scipy.misc.imsave('cifar_png/testing/{}/{}.png'.format(label, 50000 + j), im)

