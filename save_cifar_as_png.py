import os
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np

import sys

reload(sys)
sys.setdefaultencoding('utf8')

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


if __name__ == "__main__":
    file = "/Users/cemanil/Projects/Watermark/cifar-10-batches-py"
    unpacks = ['data_batch_1',
              'data_batch_2',
              'data_batch_3',
              'data_batch_4',
              'data_batch_5',
              'test_batch']

    for i, unpack in enumerate(unpacks):
        print i
        path = os.path.join(file, unpack)
        datasets = unpickle(path)
        for j in range(datasets['data'].shape[0]):
            im = datasets['data'][j,:].reshape((32,32,3), order='F').T
            im = np.rot90(im.T, k=3)
            scipy.misc.imsave('cifar_png/{}.png'.format(10000*i + j), im)


