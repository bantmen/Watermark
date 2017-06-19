import os
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.setdefaultencoding('utf8')

def unpickle(f):
    import pickle
    with open(f, 'rb') as fo:
        dct = pickle.load(fo)
    return dct

def save_dirs_as_png(root, in_dirs, out_dir):
    for i, unpack in enumerate(in_dirs):
        print i
        path = os.path.join(root, unpack)
        datasets = unpickle(path)
        for j in xrange(datasets['data'].shape[0]):
            im = datasets['data'][j,:].reshape((32,32,3), order='F').T
            im = np.rot90(im.T, k=3)
            label = datasets['labels'][j]
            scipy.misc.imsave('{}/{}/{}.png'.format(out_dir, label, 10000*i + j), im)

if __name__ == "__main__":
    root = "cifar-10-batches-py"

    trains = ['data_batch_1',
              'data_batch_2',
              'data_batch_3',
              'data_batch_4',
              'data_batch_5']
    save_dirs_as_png(root, trains, 'cifar_png/training')

    tests = ['test_batch']
    save_dirs_as_png(root, tests, 'cifar_png/testing')
