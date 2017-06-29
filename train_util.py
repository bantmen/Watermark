import os
from scipy import misc
import numpy


# generates (original_image, watermarked_image) pairs or returns the generated list if cache == True
def get_train_data(mnist_dir_path='./mnist_png/training/', watermark_dir_path='./watermark_mnist_png_random/training', cache_file=None):
    if not cache_file:
        return _get_train_data_helper(mnist_dir_path, watermark_dir_path)
    try:
        data = numpy.load('%s.npy' % cache_file)
    except IOError:
        data = list(_get_train_data_helper(mnist_dir_path, watermark_dir_path))
        numpy.save(cache_file, data)
    return data

def _get_train_data_helper(mnist_dir_path, watermark_dir_path):
    w = os.walk(mnist_dir_path)
    digit_dir_names = next(w)[1]
    for (digit_dir_path, _, digit_file_names), digit_dir_name in zip(w, digit_dir_names):
        for digit_file_name in digit_file_names:
            mnist_digit_path = os.path.join(digit_dir_path, digit_file_name)
            watermark_digit_path = os.path.join(
                watermark_dir_path, digit_dir_name, digit_file_name)
            mnist_image = misc.imread(mnist_digit_path)
            watermark_image = misc.imread(watermark_digit_path)
            yield mnist_image, watermark_image
