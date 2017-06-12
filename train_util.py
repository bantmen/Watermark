import os
from scipy import misc

def get_train_data(mnist_dir_path, watermark_dir_path):
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
