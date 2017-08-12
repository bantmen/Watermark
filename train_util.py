import os
from scipy import misc
import numpy as np


# generates (original_image, watermarked_image) pairs or returns the generated list if cache == True
def get_dataset(dataset_name, mode):
    dir_path, watermark_dir_path, cache_file = _get_dataset_info(dataset_name, mode)
    try:
        data = np.load('%s.npy' % cache_file)
    except IOError:
        data = list(_get_train_data_helper(dir_path, watermark_dir_path, dataset_name))
        np.save(cache_file, data)
    return data

def _get_train_data_helper(dir_path, watermark_dir_path, dataset_name):
    w = os.walk(dir_path)
    class_names = next(w)[1]
    for (class_dir_path, _, class_file_names), class_name in zip(w, class_names):
        for sample_file_name in class_file_names:
            # Load the original image
            sample_path = os.path.join(class_dir_path, sample_file_name)
            image = misc.imread(sample_path)
            # Load the watermarked image
            watermark_sample_path = os.path.join(
                watermark_dir_path, class_name, sample_file_name)
            watermark_image = misc.imread(watermark_sample_path)

            # cifar fix
            if dataset_name == 'cifar':
                if image.ndim != 3 or watermark_image.ndim != 3:
                    continue
                elif watermark_image.shape[2] == 2:
                    continue
                elif watermark_image.shape[2] == 4:
                    watermark_image = watermark_image[:, :, :3]

            yield image, watermark_image

def _get_dataset_info(name, mode):
    assert mode in ['training', 'testing']

    datasets = {
        'mnist': {
            'dir_path': './mnist_png/',
            'watermark_dir_path': './watermark_mnist_png_random/',
            'cache_file_prefix': './data_cache/mnist_cache_'
        },
        'cifar': {
            'dir_path': './cifar_png/',
            'watermark_dir_path': './watermark_cifar_png_random/',
            'cache_file_prefix': './data_cache/cifar_cache_'
        }
    }

    dir_path = os.path.join(datasets[name]['dir_path'], mode)
    watermark_dir_path = os.path.join(datasets[name]['watermark_dir_path'], mode)
    cache_file = datasets[name]['cache_file_prefix'] + mode
    return dir_path, watermark_dir_path, cache_file
