# Watermark

### For working with MNIST

Download and unzip https://github.com/myleott/mnist_png/blob/master/mnist_png.tar.gz

Make sure vanilla mnist data is at `./mnist_png/training/`

Run `python prepare_dataset.py ./mnist_png/training/ ./watermark_mnist_png_random/training`

Watermarked data will be placed under `./watermark_mnist_png/training`

Example output: https://drive.google.com/file/d/0B7dTwsaPqx5nZ2xLVEFLdXlvY2c/view?usp=sharing

### For working with CIFAR10

Download and unzip https://drive.google.com/open?id=0B-ujUXlVFw1zajF0eThCYjhYc0k

Make sure vanilla cifar data is at `./cifar_png/training/`

Run `python prepare_dataset.py ./cifar_png/training/ ./cifar_mnist_png_random/training`

Watermarked data will be placed under `./watermark_cifar_png/training`

