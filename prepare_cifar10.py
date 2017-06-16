import os
import random

def maybe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_watermark(in_path, out_path):
    command_str = '''convert %s -pointsize 5 -draw \
        "fill black  text 0,12 'Copyright' \
        fill white  text 1,11 'Copyright'" \
        %s'''
    os.system(command_str % (in_path, out_path))

def create_randomized_watermark(in_path, out_path):
    command_str = '''convert %s -pointsize 5 \
        -gravity %s -draw \
        "fill black  text 0,12 'Copyright' \
        fill white  text 1,11 'Copyright'" \
        %s'''
    gravity_options = ['East', 'South', 'West', 'North', 'SouthEast', 'SouthWest', 'NorthEast', 'NorthWest', 'Center']
    pick = random.randint(0, len(gravity_options) - 1)
    os.system(command_str % (in_path, gravity_options[pick], out_path))

mnist_training_dir = './cifar_png/training/'
out_dir = './cifar_mnist_png_random/training'
maybe_make_dir(out_dir)

input_walk = os.walk(mnist_training_dir)

digit_dir_names = next(input_walk)[1]

for (digit_dir_path, _, digit_files), digit_dir_name in zip(input_walk, digit_dir_names):
    out_digit_dir = os.path.join(out_dir, digit_dir_name)
    maybe_make_dir(out_digit_dir)
    for sample_file_name in digit_files:
        in_digit_path = os.path.join(mnist_training_dir, digit_dir_name, sample_file_name)
        out_digit_path = os.path.join(out_digit_dir, sample_file_name)
        create_randomized_watermark(in_digit_path, out_digit_path)
