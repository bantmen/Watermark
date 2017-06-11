import os

def maybe_make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_watermark(in_path, out_path):
    command_str = '''convert %s -pointsize 5 -draw \
        "fill black  text 0,12 'Copyright' \
        fill white  text 1,11 'Copyright'" \
        %s'''
    os.system(command_str % (in_path, out_path))

mnist_training_dir = './mnist_png/training/'
out_dir = './watermark_mnist_png/training'
maybe_make_dir(out_dir)

input_walk = os.walk(mnist_training_dir)

digit_dir_names = next(input_walk)[1]

for (digit_dir_path, _, digit_files), digit_dir_name in zip(input_walk, digit_dir_names):
    out_digit_dir = os.path.join(out_dir, digit_dir_name)
    maybe_make_dir(out_digit_dir)
    for sample_file_name in digit_files:
        in_digit_path = os.path.join(mnist_training_dir, digit_dir_name, sample_file_name)
        out_digit_path = os.path.join(out_digit_dir, sample_file_name)
        create_watermark(in_digit_path, out_digit_path)
