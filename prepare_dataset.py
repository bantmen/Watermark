import os
import random
import sys

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

in_dir = sys.argv[1]
out_dir = sys.argv[2]

maybe_make_dir(out_dir)

input_walk = os.walk(in_dir)

class_dir_names = next(input_walk)[1]

for (class_dir_path, _, class_files), class_dir_name in zip(input_walk, class_dir_names):
    print 'Preparing ', class_dir_name
    out_class_dir = os.path.join(out_dir, class_dir_name)
    maybe_make_dir(out_class_dir)
    for sample_file_name in class_files:
        in_class_path = os.path.join(in_dir, class_dir_name, sample_file_name)
        out_class_path = os.path.join(out_class_dir, sample_file_name)
        create_randomized_watermark(in_class_path, out_class_path)
