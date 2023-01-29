import os
import sys
import random
from make_data import *

input_dir = '/data/aims/sychoi/git_file_trans/STEM-Deep-Learning/New_Images/'
save_dir = '/data/aims/sychoi/im_data/MoS2/increasedSV2_74/'

# Get current working directory
# curDir = os.getcwd()

# Change directory to input_dir to select subdirectories to divide training & testing set
os.chdir(input_dir)

# List all subdirectories
all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]

# Total number of subdirectories
tot = len(all_subdirs)

# Number of testing directories to choose from
k = 

# Train and test directories
# set seed for reproducibility
random.seed(1)
train_dirs = random.sample(all_subdirs, k)
test_dirs = [t for t in all_subdirs if t not in train_dirs]

# Change directory back to previous directory
os.chdir(curDir)

'''
train_dirs = ["0", "12", "20", "17"]
test_dirs = ["8"]
'''
# total label_list = ["1vacancy", "2vacancy"]
label_list = ["1vacancy"]
parsed_dir_name = 'parsed_label_2vacancy_fix_code'
ftype = '.tiff'
 
l_shape = (256, 256)
stride = (36, 36)

one_pickle=False 
tr_fsize = 2000
ts_fsize = 400

tol = 0.05

ones_percent = 0.

# create_augments(input_dir, train_dirs, ftype)

make_data(input_dir, save_dir, train_dirs, label_list, l_shape, stride, ftype, parsed_dir_name=parsed_dir_name, \
        prefix="train", AUG=True, tol=tol, ones_pcent=ones_percent, one_save=one_pickle, fsize=tr_fsize)
make_data(input_dir, save_dir, test_dirs, label_list, l_shape, stride, ftype, parsed_dir_name=parsed_dir_name, \
        prefix="test", AUG=False, tol=tol, ones_pcent=ones_percent, one_save=one_pickle, fsize=ts_fsize)

import numpy as np

parsed_fn = save_dir + parsed_dir_name + "/test_00000.p"
check_data(parsed_fn, l_shape=l_shape)

sys.stdout = open("fix_code_2vac.txt", "w")
print("Training directories are: ")
print(*train_dirs)
print("\nTesting directories are: ")
print(*test_dirs)
sys.stdout.close()
