import sys

sys.path.insert(0, '1_preprocessing')
from make_data import *

sys.path.insert(0, '2_trainer')
from input_data import *
from models import *
from training_utilities import *

sys.path.insert(0, '3_evaluate')
from evaluate import *

from os import makedirs



"""
1. Preprocessing
"""
# Preprocessing the data before feeding them to the network
input_dir = "~/ResUNet_final/data/WSeTe/simulated/"
data_dirs = [str(i) for i in range(2)]
label_list = ["Se"]
parsed_dir_name = 'parsed_label_Se'
ftype = '.tiff'

l_shape = (256, 256)
stride = (128, 128)
one_pickle = False
tr_bs = 200
ts_bs = 20
ones_percent = .00
tol = 0.05
show_plots = False

# Create augments of original images
create_augments(input_dir, data_dirs, ftype)

# Generate data and put them in their respective directories
make_data(input_dir, label_list, data_dirs, l_shape, stride, ftye, \
        parsed_dir_name=parsed_dir_name, tr_bs=tr_bs, ts_bs=ts_bs, ones_percent=ones_percent, \
        tol=tol, show_plots=show_plots, one_save=one_pickle)

"""
2. Training
"""

# Define all variables
parent_dir = '~/ResUNet_final/'
data_dir = parent_dir + 'data/WSeTe/simulated/parsed_label_Se/'
sess_name = 'Se'
N = 256
k_fac = 16
nb_classes = 2

sess_dir = parent_dir + "results/" + sess_name + "/"
makedirs(sess_dir, exist_ok=True)

model_fn = sess_dir + "model.json"
model_weights_fn = sess_dir + "weights.h5"
diagnostics_fn = sess_dir + "diagnostics.dat"

# Construct the model and setup diagnostics
model = construct_model(N, k_fact, nb_classes, sess_dir, model_fn, model_weights_fn)
step = setup_diagnostics(diagnostics_fn)

# Start training
train(step, data_dir, N, nb_classes, model, diagnostics_fn, model_weights_fn)


"""
3. Evaluation
"""
# Define all variables
results_dir = sess_dir
data_dir = parent_dir + 'data/WSeTe/simulated/3/'

model_fn = results_dir + "model.json"
model_weights_fn = results_dir + "weights.h5"
input_file = data_dir + "input.tiff"
l_shape = (256, 256)
stride = (64, 64)
avg = 0
plot = True
save_data = False
save_dir = "./"
prefix = "Se"

# %matplotlib notebook
# store results of the evaluation process
prediction = evaluate(model_fn, model_weights_fn, input_file, l_shape, stride, avg=avg, \
        plot=plot, save_data=save_data, save_dir=save_dir, prefix=prefix)

# May decide to save this data, too
label_file_list = [data_dir + "label_{}.tiff".format("Se")]
tol = .5
nconvs - 0
r = 3
TN = 0
verbose = True

# %matplotlib notebook
# compute and display the accuracy
TP, FP, FN, TN, recall, precision, F1, bal_acc = calc_accuracy(prediction, label_file_list, tol=tol, \
        nconvs=nconvs, r=r, TN=TN, plot=plot, save_data=save_data, save_dir=save_dir, prefix=prefix, verbose=verbose)
