from input_data import *
from models import *
from training_utilities_custom import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

# Will be training for multiple sessions, that's how this system works (to label them properly)
# For simple materials like MoS2, may be possible to do it all in one go
# Need to reconfigure code for that
parent_dir = '/data/aims/sychoi/'
data_dir   = parent_dir +'im_data/MoS2/numpyDir/b1_4.77:1/parsed_label_2vacancy/'
sess_name  = 'b1_4.77:1_SV2'

# pixel width & height of image (assume square)
N          = 256    

# factor that describes how many channels we want per layer in FCN
# default value per layer * k_fac
k_fac      = 4

# of labels (defects) that we are learning at once
nb_classes = 2

# total number of steps to train on
num_steps  = 400

# Create directory to store data
from os import makedirs
sess_dir = "/data/aims/sychoi/results_MoS2/" + sess_name + "/"
makedirs(sess_dir, exist_ok=True)
checkpoint_filepath = sess_dir + 'checkpoint/'
makedirs(checkpoint_filepath, exist_ok=True)

model_weights_fn = sess_dir + "weights.h5"
model_fn         = sess_dir + "model.json"
diagnostics_fn   = sess_dir + "diagnostics.dat"

# Create model
model = construct_model(N, k_fac, nb_classes, sess_dir, model_fn, model_weights_fn)
step = setup_diagnostics(diagnostics_fn)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_filepath,
        save_weights_only = False,
        monitor = 'val_acc',
        mode = 'max',
        save_best_only = False)

callbacks_list = [checkpoint]


# Train
train(checkpoint_filepath, callbacks_list, step, data_dir, N, nb_classes, model, diagnostics_fn, model_weights_fn, num_steps=num_steps, plots=False)

