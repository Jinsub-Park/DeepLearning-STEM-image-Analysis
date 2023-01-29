# Test trained model with selected weights on real image set
from evaluate import *

# Directories
parent_dir = "/home/sychoi/"
results_dir = "/data/aims/sychoi/results_MoS2/"

# Model directory
model_dir = results_dir + "a4_sv2/" 

# Where the real images for testing are located
realImage_dir = "/data_enpl/MoS2/tif/testOnImage0/"
data_dir = parent_dir + realImage_dir

# Load model and weights
model_fn = model_dir + "model.json"
model_weights_fn = model_dir + "weights.h5"
input_file = data_dir + "input.tif"
l_shape = (512, 512)
stride = (64, 64)
avg = 0
plot = True
save_data = False
save_dir = realImage_dir
prefix = "a4_sv2"

prediction = evaluate(model_fn, model_weights_fn, input_file, l_shape, stride, avg=avg, plot=plot, save_data=save_data, save_dir=save_dir, prefix=prefix)
