# Test trained model with selected weights on real image set
from evaluate import *

# Directories
parent_dir = "/data/aims/sychoi/"
results_dir = parent_dir + "results_MoS2/"

# Model directory
model_dir = results_dir + "BEST_20201027_1vac_4800_1200/" 

# Where the real images for testing are located
simImage_dir = parent_dir + "pa74_52mrad_25pA_16us/0deg_1/"
# data_dir = parent_dir + realImage_dir

# Load model and weights
model_fn = model_dir + "model.json"
model_weights_fn = model_dir + "weights.h5"
input_file = simImage_dir + "input.tiff"
l_shape = (256, 256)
stride = (64, 64)
avg = 0
plot = True
save_data = True
save_dir = results_dir
prefix = "test_KPS_0_SV1_plotsTrue"

prediction = evaluate(model_fn, model_weights_fn, input_file, l_shape, stride, avg=avg, plot=plot, save_data=save_data, save_dir=save_dir, prefix=prefix)
