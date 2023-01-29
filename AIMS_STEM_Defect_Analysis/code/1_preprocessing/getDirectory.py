import os
import random

os.chdir('../../data/MoS2/simulated_processed/')
all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]

# Total number of subdirectories
tot = len(all_subdirs)

# Number of test directories to select
k = round(tot*0.30)

test_dir = random.sample(all_subdirs, k)
train_dir = [t for t in all_subdirs if t not in test_dir]

print("Test dirs: ")
print(*test_dir)

print("Train dirs: ")
print(*train_dir)
