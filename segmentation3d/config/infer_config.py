from easydict import EasyDict as edict

__C = edict()
cfg = __C

##################################
# general parameters
##################################
__C.general = {}

# Pick the largest connected component (cc) in segmentation
# Options:
# 1) True: pick the largest connected component
# 2) False: do not pick the largest connected component
__C.general.pick_largest_cc = True

# Remove small connected component (cc) in segmentation
# Options:
# 1) 0: Disable
# 2) a numerical number larger than 0: the threshold size of connected component
__C.general.remove_small_cc = 0

# partition type in the inference stage
# Options:
# 1) SIZE:    partition to blocks with specified size (unit: mm), set partition_size = [size_x, size_y, size_z]
# 2) DISABLE: no partition
__C.general.partition_type = 'SIZE'

# if partition type = 'SIZE'
__C.general.partition_size = [32, 32, 32]