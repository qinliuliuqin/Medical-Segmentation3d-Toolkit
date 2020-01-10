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
# 1) NUM:     partition to specified number of blocks, set partition_by_num = [num_x, num_y, num_z]
# 2) SIZE:    partition to blocks with specified size (unit: mm), set partition_by_size = [size_x, size_y, size_z]
# 4) DISABLE: no partition
__C.general.partition_type = 'DISABLE'

# if partition type = 'NUM'
__C.general.partition_by_num = [1, 1, 1]

# if partition type = 'SIZE'
__C.general.partition_by_size = [35, 35, 35]

# padding size (unit: mm) of each partition block
__C.general.partition_padding_size = [0, 0, 0]