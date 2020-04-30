from easydict import EasyDict as edict

__C = edict()
cfg = __C

##################################
# general parameters
##################################
__C.general = {}

# Run segmentation in single scale mode
# Options:
# 1) coarse: only run the coarse segmentation model
# 2) fine: only run the fine segmentation model
# 3) DISABLE: run the coarse model first and then run the fine model
__C.general.single_scale = 'DISABLE'

##################################
# Coarse model parameters
##################################
__C.coarse = {}

# the folder name containing the coarse model
__C.coarse.model_name = 'coarse'

# Pick the largest connected component (cc) in segmentation
# Options:
# 1) True: pick the largest connected component
# 2) False: do not pick the largest connected component
__C.coarse.pick_largest_cc = True

# Remove small connected component (cc) in segmentation
# Options:
# 1) 0: Disable
# 2) a numerical number larger than 0: the threshold size of connected component
__C.coarse.remove_small_cc = 0

# partition type in the inference stage
# Options:
# 1) SIZE:    partition to blocks with specified size (unit: mm), set partition_size = [size_x, size_y, size_z]
# 2) DISABLE: no partition
__C.coarse.partition_type = 'DISABLE'

# if partition type = 'SIZE', set the partition size (unit: mm).
# it is recommended to set this value as the same with the physical cropping size in the training phase
__C.coarse.partition_size = [51.2, 51.2, 51.2]

# the moving stride of the partition window. If set it as the same with the partition size, there will be no overlap
# between the partition windows. Otherwise, the value of the overlapped area will be averaged.
# it is recommended to set this value as 1/4 of the partition size in order to avoid the apparent in-consistence between
# different partition window.
__C.coarse.partition_stride = [51.2, 51.2, 51.2]

##################################
# Fine model parameters
##################################

__C.fine = {}

# the name of the folder containing the coarse model
__C.fine.model_name = 'fine'

# Pick the largest connected component (cc) in segmentation
# Options:
# 1) True: pick the largest connected component
# 2) False: do not pick the largest connected component
__C.fine.pick_largest_cc = True

# Remove small connected component (cc) in segmentation
# Options:
# 1) 0: Disable
# 2) a numerical number larger than 0: the threshold size of connected component
__C.fine.remove_small_cc = 0

# partition type in the inference stage
# Options:
# 1) SIZE:    partition to blocks with specified size (unit: mm), set partition_size = [size_x, size_y, size_z]
# 2) DISABLE: no partition
__C.fine.partition_type = 'SIZE'

# if partition type = 'SIZE', set the partition size (unit: mm).
# it is recommended to set this value as the same with the physical cropping size in the training phase
__C.fine.partition_size = [51.2, 51.2, 51.2]

# the moving stride of the partition window. If set it as the same with the partition size, there will be no overlap
# between the partition windows. Otherwise, the value of the overlapped area will be averaged.
# it is recommended to set this value as 1/4 of the partition size in order to avoid the apparent in-consistence between
# different partition window.
__C.fine.partition_stride = [51.2, 51.2, 51.2]