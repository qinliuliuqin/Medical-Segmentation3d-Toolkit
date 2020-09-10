import os
import pandas as pd


# generate dataset
train_idx = list(range(1, 62))
test_idx = list(range(62, 83))

image_folder = '/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/images'
label_folder = '/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/NIH_pancreas/label'

#image_folder = '/mnt/projects/Pancreas/label'
#label_folder = '/mnt/projects/Pancreas/label'

dataset_save_folder = '/shenlab/lab_stor6/qinliu/CT_Pancreas/dataset'
if not os.path.isdir(dataset_save_folder):
    os.makedirs(dataset_save_folder)

image_paths = os.listdir(image_folder)
image_paths.sort()

label_paths = os.listdir(label_folder)
label_paths.sort()

assert len(image_paths) == len(label_paths) == 82

train_image_paths = image_paths[:62]
train_label_paths = label_paths[:62]

test_image_paths = image_paths[62:]
test_label_paths = label_paths[62:]

train_content = []
for i in range(len(train_image_paths)):
    image_path = os.path.join(image_folder, train_image_paths[i])
    image_name = image_path.split('/')[-1]
    mask_path = os.path.join(label_folder, train_label_paths[i])
    train_content.append([image_name, image_path, mask_path])
train_df = pd.DataFrame(train_content, columns=['image_name', 'image_path', 'mask_path'])
train_df.to_csv(os.path.join(dataset_save_folder, 'train.csv'), index=False)

test_content = []
for i in range(len(test_image_paths)):
    image_path = os.path.join(image_folder, test_image_paths[i])
    image_name = image_path.split('/')[-1]
    mask_path = os.path.join(label_folder, test_label_paths[i])
    test_content.append([image_name, image_path, mask_path])
test_df = pd.DataFrame(test_content, columns=['image_name', 'image_path', 'mask_path'])
test_df.to_csv(os.path.join(dataset_save_folder, 'test.csv'), index=False)
