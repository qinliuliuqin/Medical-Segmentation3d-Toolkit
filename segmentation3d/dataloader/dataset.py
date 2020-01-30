import numpy as np
import os
import SimpleITK as sitk
from torch.utils.data import Dataset

from segmentation3d.utils.file_io import readlines
from segmentation3d.dataloader.image_tools import select_random_voxels_in_multi_class_mask, crop_image, \
    convert_image_to_tensor, get_image_frame


def read_train_txt(imlist_file):
    """ read single-modality txt file
    :param imlist_file: image list file path
    :return: a list of image path list, list of segmentation paths
    """
    lines = readlines(imlist_file)
    num_cases = int(lines[0])

    if len(lines)-1 < num_cases * 2:
        raise ValueError('too few lines in imlist file')

    im_list, seg_list = [], []
    for i in range(num_cases):
        im_path, seg_path = lines[1 + i * 2], lines[2 + i * 2]
        assert os.path.isfile(im_path), 'image not exist: {}'.format(im_path)
        assert os.path.isfile(seg_path), 'mask not exist: {}'.format(seg_path)
        im_list.append([im_path])
        seg_list.append(seg_path)

    return im_list, seg_list


class SegmentationDataset(Dataset):
    """ training data set for volumetric segmentation """

    def __init__(self, imlist_file, num_classes, spacing, crop_size, sampling_method,
                 random_translation, interpolation, crop_normalizers):
        """ constructor
        :param imlist_file: image-segmentation list file
        :param num_classes: the number of classes
        :param spacing: the resolution, e.g., [1, 1, 1]
        :param crop_size: crop size, e.g., [96, 96, 96]
        :param sampling_method: 'GLOBAL', 'MASK'
        :param random_translation: random translation
        :param interpolation: 'LINEAR' for linear interpolation, 'NN' for nearest neighbor
        :param crop_normalizers: used to normalize the image crops, one for one image modality
        """
        if imlist_file.endswith('txt'):
            self.im_list, self.seg_list = read_train_txt(imlist_file)
        else:
            raise ValueError('imseg_list must be a txt file')

        self.num_classes = num_classes

        self.spacing = np.array(spacing, dtype=np.double)
        assert self.spacing.size == 3, 'only 3-element of spacing is supported'

        self.crop_size = np.array(crop_size, dtype=np.int32)
        assert self.crop_size.size == 3, 'only 3-element of crop size is supported'

        assert sampling_method in ('CENTER', 'GLOBAL', 'MASK', 'HYBRID'), \
            'sampling_method must be CENTER, GLOBAL, MASK or HYBRID'
        self.sampling_method = sampling_method

        self.random_translation = np.array(random_translation, dtype=np.double)
        assert self.random_translation.size == 3, 'Only 3-element of random translation is supported'

        assert interpolation in ('LINEAR', 'NN'), 'interpolation must either be a LINEAR or NN'
        self.interpolation = interpolation

        assert isinstance(crop_normalizers, list), 'crop normalizers must be a list'
        self.crop_normalizers = crop_normalizers

    def __len__(self):
        """ get the number of images in this data set """
        return len(self.im_list)

    def num_modality(self):
        """ get the number of input image modalities """
        return len(self.im_list[0])

    def global_sample(self, image):
        """ random sample a position in the image
        :param image: a SimpleITK image object which should be in the RAI coordinate
        :return: a world position in the RAI coordinate
        """
        assert isinstance(image, sitk.Image)

        origin = image.GetOrigin()
        im_size_mm = [image.GetSize()[idx] * image.GetSpacing()[idx] for idx in range(3)]
        crop_size_mm = self.crop_size * self.spacing

        sp = np.array(origin, dtype=np.double)
        for i in range(3):
            if im_size_mm[i] > crop_size_mm[i]:
                sp[i] = origin[i] + np.random.uniform(0, im_size_mm[i] - crop_size_mm[i])
        center = sp + crop_size_mm / 2
        return center

    def center_sample(self, image):
        """ return the world coordinate of the image center
        :param image: a image3d object
        :return: the image center in world coordinate
        """
        assert isinstance(image, sitk.Image)

        origin = image.GetOrigin()
        end_point_voxel = [int(image.GetSize()[idx] - 1) for idx in range(3)]
        end_point_world = image.TransformIndexToPhysicalPoint(end_point_voxel)

        center = np.array([(origin[idx] + end_point_world[idx]) / 2.0 for idx in range(3)], dtype=np.double)
        return center

    def __getitem__(self, index):
        """ get a training sample - image(s) and segmentation pair
        :param index:  the sample index
        :return cropped image, cropped mask, crop frame, case name
        """
        image_paths, seg_path = self.im_list[index], self.seg_list[index]

        case_name = os.path.basename(os.path.dirname(image_paths[0]))
        case_name += '_' + os.path.basename(image_paths[0])

        # image IO
        images = []
        for image_path in image_paths:
            image = sitk.ReadImage(image_path)
            images.append(image)

        seg = sitk.ReadImage(seg_path)

        # sampling a crop center
        if self.sampling_method == 'CENTER':
            center = self.center_sample(seg)

        elif self.sampling_method == 'GLOBAL':
            center = self.global_sample(seg)

        elif self.sampling_method == 'MASK':
            centers = select_random_voxels_in_multi_class_mask(seg, 1, np.random.randint(1, self.num_classes))
            if len(centers) > 0:
                center = seg.TransformIndexToPhysicalPoint([int(centers[0][idx]) for idx in range(3)])
            else:  # if no segmentation
                center = self.global_sample(seg)

        elif self.sampling_method == 'HYBRID':
            if index % 2:
                center = self.global_sample(seg)
            else:
                centers = select_random_voxels_in_multi_class_mask(seg, 1, np.random.randint(1, self.num_classes))
                if len(centers) > 0:
                    center = seg.TransformIndexToPhysicalPoint([int(centers[0][idx]) for idx in range(3)])
                else:  # if no segmentation
                    center = self.global_sample(seg)

        else:
            raise ValueError('Only CENTER, GLOBAL, MASK and HYBRID are supported as sampling methods')

        # random translation
        center += np.random.uniform(-self.random_translation, self.random_translation, size=[3])

        # sample a crop from image and normalize it
        for idx in range(len(images)):
            images[idx] = crop_image(images[idx], center, self.crop_size, self.spacing, self.interpolation)

            if self.crop_normalizers[idx] is not None:
                images[idx] = self.crop_normalizers[idx](images[idx])

        seg = crop_image(seg, center, self.crop_size, self.spacing, 'NN')

        # image frame
        frame = get_image_frame(seg)

        # convert to tensors
        im = convert_image_to_tensor(images)
        seg = convert_image_to_tensor(seg)

        return im, seg, frame, case_name