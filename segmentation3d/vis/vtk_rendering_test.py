import os
import SimpleITK as sitk

from segmentation3d.vis.vtk_rendering import get_color_dict, vtk_surface_rendering


def test_vtk_surface_render():
    color_config_path = "/home/ql/projects/Medical-Segmentation3d-Toolkit/segmentation3d/vis/color_config.csv"
    color_dict = get_color_dict(color_config_path)

    mask_folder = '/mnt/projects/CT_Dental/results/model_0305_2020/Pre_Post_Facial_Data-Ma/'
    folders = os.listdir(mask_folder)
    for folder in folders:
        if os.path.isdir(os.path.join(mask_folder, folder)):
            print(folder)
            mask_path = os.path.join(mask_folder, folder, 'seg.mha')
            mask = sitk.ReadImage(mask_path)

            figure_save_path = os.path.join(mask_folder, '{}.png'.format(folder))
            vtk_surface_rendering(mask, color_dict, [800, 800], figure_save_path)


if __name__ == '__main__':

    test_vtk_surface_render()