import SimpleITK as sitk

from segmentation3d.vis.vtk_rendering import get_color_dict, vtk_surface_rendering

def test_vtk_surface_render():
    # Path to the .mha file
    seg_path = "/mnt/projects/CT_Dental/results/model_0305_2020/Pre_Post_Facial_Data-Ma/n03_orginImg_post/seg.mha"
    label = sitk.ReadImage(seg_path)

    color_config_path = "/home/ql/projects/Medical-Segmentation3d-Toolkit/segmentation3d/vis/color_config.csv"
    color_dict = get_color_dict(color_config_path)

    figure_save_path = "/mnt/projects/CT_Dental/results/model_0305_2020/Pre_Post_Facial_Data-Ma/n03_orginImg_post.png"
    vtk_surface_rendering(label, color_dict, [800, 800], figure_save_path)


if __name__ == '__main__':

    test_vtk_surface_render()