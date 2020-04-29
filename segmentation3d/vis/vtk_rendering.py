import numpy as np
import pandas as pd
import SimpleITK as sitk
import vtk


def get_color_dict(color_config_file):
  """
  Get color dictionary from color configuration file.
  """
  assert color_config_file.endswith('.csv')
  
  df = pd.read_csv(color_config_file)
  color_dict = {}
  for idx in range(len(df)):
    color = df.loc[idx]
    color_dict[idx] = [color['R'] / 255, color['G'] / 255, color['B'] / 255, 1.0]
  return color_dict


def get_camera(image):
  """
  Get camera according to image.
  """
  assert isinstance(image, sitk.Image)
  
  size = image.GetSize()
  camera = vtk.vtkCamera()
  camera.SetViewUp(0, 0, 1)
  
  focal_point = image.TransformIndexToPhysicalPoint([size[0] // 2, size[1] // 2, size[2] // 2])
  camera.SetFocalPoint(focal_point)
  
  position = image.TransformIndexToPhysicalPoint([size[0] // 2, -(size[1] + size[1] // 2), size[2] // 2])
  camera.SetPosition(position)
  
  camera.ComputeViewPlaneNormal()
  return camera


def vtk_surface_rendering(image, color_dict, window_size, save_figure_path=None, interact=False):
  """
  Surface rendering using VTK.
  """
  assert isinstance(image, sitk.Image)
  image_npy = sitk.GetArrayFromImage(image)
  image_npy.astype(np.uint8)
  size, spacing = image.GetSize(), image.GetSpacing()

  # Import raw data and set parameters
  importer = vtk.vtkImageImport()
  image_string = image_npy.tostring()
  importer.CopyImportVoidPointer(image_string, len(image_string))
  importer.SetDataScalarTypeToUnsignedChar()
  importer.SetDataSpacing(spacing)
  importer.SetNumberOfScalarComponents(1)
  importer.SetDataExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)
  importer.SetWholeExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

  # Extract the surface using marching cubes
  num_foreground_classes = len(color_dict.keys())
  extractor = vtk.vtkDiscreteMarchingCubes()
  extractor.SetInputConnection(importer.GetOutputPort())
  extractor.GenerateValues(num_foreground_classes, 1, num_foreground_classes)

  # Create the look up table
  clt = vtk.vtkLookupTable()
  clt.SetNumberOfTableValues(2)
  clt.Build()
  for idx in range(1, num_foreground_classes):
    color = color_dict[idx]
    clt.SetTableValue(idx - 1, color[0], color[1], color[2], color[3])

  mapper = vtk.vtkPolyDataMapper()
  mapper.SetInputConnection(extractor.GetOutputPort())
  mapper.ScalarVisibilityOn()
  mapper.SetScalarRange(0, 4)  # I still do not konw why only 4 works.
  mapper.SetLookupTable(clt)
  
  actor = vtk.vtkActor()
  actor.SetMapper(mapper)
  
  renderer = vtk.vtkRenderer()
  renderer.AddActor(actor)
  renderer.SetActiveCamera(get_camera(image))
  
  render_win = vtk.vtkRenderWindow()
  render_win.SetSize(window_size[0], window_size[1])
  render_win.AddRenderer(renderer)
  
  if interact:
    render_interactor = vtk.vtkRenderWindowInteractor()
    render_interactor.SetRenderWindow(render_win)
    render_interactor.Initialize()
    render_interactor.Start()

  # screen shot
  if save_figure_path is not None:
    w2if = vtk.vtkWindowToImageFilter()
    w2if.SetInput(render_win)
    w2if.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(save_figure_path)
    writer.SetInputData(w2if.GetOutput())
    writer.Write()