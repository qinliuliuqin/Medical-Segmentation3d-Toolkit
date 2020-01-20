import numpy as np
import SimpleITK as sitk

from segmentation3d.dataloader.image_tools import normalize_image, percentiles


class FixedNormalizer(object):
  """
  use fixed mean and stddev to normalize image intensities
  intensity = (intensity - mean) / stddev
  if clip is enabled:
      intensity = np.clip((intensity - mean) / stddev, -1, 1)
  """

  def __init__(self, mean, stddev, clip=True):
    """ constructor """
    assert stddev > 0, 'stddev must be positive'
    assert isinstance(clip, bool), 'clip must be a boolean'
    self.mean = mean
    self.stddev = stddev
    self.clip = clip

  def __call__(self, image):
    """ normalize image """
    if isinstance(image, sitk.Image):
      return normalize_image(image, self.mean, self.stddev, self.clip)

    elif isinstance(image, (list, tuple)):
      for idx, im in enumerate(image):
        assert isinstance(im, sitk.Image)
        image[idx] = normalize_image(im, self.mean, self.stddev, self.clip)
      return image

    else:
      raise ValueError('Unknown type of input. Normalizer only supports Image3d or Image3d list/tuple')

  def to_dict(self):
    """ convert parameters to dictionary """
    obj = {'type': 0, 'mean': self.mean, 'stddev': self.stddev, 'clip': self.clip}
    return obj


class AdaptiveNormalizer(object):
  """
  use the minimum and maximum percentiles to normalize image intensities
  """

  def __init__(self, min_p=1, max_p=99, clip=True, min_rand=0, max_rand=0):
    """
    constructor
    :param min_p: percentile for computing minimum value
    :param max_p: percentile for computing maximum value
    :param clip: whether to clip the intensity between min and max
    :param min_rand: the random perturbation (%) of minimum value (0-1)
    :param max_rand: the random perturbation (%) of maximum value (0-1)
    """
    assert 100 >= min_p >= 0, 'min_p must be between 0 and 100'
    assert 100 >= max_p >= 0, 'max_p must be between 0 and 100'
    assert max_p > min_p, 'max_p must be > min_p'
    assert 1 >= min_rand >= 0, 'min_rand must be between 0 and 1'
    assert 1 >= max_rand >= 0, 'max_rand must be between 0 and 1'
    assert isinstance(clip, bool), 'clip must be a boolean'
    self.min_p = min_p
    self.max_p = max_p
    self.clip = clip
    self.min_rand = min_rand
    self.max_rand = max_rand

  def normalize(self, single_image):
    """ Normalize a given image """
    assert isinstance(single_image, sitk.Image), 'image must be an image3d object'
    normalize_min, normalize_max = percentiles(single_image, [self.min_p, self.max_p])

    if self.min_rand > 0:
      offset = np.abs(normalize_min) * self.min_rand
      offset = np.random.uniform(-offset, offset)
      normalize_min += offset

    if self.max_rand > 0:
      offset = np.abs(normalize_max) * self.max_rand
      offset = np.random.uniform(-offset, offset)
      normalize_max += offset

    normalize_mean = (normalize_min + normalize_max) / 2.0
    normalize_stddev = max(1e-6, np.abs(normalize_max - normalize_min) / 2.0)

    return normalize_image(single_image, normalize_mean, normalize_stddev, clip=self.clip)

  def __call__(self, image):
    """ normalize image """
    if isinstance(image, sitk.Image):
      return self.normalize(image)

    elif isinstance(image, (list, tuple)):
      for idx, im in enumerate(image):
        assert isinstance(im, sitk.Image)
        image[idx] = self.normalize(im)
      return image

    else:
      raise ValueError('Unknown type of input. Normalizer only supports Image3d or Image3d list/tuple')

  def to_dict(self):
    """ convert parameters to dictionary """
    obj = {'type': 1, 'min_p': self.min_p, 'max_p': self.max_p, 'clip': self.clip}
    return obj