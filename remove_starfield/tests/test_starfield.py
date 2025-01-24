from copy import deepcopy

from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import pytest

import remove_starfield
from .test_core import starfield


def test_starfield_save_load(starfield, tmp_path):
    save_path = tmp_path / "starfield"
    starfield.save(save_path)
    starfield2 = remove_starfield.Starfield.load(save_path)
    
    np.testing.assert_equal(starfield.starfield, starfield2.starfield)
    np.testing.assert_equal(starfield.attribution, starfield2.attribution)
    np.testing.assert_equal(starfield.frame_count, starfield2.frame_count)
    assert str(starfield.wcs) == str(starfield2.wcs)


@pytest.mark.array_compare(file_format='fits', atol=1e-18)
def test_starfield_subtract_from_image(starfield):
    test_file = remove_starfield.utils.test_data_path(
        'WISPR_files_preprocessed_quarter_size_L3',
        'psp_L3_wispr_20221206T223017_V1_1221.fits')
    
    subtracted = starfield.subtract_from_image(test_file)
    
    assert subtracted.source_file == test_file
    
    return np.stack((
        subtracted.source_data, subtracted.blurred_data,
        subtracted.starfield_sample, subtracted.subtracted))


@pytest.mark.array_compare(file_format='fits', atol=1e-18,
                           filename='test_starfield_subtract_from_image.fits')
def test_starfield_subtract_from_image_3D(starfield):
    # Ensure we handle multidimensional starfields and images
    test_file = remove_starfield.utils.test_data_path(
        'WISPR_files_preprocessed_quarter_size_L3',
        'psp_L3_wispr_20221206T223017_V1_1221.fits')
    
    starfield = deepcopy(starfield)
    starfield.starfield = np.stack((starfield.starfield, starfield.starfield))
    
    data, hdr = fits.getdata(test_file, header=True)
    wcs = WCS(hdr, key='A')
    
    class ImageHolderLike:
        def __init__(self, data, wcs, meta):
            self.data = data
            self.wcs = wcs
            self.meta = meta
    
    stacked_input = ImageHolderLike(np.stack((data, data)), wcs, hdr)
    
    subtracted = starfield.subtract_from_image(stacked_input)
    
    np.testing.assert_array_equal(
        subtracted.source_data[0], subtracted.source_data[1])
    np.testing.assert_array_equal(
        subtracted.blurred_data[0], subtracted.blurred_data[1])
    np.testing.assert_array_equal(
        subtracted.starfield_sample[0], subtracted.starfield_sample[1])
    np.testing.assert_array_equal(
        subtracted.subtracted[0], subtracted.subtracted[1])
    
    return np.stack((
        subtracted.source_data[0], subtracted.blurred_data[0],
        subtracted.starfield_sample[0], subtracted.subtracted[0]))


def test_starfield_subtract_from_image_ImageHolder(starfield):
    # Ensure we get the same result when passing in the
    # image-to-be-subtracted-from as a file path or an ImageHolder
    test_file = remove_starfield.utils.test_data_path(
        'WISPR_files_preprocessed_quarter_size_L3',
        'psp_L3_wispr_20221206T223017_V1_1221.fits')
    
    data, hdr = fits.getdata(test_file, header=True)
    wcs = WCS(hdr, key='A')
    
    class ImageHolderLike:
        def __init__(self, data, wcs, meta):
            self.data = data
            self.wcs = wcs
            self.meta = meta
    
    class ImageProcessorCantLoad(remove_starfield.ImageProcessor):
        def load_image(self, filename):
            raise RuntimeError("This should not run")
    
    subtracted = starfield.subtract_from_image(
        ImageHolderLike(data, wcs, hdr),
        processor=ImageProcessorCantLoad())
    
    subtracted_from_string = starfield.subtract_from_image(test_file)
    
    np.testing.assert_array_equal(
        subtracted.subtracted, subtracted_from_string.subtracted)


@pytest.mark.mpl_image_compare
def test_starfield_plot(starfield):
    fig, ax = plt.subplots(1, 1)
    starfield.plot(ax, grid=True)
    return fig


@pytest.mark.mpl_image_compare
def test_starfield_plot_frame_count(starfield):
    fig, ax = plt.subplots(1, 1)
    starfield.plot_frame_count(ax, grid=False)
    return fig


@pytest.mark.mpl_image_compare
def test_starfield_plot_attribution(starfield):
    fig, ax = plt.subplots(1, 1)
    starfield.plot_attribution(ax, grid=False)
    return fig