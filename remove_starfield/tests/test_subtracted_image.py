from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import pytest

import remove_starfield
from .test_core import starfield


@pytest.fixture(scope='session')
def subtracted_image(starfield):
    test_file = remove_starfield.utils.test_data_path(
        'WISPR_files_preprocessed_quarter_size_L3',
        'psp_L3_wispr_20221206T223017_V1_1221.fits')
    
    return starfield.subtract_from_image(test_file)


@pytest.mark.mpl_image_compare
def test_subtracted_image_plot_comparison(subtracted_image):
    subtracted_image.plot_comparison()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_subtracted_image_plot_comparison_bwr(subtracted_image):
    subtracted_image.plot_comparison(bwr=True, vmin=-3e-12, vmax=3e-12)
    return plt.gcf()