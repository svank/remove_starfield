import os
import warnings

from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import pytest

import remove_starfield


@pytest.fixture(scope='session')
def starfield():
    return _calc_starfield(.1)


def _calc_starfield(target_mem_usage, shuffle=True):
    dir = remove_starfield.utils.test_data_path(
        'WISPR_files_preprocessed_quarter_size_L3')
    files = sorted(os.listdir(dir))
    files = [os.path.join(dir, file) for file in files]
    
    with warnings.catch_warnings():
        warnings.filterwarnings(
                action='ignore', message=".*'BLANK' keyword.*")
        warnings.filterwarnings(
                action='ignore', message=".*datfix.*")
        
        starfield = remove_starfield.build_starfield_estimate(
                files, attribution=True, frame_count=True,
                dec_bounds=(-19, 24), ra_bounds=(149, 188),
                map_scale=.2, target_mem_usage=target_mem_usage,
                shuffle=shuffle,
                reducer=remove_starfield.reducers.GaussianReducer(min_size=40))
    
    return starfield


@pytest.mark.array_compare(file_format='fits', atol=1e-18)
def test_build_starfield_estimate(starfield):
    return np.stack(
        (starfield.starfield, starfield.attribution, starfield.frame_count))


@pytest.mark.mpl_image_compare(style="default")
def test_build_starfield_estimate_custom_wcs():
    # Make sure this runs and the custom starfield gets used, and also test
    # an oblique projection
    dir = remove_starfield.utils.test_data_path(
        'WISPR_files_tiny_covers_sky')
    files = sorted(os.listdir(dir))
    files = [os.path.join(dir, file) for file in files]
    
    map_scale = 1
    shape = [int(np.floor(50 / map_scale)), int(np.floor(360 / map_scale))]
    starfield_wcs = WCS(naxis=2)
    crpix = [shape[1] / 2 + .5, shape[0] / 2 + .5]
    starfield_wcs.wcs.crpix = crpix
    starfield_wcs.wcs.crval = 90, 30
    starfield_wcs.wcs.cdelt = map_scale, map_scale
    starfield_wcs.wcs.ctype = 'RA---CAR', 'DEC--CAR'
    starfield_wcs.wcs.cunit = 'deg', 'deg'
    starfield_wcs.array_shape = shape
    
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore', message=".*'BLANK' keyword.*")
        warnings.filterwarnings(
            action='ignore', message=".*datfix.*")
        
        starfield = remove_starfield.build_starfield_estimate(
            files, attribution=True, frame_count=True,
            target_mem_usage=.1,
            shuffle=True, starfield_wcs=starfield_wcs,
            reducer=remove_starfield.reducers.MeanReducer(min_size=1))
        
        assert starfield.wcs is starfield_wcs
        assert list(starfield.wcs.wcs.cdelt) == list(starfield_wcs.wcs.cdelt)
        assert list(starfield.wcs.array_shape) == list(starfield_wcs.array_shape)
        assert list(starfield.starfield.shape) == shape
    
    fig, ax = plt.subplots(1, 1)
    starfield.plot_frame_count(ax, grid=True)
    return fig


def test_build_starfield_estimate_multiple_chunks(starfield):
    starfield2 = _calc_starfield(.01)
    
    np.testing.assert_equal(starfield.starfield, starfield2.starfield)
    np.testing.assert_equal(starfield.attribution, starfield2.attribution)
    np.testing.assert_equal(starfield.frame_count, starfield2.frame_count)
    assert str(starfield.wcs) == str(starfield2.wcs)


def test_build_starfield_estimate_shuffle(starfield):
    # Verify that the shuffling of the input images (to balance the
    # multiprocessing) doesn't affect the outputs
    starfield2 = _calc_starfield(.1, shuffle=False)
    
    np.testing.assert_equal(starfield.starfield, starfield2.starfield)
    np.testing.assert_equal(starfield.attribution, starfield2.attribution)
    np.testing.assert_equal(starfield.frame_count, starfield2.frame_count)
    assert str(starfield.wcs) == str(starfield2.wcs)


def test_build_starfield_estimate_wcs(starfield):
    assert np.all(starfield.wcs.wcs.crval == (180, 0))
    assert np.all(starfield.wcs.wcs.crpix == (156.5, 96.5))
    assert np.all(starfield.wcs.wcs.cdelt == (.2, .2))
    assert list(starfield.wcs.wcs.ctype) == ['RA---CAR', 'DEC--CAR']

