import os
import warnings

from astropy.wcs import WCS
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


@pytest.mark.array_compare(file_format='fits')
def test_build_starfield_estimate(starfield):
    return np.stack(
        (starfield.starfield, starfield.attribution, starfield.frame_count))


def test_build_starfield_estimate_custom_wcs():
    # Just make sure this runs and the custom starfield gets used
    dir = remove_starfield.utils.test_data_path(
        'WISPR_files_preprocessed_quarter_size_L3')
    files = sorted(os.listdir(dir))
    files = [os.path.join(dir, file) for file in files]
    
    shape = [int(180 / 1), int(360 / 1)]
    starfield_wcs = WCS(naxis=2)
    # n.b. it seems the RA wrap point is chosen so there's 180 degrees
    # included on either side of crpix
    crpix = [shape[1] / 2 + .5, shape[0] / 2 + .5]
    starfield_wcs.wcs.crpix = crpix
    starfield_wcs.wcs.crval = 180, 0
    starfield_wcs.wcs.cdelt = 1, 1
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
            dec_bounds=(-19, 24), ra_bounds=(149, 188),
            map_scale=.2, target_mem_usage=.1,
            shuffle=True, starfield_wcs=starfield_wcs,
            reducer=remove_starfield.reducers.GaussianReducer(min_size=40))
        
        assert starfield.wcs is starfield_wcs
        assert list(starfield.wcs.wcs.cdelt) == list(starfield_wcs.wcs.cdelt)
        assert list(starfield.wcs.array_shape) == list(starfield_wcs.array_shape)
        assert list(starfield.starfield.shape) == shape


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

